"""记忆持久化（T1.1/T1.3，FR-603、FR-410）。

把 :class:`~app.services.llm.memory.Memory` 与 ``conversation`` / ``chat_round`` / ``message``
三张表互转。持久化是"刷新后恢复到待回答的检查点"（FR-410）与"重开后继续对话"的基础。

**实现取舍**：保存时整体重写该会话的轮次（删除后重新插入）。
单用户、单会话的数据量是数十轮量级，整体重写的代码复杂度远低于差分更新，
而差分更新一旦有 bug 就是**静默的数据错位**——这类 bug 的代价比多写几毫秒 IO 高得多。
数据量真的上来了再换成追加式。
"""

import json
import uuid
from typing import Any

from sqlalchemy import delete, select

from app.contracts.schemas import ConversationKind
from app.infra.storage.db import Database
from app.infra.storage.models import ChatRoundRow, ConversationRow, MessageRow, utcnow
from app.services.llm.memory import (
    CONTENT_KEY,
    REASONING_KEY,
    ROLE_KEY,
    TOOL_CALLS_KEY,
    ChatRound,
    Memory,
    MessageEnvelope,
    ProtocolMessage,
)

TOOL_CALLS_STORAGE_KEY = "_tool_calls"
"""``tool_calls`` 在库中单独存列，读取时合并回消息字典。"""


class MemoryRepository:
    """会话记忆的读写。"""

    def __init__(self, db: Database) -> None:
        self._db = db

    # ------------------------------------------------------------------ #
    # 会话
    # ------------------------------------------------------------------ #

    def ensure_conversation(
        self,
        kind: ConversationKind,
        *,
        subject_id: str | None = None,
        conversation_id: str | None = None,
    ) -> str:
        """确保会话存在并返回其 id；同一主题同一用途只应有一条（FR-602）。"""
        with self._db.session() as session:
            if conversation_id is not None:
                row = session.get(ConversationRow, conversation_id)
                if row is not None:
                    return row.id

            if subject_id is not None:
                existing = session.scalar(
                    select(ConversationRow).where(
                        ConversationRow.subject_id == subject_id,
                        ConversationRow.kind == kind.value,
                    )
                )
                if existing is not None:
                    return existing.id

            conversation = ConversationRow(
                id=conversation_id or str(uuid.uuid4()),
                kind=kind.value,
                subject_id=subject_id,
            )
            session.add(conversation)
            return conversation.id

    def conversation_id_for(self, subject_id: str, kind: ConversationKind) -> str | None:
        with self._db.session() as session:
            return session.scalar(
                select(ConversationRow.id).where(
                    ConversationRow.subject_id == subject_id,
                    ConversationRow.kind == kind.value,
                )
            )

    def delete_conversation(self, conversation_id: str) -> None:
        """删除会话及其轮次与消息（外键级联）。"""
        with self._db.session() as session:
            session.execute(delete(ConversationRow).where(ConversationRow.id == conversation_id))

    # ------------------------------------------------------------------ #
    # 记忆
    # ------------------------------------------------------------------ #

    def save(self, conversation_id: str, memory: Memory) -> int:
        """整体重写会话记忆，返回写入的轮次数。"""
        rounds = [round_.clone() for round_ in memory.rounds]

        with self._db.session() as session:
            conversation = session.get(ConversationRow, conversation_id)
            if conversation is None:
                raise KeyError(f"会话不存在: {conversation_id}")

            # 先删旧轮次（外键级联删除消息），再按当前快照重建
            session.execute(delete(ChatRoundRow).where(ChatRoundRow.conversation_id == conversation_id))

            for index, round_ in enumerate(rounds):
                round_row = ChatRoundRow(conversation_id=conversation_id, round_index=index)
                session.add(round_row)
                session.flush()  # 需要 round_row.id

                for envelope in round_.messages:
                    session.add(
                        MessageRow(
                            round_id=round_row.id,
                            role=envelope.role or "",
                            content=envelope.content,
                            reasoning_content=_read_optional(envelope.data, REASONING_KEY),
                            tool_calls_json=_dump_tool_calls(envelope.data),
                            is_summary=envelope.is_summary,
                            token_count=envelope.token_count,
                        )
                    )

            conversation.round_count = len(rounds)
            session.add(conversation)

        return len(rounds)

    def load(self, conversation_id: str) -> Memory:
        """读取会话记忆；会话不存在时返回空记忆。"""
        with self._db.session() as session:
            round_rows = list(
                session.scalars(
                    select(ChatRoundRow)
                    .where(ChatRoundRow.conversation_id == conversation_id)
                    .order_by(ChatRoundRow.round_index)
                )
            )
            if not round_rows:
                return Memory()

            round_ids = [row.id for row in round_rows]
            message_rows = list(
                session.scalars(
                    select(MessageRow)
                    .where(MessageRow.round_id.in_(round_ids))
                    .order_by(MessageRow.round_id, MessageRow.id)
                )
            )

        grouped: dict[int, list[MessageEnvelope]] = {round_id: [] for round_id in round_ids}
        for row in message_rows:
            grouped[row.round_id].append(
                MessageEnvelope(
                    _to_protocol_message(row),
                    is_summary=row.is_summary,
                    token_count=row.token_count,
                )
            )

        return Memory([ChatRound(grouped[row.id]) for row in round_rows])

    def mark_compressed(self, conversation_id: str) -> None:
        """记录一次压缩发生的时间（FR-604 的可观测性）。"""
        with self._db.session() as session:
            conversation = session.get(ConversationRow, conversation_id)
            if conversation is not None:
                conversation.compressed_at = utcnow()
                session.add(conversation)


def _read_optional(message: ProtocolMessage, key: str) -> str | None:
    value = message.get(key)
    return value if isinstance(value, str) and value else None


def _dump_tool_calls(message: ProtocolMessage) -> str | None:
    tool_calls = message.get(TOOL_CALLS_KEY)
    if not tool_calls:
        return None
    return json.dumps(tool_calls, ensure_ascii=False)


def _to_protocol_message(row: MessageRow) -> ProtocolMessage:
    """把数据库行还原为协议消息；空的 reasoning / tool_calls 不写回，避免出现 null 字段。"""
    message: ProtocolMessage = {ROLE_KEY: row.role, CONTENT_KEY: row.content}

    if row.reasoning_content:
        message[REASONING_KEY] = row.reasoning_content

    if row.tool_calls_json:
        try:
            loaded: Any = json.loads(row.tool_calls_json)
        except json.JSONDecodeError:
            loaded = None
        if loaded:
            message[TOOL_CALLS_KEY] = loaded

    return message
