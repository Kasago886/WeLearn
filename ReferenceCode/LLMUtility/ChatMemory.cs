using System;
using System.Collections.Generic;
using System.Text.Json;

namespace KasagoCalendar.Utility.LLMUtility;

/// <summary>
/// 对话记忆：由若干轮次（<see cref="ChatRound"/>）组成，可直接展开为请求体所需的 messages 数组。
/// </summary>
public class Memory
{
    /// <summary>轮次列表（对外公开，供上下文压缩等上层逻辑使用）</summary>
    public readonly List<ChatRound> roundList = new();

    /// <summary>跨线程操作记忆（如后台上下文压缩）时请锁定此对象</summary>
    public object SyncRoot { get; } = new();

    /// <summary>轮次数量</summary>
    public int RoundCount => roundList.Count;

    /// <summary>最后一轮；记忆为空时抛出异常</summary>
    public ChatRound LastRound => roundList.Count > 0
        ? roundList[roundList.Count - 1]
        : throw new InvalidOperationException("Memory is empty");

    /// <summary>全部消息，按时间顺序展开</summary>
    public List<Dictionary<string, object?>> MessageList
    {
        get
        {
            var messages = new List<Dictionary<string, object?>>();
            foreach (var round in roundList) messages.AddRange(round.messageList);
            return messages;
        }
    }

    /// <summary>全部消息的 JSON 表示（日志/调试用）</summary>
    public string SerializedMessageList => JsonSerializer.Serialize(MessageList);

    /// <summary>消息总条数</summary>
    public int MessageCount
    {
        get
        {
            int count = 0;
            foreach (var round in roundList) count += round.Count;
            return count;
        }
    }

    /// <summary>深拷贝（轮次逐一深拷贝，修改副本不影响原记忆）</summary>
    public Memory CloneMemory
    {
        get
        {
            var clone = new Memory();
            foreach (var round in roundList) clone.roundList.Add(round.CloneRound);
            return clone;
        }
    }

    /// <summary>向最后一轮追加消息</summary>
    public void AddMessageToLastRound(Dictionary<string, object?> messageDict) => LastRound.AddMessage(messageDict);

    /// <summary>向最后一轮插入消息</summary>
    public void InsertMessageToLastRound(int index, Dictionary<string, object?> messageDict)
        => LastRound.InsertMessage(index, messageDict);

    /// <summary>在末尾新增一轮</summary>
    public ChatRound AddNewRound() => InsertNewRound(roundList.Count);

    /// <summary>在指定位置新增一轮（索引会被夹紧到合法范围）</summary>
    public ChatRound InsertNewRound(int index)
    {
        var round = new ChatRound();
        roundList.Insert(Math.Clamp(index, 0, roundList.Count), round);
        return round;
    }

    /// <summary>移除指定范围的轮次；参数越界时不做任何操作</summary>
    public void RemoveRounds(int startRoundIndex, int roundCount)
    {
        if (startRoundIndex < 0 || roundCount <= 0) return;
        if (startRoundIndex + roundCount > roundList.Count) return;

        roundList.RemoveRange(startRoundIndex, roundCount);
    }

    /// <summary>用一条消息替换指定范围的轮次（上下文压缩用）</summary>
    public void ReplaceRoundsWithMessage(int startRoundIndex, int roundCount, Dictionary<string, object?> message)
    {
        RemoveRounds(startRoundIndex, roundCount);
        InsertNewRound(startRoundIndex).AddMessage(message);
    }
}
