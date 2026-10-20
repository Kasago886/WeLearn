"""间隔复习排期（T4.1，FR-505、FR-506）。

**为什么单独成模块**：排期是纯计算（箱号 → 下次到期日），没有 IO，也没有依赖。
把它写进 ``ReviewService`` 会让"某天该复习哪几个节点"变成只能通过一次完整复习
会话才验证得到；抽成纯函数后，阶梯推进、封顶、失败重置都能直接断言。

**为什么用 Leitner 箱号而不是艾宾浩斯曲线**：箱号只有"通过了进一档 / 没通过回
第一档"两条规则，用户看得懂、算得出；而曲线需要按每次作答的精确间隔累计，
一旦中间漏了几次复习就失去意义。

**为什么到期日要落库而不是每次现算**：现算只能用"最近修改时间 + 间隔"，节点
改标题、改要点都会把最近修改时间推后，复习计划就会凭空推迟。``review_due_at``
只在真正复习通过/失败时写入，才代表"下一次该复习的时间"。

全部时间都是 **naive UTC**（与库中存储约定一致），调用方负责保持一致。
"""

from datetime import datetime, timedelta

REVIEW_INTERVALS_DAYS: tuple[int, ...] = (1, 3, 7, 21, 60)
"""Leitner 阶梯（天）：箱号 = 下标 + 1。

首档取 1 天：刚学会的内容第二天就回顾一次，遗忘最快的就是这一段；
之后每通过一次间隔拉长，封顶 60 天——再长就接近"不再复习"，对概念学习没有意义。
"""

FIRST_BOX = 1
MAX_BOX = len(REVIEW_INTERVALS_DAYS)


def interval_days(box: int | None) -> int:
    """某个箱号对应的间隔天数（越界按边界收敛）。"""
    index = min(max(box or FIRST_BOX, FIRST_BOX), MAX_BOX) - 1
    return REVIEW_INTERVALS_DAYS[index]


def schedule_after_mastery(now: datetime) -> tuple[int, datetime]:
    """首次掌握：进入第一档，第二天检查一次。"""
    box = FIRST_BOX
    return box, now + timedelta(days=interval_days(box))


def schedule_after_pass(box: int | None, now: datetime) -> tuple[int, datetime]:
    """复习通过：升一档（封顶最高档），到期日按新档位推后。"""
    new_box = min((box or 0) + 1, MAX_BOX)
    return new_box, now + timedelta(days=interval_days(new_box))


def schedule_after_fail(now: datetime) -> tuple[int, datetime]:
    """复习不通过：回到第一档。刚刚证明记不住了，再长的间隔都是自欺欺人。"""
    box = FIRST_BOX
    return box, now + timedelta(days=interval_days(box))


def is_due(due_at: datetime | None, now: datetime) -> bool:
    """是否已到期。没有排期的节点不算到期（是否该补排期由调用方决定）。"""
    return due_at is not None and due_at <= now
