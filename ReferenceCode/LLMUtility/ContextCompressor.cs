using System;
using System.Collections.Generic;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace KasagoCalendar.Utility.LLMUtility;

/// <summary>压缩摘要的解析结果</summary>
public enum SummaryReadResult
{
    /// <summary>解析成功，可以应用</summary>
    Ok,
    /// <summary>模型明确表示没有可保留的信息（空摘要），无需重试</summary>
    Empty,
    /// <summary>格式不符（含解析异常），可重试</summary>
    Invalid,
}

/// <summary>
/// 上下文压缩：把记忆中最早的一段「非 system」轮次交给 LLM 生成摘要，并在校验后替换回记忆。
/// <para>
/// 这里集中处理三件容易出错的事，使上层（KasaAlpha）无需再关心：
/// 1. 范围计算：跳过开头的 system 轮次，并限制单次压缩的轮次数；
/// 2. 线程安全：读写记忆一律持有 <see cref="Memory.SyncRoot"/> 锁，避免与主对话并发改写 <c>roundList</c>；
/// 3. 过期校验：压缩期间记忆会持续新增轮次，只有待压缩范围的内容未被改动时才允许替换，
///    因此“压缩完成前用户又说了几句”不会再导致整次压缩被丢弃。
/// </para>
/// </summary>
public static class ContextCompressor
{
    /// <summary>一次压缩任务的计划</summary>
    public readonly struct Plan
    {
        /// <summary>待压缩范围在记忆中的起始轮次下标</summary>
        public int StartRoundIndex { get; init; }

        /// <summary>待压缩的轮次数</summary>
        public int RoundCount { get; init; }

        /// <summary>发给压缩模型的完整输入（complete_messages + messages_to_compress）</summary>
        public string UserInput { get; init; }

        /// <summary>待压缩范围的内容指纹，用于应用前校验范围是否仍与计划一致</summary>
        public string Signature { get; init; }
    }

    /// <summary>摘要回复的 JSON 结构</summary>
    sealed class CompressResponse
    {
        [JsonPropertyName("compressed_context")] public string? CompressedContext { get; set; }
    }

    /// <summary>
    /// 生成压缩计划。
    /// 当前轮次不足、或去掉 system 轮次后没有可压缩内容时返回 null。
    /// </summary>
    /// <param name="memory">目标记忆</param>
    /// <param name="thresholdRoundCount">触发压缩的轮次数阈值</param>
    /// <param name="compressRoundCount">单次最多压缩的轮次数</param>
    public static Plan? TryCreatePlan(Memory memory, int thresholdRoundCount, int compressRoundCount)
    {
        if (memory is null || compressRoundCount <= 0) return null;

        lock (memory.SyncRoot)
        {
            if (memory.RoundCount < thresholdRoundCount) return null;

            int start = CountLeadingSystemRounds(memory);
            int count = Math.Min(compressRoundCount, memory.RoundCount - start);
            if (count <= 0) return null;

            List<Dictionary<string, object?>> range = CollectMessages(memory, start, count);

            return new Plan
            {
                StartRoundIndex = start,
                RoundCount = count,
                Signature = ComputeSignature(range),
                UserInput = JsonSerializer.Serialize(new
                {
                    complete_messages = memory.MessageList,
                    messages_to_compress = range,
                }),
            };
        }
    }

    /// <summary>
    /// 从模型回复中解析摘要。
    /// 兼容严格的 <c>{"compressed_context":"..."}</c>、markdown 代码块包裹以及前后夹带说明文字的情况。
    /// </summary>
    public static SummaryReadResult TryReadSummary(string? content, out string summary)
    {
        summary = string.Empty;
        if (string.IsNullOrWhiteSpace(content)) return SummaryReadResult.Invalid;

        SummaryReadResult result = ReadJson(content, out summary);
        if (result != SummaryReadResult.Invalid) return result;

        // 退化为截取最外层 {} 再解析（应对 ```json 包裹或前后说明文字）
        int start = content.IndexOf('{');
        int end = content.LastIndexOf('}');
        if (start >= 0 && end > start)
        {
            result = ReadJson(content[start..(end + 1)], out summary);
            if (result != SummaryReadResult.Invalid) return result;
        }

        summary = string.Empty;
        return SummaryReadResult.Invalid;
    }

    /// <summary>
    /// 把摘要写回记忆（替换掉计划中的轮次）。
    /// 范围越界或内容已变化（过期）时返回 false 且不修改记忆。
    /// </summary>
    public static bool TryApply(Memory memory, Plan plan, string summary)
    {
        if (memory is null || string.IsNullOrWhiteSpace(summary)) return false;

        lock (memory.SyncRoot)
        {
            if (plan.StartRoundIndex < 0 || plan.RoundCount <= 0) return false;
            if (plan.StartRoundIndex + plan.RoundCount > memory.RoundCount) return false;

            // 校验待压缩范围是否仍是计划时的内容（期间可能已被另一次压缩替换）
            List<Dictionary<string, object?>> range = CollectMessages(memory, plan.StartRoundIndex, plan.RoundCount);
            if (!string.Equals(ComputeSignature(range), plan.Signature, StringComparison.Ordinal)) return false;

            memory.RemoveRounds(plan.StartRoundIndex, plan.RoundCount);
            memory.InsertNewRound(plan.StartRoundIndex).AddMessage(BuildSummaryMessage(summary));
            return true;
        }
    }

    /// <summary>
    /// 构造写入记忆的摘要消息。
    /// 用 user 角色承载背景信息，并明确标注它不是新的用户请求，避免模型误判。
    /// </summary>
    public static Dictionary<string, object?> BuildSummaryMessage(string summary) => new()
    {
        [ChatMessage.RoleKey] = RoleOption.User.ToRoleString(),
        [ChatMessage.ContentKey] =
            "[历史对话压缩摘要]\n" +
            "以下内容是对更早对话的压缩摘要，仅作为背景信息，不是新的用户请求，也不是 assistant 的回复。\n" +
            "若与后续消息冲突，以后续消息为准。\n\n" +
            "<summary>\n" + summary + "\n</summary>\n" +
            "[/历史对话压缩摘要]",
    };

    /// <summary>统计记忆开头连续的 system 轮次数量</summary>
    public static int CountLeadingSystemRounds(Memory memory)
    {
        int count = 0;
        while (count < memory.RoundCount && memory.roundList[count].IsSystemRound) count++;
        return count;
    }

    static List<Dictionary<string, object?>> CollectMessages(Memory memory, int startRoundIndex, int roundCount)
    {
        var messages = new List<Dictionary<string, object?>>();
        for (int i = startRoundIndex; i < startRoundIndex + roundCount; i++)
            messages.AddRange(memory.roundList[i].messageList);

        return messages;
    }

    /// <summary>解析一个 JSON 对象：结构不符返回 Invalid，摘要为空返回 Empty</summary>
    static SummaryReadResult ReadJson(string json, out string summary)
    {
        summary = string.Empty;

        CompressResponse? parsed;
        try
        {
            parsed = JsonSerializer.Deserialize<CompressResponse>(json);
        }
        catch (JsonException)
        {
            return SummaryReadResult.Invalid;
        }

        if (parsed?.CompressedContext is null) return SummaryReadResult.Invalid;

        string text = parsed.CompressedContext.Trim();
        if (text.Length == 0) return SummaryReadResult.Empty;

        summary = text;
        return SummaryReadResult.Ok;
    }

    static string ComputeSignature(IReadOnlyList<Dictionary<string, object?>> messages)
    {
        byte[] hash = SHA256.HashData(Encoding.UTF8.GetBytes(JsonSerializer.Serialize(messages)));
        return Convert.ToHexString(hash);
    }
}
