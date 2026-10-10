using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;

namespace KasagoCalendar.Utility.LLMUtility;

/// <summary>SSE 行的语义分类</summary>
internal enum SseLineKind
{
    /// <summary>与本模块无关的行（空行、心跳注释、event:/id:/retry: 等字段）</summary>
    Ignore,
    /// <summary>携带数据载荷的 data 行</summary>
    Data,
    /// <summary>流结束标记 [DONE]</summary>
    Done,
}

/// <summary>
/// Server-Sent Events 解析器。
/// 统一处理空行、注释/心跳行（以 ':' 开头）、其他字段行以及结束标记 [DONE]，
/// 使上层只需关心 JSON 载荷，避免单个非 JSON 行导致整轮请求失败。
/// </summary>
internal static class SseReader
{
    const string DataPrefix = "data:";
    const string DonePayload = "[DONE]";

    /// <summary>解析单行内容</summary>
    public static SseLineKind Parse(string? rawLine, out string payload)
    {
        payload = string.Empty;
        if (string.IsNullOrWhiteSpace(rawLine)) return SseLineKind.Ignore;

        string line = rawLine.Trim();
        if (!line.StartsWith(DataPrefix, StringComparison.Ordinal)) return SseLineKind.Ignore;

        string data = line[DataPrefix.Length..].Trim();
        if (data.Length == 0) return SseLineKind.Ignore;
        if (string.Equals(data, DonePayload, StringComparison.Ordinal)) return SseLineKind.Done;

        payload = data;
        return SseLineKind.Data;
    }

    /// <summary>逐行读取事件流，产出 data 载荷；遇到 [DONE] 或流结束即停止</summary>
    public static async IAsyncEnumerable<string> ReadPayloadsAsync(
        StreamReader reader,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        while (true)
        {
            cancellationToken.ThrowIfCancellationRequested();

            string? line = await reader.ReadLineAsync(cancellationToken);
            if (line is null) yield break;

            switch (Parse(line, out string payload))
            {
                case SseLineKind.Data:
                    yield return payload;
                    break;
                case SseLineKind.Done:
                    yield break;
            }
        }
    }
}
