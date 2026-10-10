using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json.Serialization;

namespace KasagoCalendar.Utility.LLMUtility;

/// <summary>工具注册表：LLM 可调用的工具声明，以及工具名到实现的映射</summary>
public class Tools
{
    /// <summary>工具声明（发往 LLM 的 function 定义，通常为匿名对象，结构与 OpenAI function 一致）</summary>
    public List<object> tools = new();

    /// <summary>工具名 → 实现</summary>
    public Dictionary<string, ToolFunc> toolCallMap = new();

    /// <summary>按名称查找工具实现，不存在时返回 null</summary>
    public ToolFunc? GetFuncByName(string name)
        => name is not null && toolCallMap.TryGetValue(name, out var func) ? func : null;
}

/// <summary>工具实现</summary>
public class ToolFunc
{
    /// <summary>执行体：入参为已解析的工具参数，返回可序列化结果</summary>
    public Func<Dictionary<string, object>, object> ResolveFunc { get; set; }

    public ToolFunc(Func<Dictionary<string, object>, object> resolveFunc) => ResolveFunc = resolveFunc;

    public object Invoke(Dictionary<string, object> args) => ResolveFunc.Invoke(args);
}

/// <summary>一次工具调用（流式场景下为按 index 合并后的完整调用）</summary>
public class ToolCallDelta
{
    [JsonPropertyName("index")] public int? Index { get; set; }
    [JsonPropertyName("id")] public string? ID { get; set; }
    [JsonPropertyName("type")] public string? Type { get; set; }
    [JsonPropertyName("function")] public FunctionCallDelta? Function { get; set; }
}

/// <summary>工具调用中的 function 字段</summary>
public class FunctionCallDelta
{
    [JsonPropertyName("name")] public string? Name { get; set; }
    [JsonPropertyName("arguments")] public string? Arguments { get; set; }
}

/// <summary>
/// 流式工具调用累积器。
/// 同一个 index 的工具调用会分多个数据块到达：id/name 通常只在首块出现，arguments 需要按顺序拼接。
/// </summary>
public sealed class ToolCallAccumulator
{
    readonly Dictionary<int, ToolCallDelta> byIndex = new();

    /// <summary>是否存在工具调用</summary>
    public bool HasAny => byIndex.Count > 0;

    /// <summary>合并一个数据块中携带的工具调用增量</summary>
    public void Merge(IEnumerable<ToolCallDelta>? deltas)
    {
        if (deltas is null) return;

        foreach (var delta in deltas)
        {
            // 部分服务端不下发 index，此时按已有数量顺延
            int index = delta.Index ?? byIndex.Count;

            if (!byIndex.TryGetValue(index, out var target))
            {
                target = new ToolCallDelta { Index = index };
                byIndex[index] = target;
            }

            if (!string.IsNullOrEmpty(delta.ID)) target.ID ??= delta.ID;
            if (!string.IsNullOrEmpty(delta.Type)) target.Type ??= delta.Type;

            if (delta.Function is null) continue;

            target.Function ??= new FunctionCallDelta();
            if (!string.IsNullOrEmpty(delta.Function.Name)) target.Function.Name ??= delta.Function.Name;
            if (!string.IsNullOrEmpty(delta.Function.Arguments)) target.Function.Arguments += delta.Function.Arguments;

            // 参数为空时也保持空串，避免回传的 assistant 消息出现 "arguments": null
            target.Function.Arguments ??= string.Empty;
        }
    }

    /// <summary>按 index 升序输出合并后的工具调用</summary>
    public List<ToolCallDelta> Build() => byIndex.Values.OrderBy(call => call.Index ?? 0).ToList();

    /// <summary>清空累积状态</summary>
    public void Clear() => byIndex.Clear();
}
