using System;
using System.Collections.Generic;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace KasagoCalendar.Utility.LLMUtility;

#region 请求模型

/// <summary>一次 chat/completions 调用所需的全部输入</summary>
public sealed class ChatCompletionRequest
{
    /// <summary>API 基础地址（例如 https://api.deepseek.com 或 https://host/v1）</summary>
    public required string BaseUrl { get; init; }

    /// <summary>API Key</summary>
    public required string ApiKey { get; init; }

    /// <summary>模型名</summary>
    public required string Model { get; init; }

    /// <summary>会话消息列表</summary>
    public required IReadOnlyList<Dictionary<string, object?>> Messages { get; init; }

    /// <summary>工具声明；为空时不向服务端下发 tools 字段</summary>
    public IReadOnlyList<object>? Tools { get; init; }

    /// <summary>是否开启思维链</summary>
    public bool Thinking { get; init; }
}

/// <summary>请求体构建</summary>
internal static class LlmRequestBuilder
{
    /// <summary>构建 chat/completions 的请求体 JSON</summary>
    public static string Build(ChatCompletionRequest request, bool stream)
    {
        var body = new Dictionary<string, object?>
        {
            ["model"] = request.Model,
            ["messages"] = request.Messages,
            ["stream"] = stream,
            ["thinking"] = new { type = request.Thinking ? "enabled" : "disabled" },
        };

        // 仅在确有工具时下发，避免部分服务端不接受 "tools": null
        if (request.Tools is { Count: > 0 }) body["tools"] = request.Tools;

        return JsonSerializer.Serialize(body);
    }
}

#endregion

#region 响应模型

/// <summary>流式响应的单个数据块</summary>
public class StreamChunk
{
    [JsonPropertyName("choices")] public List<StreamChoice>? Choices { get; set; }
    [JsonPropertyName("usage")] public TokenUsage? Usage { get; set; }
}

/// <summary>流式响应中的候选</summary>
public class StreamChoice
{
    [JsonPropertyName("delta")] public StreamDelta? Delta { get; set; }
    [JsonPropertyName("finish_reason")] public string? FinishReason { get; set; }
}

/// <summary>流式响应中的增量内容</summary>
public class StreamDelta
{
    [JsonPropertyName("content")] public string? Content { get; set; }
    [JsonPropertyName("reasoning_content")] public string? ReasoningContent { get; set; }
    [JsonPropertyName("tool_calls")] public List<ToolCallDelta>? ToolCalls { get; set; }
}

/// <summary>token 用量</summary>
public sealed class TokenUsage
{
    [JsonPropertyName("prompt_tokens")] public int? PromptTokens { get; set; }
    [JsonPropertyName("completion_tokens")] public int? CompletionTokens { get; set; }
    [JsonPropertyName("total_tokens")] public int? TotalTokens { get; set; }
    [JsonPropertyName("prompt_cache_hit_tokens")] public int? PromptCacheHitTokens { get; set; }
    [JsonPropertyName("completion_tokens_details")] public TokenUsageDetails? CompletionTokensDetails { get; set; }

    /// <summary>从非流式响应根节点读取 usage；字段缺失时按 0 处理</summary>
    public static TokenUsage? FromJson(JsonElement root)
    {
        if (!root.TryGetProperty("usage", out var usage) || usage.ValueKind != JsonValueKind.Object) return null;

        return new TokenUsage
        {
            PromptTokens = ReadInt(usage, "prompt_tokens"),
            CompletionTokens = ReadInt(usage, "completion_tokens"),
            TotalTokens = ReadInt(usage, "total_tokens"),
            PromptCacheHitTokens = ReadInt(usage, "prompt_cache_hit_tokens"),
            CompletionTokensDetails = usage.TryGetProperty("completion_tokens_details", out var details) && details.ValueKind == JsonValueKind.Object
                ? new TokenUsageDetails { ReasoningTokens = ReadInt(details, "reasoning_tokens") }
                : null,
        };
    }

    static int? ReadInt(JsonElement element, string name)
        => element.TryGetProperty(name, out var value) && value.ValueKind == JsonValueKind.Number && value.TryGetInt32(out int number)
            ? number
            : null;
}

/// <summary>completion_tokens_details</summary>
public sealed class TokenUsageDetails
{
    [JsonPropertyName("reasoning_tokens")] public int? ReasoningTokens { get; set; }
}

#endregion

#region 异常

/// <summary>服务端返回错误响应（附带原始响应体，便于定位）</summary>
public sealed class LlmResponseException : Exception
{
    /// <summary>构造</summary>
    public LlmResponseException(string message) : base(message) { }
}

#endregion
