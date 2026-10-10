using System;
using System.Collections.Generic;
using System.Text.Json;

namespace KasagoCalendar.Utility.LLMUtility;

/// <summary>
/// 协议消息的构造与读取。
/// 消息统一使用 <c>Dictionary&lt;string, object?&gt;</c> 承载，从而可以原样序列化进请求体，
/// 并保留服务端返回的额外字段（如 tool_calls）。
/// </summary>
public static class ChatMessage
{
    /// <summary>role 字段名</summary>
    public const string RoleKey = "role";
    /// <summary>content 字段名</summary>
    public const string ContentKey = "content";
    /// <summary>reasoning_content 字段名（思维链）</summary>
    public const string ReasoningContentKey = "reasoning_content";
    /// <summary>tool_calls 字段名</summary>
    public const string ToolCallsKey = "tool_calls";
    /// <summary>tool_call_id 字段名</summary>
    public const string ToolCallIdKey = "tool_call_id";

    /// <summary>system 消息</summary>
    public static Dictionary<string, object?> System(string content) => new()
    {
        [RoleKey] = RoleOption.System.ToRoleString(),
        [ContentKey] = content,
    };

    /// <summary>user 消息</summary>
    public static Dictionary<string, object?> User(string content) => new()
    {
        [RoleKey] = RoleOption.User.ToRoleString(),
        [ContentKey] = content,
    };

    /// <summary>
    /// assistant 消息。
    /// 空的 reasoning_content / tool_calls 不会写入，避免出现 <c>"tool_calls": null</c> 这类部分服务端不接受的字段。
    /// </summary>
    public static Dictionary<string, object?> Assistant(string? content, string? reasoningContent = null, object? toolCalls = null)
    {
        var message = new Dictionary<string, object?>
        {
            [RoleKey] = RoleOption.Assistant.ToRoleString(),
            [ContentKey] = content ?? string.Empty,
        };

        if (!string.IsNullOrEmpty(reasoningContent)) message[ReasoningContentKey] = reasoningContent;
        if (toolCalls is not null) message[ToolCallsKey] = toolCalls;
        return message;
    }

    /// <summary>tool 结果消息（toolCallId 需与 assistant 的 tool_calls[].id 对应）</summary>
    public static Dictionary<string, object?> Tool(string toolCallId, string? content) => new()
    {
        [RoleKey] = RoleOption.Tool.ToRoleString(),
        [ToolCallIdKey] = toolCallId,
        [ContentKey] = content ?? string.Empty,
    };

    /// <summary>读取 role；兼容记忆深拷贝后 role 变成 <see cref="JsonElement"/> 的情况</summary>
    public static string? GetRole(Dictionary<string, object?> message)
    {
        if (message is null || !message.TryGetValue(RoleKey, out var role)) return null;

        return role switch
        {
            string value => value,
            JsonElement element when element.ValueKind == JsonValueKind.String => element.GetString(),
            _ => null,
        };
    }

    /// <summary>判断消息是否为指定角色</summary>
    public static bool HasRole(Dictionary<string, object?> message, RoleOption role)
        => string.Equals(GetRole(message), role.ToRoleString(), StringComparison.Ordinal);
}
