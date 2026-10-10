using System;
using System.Collections.Generic;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

namespace KasagoCalendar.Utility.LLMUtility;

/// <summary>Agent 的非流式请求实现（一次性返回完整响应）</summary>
public partial class Agent
{
    /// <summary>
    /// 执行一轮非流式请求，自动完成工具调用及其后续轮次。
    /// </summary>
    /// <param name="thinking">是否开启思维链</param>
    /// <param name="timeoutSeconds">单次 HTTP 请求的超时时间</param>
    public async Task<AgentResponse> OneRoundWithoutPromptAsync(bool thinking = false, int timeoutSeconds = 100)
    {
        using var timeoutSource = new CancellationTokenSource(TimeSpan.FromSeconds(timeoutSeconds));
        try
        {
            LlmLog.Write($"[OneRoundLog]\n<Memory>\n{memory.SerializedMessageList}");

            var request = BuildRequest(thinking);
            string responseJson = await LlmTransport.SendBlockingAsync(request, timeoutSource.Token);
            LlmLog.Write($"[OneRoundLogResponse]\n{responseJson}");

            using var document = JsonDocument.Parse(responseJson);
            JsonElement root = document.RootElement;
            JsonElement message = ReadFirstMessage(root);

            var assistantMessage = JsonSerializer.Deserialize<Dictionary<string, object?>>(message.GetRawText());
            if (assistantMessage is null || ChatMessage.GetRole(assistantMessage) is null)
                throw new LlmResponseException("响应中的 message 无法解析为对话消息。");

            memory.AddMessageToLastRound(assistantMessage);

            RecordUsage(TokenUsage.FromJson(root));

            string content = ReadString(message, ChatMessage.ContentKey) ?? string.Empty;
            string? reasoning = ReadString(message, ChatMessage.ReasoningContentKey);

            var reasoningContent = new List<string>();
            if (!string.IsNullOrEmpty(reasoning)) reasoningContent.Add(reasoning);

            var toolCalls = ParseToolCalls(message);
            if (toolCalls.Count > 0)
            {
                if (!await ExecuteToolCallsAsync(toolCalls, reasoningContent, null))
                    return AgentResponse.Failure("模型请求调用工具，但 Agent 未注册任何工具。");

                var followUp = await OneRoundWithoutPromptAsync(thinking, timeoutSeconds);
                reasoningContent.AddRange(followUp.reasoningContent);

                // 后续轮次失败时同样要向上层报错，避免错误被当成正常回答
                if (followUp.error)
                    return new AgentResponse
                    {
                        content = followUp.content,
                        reasoningContent = reasoningContent,
                        error = true,
                    };

                content = followUp.content;
            }

            service.LastRequestSucceeded = true;
            return AgentResponse.Success(content, reasoningContent);
        }
        catch (OperationCanceledException)
        {
            service.LastRequestSucceeded = false;
            LlmLog.Write(TIMEOUT_ERROR);
            return AgentResponse.Timeout();
        }
        catch (Exception ex)
        {
            service.LastRequestSucceeded = false;
            LlmLog.Write($"Agent Request Error:{ex.Message}");
            return AgentResponse.Failure($"Agent Request Error:{ex.Message}");
        }
    }

    /// <summary>读取 choices[0].message</summary>
    static JsonElement ReadFirstMessage(JsonElement root)
    {
        if (!root.TryGetProperty("choices", out var choices) || choices.ValueKind != JsonValueKind.Array || choices.GetArrayLength() == 0)
            throw new LlmResponseException("响应中缺少 choices[0]。");

        if (!choices[0].TryGetProperty("message", out var message) || message.ValueKind != JsonValueKind.Object)
            throw new LlmResponseException("响应中缺少 choices[0].message。");

        return message;
    }

    /// <summary>读取字符串字段，非字符串或缺失时返回 null</summary>
    static string? ReadString(JsonElement element, string name)
        => element.TryGetProperty(name, out var value) && value.ValueKind == JsonValueKind.String
            ? value.GetString()
            : null;

    /// <summary>解析 message.tool_calls；部分服务端不下发 index，此时按顺序补全</summary>
    static List<ToolCallDelta> ParseToolCalls(JsonElement message)
    {
        var toolCalls = new List<ToolCallDelta>();
        if (!message.TryGetProperty(ChatMessage.ToolCallsKey, out var element) || element.ValueKind != JsonValueKind.Array)
            return toolCalls;

        int index = 0;
        foreach (var item in element.EnumerateArray())
        {
            var call = JsonSerializer.Deserialize<ToolCallDelta>(item.GetRawText());
            if (call is null) continue;

            call.Index ??= index;
            toolCalls.Add(call);
            index++;
        }
        return toolCalls;
    }
}
