using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

namespace KasagoCalendar.Utility.LLMUtility;

/// <summary>Agent 的流式请求实现（SSE）</summary>
public partial class Agent
{
    /// <summary>
    /// 执行一轮流式请求：逐块回调 <paramref name="dealingAction"/>，自动完成工具调用及其后续轮次。
    /// </summary>
    /// <param name="dealingAction">增量回调；每次仅包含本次新增的 content / reasoningContent</param>
    /// <param name="thinking">是否开启思维链</param>
    /// <param name="timeoutSeconds">单次 HTTP 请求的超时时间</param>
    public async Task<AgentResponse> OneRoundWithoutPromptStreamingAsync(
        Action<AgentResponse>? dealingAction,
        bool thinking = false,
        int timeoutSeconds = 60)
    {
        LlmLog.Write($"[OneRoundStreamingLog Start]\n<Memory>\n{memory.SerializedMessageList}");

        using var timeoutSource = new CancellationTokenSource(TimeSpan.FromSeconds(timeoutSeconds));
        try
        {
            var request = BuildRequest(thinking);

            using var response = await LlmTransport.SendStreamingAsync(request, timeoutSource.Token);
            using var stream = await response.Content.ReadAsStreamAsync(timeoutSource.Token);
            using var reader = new StreamReader(stream, Encoding.UTF8);

            var toolCalls = new ToolCallAccumulator();
            var reasoningContent = new List<string>();
            var contentBuilder = new StringBuilder();
            var reasoningBuilder = new StringBuilder();
            TokenUsage? usage = null;

            await foreach (string payload in SseReader.ReadPayloadsAsync(reader, timeoutSource.Token))
            {
                LlmLog.Write($"[OneRoundStreamingLog Streaming]\n<Response>\ndata: {payload}");

                StreamChunk? chunk;
                try
                {
                    chunk = JsonSerializer.Deserialize<StreamChunk>(payload);
                }
                catch (JsonException ex)
                {
                    // 单个数据块异常不应中断整轮对话，记录后跳过
                    LlmLog.Write($"[OneRoundStreamingLog ChunkError] {ex.Message}\n{payload}");
                    continue;
                }

                if (chunk?.Usage is not null) usage = chunk.Usage;

                var delta = chunk?.Choices?.FirstOrDefault()?.Delta;
                if (delta is null) continue;

                toolCalls.Merge(delta.ToolCalls);

                string? contentPart = delta.Content;
                string? reasoningPart = delta.ReasoningContent;

                if (contentPart is { Length: > 0 }) contentBuilder.Append(contentPart);
                if (reasoningPart is { Length: > 0 }) reasoningBuilder.Append(reasoningPart);

                if (dealingAction is not null && (contentPart is { Length: > 0 } || reasoningPart is { Length: > 0 }))
                {
                    var parts = new List<string>(1);
                    if (reasoningPart is { Length: > 0 }) parts.Add(reasoningPart);

                    dealingAction(new AgentResponse
                    {
                        content = contentPart ?? string.Empty,
                        reasoningContent = parts,
                        error = false,
                    });
                }
            }

            string finalContent = contentBuilder.ToString();
            string finalReasoning = reasoningBuilder.ToString();
            var finalToolCalls = toolCalls.Build();

            LlmLog.Write(
                $"[OneRoundStreamingLog Final]\n" +
                $"<FinalContent>\n{finalContent}\n" +
                $"<FinalReasoning>\n{finalReasoning}");

            if (finalReasoning.Length > 0) reasoningContent.Add(finalReasoning);

            // assistant 消息（含工具调用）必须先写回记忆，其后才能追加 tool 结果
            memory.AddMessageToLastRound(ChatMessage.Assistant(
                finalContent,
                finalReasoning.Length > 0 ? finalReasoning : null,
                finalToolCalls.Count > 0 ? finalToolCalls : null));

            RecordUsage(usage);

            if (finalToolCalls.Count > 0)
            {
                if (!await ExecuteToolCallsAsync(finalToolCalls, reasoningContent, dealingAction))
                    return AgentResponse.Failure("模型请求调用工具，但 Agent 未注册任何工具。");

                var followUp = await OneRoundWithoutPromptStreamingAsync(dealingAction, thinking, timeoutSeconds);
                reasoningContent.AddRange(followUp.reasoningContent);

                // 后续轮次失败时同样要向上层报错，避免错误被当成正常回答
                if (followUp.error)
                    return new AgentResponse
                    {
                        content = followUp.content,
                        reasoningContent = reasoningContent,
                        error = true,
                    };

                finalContent = followUp.content;
            }

            service.LastRequestSucceeded = true;
            return AgentResponse.Success(finalContent, reasoningContent);
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
}
