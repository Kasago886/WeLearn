using System;
using System.Collections.Generic;
using System.Text.Json;
using System.Threading.Tasks;

namespace KasagoCalendar.Utility.LLMUtility;

/// <summary>
/// 对话代理：持有对话记忆与工具集，完成一轮完整的「请求 → 工具调用 → 再请求」流程。
/// <para>
/// 建议通过 <see cref="Clone"/> 在副本上执行，成功后再由上层提交记忆，
/// 这样失败（超时/报错）的一轮不会污染上下文。
/// </para>
/// </summary>
public partial class Agent
{
    /// <summary>请求超时（或被取消）时的错误内容</summary>
    public const string TIMEOUT_ERROR = "Agent Request Timeout";

    /// <summary>对话记忆（对外公开，供上下文压缩等上层逻辑使用）</summary>
    public Memory memory = new();

    /// <summary>可用工具集；为 null 表示本 Agent 不启用工具调用</summary>
    public Tools? tools = null;

    readonly ILlmService service;

    /// <param name="systemPrompt">系统提示词；非空时作为第一轮记忆写入</param>
    /// <param name="tools">可用工具集</param>
    /// <param name="service">LLM 服务；默认使用全局单例 <see cref="LLMUtility.Instance"/></param>
    public Agent(string? systemPrompt = null, Tools? tools = null, ILlmService? service = null)
    {
        this.service = service ?? LLMUtility.Instance;
        this.tools = tools;

        if (!string.IsNullOrWhiteSpace(systemPrompt))
            memory.AddNewRound().AddMessage(ChatMessage.System(systemPrompt));
    }

    /// <summary>复制一个记忆独立、工具与服务共享的新 Agent</summary>
    public Agent Clone() => new(null, tools, service) { memory = memory.CloneMemory };

    #region 对外入口

    /// <summary>提交用户输入并执行一轮流式请求（含工具调用与后续轮次）</summary>
    /// <param name="userPrompt">用户输入；为空时不会新增轮次</param>
    /// <param name="action">增量回调，每次只包含本次新增的内容</param>
    /// <param name="thinking">是否开启思维链</param>
    /// <param name="timeoutSeconds">单次 HTTP 请求的超时时间</param>
    public async Task<AgentResponse> OneRoundStreamingAsync(
        string userPrompt,
        Action<AgentResponse>? action = null,
        bool thinking = false,
        int timeoutSeconds = 60)
    {
        AppendUserPrompt(userPrompt);
        return await OneRoundWithoutPromptStreamingAsync(action, thinking, timeoutSeconds);
    }

    /// <summary>提交用户输入并执行一轮非流式请求（含工具调用与后续轮次）</summary>
    public async Task<AgentResponse> OneRoundAsync(string userPrompt, bool thinking = false, int timeoutSeconds = 100)
    {
        AppendUserPrompt(userPrompt);
        return await OneRoundWithoutPromptAsync(thinking, timeoutSeconds);
    }

    #endregion

    #region 共用逻辑

    void AppendUserPrompt(string userPrompt)
    {
        if (string.IsNullOrEmpty(userPrompt)) return;
        memory.AddNewRound().AddMessage(ChatMessage.User(userPrompt));
    }

    ChatCompletionRequest BuildRequest(bool thinking) => new()
    {
        BaseUrl = service.APIUrl,
        ApiKey = service.APIKey,
        Model = service.ModelName,
        Messages = memory.MessageList,
        Tools = tools?.tools,
        Thinking = thinking,
    };

    /// <summary>记录 token 用量</summary>
    void RecordUsage(TokenUsage? usage)
    {
        if (usage is null) return;

        service.RecordUsage(
            usage.TotalTokens ?? 0,
            usage.PromptTokens ?? 0,
            usage.PromptCacheHitTokens ?? 0,
            usage.CompletionTokens ?? 0,
            usage.CompletionTokensDetails?.ReasoningTokens ?? 0);
    }

    /// <summary>
    /// 并行执行工具调用，按 index 顺序把结果写回记忆，并把过程追加到思维链。
    /// </summary>
    /// <returns>未注册任何工具时返回 false（无法继续本轮对话）</returns>
    async Task<bool> ExecuteToolCallsAsync(
        IReadOnlyList<ToolCallDelta> toolCalls,
        List<string> reasoningContent,
        Action<AgentResponse>? onEvent)
    {
        if (tools is null || tools.toolCallMap.Count == 0)
        {
            LlmLog.Write("收到工具调用，但 Agent 未注册任何工具。");
            return false;
        }

        // 并行执行（工具内部不应依赖执行顺序）
        var results = new string?[toolCalls.Count];
        var pending = new List<Task>(toolCalls.Count);
        for (int i = 0; i < toolCalls.Count; i++)
        {
            int index = i;
            pending.Add(Task.Run(() => results[index] = InvokeTool(toolCalls[index])));
        }
        await Task.WhenAll(pending);

        for (int i = 0; i < toolCalls.Count; i++)
        {
            var call = toolCalls[i];
            string name = call.Function?.Name ?? string.Empty;
            if (string.IsNullOrEmpty(call.ID))
                LlmLog.Write($"工具调用 {name} 缺少 id，tool 消息可能无法与服务端记录对应。");

            memory.AddMessageToLastRound(ChatMessage.Tool(call.ID ?? string.Empty, results[i]));

            string info = $"\n调用工具{name}，返回结果：{results[i]}";
            reasoningContent.Add(info);
            onEvent?.Invoke(new AgentResponse
            {
                content = string.Empty,
                reasoningContent = new List<string> { info },
                error = false,
            });
        }

        return true;
    }

    /// <summary>执行单个工具；任何异常都转换为可读的字符串结果返回给模型</summary>
    string InvokeTool(ToolCallDelta call)
    {
        string name = call.Function?.Name ?? string.Empty;

        var func = tools!.GetFuncByName(name);
        if (func is null) return $"未找到方法 {name}";

        var args = ParseToolArguments(call.Function?.Arguments, out string? argumentError);
        if (argumentError is not null) return argumentError;

        try
        {
            return func.Invoke(args)?.ToString() ?? string.Empty;
        }
        catch (Exception ex)
        {
            return $"工具 {name} 执行异常: {ex.Message}";
        }
    }

    /// <summary>解析工具参数 JSON；空参数视为空字典，解析失败返回错误信息（作为工具结果反馈给模型）</summary>
    static Dictionary<string, object> ParseToolArguments(string? argumentsJson, out string? error)
    {
        error = null;
        if (string.IsNullOrWhiteSpace(argumentsJson)) return new Dictionary<string, object>();

        try
        {
            return JsonSerializer.Deserialize<Dictionary<string, object>>(argumentsJson)
                ?? new Dictionary<string, object>();
        }
        catch (JsonException ex)
        {
            error = $"工具参数解析失败: {ex.Message}。原始参数: {argumentsJson}";
            return new Dictionary<string, object>();
        }
    }

    #endregion
}
