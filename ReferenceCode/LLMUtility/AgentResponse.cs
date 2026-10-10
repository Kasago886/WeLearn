using System.Collections.Generic;

namespace KasagoCalendar.Utility.LLMUtility;

/// <summary>一次 Agent 请求的结果</summary>
public struct AgentResponse
{
    /// <summary>正文内容；出错时为错误描述</summary>
    public string content;

    /// <summary>思维链内容（含工具调用过程的记录）</summary>
    public List<string> reasoningContent;

    /// <summary>是否出错（超时、网络错误、协议错误等）</summary>
    public bool error;

    /// <summary>成功响应</summary>
    public static AgentResponse Success(string content, List<string> reasoningContent) => new()
    {
        content = content,
        reasoningContent = reasoningContent,
        error = false,
    };

    /// <summary>失败响应</summary>
    public static AgentResponse Failure(string message) => new()
    {
        content = message,
        reasoningContent = new List<string>(),
        error = true,
    };

    /// <summary>超时响应</summary>
    public static AgentResponse Timeout() => Failure(Agent.TIMEOUT_ERROR);
}
