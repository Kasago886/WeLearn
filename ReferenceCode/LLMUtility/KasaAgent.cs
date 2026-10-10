using System;
using System.Threading.Tasks;

namespace KasagoCalendar.Utility.LLMUtility;

/// <summary>
/// 带重试的 Agent 调度器。
/// <para>职责：</para>
/// <list type="bullet">
/// <item>保证同一时刻只有一轮对话在执行（<see cref="Ready"/>）；</item>
/// <item>失败时最多重试 <see cref="MaxRetries"/> 次；</item>
/// <item>仅在成功时把新记忆提交回 <see cref="mainAgent"/>，失败轮次不会污染上下文。</item>
/// </list>
/// </summary>
internal class KasaAgent
{
    readonly object stateLock = new();
    bool running;

    /// <summary>被调度的 Agent；成功的一轮会在此提交新记忆</summary>
    public Agent mainAgent;

    /// <summary>是否空闲（可接受新的用户输入）</summary>
    public bool Ready
    {
        get { lock (stateLock) return !running; }
    }

    /// <summary>失败后的最大重试次数</summary>
    public int MaxRetries { get; }

    /// <summary>不可恢复错误时的通知方式，默认弹窗（可替换为日志或自定义 UI）</summary>
    public Action<string> Notify { get; set; } = UiNotifier.Show;

    /// <summary>构造</summary>
    public KasaAgent(Agent mainAgent, int maxRetries)
    {
        this.mainAgent = mainAgent ?? throw new ArgumentNullException(nameof(mainAgent));
        MaxRetries = Math.Max(0, maxRetries);
    }

    /// <summary>
    /// 处理一次用户输入。
    /// 已有一轮在执行时会直接忽略本次输入（不会排队），保证记忆不会并发被改写。
    /// </summary>
    /// <param name="userInput">用户输入</param>
    /// <param name="dealingNewtokenAction">流式增量回调</param>
    /// <param name="dealingFullresponseAction">成功时的回调（在记忆提交之后调用）</param>
    /// <param name="dealingErrorAction">每次失败时的回调</param>
    /// <param name="thinking">是否开启思维链</param>
    /// <param name="timeoutSeconds">单次请求超时时间</param>
    public async Task UserInput(
        string userInput,
        Action<AgentResponse>? dealingNewtokenAction = null,
        Func<AgentResponse, Task>? dealingFullresponseAction = null,
        Func<AgentResponse, Task>? dealingErrorAction = null,
        bool thinking = true,
        int timeoutSeconds = 60)
    {
        if (string.IsNullOrEmpty(userInput)) return;

        if (!TryEnter())
        {
            LlmLog.Write("KasaAgent 正忙，忽略本次输入。");
            return;
        }

        try
        {
            await RunWithRetryAsync(userInput, dealingNewtokenAction, dealingFullresponseAction, dealingErrorAction, thinking, timeoutSeconds);
        }
        finally
        {
            Exit();
        }
    }

    /// <summary>执行一轮并在失败时重试；使用循环而非递归，重试期间保持忙碌状态</summary>
    async Task RunWithRetryAsync(
        string userInput,
        Action<AgentResponse>? dealingNewtokenAction,
        Func<AgentResponse, Task>? dealingFullresponseAction,
        Func<AgentResponse, Task>? dealingErrorAction,
        bool thinking,
        int timeoutSeconds)
    {
        for (int attempt = 0; ; attempt++)
        {
            // 在副本上执行，成功后再提交，避免失败轮次污染记忆
            var agent = mainAgent.Clone();
            var response = await agent.OneRoundStreamingAsync(userInput, dealingNewtokenAction, thinking, timeoutSeconds);

            if (!response.error)
            {
                mainAgent = agent;
                await InvokeAsync(dealingFullresponseAction, response);
                return;
            }

            await InvokeAsync(dealingErrorAction, response);

            if (attempt >= MaxRetries)
            {
                Notify($"KasaAgent Error: 重试 {attempt} 次后仍然失败（MaxRetries = {MaxRetries}）。错误详情: {response.content}");
                return;
            }

            LlmLog.Write($"KasaAgent 第 {attempt + 1} 次重试。原因: {response.content}");
        }
    }

    static async Task InvokeAsync(Func<AgentResponse, Task>? callback, AgentResponse response)
    {
        if (callback is null) return;

        var task = callback(response);
        if (task is not null) await task;
    }

    /// <summary>尝试进入忙碌状态；已在执行时返回 false</summary>
    bool TryEnter()
    {
        lock (stateLock)
        {
            if (running) return false;
            running = true;
            return true;
        }
    }

    /// <summary>退出忙碌状态</summary>
    void Exit()
    {
        lock (stateLock) running = false;
    }
}
