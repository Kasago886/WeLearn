using System;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Threading;
using System.Threading.Tasks;

namespace KasagoCalendar.Utility.LLMUtility;

/// <summary>
/// LLM 服务门面（进程内单例）：集中管理连接配置、API Key、余额查询与 token 用量统计。
/// 具体对话逻辑见 <see cref="Agent"/>。
/// </summary>
public partial class LLMUtility : ILlmService
{
    /// <summary>
    /// 本模块的日志文件名。
    /// 在进程启动时固定，避免同一轮对话的日志被拆分到多个文件里。
    /// </summary>
    public static readonly string LOG_NAME = "log_llmutil_" + DateTime.Now.ToString("yyyy_MM_dd_HH_mm_ss");

    /// <summary>全局单例</summary>
    public static LLMUtility Instance { get; } = new();

    /// <summary>DeepSeek 账户余额接口（服务商专有接口，与聊天端点不同）</summary>
    const string BalanceUrl = "https://api.deepseek.com/user/balance";

    static readonly HttpClient BalanceHttpClient = new() { Timeout = TimeSpan.FromSeconds(30) };

    /// <summary>当前 API 基础地址</summary>
    public string APIUrl { get; private set; } = DefaultAPIUrl;

    /// <summary>当前模型名</summary>
    public string ModelName { get; private set; } = DefaultModelName;

    /// <summary>当前 API Key；未配置时为 <see cref="NO_API_KEY"/></summary>
    public string APIKey { get; private set; } = NO_API_KEY;

    readonly object usageLock = new();
    Usage usageField;

    /// <summary>累计 token 用量（读取时加锁，返回一致快照）</summary>
    public Usage usage
    {
        get { lock (usageLock) return usageField; }
    }

    int lastRequestSucceeded;

    /// <summary>最近一次请求是否成功</summary>
    public bool LastRequestSucceeded
    {
        get => Volatile.Read(ref lastRequestSucceeded) != 0;
        set => Volatile.Write(ref lastRequestSucceeded, value ? 1 : 0);
    }

    /// <summary>构造：初始化 API Key</summary>
    public LLMUtility() => GetAPIKey();

    /// <summary>token 用量统计</summary>
    public struct Usage
    {
        /// <summary>总 token 数</summary>
        public int total;
        /// <summary>输入 token 数</summary>
        public int prompt;
        /// <summary>输入中命中缓存的 token 数</summary>
        public int promptCached;
        /// <summary>输出 token 数</summary>
        public int completion;
        /// <summary>输出中用于思考的 token 数</summary>
        public int reasoning;
    }

    /// <summary>创建新 Agent（使用本单例作为 LLM 服务）</summary>
    /// <param name="systemPrompt">系统提示词</param>
    /// <param name="tools">可用工具集</param>
    public Agent GetNewAgent(string? systemPrompt = null, Tools? tools = null) => new(systemPrompt, tools, this);

    /// <summary>累计 token 用量（线程安全）</summary>
    public void RecordUsage(int total, int prompt, int promptCached, int completion, int reasoning)
    {
        lock (usageLock)
        {
            usageField.total += total;
            usageField.prompt += prompt;
            usageField.promptCached += promptCached;
            usageField.completion += completion;
            usageField.reasoning += reasoning;
        }
    }

    /// <summary>重置用量统计</summary>
    public void ResetUsage()
    {
        lock (usageLock) usageField = new Usage();
    }

    /// <summary>
    /// 查询账户余额，返回原始 JSON 字符串。
    /// 未配置 API Key 时返回 <see cref="NO_API_KEY"/>；请求失败时返回服务端响应体。
    /// </summary>
    public async Task<string> GetBalanceInfoAsync()
    {
        if (APIKey == NO_API_KEY) return NO_API_KEY;

        using var request = new HttpRequestMessage(HttpMethod.Get, BalanceUrl);
        request.Headers.Accept.Add(new MediaTypeWithQualityHeaderValue("application/json"));
        request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", APIKey);

        try
        {
            using var response = await BalanceHttpClient.SendAsync(request);
            string body = await response.Content.ReadAsStringAsync();

            if (!response.IsSuccessStatusCode)
            {
                LastRequestSucceeded = false;
                UiNotifier.Show($"余额查询失败({(int)response.StatusCode}): {body}");
            }

            return body;
        }
        catch (Exception ex)
        {
            LastRequestSucceeded = false;
            UiNotifier.Show($"余额查询异常: {ex.Message}");
            return $"Balance request error: {ex.Message}";
        }
    }
}
