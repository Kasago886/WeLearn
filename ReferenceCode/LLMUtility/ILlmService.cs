namespace KasagoCalendar.Utility.LLMUtility;

/// <summary>
/// <see cref="Agent"/> 运行所需的 LLM 服务契约：连接配置与用量统计。
/// 默认实现为全局单例 <see cref="LLMUtility"/>；测试时可注入自定义实现（如离线桩）。
/// </summary>
public interface ILlmService
{
    /// <summary>API 基础地址</summary>
    string APIUrl { get; }

    /// <summary>API Key</summary>
    string APIKey { get; }

    /// <summary>模型名</summary>
    string ModelName { get; }

    /// <summary>最近一次请求是否成功</summary>
    bool LastRequestSucceeded { get; set; }

    /// <summary>累计 token 用量</summary>
    void RecordUsage(int total, int prompt, int promptCached, int completion, int reasoning);
}
