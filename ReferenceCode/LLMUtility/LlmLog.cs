namespace KasagoCalendar.Utility.LLMUtility;

/// <summary>
/// LLM 模块统一日志入口。
/// 日志属于旁路设施：写日志失败（例如队列溢出）不应中断对话主流程。
/// </summary>
internal static class LlmLog
{
    /// <summary>写入 LLM 日志；失败时静默忽略</summary>
    public static void Write(string? message)
    {
        try
        {
            LogManager.AddLog(LLMUtility.LOG_NAME, message);
        }
        catch
        {
            // 忽略日志写入失败
        }
    }
}
