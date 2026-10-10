using System.Windows;

namespace KasagoCalendar.Utility.LLMUtility;

/// <summary>
/// 轻量用户通知：把消息安全地投递到 UI 线程弹窗（可从任意线程调用）。
/// 在没有 UI 的环境（如单元测试）中退化为写日志。
/// </summary>
internal static class UiNotifier
{
    /// <summary>弹窗提示；无 UI 时写入日志</summary>
    public static void Show(string message)
    {
        var dispatcher = Application.Current?.Dispatcher;
        if (dispatcher is null)
        {
            LlmLog.Write(message);
            return;
        }

        if (dispatcher.CheckAccess()) MessageBox.Show(message);
        else dispatcher.Invoke(() => MessageBox.Show(message));
    }
}
