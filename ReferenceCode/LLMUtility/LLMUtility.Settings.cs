namespace KasagoCalendar.Utility.LLMUtility;

/// <summary>LLMUtility 的配置与凭据部分</summary>
public partial class LLMUtility
{
    /// <summary>LLM 连接设置</summary>
    public struct Settings
    {
        /// <summary>API 基础地址</summary>
        public string APIUrl { get; set; }
        /// <summary>模型名</summary>
        public string ModelName { get; set; }
    }

    /// <summary>默认 API 地址</summary>
    public const string DefaultAPIUrl = "https://api.deepseek.com";

    /// <summary>默认模型名</summary>
    public const string DefaultModelName = "deepseek-flash";

    /// <summary>未配置 API Key 时的占位值</summary>
    public const string NO_API_KEY = "No API Key";

    const string SettingsPath = "llm/settings.json";
    const string DefaultAPIKeyPath = "llm/APIKey.txt";

    static readonly Settings DefaultSettings = new() { APIUrl = DefaultAPIUrl, ModelName = DefaultModelName };

    /// <summary>
    /// 读取设置。
    /// 文件不存在时写入默认设置；字段缺失/为空/格式错误时回退默认值，保证返回值始终可用。
    /// </summary>
    public Settings GetSettings()
    {
        if (!DataManager.DataExists(SettingsPath))
            DataManager.WriteJson(SettingsPath, DefaultSettings);

        var settings = DataManager.ReadJson(SettingsPath, out Settings? loaded) && loaded is { } value
            ? value
            : DefaultSettings;

        APIUrl = string.IsNullOrWhiteSpace(settings.APIUrl) ? DefaultAPIUrl : settings.APIUrl.Trim();
        ModelName = string.IsNullOrWhiteSpace(settings.ModelName) ? DefaultModelName : settings.ModelName.Trim();

        return new Settings { APIUrl = APIUrl, ModelName = ModelName };
    }

    /// <summary>保存设置并返回生效后的值</summary>
    public Settings SetSettings(Settings settings)
    {
        DataManager.WriteJson(SettingsPath, settings);
        LlmLog.Write($"SetSettings: APIUrl={settings.APIUrl}, ModelName={settings.ModelName}");
        return GetSettings();
    }

    /// <summary>
    /// 读取 API Key。
    /// 默认路径下的文件不存在时会创建占位文件，并返回 <see cref="NO_API_KEY"/>。
    /// </summary>
    /// <param name="filePath">自定义路径；为空时使用默认路径</param>
    public string GetAPIKey(string? filePath = null)
    {
        string path = string.IsNullOrWhiteSpace(filePath) ? DefaultAPIKeyPath : filePath;

        if (!DataManager.DataExists(path))
        {
            if (path == DefaultAPIKeyPath) DataManager.WriteData(path, NO_API_KEY);
            APIKey = NO_API_KEY;
            return APIKey;
        }

        APIKey = DataManager.ReadData(path, out string content) ? content.Trim() : NO_API_KEY;
        if (APIKey.Length == 0) APIKey = NO_API_KEY;

        return APIKey;
    }

    /// <summary>保存 API Key 并返回生效值；日志中不记录密钥明文</summary>
    public string SetAPIKey(string newAPIKey)
    {
        DataManager.WriteData(DefaultAPIKeyPath, newAPIKey ?? string.Empty);
        LlmLog.Write("SetAPIKey: API Key 已更新（内容不记录）");
        return GetAPIKey();
    }
}
