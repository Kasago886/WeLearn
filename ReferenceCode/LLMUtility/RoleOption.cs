namespace KasagoCalendar.Utility.LLMUtility;

/// <summary>对话消息在 chat/completions 协议中的角色</summary>
public enum RoleOption
{
    /// <summary>用户</summary>
    User = 0,
    /// <summary>助手</summary>
    Assistant = 1,
    /// <summary>系统提示</summary>
    System = 2,
    /// <summary>工具执行结果</summary>
    Tool = 3,
}

/// <summary><see cref="RoleOption"/> 与协议 role 字符串之间的转换</summary>
public static class RoleOptionExtensions
{
    static readonly string[] RoleNames = ["user", "assistant", "system", "tool"];

    /// <summary>转换为协议中的 role 字符串；取值越界时回退为 user</summary>
    public static string ToRoleString(this RoleOption role)
    {
        int index = (int)role;
        return index >= 0 && index < RoleNames.Length ? RoleNames[index] : RoleNames[0];
    }
}
