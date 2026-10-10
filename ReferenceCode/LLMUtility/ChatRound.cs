using System.Collections.Generic;
using System.Text.Json;

namespace KasagoCalendar.Utility.LLMUtility;

/// <summary>
/// 一轮对话：一次用户输入，以及它引发的全部消息
/// （user → assistant(可含 tool_calls) → tool... → assistant）。
/// </summary>
public class ChatRound
{
    /// <summary>本轮消息列表，元素为协议原始字段（role/content/reasoning_content/tool_calls...）</summary>
    public List<Dictionary<string, object?>> messageList = new();

    /// <summary>消息条数</summary>
    public int Count => messageList.Count;

    /// <summary>序列化后的消息列表（日志/调试用）</summary>
    public string SerializedMessageList => JsonSerializer.Serialize(messageList);

    /// <summary>深拷贝：通过 JSON 往返，保证与协议结构完全一致</summary>
    public ChatRound CloneRound => new()
    {
        messageList = JsonSerializer.Deserialize<List<Dictionary<string, object?>>>(SerializedMessageList) ?? new(),
    };

    /// <summary>是否为 system 轮次（首条消息 role = system）</summary>
    public bool IsSystemRound
        => messageList.Count > 0 && ChatMessage.HasRole(messageList[0], RoleOption.System);

    /// <summary>在末尾追加消息</summary>
    public void AddMessage(Dictionary<string, object?> messageDict) => InsertMessage(messageList.Count, messageDict);

    /// <summary>在指定位置插入消息；索引越界时不执行任何操作</summary>
    public void InsertMessage(int index, Dictionary<string, object?> messageDict)
    {
        if (messageDict is null) return;
        if (index < 0 || index > messageList.Count) return;

        messageList.Insert(index, messageDict);
    }
}
