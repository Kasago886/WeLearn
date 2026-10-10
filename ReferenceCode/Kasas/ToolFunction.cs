using KasagoCalendar.Utility;
using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace KasagoCalendar.Kasas
{
    internal class ToolFunction
    {
        /// <summary>
        /// 获取日期
        /// </summary>
        /// <returns></returns>
        public static string GetDate()
        {
            return TimeUtility.GetDate();
        }
        /// <summary>
        /// 获取时刻
        /// </summary>
        /// <returns></returns>
        public static string GetTime()
        {
            return TimeUtility.GetTime();
        }

        #region // ==================== 文件管理工具实现（限定 llm/normal_agent/memory 目录） ====================

        const string MEMORY_ROOT = "llm/normal_agent/memory";

        /// <summary>
        /// 将 memory 目录内的文件名/子路径规整为限定范围内的相对路径，越界时抛异常
        /// </summary>
       public static string NormalizeMemoryPath(string fileName)
        {
            string relative = (fileName ?? string.Empty).Trim().TrimStart('/', '\\');
            if (relative.Length == 0)
                throw new ArgumentException("文件名不能为空");

            string localPath = MEMORY_ROOT + "/" + relative;
            string memoryRoot = Path.GetFullPath(DataManager.GetUnsafeFullPath(MEMORY_ROOT));
            string full = Path.GetFullPath(DataManager.GetUnsafeFullPath(localPath));
            if (!full.StartsWith(memoryRoot + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase))
            {
                throw new ArgumentException($"文件名越出 memory 目录: {fileName}");
            }
            return localPath;
        }

        /// <summary>
        /// 列表：memory 目录下的所有文件
        /// </summary>
       public static string ListMemoryFiles(bool recursive = false)
        {
            try
            {
                var files = DataManager.ListDataFiles(MEMORY_ROOT, "*", recursive);
                if (files.Length == 0) return "(memory 目录为空)";
                const string prefix = MEMORY_ROOT + "/";
                return string.Join("\n", files.Select(f => f.StartsWith(prefix) ? f[prefix.Length..] : f));
            }
            catch (Exception ex) { return $"错误: {ex.Message}"; }
        }

        /// <summary>
        /// 读取：memory 目录下某文件内容
        /// </summary>
       public static string ReadMemoryFile(string fileName)
        {
            try
            {
                string path = NormalizeMemoryPath(fileName);
                if (!DataManager.DataExists(path)) return $"文件不存在: {fileName}";
                return DataManager.ReadData(path, out string content) ? content : $"读取失败: {fileName}";
            }
            catch (Exception ex) { return $"错误: {ex.Message}"; }
        }

        /// <summary>
        /// 写入（新建或覆写）：memory 目录下某文件
        /// </summary>
       public static string WriteMemoryFile(string fileName, string content)
        {
            try
            {
                string path = NormalizeMemoryPath(fileName);
                DataManager.WriteData(path, content ?? "");
                return $"已写入: {fileName}";
            }
            catch (Exception ex) { return $"错误: {ex.Message}"; }
        }

        /// <summary>
        /// 追加：memory 目录下某文件
        /// </summary>
       public static string AppendMemoryFile(string fileName, string content)
        {
            try
            {
                string path = NormalizeMemoryPath(fileName);
                DataManager.AppendData(path, content ?? "");
                return $"已追加: {fileName}";
            }
            catch (Exception ex) { return $"错误: {ex.Message}"; }
        }

        /// <summary>
        /// 删除：memory 目录下某文件
        /// </summary>
       public static string DeleteMemoryFile(string fileName)
        {
            try
            {
                string path = NormalizeMemoryPath(fileName);
                return DataManager.DeleteData(path) ? $"已删除: {fileName}" : $"文件不存在: {fileName}";
            }
            catch (Exception ex) { return $"错误: {ex.Message}"; }
        }

        /// <summary>
        /// 判断：memory 目录下某文件是否存在
        /// </summary>
       public static string MemoryFileExists(string fileName)
        {
            try
            {
                string path = NormalizeMemoryPath(fileName);
                return DataManager.DataExists(path) ? "存在" : "不存在";
            }
            catch (Exception ex) { return $"错误: {ex.Message}"; }
        }

        /// <summary>
        /// 搜索：在 memory 目录内搜索关键词，返回 文件:行号:内容
        /// </summary>
       public static string SearchMemoryFiles(string keyword, bool recursive = true)
        {
            try
            {
                if (string.IsNullOrEmpty(keyword)) return "错误: 关键词不能为空";
                var files = DataManager.ListDataFiles(MEMORY_ROOT, "*", recursive);
                var results = new List<string>();
                const string prefix = MEMORY_ROOT + "/";
                foreach (var file in files)
                {
                    var hits = DataManager.SearchLines(file, keyword);
                    if (hits.Length == 0) continue;
                    string name = file.StartsWith(prefix) ? file[prefix.Length..] : file;
                    foreach (var hit in hits)
                    {
                        results.Add($"{name}:{hit.LineNumber}: {hit.Content}");
                    }
                }
                return results.Count == 0 ? "未找到匹配内容" : string.Join("\n", results);
            }
            catch (Exception ex) { return $"错误: {ex.Message}"; }
        }

        #endregion
    }

    internal class ToolUtil
    {
        /// <summary>
        /// 从工具参数中取字符串
        /// </summary>
        public static string? GetStringArg(Dictionary<string, object> args, string key)
        {
            if (args.TryGetValue(key, out object? val) && val is JsonElement je && je.ValueKind == JsonValueKind.String)
            {
                return je.GetString();
            }
            return null;
        }

        /// <summary>
        /// 从工具参数中取布尔值
        /// </summary>
        public static bool GetBoolArg(Dictionary<string, object> args, string key, bool defaultValue = false)
        {
            if (args.TryGetValue(key, out object? val) && val is JsonElement je)
            {
                if (je.ValueKind == JsonValueKind.True) return true;
                if (je.ValueKind == JsonValueKind.False) return false;
            }
            return defaultValue;
        }
    }
}
