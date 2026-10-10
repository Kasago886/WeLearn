using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection.Metadata;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace KasagoCalendar.Utility
{
    class DataManager
    {
        static readonly string rootPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "data");

        /// <summary>
        /// 数据根目录的完整路径
        /// </summary>
        public static string RootPath => rootPath;

        /// <summary>
        /// JSON 序列化选项：缩进输出便于人工与智能体阅读，反序列化忽略属性名大小写
        /// </summary>
        static readonly JsonSerializerOptions jsonOptions = new()
        {
            WriteIndented = true,
            PropertyNameCaseInsensitive = true
        };

        static DataManager()
        {
            if (!Directory.Exists(rootPath))
            {
                Directory.CreateDirectory(rootPath);
            }
        }

        /// <summary>
        /// 判断本地路径是否安全（解析后必须位于数据根目录之内，且不是根目录本身）
        /// </summary>
        /// <param name="localPath"></param>
        /// <returns>安全返回true</returns>
        public static bool IsSafeLocalPath(string localPath)
        {
            if (string.IsNullOrWhiteSpace(localPath)) return false;
            try
            {
                string fullRoot = Path.GetFullPath(rootPath);
                string fullPath = Path.GetFullPath(GetUnsafeFullPath(localPath));
                return fullPath.StartsWith(fullRoot + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase);
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// 获取不安全的完整路径，可能不存在；路径越出数据根目录时抛出异常
        /// </summary>
        /// <param name="localPath"></param>
        /// <returns>标准化后的完整路径</returns>
        public static string GetUnsafeFullPath(string localPath)
        {
            string filePath = Path.Combine(rootPath, Path.Combine(localPath.Split('/')));
            string fullRoot = Path.GetFullPath(rootPath);
            string fullPath = Path.GetFullPath(filePath);
            // 越界防护：仅允许访问根目录内的路径，禁止根目录本身或越出根目录
            if (!fullPath.StartsWith(fullRoot + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase))
            {
                throw new ArgumentException($"路径越出数据根目录: {localPath}");
            }
            return fullPath;
        }
        /// <summary>
        /// 获取安全的完整路径，自动创建不存在的目录；路径越出数据根目录时抛出异常
        /// </summary>
        /// <param name="localPath"></param>
        /// <returns></returns>
        public static string GetSafeFullPath(string localPath)
        {
            string filePath = GetUnsafeFullPath(localPath);
            string? dir = Path.GetDirectoryName(filePath);
            if (!string.IsNullOrEmpty(dir))
            {
                Directory.CreateDirectory(dir);
            }
            return filePath;
        }

        /// <summary>
        /// 覆写数据
        /// </summary>
        /// <param name="localPath"></param>
        /// <param name="content"></param>
        public static void WriteData(string localPath, string content)
        {
            string filePath = GetSafeFullPath(localPath);
            File.WriteAllText(filePath, content);
        }
        /// <summary>
        /// 追加数据
        /// </summary>
        /// <param name="localPath"></param>
        /// <param name="content"></param>
        public static void AppendData(string localPath, string content)
        {
            string filePath = GetSafeFullPath(localPath);
            File.AppendAllText(filePath, content);
        }
        /// <summary>
        /// 是否存在数据文件
        /// </summary>
        /// <param name="localPath"></param>
        /// <returns></returns>
        public static bool DataExists(string localPath)
        {
            if (!IsSafeLocalPath(localPath)) return false;
            string filePath = GetUnsafeFullPath(localPath);
            return File.Exists(filePath);
        }
        /// <summary>
        /// 读取数据
        /// </summary>
        /// <param name="localPath"></param>
        /// <param name="content">读取到的数据，不存在时为空</param>
        /// <returns>数据存在时返回true</returns>
        public static bool ReadData(string localPath, out string content)
        {
            content = string.Empty;
            if (!DataExists(localPath)) return false;

            string filePath = GetSafeFullPath(localPath);
            content = File.ReadAllText(filePath);
            return true;
        }

        /// <summary>
        /// 删除数据文件
        /// </summary>
        /// <param name="localPath"></param>
        /// <returns>删除成功返回true，文件不存在时返回false</returns>
        public static bool DeleteData(string localPath)
        {
            if (!IsSafeLocalPath(localPath)) return false;
            string filePath = GetUnsafeFullPath(localPath);
            if (!File.Exists(filePath)) return false;
            File.Delete(filePath);
            return true;
        }

        /// <summary>
        /// 是否存在数据目录
        /// </summary>
        /// <param name="localDir">本地目录，如 "llm" 或 "llm/sub"</param>
        /// <returns>目录存在返回true</returns>
        public static bool DataDirectoryExists(string localDir)
        {
            if (!IsSafeLocalPath(localDir)) return false;
            string dirPath = GetUnsafeFullPath(localDir);
            return Directory.Exists(dirPath);
        }

        /// <summary>
        /// 删除数据目录及其下所有内容（递归）
        /// </summary>
        /// <param name="localDir">本地目录，如 "llm" 或 "llm/sub"</param>
        /// <returns>删除成功返回true，目录不存在时返回false</returns>
        /// <remarks>会校验路径，防止误删数据根目录或越出根目录</remarks>
        public static bool DeleteDirectory(string localDir)
        {
            if (!IsSafeLocalPath(localDir)) return false;

            string fullDir = GetUnsafeFullPath(localDir);
            if (!Directory.Exists(fullDir)) return false;

            Directory.Delete(fullDir, true);
            return true;
        }

        /// <summary>
        /// 清空数据文件内容（保留文件本身）
        /// </summary>
        /// <param name="localPath"></param>
        public static void ClearData(string localPath)
        {
            WriteData(localPath, string.Empty);
        }

        /// <summary>
        /// 列出指定本地目录下的所有数据文件
        /// </summary>
        /// <param name="localDir">本地目录，如 "llm" 或 "llm/sub"</param>
        /// <param name="searchPattern">搜索模式，如 "*.json"</param>
        /// <param name="recursive">是否递归子目录</param>
        /// <returns>以 / 分隔的相对路径数组（可直接用于其他 DataManager 方法），目录不存在时返回空数组</returns>
        public static string[] ListDataFiles(string localDir, string searchPattern = "*", bool recursive = false)
        {
            if (!IsSafeLocalPath(localDir)) return Array.Empty<string>();
            string dirPath = GetUnsafeFullPath(localDir);
            if (!Directory.Exists(dirPath)) return Array.Empty<string>();

            SearchOption option = recursive ? SearchOption.AllDirectories : SearchOption.TopDirectoryOnly;
            return Directory.GetFiles(dirPath, searchPattern, option)
                .Select(file => Path.GetRelativePath(rootPath, file).Replace('\\', '/'))
                .ToArray();
        }

        /// <summary>
        /// 读取 JSON 数据并反序列化为对象
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="localPath"></param>
        /// <param name="value">反序列化结果，不存在或解析失败时为默认值</param>
        /// <returns>读取并解析成功返回true</returns>
        public static bool ReadJson<T>(string localPath, out T? value)
        {
            value = default;
            if (!ReadData(localPath, out string content)) return false;
            try
            {
                value = JsonSerializer.Deserialize<T>(content, jsonOptions);
                return true;
            }
            catch (JsonException)
            {
                return false;
            }
        }

        /// <summary>
        /// 将对象序列化为 JSON 并覆写数据文件（用于新增或整体修改结构化数据）
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="localPath"></param>
        /// <param name="value"></param>
        public static void WriteJson<T>(string localPath, T value)
        {
            string content = JsonSerializer.Serialize(value, jsonOptions);
            WriteData(localPath, content);
        }

        /// <summary>
        /// 读取-修改-写回：对已存在的 JSON 数据执行局部更新；文件不存在时以新对象创建
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="localPath"></param>
        /// <param name="updater">修改回调，在此回调中修改对象字段</param>
        /// <returns>成功返回true；文件存在但解析失败时返回false且不覆盖原数据</returns>
        public static bool UpdateJson<T>(string localPath, Action<T> updater) where T : new()
        {
            if (!IsSafeLocalPath(localPath)) return false;

            T value;
            if (!DataExists(localPath))
            {
                value = new T();
            }
            else if (!ReadJson(localPath, out T? existing))
            {
                return false;
            }
            else
            {
                value = existing!;
            }

            updater(value);
            WriteJson(localPath, value);
            return true;
        }

        /// <summary>
        /// 读取数据文件的所有行
        /// </summary>
        /// <param name="localPath"></param>
        /// <returns>行数组，文件不存在时返回空数组</returns>
        public static string[] ReadLines(string localPath)
        {
            if (!DataExists(localPath)) return Array.Empty<string>();
            return File.ReadAllLines(GetUnsafeFullPath(localPath));
        }

        /// <summary>
        /// 追加一行（保证新行从新行开始，自动附加换行符）
        /// </summary>
        /// <param name="localPath"></param>
        /// <param name="line"></param>
        public static void AppendLine(string localPath, string line)
        {
            // 若文件已有内容且末尾不是换行符，先补一个换行，避免新行与上一行合并
            if (DataExists(localPath) && !File.ReadAllText(GetUnsafeFullPath(localPath)).EndsWith("\n"))
            {
                AppendData(localPath, Environment.NewLine);
            }
            AppendData(localPath, line + Environment.NewLine);
        }

        /// <summary>
        /// 读取指定行（行号从 1 开始）
        /// </summary>
        /// <param name="localPath"></param>
        /// <param name="lineNumber">行号，从 1 开始</param>
        /// <param name="line">读到的行内容，越界或文件不存在时为空</param>
        /// <returns>成功返回true</returns>
        public static bool ReadLine(string localPath, int lineNumber, out string line)
        {
            line = string.Empty;
            string[] lines = ReadLines(localPath);
            if (lineNumber < 1 || lineNumber > lines.Length) return false;
            line = lines[lineNumber - 1];
            return true;
        }

        /// <summary>
        /// 覆写指定行（行号从 1 开始）
        /// </summary>
        /// <param name="localPath"></param>
        /// <param name="lineNumber">行号，从 1 开始</param>
        /// <param name="line">新的行内容</param>
        /// <returns>越界或文件不存在时返回false</returns>
        public static bool WriteLine(string localPath, int lineNumber, string line)
        {
            var (lines, newline, endsWithNewline) = LoadLines(localPath);
            if (lineNumber < 1 || lineNumber > lines.Length) return false;
            lines[lineNumber - 1] = line;
            SaveLines(localPath, lines, newline, endsWithNewline);
            return true;
        }

        /// <summary>
        /// 在指定位置插入一行（行号从 1 开始，可插入到末尾即 length+1 处）
        /// </summary>
        /// <param name="localPath"></param>
        /// <param name="lineNumber">插入位置，从 1 开始</param>
        /// <param name="line">要插入的行内容</param>
        /// <returns>位置越界时返回false；文件不存在时视为空文件创建</returns>
        public static bool InsertLine(string localPath, int lineNumber, string line)
        {
            var (lines, newline, endsWithNewline) = LoadLines(localPath);
            if (lineNumber < 1 || lineNumber > lines.Length + 1) return false;

            var newLines = new string[lines.Length + 1];
            Array.Copy(lines, 0, newLines, 0, lineNumber - 1);
            newLines[lineNumber - 1] = line;
            Array.Copy(lines, lineNumber - 1, newLines, lineNumber, lines.Length - lineNumber + 1);
            SaveLines(localPath, newLines, newline, endsWithNewline);
            return true;
        }

        /// <summary>
        /// 删除指定行（行号从 1 开始）
        /// </summary>
        /// <param name="localPath"></param>
        /// <param name="lineNumber">行号，从 1 开始</param>
        /// <returns>越界或文件不存在时返回false</returns>
        public static bool DeleteLine(string localPath, int lineNumber)
        {
            var (lines, newline, endsWithNewline) = LoadLines(localPath);
            if (lineNumber < 1 || lineNumber > lines.Length) return false;

            var newLines = new string[lines.Length - 1];
            Array.Copy(lines, 0, newLines, 0, lineNumber - 1);
            Array.Copy(lines, lineNumber, newLines, lineNumber - 1, lines.Length - lineNumber);
            SaveLines(localPath, newLines, newline, endsWithNewline);
            return true;
        }

        /// <summary>
        /// 在文件中搜索关键词，返回所有匹配行的内容与行号
        /// </summary>
        /// <param name="localPath"></param>
        /// <param name="keyword">要搜索的关键词</param>
        /// <param name="caseSensitive">是否区分大小写，默认不区分</param>
        /// <returns>匹配行数组（含行号与内容），文件不存在或关键词为空时返回空数组</returns>
        public static LineHit[] SearchLines(string localPath, string keyword, bool caseSensitive = false)
        {
            if (string.IsNullOrEmpty(keyword)) return Array.Empty<LineHit>();

            string[] lines = ReadLines(localPath);
            var results = new List<LineHit>();
            StringComparison comparison = caseSensitive ? StringComparison.Ordinal : StringComparison.OrdinalIgnoreCase;

            for (int i = 0; i < lines.Length; i++)
            {
                if (lines[i].Contains(keyword, comparison))
                {
                    results.Add(new LineHit(i + 1, lines[i]));
                }
            }
            return results.ToArray();
        }

        /// <summary>
        /// 读取全部行，并记录换行符与末尾换行状态，便于改写时保留原格式
        /// </summary>
        static (string[] lines, string newline, bool endsWithNewline) LoadLines(string localPath)
        {
            if (!DataExists(localPath))
            {
                return (Array.Empty<string>(), Environment.NewLine, false);
            }

            string content = File.ReadAllText(GetUnsafeFullPath(localPath));
            string newline = DetectNewline(content);
            bool endsWithNewline = content.EndsWith("\n");
            string[] lines = content.Split(new[] { "\r\n", "\n" }, StringSplitOptions.None);
            if (endsWithNewline && lines.Length > 0)
            {
                Array.Resize(ref lines, lines.Length - 1); // 去掉末尾空元素
            }
            return (lines, newline, endsWithNewline);
        }

        /// <summary>
        /// 按原格式写回全部行
        /// </summary>
        static void SaveLines(string localPath, string[] lines, string newline, bool endsWithNewline)
        {
            if (lines.Length == 0)
            {
                WriteData(localPath, string.Empty);
                return;
            }

            string content = string.Join(newline, lines);
            if (endsWithNewline) content += newline;
            WriteData(localPath, content);
        }

        /// <summary>
        /// 探测文件中使用的换行符
        /// </summary>
        static string DetectNewline(string content)
        {
            if (content.Contains("\r\n")) return "\r\n";
            if (content.Contains('\n')) return "\n";
            return Environment.NewLine;
        }
    }

    /// <summary>
    /// 关键词搜索结果：某一行匹配时返回的行号与内容
    /// </summary>
    public struct LineHit
    {
        /// <summary>
        /// 行号（从 1 开始）
        /// </summary>
        public int LineNumber { get; set; }

        /// <summary>
        /// 该行内容
        /// </summary>
        public string Content { get; set; }

        public LineHit(int lineNumber, string content)
        {
            LineNumber = lineNumber;
            Content = content;
        }
    }
}
