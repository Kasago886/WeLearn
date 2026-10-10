using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;

namespace KasagoCalendar.Utility
{
    public class LogManager
    {
        // 日志根目录
        private static readonly string LogDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Logs");
        private static readonly string ErrorLogPath = Path.Combine(LogDir, "LogManagerErrorLog.txt");

        // 写入队列
        private struct LogStruct(string _filePath, string _message, uint _retryCount = 0)
        {
            public string FilePath = _filePath;
            public string Message = _message;
            public uint RetryCount = _retryCount;
        }
        private static readonly BlockingCollection<LogStruct> _queue = new();
        private const int MaxQueueCount = 1000;
        private static readonly Task _worker;

        static LogManager()
        {
            // 文件夹创建
            if (!Directory.Exists(LogDir))
                Directory.CreateDirectory(LogDir);

            // 写入线程
            _worker = Task.Run(ProcessQueue);
        }

        public static void AddLog(string logName, string? message)
        {
            if (_queue.Count >= MaxQueueCount)
            {
                throw new Exception("Log requests are too many. Forced Rejection.");
            }

            // 添加后缀
            if (!logName.EndsWith(".txt", StringComparison.OrdinalIgnoreCase)) // OrdinalIgnoreCase忽略大小写
                logName += ".txt";

            string filePath = Path.Combine(LogDir, logName);
            string finalMessage = $"{DateTime.Now:yyyy-MM-dd HH:mm:ss.fff}\n{message}\n\n";

            _queue.Add(new LogStruct(filePath, finalMessage));
        }

        private static void ProcessQueue()
        {
            foreach (var item in _queue.GetConsumingEnumerable())
            {
                string filePath = item.FilePath;
                string message = item.Message;
                uint retryCount = item.RetryCount;
                try
                {
                    using (StreamWriter writer = new StreamWriter(filePath, true, Encoding.UTF8)) // true 表示追加
                    {
                        writer.WriteLine(message);
                    }
                }
                catch (IOException ex)
                {
                    if (retryCount < 3)
                    {
                        _queue.Add(new LogStruct(filePath, message, retryCount + 1));
                    }
                    else if (retryCount == 3)
                    {
                        string errorMsg = $"{DateTime.Now:yyyy-MM-dd HH:mm:ss.fff}\n" +
                            $"{ex.Message}\n" +
                            $"Origin LogStruct:\n" +
                            $"FilePath:\t{filePath}\n" +
                            $"Message:\t{message}\n" +
                            $"RetryCount:\t{retryCount}\n\n";
                        _queue.Add(new LogStruct(ErrorLogPath, errorMsg, retryCount + 1));
                    }
                    else
                    {
                        //throw;
                        MessageBox.Show($"LogError:\n" +
                            $"{ex.Message}\n" +
                            $"Origin LogStruct:\n" +
                            $"FilePath:\t{filePath}\n" +
                            $"Message:\t{message}\n" +
                            $"RetryCount:\t{retryCount}");
                    }
                }
                catch (Exception ex)
                {
                    if (retryCount == 0)
                    {
                        string errorMsg = $"{DateTime.Now:yyyy-MM-dd HH:mm:ss.fff}\n" +
                            $"{ex.Message}\n" +
                            $"Origin LogStruct:\n" +
                            $"FilePath:\t{filePath}\n" +
                            $"Message:\t{message}\n" +
                            $"RetryCount:\t{retryCount}\n\n";
                        _queue.Add(new LogStruct(ErrorLogPath, errorMsg, 999));
                    }
                    else
                    {
                        //throw;
                        MessageBox.Show($"LogError:\n" +
                            $"{ex.Message}\n" +
                            $"Origin LogStruct:\n" +
                            $"FilePath:\t{filePath}\n" +
                            $"Message:\t{message}\n" +
                            $"RetryCount:\t{retryCount}");
                    }
                }
            }
        }
    }
}
