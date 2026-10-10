using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using System.Windows.Media.Animation;
using static KasagoCalendar.Kasas.ToolFunction;
using static KasagoCalendar.Kasas.ToolUtil;
using KasagoCalendar.Utility;
using KasagoCalendar.Utility.LLMUtility;

namespace KasagoCalendar.Kasas
{
    internal class ToolPreset
    {
        public static Tools TestTools
        {
            get
            {
                static string GetDateTest() => "2026-03-05";
                static string GetWeatherTest(string location, string date) => $"Weather in {location}: Cloudy 7~13°C";

                var getDateToolContent = new
                {
                    type = "function",
                    function = new
                    {
                        name = nameof(GetDateTest),
                        description = "Get the current date",
                        parameters = new
                        {
                            type = "object",
                            properties = new { }
                        }
                    }
                };
                var getWeatherToolContent = new
                {
                    type = "function",
                    function = new
                    {
                        name = nameof(GetWeatherTest),
                        description = "Get weather of a location, the user should supply the location and date.",
                        parameters = new
                        {
                            type = "object",
                            properties = new
                            {
                                location = new { type = "string", description = "The city name" },
                                date = new { type = "string", description = "The date in format YYYY-mm-dd" }
                            },
                            required = new[] { "location", "date" }
                        }
                    }
                };

                Tools tools = new()
                {
                    tools = [getDateToolContent, getWeatherToolContent],
                    toolCallMap =
                {
                    [nameof(GetDateTest)] = new((x)=>GetDateTest()),
                    [nameof(GetWeatherTest)] = new((x)=>{
                        JsonElement? location = x["location"] as JsonElement?;
                        JsonElement? date = x["date"] as JsonElement?;
                        return GetWeatherTest(location?.GetString()!, date?.GetString()!);
                        })
                }
                };
                return tools;
            }
        }
        public static Tools KasaAlphaTools
        {
            get
            {
                var getDateTool = new
                {
                    type = "function",
                    function = new
                    {
                        name = nameof(GetDate),
                        description = "Get the current date (yyyy-MM-dd)",
                        parameters = new
                        {
                            type = "object",
                            properties = new { }
                        }
                    }
                };
                var getTimeTool = new
                {
                    type = "function",
                    function = new
                    {
                        name = nameof(GetTime),
                        description = "Get the current time (HH:mm:ss ddd)",
                        parameters = new
                        {
                            type = "object",
                            properties = new { }
                        }
                    }
                };

                #region // ===== 文件管理工具（仅允许操作 llm/normal_agent/memory 目录内） =====
                var listMemoryFilesTool = new
                {
                    type = "function",
                    function = new
                    {
                        name = nameof(ListMemoryFiles),
                        description = "List files in the agent memory directory (llm/normal_agent/memory).",
                        parameters = new
                        {
                            type = "object",
                            properties = new
                            {
                                recursive = new { type = "boolean", description = "Whether to include files in subdirectories" }
                            }
                        }
                    }
                };
                var readMemoryFileTool = new
                {
                    type = "function",
                    function = new
                    {
                        name = nameof(ReadMemoryFile),
                        description = "Read the full content of a file in the agent memory directory.",
                        parameters = new
                        {
                            type = "object",
                            properties = new
                            {
                                fileName = new { type = "string", description = "File name or subpath within memory (e.g. notes/2026-08-16.md)" }
                            },
                            required = new[] { "fileName" }
                        }
                    }
                };
                var writeMemoryFileTool = new
                {
                    type = "function",
                    function = new
                    {
                        name = nameof(WriteMemoryFile),
                        description = "Create a new file or overwrite an existing file with the given content in the agent memory directory.",
                        parameters = new
                        {
                            type = "object",
                            properties = new
                            {
                                fileName = new { type = "string", description = "File name or subpath within memory" },
                                content = new { type = "string", description = "Full content to write" }
                            },
                            required = new[] { "fileName", "content" }
                        }
                    }
                };
                var appendMemoryFileTool = new
                {
                    type = "function",
                    function = new
                    {
                        name = nameof(AppendMemoryFile),
                        description = "Append content to a file in the agent memory directory (creates the file if missing).",
                        parameters = new
                        {
                            type = "object",
                            properties = new
                            {
                                fileName = new { type = "string", description = "File name or subpath within memory" },
                                content = new { type = "string", description = "Content to append" }
                            },
                            required = new[] { "fileName", "content" }
                        }
                    }
                };
                var deleteMemoryFileTool = new
                {
                    type = "function",
                    function = new
                    {
                        name = nameof(DeleteMemoryFile),
                        description = "Delete a file in the agent memory directory.",
                        parameters = new
                        {
                            type = "object",
                            properties = new
                            {
                                fileName = new { type = "string", description = "File name or subpath within memory" }
                            },
                            required = new[] { "fileName" }
                        }
                    }
                };
                var memoryFileExistsTool = new
                {
                    type = "function",
                    function = new
                    {
                        name = nameof(MemoryFileExists),
                        description = "Check whether a file exists in the agent memory directory.",
                        parameters = new
                        {
                            type = "object",
                            properties = new
                            {
                                fileName = new { type = "string", description = "File name or subpath within memory" }
                            },
                            required = new[] { "fileName" }
                        }
                    }
                };
                var searchMemoryFilesTool = new
                {
                    type = "function",
                    function = new
                    {
                        name = nameof(SearchMemoryFiles),
                        description = "Search a keyword across files in the agent memory directory, returning file, line number and content of each match.",
                        parameters = new
                        {
                            type = "object",
                            properties = new
                            {
                                keyword = new { type = "string", description = "Keyword to search (case-insensitive)" },
                                recursive = new { type = "boolean", description = "Whether to search subdirectories (default true)" }
                            },
                            required = new[] { "keyword" }
                        }
                    }
                };
                #endregion

                Tools tools = new()
                {
                    tools = [
                        getDateTool,
                    getTimeTool,
                    listMemoryFilesTool,
                    readMemoryFileTool,
                    writeMemoryFileTool,
                    appendMemoryFileTool,
                    deleteMemoryFileTool,
                    memoryFileExistsTool,
                    searchMemoryFilesTool
                    ],
                    toolCallMap =
                {
                    [nameof(GetDate)] = new((x)=>GetDate()),
                    [nameof(GetTime)] = new((x)=>GetTime()),
                    [nameof(ListMemoryFiles)] = new((x)=>ListMemoryFiles(GetBoolArg(x, "recursive"))),
                    [nameof(ReadMemoryFile)] = new((x)=>ReadMemoryFile(GetStringArg(x, "fileName") ?? "")),
                    [nameof(WriteMemoryFile)] = new((x)=>WriteMemoryFile(GetStringArg(x, "fileName") ?? "", GetStringArg(x, "content") ?? "")),
                    [nameof(AppendMemoryFile)] = new((x)=>AppendMemoryFile(GetStringArg(x, "fileName") ?? "", GetStringArg(x, "content") ?? "")),
                    [nameof(DeleteMemoryFile)] = new((x)=>DeleteMemoryFile(GetStringArg(x, "fileName") ?? "")),
                    [nameof(MemoryFileExists)] = new((x)=>MemoryFileExists(GetStringArg(x, "fileName") ?? "")),
                    [nameof(SearchMemoryFiles)] = new((x)=>SearchMemoryFiles(GetStringArg(x, "keyword") ?? "", GetBoolArg(x, "recursive", true))),
                }
                };
                return tools;
            }
        }
    }
}
