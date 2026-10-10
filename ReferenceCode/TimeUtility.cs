using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace KasagoCalendar.Utility
{
    internal class TimeUtility
    {
        /// <summary>
        /// 获取当前日期的字符串表示，格式为 "yyyy-MM-dd"
        /// </summary>
        /// <returns></returns>
        static public string GetDate() => DateTime.Now.ToString("yyyy-MM-dd");

        /// <summary>
        /// 获取当前时间的字符串表示，格式为 "HH:mm:ss ddd"
        /// </summary>
        /// <returns></returns>
        static public string GetTime() => DateTime.Now.ToString("HH:mm:ss ddd");

        /// <summary>
        /// 获取当前时间的字符串表示，格式为 "yyyy-MM-dd HH:mm:ss ddd"
        /// </summary>
        /// <returns></returns>
        static public string GetCurrentTime() => DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss ddd");
    }
}
