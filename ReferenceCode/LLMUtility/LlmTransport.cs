using System;
using System.Net;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace KasagoCalendar.Utility.LLMUtility;

/// <summary>LLM 服务端返回非 2xx 状态码时抛出，携带原始响应体便于定位问题</summary>
public sealed class LlmRequestException : Exception
{
    /// <summary>HTTP 状态码</summary>
    public int StatusCode { get; }

    /// <summary>原始响应体</summary>
    public string ResponseBody { get; }

    /// <summary>构造</summary>
    public LlmRequestException(int statusCode, string responseBody)
        : base($"LLM 请求失败 ({(HttpStatusCode)statusCode}/{statusCode}): {Truncate(responseBody)}")
    {
        StatusCode = statusCode;
        ResponseBody = responseBody;
    }

    static string Truncate(string text) => text.Length <= 500 ? text : text[..500] + "…";
}

/// <summary>
/// LLM HTTP 传输层：负责与 OpenAI 兼容端点通信。
/// <list type="bullet">
/// <item>进程内共享单个 <see cref="HttpClient"/>，避免连接/端口耗尽；</item>
/// <item>超时统一由调用方传入的 <see cref="CancellationToken"/> 控制（HttpClient 自身超时关闭）；</item>
/// <item>失败时回传响应体，便于诊断鉴权、额度、参数等问题。</item>
/// </list>
/// </summary>
internal static class LlmTransport
{
    static readonly HttpClient Http = new() { Timeout = Timeout.InfiniteTimeSpan };

    /// <summary>
    /// 发起流式请求并返回尚未读取的响应。
    /// 调用方负责 Dispose 返回的响应（Dispose 会一并释放响应流）。
    /// </summary>
    public static async Task<HttpResponseMessage> SendStreamingAsync(ChatCompletionRequest request, CancellationToken cancellationToken)
    {
        using var message = CreateMessage(request, stream: true);
        var response = await Http
            .SendAsync(message, HttpCompletionOption.ResponseHeadersRead, cancellationToken)
            .ConfigureAwait(false);

        try
        {
            await ThrowIfFailedAsync(response, cancellationToken).ConfigureAwait(false);
            return response;
        }
        catch
        {
            response.Dispose();
            throw;
        }
    }

    /// <summary>发起非流式请求并返回响应体 JSON</summary>
    public static async Task<string> SendBlockingAsync(ChatCompletionRequest request, CancellationToken cancellationToken)
    {
        using var message = CreateMessage(request, stream: false);
        using var response = await Http
            .SendAsync(message, HttpCompletionOption.ResponseContentRead, cancellationToken)
            .ConfigureAwait(false);

        await ThrowIfFailedAsync(response, cancellationToken).ConfigureAwait(false);
        return await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);
    }

    static HttpRequestMessage CreateMessage(ChatCompletionRequest request, bool stream)
    {
        var message = new HttpRequestMessage(HttpMethod.Post, BuildEndpoint(request.BaseUrl))
        {
            Content = new StringContent(LlmRequestBuilder.Build(request, stream), Encoding.UTF8, "application/json"),
        };
        message.Headers.Authorization = new AuthenticationHeaderValue("Bearer", request.ApiKey);
        return message;
    }

    /// <summary>
    /// 拼接 chat/completions 端点。
    /// 直接拼接而非使用 BaseAddress + 绝对路径，从而保留 BaseUrl 中的子路径（如 /v1）。
    /// </summary>
    static Uri BuildEndpoint(string baseUrl)
    {
        string url = (baseUrl ?? string.Empty).Trim().TrimEnd('/');
        if (url.Length == 0) throw new InvalidOperationException("未配置 LLM API 地址。");

        return new Uri(url + "/chat/completions", UriKind.Absolute);
    }

    static async Task ThrowIfFailedAsync(HttpResponseMessage response, CancellationToken cancellationToken)
    {
        if (response.IsSuccessStatusCode) return;

        string body = string.Empty;
        try
        {
            body = await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);
        }
        catch
        {
            // 读取错误响应体失败时忽略，仅上报状态码
        }

        throw new LlmRequestException((int)response.StatusCode, body);
    }
}
