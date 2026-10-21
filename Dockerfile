# 多阶段构建：前端产物编进镜像，运行期只保留 Python 与静态文件。
# 目标形态是 D1「一条命令启动」——用户不需要装 Node，也不需要同时起两个进程。

FROM node:22-alpine AS web
WORKDIR /build/web
# 先只拷清单文件：依赖没变时这一层可以命中缓存，改代码不必重装依赖
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    WELEARN_HOST=0.0.0.0 \
    WELEARN_PORT=8000 \
    WELEARN_DATA_DIR=/data

WORKDIR /app
COPY pyproject.toml README.md ./
COPY app ./app
COPY scripts ./scripts
RUN pip install .

# 前端产物放在后端默认查找的位置（仓库内的 web/dist）
COPY --from=web /build/web/dist ./web/dist

# 非 root 运行：容器里没有理由用 root 写自己的数据目录
RUN useradd --create-home --uid 10001 welearn \
    && mkdir -p /data \
    && chown -R welearn:welearn /data /app
USER welearn

VOLUME ["/data"]
EXPOSE 8000

# 用 Python 自带 urllib 探活，避免为健康检查额外装 curl
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)"

CMD ["python", "-m", "app.main"]
