# WeLearn

个性化概念学习 Web 应用：用户指定要学习的概念，系统先确认知识基础，再生成个性化学习路径，最后按路径逐节点进行问答式教学（每个节点通过检查点提问确认掌握后才继续）。

- 需求规格：[docs/需求规格说明书.md](docs/需求规格说明书.md)
- 实施计划（当前现状与剩余规划）：[docs/实施计划.md](docs/实施计划.md)
- 实施记录（各阶段历史与交付证据）：[docs/实施记录.md](docs/实施记录.md)
- 接口契约（当前有效）：[docs/接口契约.md](docs/接口契约.md)
- 契约变更记录（版本演进与实测结论）：[docs/接口契约变更记录.md](docs/接口契约变更记录.md)
- Spike 实测：[docs/Spike实测结果.md](docs/Spike实测结果.md)
- 设计参考：[ReferenceCode/](ReferenceCode)（C# 实现，仅作参考，不作依赖）

## 环境准备

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
Copy-Item .env.example .env   # 填写 WELEARN_API_KEY
```

## 运行

后端启动后**同时提供前端页面**（`web/dist` 由后端托管），不需要另起前端进程：

```powershell
cd web; npm run build; cd ..        # 只需一次，或前端有改动后重跑
.\.venv\Scripts\python.exe -m app.main
# 浏览器打开 http://127.0.0.1:8000
```

开发期仍推荐前后端分离（vite 有 HMR）：见下面「前端（web/）」一节。
没有 `web/dist` 时后端照常提供接口，只是访问 `/` 会 404——开发期前端走 vite 就是这个状态。

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:create_app --factory --reload
```

健康检查：`GET http://127.0.0.1:8000/api/health`

### 环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `WELEARN_HOST` / `WELEARN_PORT` | `127.0.0.1` / `8000` | 监听地址与端口 |
| `WELEARN_DATA_DIR` | `./data` | 数据目录（SQLite、日志、备份），**整目录复制即可搬迁** |
| `WELEARN_WEB_DIST` | `<仓库>/web/dist` | 前端产物位置，仅在自定义打包布局时需要 |

密钥、模型、访问口令等业务配置都在**设置页**里改（存进数据库），不放环境变量；
`.env` 只用于首次配置 `WELEARN_API_KEY`。

## 部署（Docker）

```bash
docker compose up -d --build      # 首次或改代码后
docker compose logs -f welearn    # 看日志
docker compose down               # 停止；数据在 ./data，不随容器消失
```

- 镜像分两阶段构建：`node:22` 编前端 → `python:3.12-slim` 跑后端，用户机器不需要装 Node。
- 进程以非 root（uid 10001）运行，`./data` 用 bind mount 挂进 `/data`，便于直接备份与升级。
- 默认只绑定 `127.0.0.1:8000`；要从别的机器访问，设 `WELEARN_BIND=0.0.0.0` 并**同时**在设置页配置访问口令。
- Linux 上若 `./data` 不属于 uid 10001，容器会以只读权限启动失败：先 `sudo chown -R 10001:10001 ./data`（Docker Desktop 不需要）。
- `env_file: .env` 声明为可选（需要 Compose ≥ 2.24）：没有 `.env` 也能启动，只是首次得在设置页填 Key。

## 备份与恢复

```powershell
.\.venv\Scripts\python.exe scripts\backup_db.py backup            # 备份到 <数据目录>/backups/welearn-<时间戳>.db
.\.venv\Scripts\python.exe scripts\backup_db.py verify            # 校验最新备份（也可 --file 指定）
.\.venv\Scripts\python.exe scripts\backup_db.py restore --file <备份文件>
```

服务运行中也能备份：用的是 SQLite 在线备份 API，拿到的是包含 WAL 已提交内容的一致视图
（直接复制 `welearn.db` 会丢掉最近提交的数据）。恢复前会把原库改名留档为
`welearn.db.before-restore-<时间戳>`，备份校验不过则拒绝覆盖。

升级迁移不需要额外操作：新版本启动时按 `SCHEMA_VERSION` 自动做幂等列级升级，老库原地升到最新版本。

## 前端（web/）

React 19 + Vite + TypeScript，不依赖 UI 组件库；按页面拆分为主题 / 测评 / 路径 / 教学 / 设置。

```powershell
cd web
npm install
npm run dev          # 开发服务器 http://localhost:5173，/api 自动代理到 127.0.0.1:8000
npm run build        # tsc -b && vite build
npm run lint         # oxlint
```

后端未启动时前端会给出"无法连接到后端服务"的可操作提示，而不是白屏。

### 不开真实额度也能联调

`scripts/dev_stub_server.py` 用假 LLM 起一个**真实**服务端，让整条链路
（SSE 分帧、事件处理、三选一、刷新恢复）可以零成本反复跑：

```powershell
# 终端 1
.\.venv\Scripts\python.exe scripts\dev_stub_server.py
# 终端 2
cd web; npm run dev
```

桩的判定规则是"回答 ≥40 字算通过"，因此可以手动制造
"未通过 → 换角度重讲 → 重试耗尽 → 三选一"的完整路径。

## 校验

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m mypy app scripts
cd web; npm run build; npm run lint
```

性能校验（真实 uvicorn + 假 LLM，不消耗额度；结果只反映本机瓶颈）：

```powershell
.\.venv\Scripts\python.exe scripts\perf_check.py
```

它报告首 token 延迟与本地接口（主题列表、状态快照、档案页、检索）的 p50/p95，
与预算（NFR-101 首 token ≤ 3s、NFR-102 本地接口 ≤ 300ms）逐项对照。

## 真实 API 冒烟测试

```powershell
# 需要 .env 中已配置 WELEARN_API_KEY
.\.venv\Scripts\python.exe scripts\llm_smoke.py
# 附带录制响应样本，供离线回放（T0.7）
.\.venv\Scripts\python.exe scripts\llm_smoke.py --record
```

未配置 Key 时会给出引导并以退出码 2 退出（便于区分"没配密钥"与"接口故障"）。
`--record` 会把真实 SSE 流、非流式响应、JSON Output 响应、模型列表与余额写入
`tests/fixtures/llm/recorded/`；这些样本由 `tests/test_recorded_fixtures.py` 回放验证，
并强制检查其中不含密钥。

## Spike 实测（真实额度）

离线测试证明不了"官方多久返回一次空内容、判定器会不会放水、思考档位值不值"，
这些要用真实调用抽样（结论见 [docs/Spike实测结果.md](docs/Spike实测结果.md)）：

```powershell
# 全部跑一遍（约 30 次调用），或只跑其中一个
.\.venv\Scripts\python.exe scripts\spike_real.py --out docs/Spike实测结果.md
.\.venv\Scripts\python.exe scripts\spike_real.py --spike b
```

默认用设置里的 `model_default`（`deepseek-flash`）：Spike 要回答的是"默认配置下表现如何"，
换个模型测出来的结论对默认用户没有意义。
