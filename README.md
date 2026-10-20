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

```powershell
.\.venv\Scripts\python.exe -m app.main
# 或
.\.venv\Scripts\python.exe -m uvicorn app.main:create_app --factory --reload
```

健康检查：`GET http://127.0.0.1:8000/api/health`

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
```

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
