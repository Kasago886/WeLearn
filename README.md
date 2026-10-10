# WeLearn

个性化概念学习 Web 应用：用户指定要学习的概念，系统先确认知识基础，再生成个性化学习路径，最后按路径逐节点进行问答式教学（每个节点通过检查点提问确认掌握后才继续）。

- 需求规格：[docs/需求规格说明书.md](docs/需求规格说明书.md)
- 实施计划：[docs/实施计划.md](docs/实施计划.md)
- 接口契约：[docs/接口契约.md](docs/接口契约.md)
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
# 附带录制响应样本，供离线测试回放（T0.7）
.\.venv\Scripts\python.exe scripts\llm_smoke.py --record
```
