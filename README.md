# Agent 应用后端

本仓库实现 Kratos 智能健身 Agent 的后端。主要负责 Agent 运行时、工具封装、长期/短期记忆存储以及与前端的 REST 接口对接等。

## 快速开始

前置条件：Python 3.10+。

在仓库根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env

# 通过 ssh 隧道连接远程 MySQL
ssh -L 3306:192.0.2.1:3306 root@192.0.2.1

# 启动服务器，默认端口 8000
uvicorn app.main:app --reload --reload-dir app --host 0.0.0.0 --port 8000

# 验证连接，应返回：{"status":"ok","db":true}
curl -sS http://127.0.0.1:8000/health
```

## 项目结构

- `app/main.py`: FastAPI 启动入口。
- `app/core/config.py`: 配置与环境变量。
- `app/db/session.py`: SQLAlchemy engine 与 `SessionLocal`。
- `app/api/v1/endpoints/`: 放置 REST 接口的路由实现。
- `app/models/`: ORM 模型。
- `app/schemas/`: 请求/响应的 Pydantic 模式。
- `app/services/`: 业务逻辑、工具封装。
- `app/agent/graph.py`: LangGraph 状态流。

## 测试

使用 `pytest` 组织单元与集成测试，测试代码放在 `tests/` 目录。
