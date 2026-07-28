# Kratos Agent Backend  [![CI](https://github.com/ChanHsing1972/kratos-backend/actions/workflows/ci.yml/badge.svg)](https://github.com/ChanHsing1972/kratos-backend/actions/workflows/ci.yml)  [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Kratos 智能健身 Agent 的后端服务。负责 Agent 运行时、工具封装、记忆存储以及 REST API。

## 快速开始

前置条件：Python 3.10+。

在仓库根目录执行：

```bash
python3 -m venv .venv

source .venv/bin/activate # macOS/Linux 
.venv/Scripts/activate # Windows

pip install -r requirements.txt

cp .env.example .env

# 如需通过 SSH 隧道连接远程 PostgreSQL，请把示例主机替换为自己的部署主机
ssh -f -N -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -L 5432:localhost:5432 user@example-host

# 如需通过 SSH 隧道连接自托管大模型端口
ssh -f -N -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -L 4141:localhost:4141 user@example-host

# 如果使用 autossh，请按自己的主机和转发目标替换占位符
autossh -M 0 -f -N \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -o ExitOnForwardFailure=yes \
  -L 5432:localhost:5432 \
  -L 4141:localhost:4141 \
  user@example-host

# 启动服务器，默认端口 8000
uvicorn app.main:app --reload --reload-dir app --host 0.0.0.0 --port 8000

# 验证连接，应返回：{"status":"ok","db":true}
curl -sS http://127.0.0.1:8000/health
(win的要去掉 -sS)
```

## Apple Health 同步

`sports` iOS App 的 HealthKit 数据现在直接同步到本后端，不再需要单独启动 `sports/app` 或 8001 端口。

1. 先用 Agent 账号登录获取 JWT：

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username":"YOUR_USERNAME","password":"YOUR_PASSWORD"}'
```

2. 使用 Bearer token 上传 Apple Health 摘要：

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/apple-health/sync" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "source": "apple_health",
    "synced_at": "2026-06-05T12:00:00Z",
    "daily_summary": {
      "date": "2026-06-05",
      "steps": 8500,
      "active_energy_kcal": 520.5,
      "latest_heart_rate_bpm": 78,
      "hrv_sdnn_ms": 45.2,
      "sleep_minutes": 430,
      "vo2_max": 42.5,
      "blood_oxygen_percentage": 97
    },
    "workouts": []
  }'
```

同步成功后，后端会写入 `apple_health_syncs` 原始同步记录，并 upsert 当前用户同一天的 `health_metrics`，因此 Web 前端和 Agent 上下文会读取到最新健康数据。

## 测试 Agent

```bash
source .venv/bin/activate # macOS/Linux 
.venv/Scripts/activate # Windows

pip install -r requirements.txt

# 如需连接自托管大模型端口，请把示例主机替换为自己的部署主机
ssh -f -N -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -L 4141:localhost:4141 user@example-host

# 直接运行 Agent 主程序，观察日志输出
python -m app.agent.main
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
