# 前端工具管理接口说明

本文档只说明前端实现“用户管理 Agent 工具”功能时需要调用的接口。

基础前缀：

```http
/api/v1
```

所有接口都需要登录态：

```http
Authorization: Bearer <access_token>
```

---

## 1. 获取当前用户可管理的工具列表

- `GET /api/v1/tools`

### 用途

进入“工具管理”页面时调用。用于展示所有 Agent 工具、分类、是否启用、健康状态、失败次数等。

### 请求

无请求体。

### 返回示例

```json
[
  {
    "id": 1,
    "name": "tavily_search",
    "description": "联网搜索新闻、实时信息和网页资料。",
    "category": "search",
    "enabled": true,
    "requires_api_key": true,
    "health_status": "healthy",
    "failure_count": 0,
    "api_key_configured": true,
    "created_at": "2026-05-20T12:00:00",
    "updated_at": "2026-05-20T12:00:00"
  }
]
```

### 前端主要使用字段

| 字段 | 说明 |
| --- | --- |
| `name` | 工具唯一名称，后续更新接口要用 |
| `description` | 工具描述 |
| `category` | 工具分类，可用于分组展示 |
| `enabled` | 当前用户是否启用该工具 |
| `requires_api_key` | 是否需要后端配置 API Key |
| `api_key_configured` | 后端是否已经配置好 API Key |
| `health_status` | 健康状态 |
| `failure_count` | 工具失败次数 |

### `health_status` 取值

```text
unknown      未知
healthy      正常
degraded     异常但可用
unavailable  不可用，通常是缺少 API Key
disabled     已禁用
```

---

## 2. 启用或禁用单个工具

- `PATCH /api/v1/tools/{tool_name}`

### 用途

用户点击某个工具的开关时调用。

### 路径参数

| 参数 | 说明 |
| --- | --- |
| `tool_name` | 工具名称，例如 `tavily_search` |

### 请求示例：禁用工具

```json
{
  "enabled": false
}
```

### 请求示例：启用工具

```json
{
  "enabled": true
}
```

### 返回示例

```json
{
  "id": 1,
  "name": "tavily_search",
  "description": "联网搜索新闻、实时信息和网页资料。",
  "category": "search",
  "enabled": false,
  "requires_api_key": true,
  "health_status": "disabled",
  "failure_count": 0,
  "api_key_configured": true,
  "created_at": "2026-05-20T12:00:00",
  "updated_at": "2026-05-20T12:10:00"
}
```

---

## 3. 批量保存工具启用状态

- `PATCH /api/v1/tools`

### 用途

如果前端页面是“多个开关修改后点击保存”，使用这个接口。

### 请求示例

```json
{
  "tools": [
    {
      "name": "tavily_search",
      "enabled": true
    },
    {
      "name": "weather_fitness_advisor",
      "enabled": false
    },
    {
      "name": "diet_plan_generator",
      "enabled": true
    }
  ]
}
```

### 返回

返回更新后的完整工具列表，结构同 `GET /api/v1/tools`。

---

## 4. 更新工具健康状态

- `POST /api/v1/tools/{tool_name}/health`

### 用途

一般前端不一定需要调用。可以用于管理员页面或调试页面，手动标记工具健康状态。

### 请求示例

```json
{
  "health_status": "degraded",
  "failure_delta": 1
}
```

### 字段说明

| 字段 | 说明 |
| --- | --- |
| `health_status` | 新的健康状态 |
| `failure_delta` | 失败次数增加值，默认 0 |

### 返回

返回更新后的单个工具配置。

---

## 5. 重置工具失败次数

- `POST /api/v1/tools/{tool_name}/failures/reset`

### 用途

前端提供“重置失败次数”按钮时调用。

### 请求

无请求体。

### 返回示例

```json
{
  "id": 1,
  "name": "tavily_search",
  "description": "联网搜索新闻、实时信息和网页资料。",
  "category": "search",
  "enabled": true,
  "requires_api_key": true,
  "health_status": "healthy",
  "failure_count": 0,
  "api_key_configured": true,
  "created_at": "2026-05-20T12:00:00",
  "updated_at": "2026-05-20T12:20:00"
}
```

---

## 推荐前端页面逻辑

### 页面加载

```text
GET /api/v1/tools
```

然后按 `category` 分组展示。

推荐分类展示：

| category | 说明 |
| --- | --- |
| `search` | 搜索 |
| `fitness` | 健身计算/训练建议 |
| `fitness_knowledge` | 健身知识库 |
| `diet` | 饮食/食谱 |
| `weather` | 天气 |
| `location` | 地理位置/路线 |

### 用户切换单个开关

```text
PATCH /api/v1/tools/{tool_name}
```

请求：

```json
{
  "enabled": true
}
```

或者：

```json
{
  "enabled": false
}
```

### 用户点击“保存全部”

```text
PATCH /api/v1/tools
```

请求：

```json
{
  "tools": [
    { "name": "tavily_search", "enabled": true },
    { "name": "diet_plan_generator", "enabled": false }
  ]
}
```

---

## 注意事项

1. 如果 `requires_api_key=true` 且 `api_key_configured=false`，前端可以提示：

```text
该工具需要后端配置 API Key，当前不可用。
```

2. 如果 `health_status=disabled`，表示用户主动禁用了工具。

3. 如果 `health_status=unavailable`，通常表示缺少 API Key。

4. 如果用户禁用了某个工具，之后调用 Agent Chat 时，Agent 将不会看到这个工具，也不会调用它。

5. Agent Chat 接口仍然是：

```http
POST /api/v1/agent/chat
```

工具管理接口只负责控制 Agent 可用工具列表。
