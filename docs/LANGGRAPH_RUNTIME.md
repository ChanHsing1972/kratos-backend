# LangGraph 运行时迁移

## 唯一流程定义

普通聊天与 SSE 聊天均通过 `LangGraphAgent` 调用 `app/agent/graph.py` 中编译的同一张 StateGraph。所有业务节点的选择、工具循环、反思与重规划由图的边控制；原 `AgentRunner` 及手写调度循环已删除。CLI 也调用相同运行时。

主要路径是 `intent → router → plan → reason → act/finish → reason → generate → reflect → end → finalize`。Router 可以选择轻量 workflow 或专项能力分支；轻量流程无法直接回答时回到 plan。反思失败且还有重规划额度时，经 prepare_replan 清除旧回答的流式展示，再回到 plan。

`runtime.py` 只调用编译图并适配事件迭代，不决定下一业务节点。`nodes/factory.py` 构造原有业务节点，`events.py` 保留客户端进度文案和轻量回答的展示分块，`execution.py` 保存每次调用独立的取消信号、回调、事件累积及步数预算。缓存的是图与节点，运行上下文不跨请求共享。

服务层原有确定性快捷回复保留在原位置，它们不启动多节点 Agent 流程。身份校验、工具/Skill 范围、会话恢复、运行记录幂等预留、媒体补齐、数据库持久化和后台收尾也保留原职责。此迁移没有修改前端、API 路径、数据库结构或模型提示词。

## 状态与持久化边界

图通过一个覆盖写入的 `session` 通道承接现有 SessionState，避免将业务节点原地修改的消息列表再次交给 reducer 合并。每个节点返回更新后的 session，供 LangGraph 的条件边选择后继节点。业务节点保持串行执行。

取消检查位于节点前后以及工具/回答事件输出时，仍是协作式取消；同步模型或工具调用不能被强制中断。原有业务节点执行上限默认是 80，保留原错误文本。LangGraph 的 recursion_limit 另留出内部控制节点所需的额度，避免默认图步数限制提前截断正常回答。

本次没有启用 LangGraph checkpointer、节点重放或自动节点重试。会话与记忆仍按原有数据库路径持久化。后续启用检查点前，需要将 SessionState.tools 中的实际工具实例移至运行时依赖，并检查外部写入的幂等性；不能把当前图直接视为可跨进程恢复的工作流。

## 用户可见合同

原有 status、thought、action、observation、reflection、error、answer_delta、answer_replace、final 和 done 事件，以及 session_id、run_id、trace_id 和结构化结果字段保持兼容。生成过程中持续输出增量；反思重规划会替换旧回答；收尾后的规范答案与最终卡片仍由现有服务层发送。取消与模型错误继续使用原路由的错误与结束事件，客户端重连的事件回放仍由 LiveAgentStream 负责。

## 验证

`tests/fixtures/legacy_agent_contracts.json` 从迁移前提交 `dda0f87c9bab22cba8f529e19297e75c6d170bf6` 的 AgentRunner 实际运行录制而来，使用 `tests/test_graph_runtime.py` 中相同的可控业务节点，共 12 种场景、同步与流式两种模式。EndNode 使用真实实现。对照包括完整事件序列、业务节点调用顺序、结构化结果、会话历史、记忆与最终状态；仅忽略执行耗时和自动生成的时间戳。

场景覆盖标准流程、工具成功/失败、重规划、质量检查额度耗尽、跳过反思、轻量流程直接回答/回退、专项分支、收尾规范化、缺省 Router 和已耗尽的重规划额度。另有取消、异常、步数上限、并发上下文隔离，以及真实 FastAPI 路由与 service + graph 相连的 SSE、错误和取消协议测试。测试使用可控模型输出，不向线上数据库写入。

运行 `PYTHONPATH=. python -m pytest -q`。图的实现已在 LangGraph 1.1.6 与线上现有的 1.2.14 验证。发布依赖固定为 `langgraph==1.2.14`，避免编排迁移同时改变线上框架版本；升级前应重跑这些合同测试。

## 已知的原有行为

真实业务节点联调确认：workflow 直接回答时未追加 AIMessage，现有 EndNode 会将末条用户消息记为历史的 ai_ans。旧 AgentRunner 录制的 workflow 合同具有相同行为。本次保留该语义以限定迁移范围，后续可独立修复轻量记忆查询的归档问题。
