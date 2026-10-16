"""构造业务节点；流程连线只在 app.agent.graph 中定义。"""

from dataclasses import dataclass

from app.agent.nodes.act_node import ActNode
from app.agent.nodes.end_node import EndNode
from app.agent.nodes.finish_node import FinishNode
from app.agent.nodes.generate_node import GenerateNode
from app.agent.nodes.intent_node import IntentNode
from app.agent.nodes.multi_agent_node import MultiAgentNode
from app.agent.nodes.plan_node import PlanNode
from app.agent.nodes.reason_node import ReasonNode
from app.agent.nodes.reflect_node import ReflectNode
from app.agent.nodes.router_node import RouterNode
from app.agent.nodes.workflow_node import WorkflowNode
from app.core.config import settings


@dataclass
class AgentNodeSet:
    """一次运行所需的节点集合。

    以数据类集中持有节点，方便测试注入假节点，也避免运行时构造函数出现过长参数列表。
    """

    intent: IntentNode
    router: RouterNode
    workflow: WorkflowNode
    multi_agent: MultiAgentNode
    plan: PlanNode
    reason: ReasonNode
    act: ActNode
    finish: FinishNode
    generate: GenerateNode
    reflect: ReflectNode
    end: EndNode


def build_agent_nodes(llm) -> AgentNodeSet:
    """用同一个 LLM 实例构造完整节点集合。"""

    return AgentNodeSet(
        intent=IntentNode(llm),
        router=RouterNode(llm),
        workflow=WorkflowNode(llm),
        multi_agent=MultiAgentNode(llm),
        plan=PlanNode(llm),
        reason=ReasonNode(llm),
        act=ActNode(llm),
        finish=FinishNode(llm),
        generate=GenerateNode(llm),
        reflect=ReflectNode(llm),
        end=EndNode(llm if settings.AGENT_ENABLE_MEMORY_SUMMARY_LLM else None),
    )
