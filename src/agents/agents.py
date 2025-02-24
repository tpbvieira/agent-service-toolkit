from dataclasses import dataclass

from langgraph.graph.state import CompiledStateGraph

from agents.bg_task_agent.bg_task_agent import bg_task_agent
from agents.chatbot import chatbot
from agents.code_reviewer import code_reviewer
from agents.rag import rag
from agents.research_assistant import research_assistant
from schemas import AgentInfo

DEFAULT_AGENT = "rag"


@dataclass
class Agent:
    description: str
    graph: CompiledStateGraph


agents: dict[str, Agent] = {
    "bg-task-agent": Agent(description="A background task agent.", graph=bg_task_agent),
    "code-reviewer": Agent(description="A Pytho Code Reviewer.", graph=code_reviewer),
    "chatbot": Agent(description="A simple chatbot.", graph=chatbot),
    "rag": Agent(description="A simple chatbot that relies on RAG.", graph=rag),
    "research-assistant": Agent(
        description="A research assistant with web search and calculator.", graph=research_assistant
    ),
}


def get_agent(agent_id: str) -> CompiledStateGraph:
    return agents[agent_id].graph


def get_all_agent_info() -> list[AgentInfo]:
    return [
        AgentInfo(key=agent_id, description=agent.description) for agent_id, agent in agents.items()
    ]
