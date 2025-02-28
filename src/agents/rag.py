import logging
import warnings
from datetime import datetime
from typing import Literal

import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core._api import LangChainBetaWarning
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from client.client import AgentClientError
from core import get_model, settings
from db.agent_model import DatabaseManager

warnings.filterwarnings("ignore", category=LangChainBetaWarning)

logger = logging.getLogger(__name__)
# Set the log level to INFO
logger.setLevel(logging.INFO)
# Prevent duplicate logs
logger.propagate = False  
# Check if the logger already has handlers to prevent duplicate entries
if not logger.handlers:
    # Add a handler (e.g., to console) if one doesn't already exist.
    handler = logging.StreamHandler()  # Sends logs to the console
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class AgentState(MessagesState, total=False):
    """`total=False` is PEP589 specs.

    documentation: https://typing.readthedocs.io/en/latest/spec/typeddict.html#totality
    """


current_date = datetime.now().strftime("%B %d, %Y")
base_system_prompt = f"""
    Você é um assistente prestativo, com habilidade para recuperar informações de ferramentas,
    se você decidir que é necessário para fornecer uma boa resposta.
    A data de hoje é {current_date}.
    """

# web_search = DuckDuckGoSearchResults(name="WebSearch")

@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    logger.info("#> retrieve")
    retrieved_docs = DatabaseManager().get_vector_store("resolucoes_embd").similarity_search(query, k=5)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


# tools_list = [web_search, retrieve]
tools_list = [retrieve]


def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    logger.info("#> wrap_model")
    preprocessor = RunnableLambda(
        lambda state: [SystemMessage(content=base_system_prompt)] + state["messages"],
        name="StateModifier",
    )
    model = model.bind_tools(tools_list)

    return preprocessor | model


# Step 1: Generate an AIMessage that may include a tool-call to be sent.
def query_or_respond(state: MessagesState, config: RunnableConfig) -> AgentState:
    """Generate tool call for retrieval or respond."""
    logger.info("#> query_or_respond")
    model = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_with_tools = wrap_model(model)
    try:
        response = model_with_tools.invoke(state, config)
    except (AgentClientError, Exception) as e:
        logger.error("#> query_or_respond > error: %s", e)
        response = "Unexpected error."
    return {"messages": [response]}


# Step 2: Execute the retrieval.
tools = ToolNode(tools_list)


# Step 3: Generate a response using the retrieved content.
def generate(state: MessagesState, config: RunnableConfig) -> AgentState:
    """Generate answer."""
    logger.info("#> generate")
    # Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break # only the last tool message
    tool_messages = recent_tool_messages[::-1]

    # Format into prompt
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_prompt = f"""
        Use as seguintes partes do contexto recuperado para responder à pergunta.
        Se você não souber a resposta, diga que não sabe.
        Use no máximo três frases e mantenha a resposta concisa.
        Partes do contexto recuperado:
        \n\n
        {docs_content}"""
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    rag_prompt = [SystemMessage(base_system_prompt + system_prompt)] + conversation_messages
    logger.info("#> generate > rag_prompt: %s", rag_prompt)
    # Run
    llm = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    response = llm.invoke(rag_prompt, config)
    return {"messages": [response]}


# # After "model", if there are tool calls, run "tools". Otherwise END.
# def pending_tool_calls(state: AgentState) -> Literal["tools", "done"]:
#     logger.info("#> pending_tool_calls")
#     last_message = state["messages"][-1]
#     if not isinstance(last_message, AIMessage):
#         raise TypeError(f"Expected AIMessage, got {type(last_message)}")
#     if last_message.tool_calls:
#         return "tools"
#     return "done"


# Load and chunk contents of the blog
logger.info("#> WebBaseLoader")
urls = [
    "https://informacoes.anatel.gov.br/legislacao/resolucoes/2024/1965-resolucao-767",  # cyber
    "https://informacoes.anatel.gov.br/legislacao/resolucoes/2020/1497-resolucao-740",  # cyber
    "https://informacoes.anatel.gov.br/legislacao/resolucoes/2024/1990-resolucao-771",  # sei
    "https://informacoes.anatel.gov.br/legislacao/resolucoes/2017/943-resolucao-682",   # sei
    "https://informacoes.anatel.gov.br/legislacao/resolucoes/2023/1900-resolucao-765"   # rgc
]

logger.info("#> WebBaseLoader > loading...")
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=256, chunk_overlap=64
)
doc_splits = text_splitter.split_documents(docs_list)
vector_store = DatabaseManager().get_vector_store("resolucoes_embd")
# Index chunks
_ = vector_store.add_documents(documents=doc_splits)
logger.info("#> WebBaseLoader > loaded and indexed")

# # Define the graph
# agent = StateGraph(AgentState)
# agent.add_node("model", acall_model)
# agent.add_node("tools", ToolNode(tools))

# agent.set_entry_point("model")

# # Always run "model" after "tools"
# agent.add_edge("tools", "model")

# # Connect the graph
# agent.add_edge("model", END)
# agent.add_conditional_edges("model", pending_tool_calls, {"tools": "tools", "done": END})

# rag = agent.compile(checkpointer=MemorySaver())

logger.info("#> StateGraph(MessagesState)")
graph_builder = StateGraph(MessagesState)
graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)

graph_builder.set_entry_point("query_or_respond")
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)

rag = graph_builder.compile(checkpointer=MemorySaver())

# Get the PNG image binary data
png = rag.get_graph().draw_mermaid_png()

# Save the binary PNG data to a file in /tmp
file_path = "/app/graph.png"
with open(file_path, "wb") as f:
    f.write(png)
