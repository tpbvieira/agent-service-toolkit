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

from core import get_model, settings
from db.agent_model import DatabaseManager

warnings.filterwarnings("ignore", category=LangChainBetaWarning)
logger = logging.getLogger(__name__)

# Set the log level to INFO
logger.setLevel(logging.INFO)

# Add a handler (e.g., to console) if one doesn't already exist.  
# This is crucial; otherwise, you won't see any log output.
handler = logging.StreamHandler()  # Sends logs to the console
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class AgentState(MessagesState, total=False):
    """`total=False` is PEP589 specs.

    documentation: https://typing.readthedocs.io/en/latest/spec/typeddict.html#totality
    """


current_date = datetime.now().strftime("%B %d, %Y")
instructions = f"""
    You are a helpful assistant with the ability to search the web.
    Today's date is {current_date}.

    NOTE: THE USER CAN'T SEE THE TOOL RESPONSE.

    A few things to remember:
    - Please include markdown-formatted links to any citations used in your response. Only include one
    or two citations per response unless more are needed. ONLY USE LINKS RETURNED BY THE TOOLS.
    """

web_search = DuckDuckGoSearchResults(name="WebSearch")

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
        lambda state: [SystemMessage(content=instructions)] + state["messages"],
        name="StateModifier",
    )
    model = model.bind_tools(tools_list)

    return preprocessor | model

# Step 1: Generate an AIMessage that may include a tool-call to be sent.
async def query_or_respond(state: MessagesState, config: RunnableConfig) -> AgentState:
    """Generate tool call for retrieval or respond."""
    logger.info("#> query_or_respond")
    model = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(model)
    response = await model_runnable.ainvoke(state, config)

    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# Step 2: Execute the retrieval.
tools = ToolNode(tools_list)


# Step 3: Generate a response using the retrieved content.
async def generate(state: MessagesState, config: RunnableConfig) -> AgentState:
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
    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        f"{docs_content}"
    )
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    # Run
    model = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(model)
    response = await model_runnable.ainvoke(prompt)
    return {"messages": [response]}


# After "model", if there are tool calls, run "tools". Otherwise END.
def pending_tool_calls(state: AgentState) -> Literal["tools", "done"]:
    logger.info("#> pending_tool_calls")
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        raise TypeError(f"Expected AIMessage, got {type(last_message)}")
    if last_message.tool_calls:
        return "tools"
    return "done"


# Load and chunk contents of the blog
logger.info("#> WebBaseLoader")
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)
vector_store = DatabaseManager().get_vector_store("resolucoes_embd")
# Index chunks
_ = vector_store.add_documents(documents=all_splits)


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

graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)

rag = graph_builder.compile(checkpointer=MemorySaver())
