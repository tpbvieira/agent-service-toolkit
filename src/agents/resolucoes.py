import hashlib
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
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class AgentState(MessagesState, total=False):
    """`total=False` is PEP589 specs.

    documentation: https://typing.readthedocs.io/en/latest/spec/typeddict.html#totality
    """


current_date = datetime.now().strftime("%B %d, %Y")
base_system_prompt = f"""
    Você é um assistente prestativo, capaz de atender prompts diversos, mas com habilidade 
    específica de recuperar informações de resoluções da anatel por meio de chamadas a uma 
    ferramenta, sempre você decidir que é necessário para fornecer uma boa resposta.
    A data de hoje é {current_date}.
    """

# web_search = DuckDuckGoSearchResults(name="WebSearch")


@tool(response_format="content_and_artifact")
def resolution_retrieval(query: str):
    """Retrieve information related to a query about Anatel's Resolutions."""
    logger.info("#> resolution_retrieval")
    retrieved_docs = (
        DatabaseManager().get_vector_store("resolucoes_embd").similarity_search(query, k=5)
    )
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}") for doc in retrieved_docs
    )
    return serialized, retrieved_docs


# tools_list = [web_search, resolution_retrieval]
tools_list = [resolution_retrieval]


def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    """Wrap the model with a preprocessor that adds a system message to the state."""
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
    except AgentClientError as e:
        logger.error("#> query_or_respond > error: %s", e)
        response = AIMessage(content="Unexpected error, sorry! Please try again latter.")
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
    generation_prompt = f"""
        Use as seguintes partes do contexto recuperado para atender a instrução.
        Se você não souber a resposta, diga que não sabe.
        Use no máximo três frases e mantenha a resposta concisa.
        Partes do contexto recuperado:
        \n\n
        {docs_content}"""
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system") or (message.type == "ai" and not message.tool_calls)
    ]
    resolucoes_prompt = [
        SystemMessage(base_system_prompt + generation_prompt)
    ] + conversation_messages
    logger.info("#> generate > resolucoes_prompt: %s", resolucoes_prompt)

    # Run
    llm = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    response = llm.invoke(resolucoes_prompt, config)
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
    # accessibility
    "https://informacoes.anatel.gov.br/legislacao/resolucoes/2016/905-resolucao-n-667",
    # universalization 
    "https://informacoes.anatel.gov.br/legislacao/resolucoes/2022/1689-resolucao-754",
    # rgg
    "https://informacoes.anatel.gov.br/legislacao/resolucoes/2023/1900-resolucao-765",
]

logger.info("#> WebBaseLoader > loading vector database of resolucoes...")
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]
CHUNK_SIZE = 512
CHUNK_OVERLAP = CHUNK_SIZE // 5
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
)
doc_chunks = text_splitter.split_documents(docs_list)

def generate_doc_id(doc):
    """Generate a unique ID based on document content."""
    return hashlib.sha256(doc.page_content.encode()).hexdigest()  # Hash content as ID


# Generate document IDs
chunk_id_map = {}
unique_chunks = []
for chunk in doc_chunks:
    chunk_id = generate_doc_id(chunk)
    if chunk_id not in chunk_id_map:
        chunk_id_map[chunk_id] = chunk
        unique_chunks.append(chunk)

# Get unique IDs and documents
unique_chunl_ids = list(chunk_id_map.keys())
unique_chunks = list(chunk_id_map.values())

# Index chunks
vector_store = DatabaseManager().get_vector_store("resolucoes_embd")
indexed = vector_store.add_documents(documents=unique_chunks, ids=unique_chunl_ids)
logger.info("#> WebBaseLoader > Indexed %s chunks", len(indexed))

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

resolucoes_graph = graph_builder.compile(checkpointer=MemorySaver())

# Get the PNG image binary data
png = resolucoes_graph.get_graph().draw_mermaid_png()
# Save the binary PNG data to a file in /tmp
file_path = "/app/graph.png"
with open(file_path, "wb") as f:
    f.write(png)
