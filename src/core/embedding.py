from functools import cache
from typing import TypeAlias

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from schemas.models import (
    GoogleModelName,
)

_MODEL_TABLE = {
    GoogleModelName.GEMINI_15_FLASH: "gemini-1.5-flash",
    GoogleModelName.GEMINI_2_FLASH: "gemini-2.0-flash"
}

EmbdModel: TypeAlias = GoogleGenerativeAIEmbeddings

@cache
def get_embedding_model() -> EmbdModel:
    # NOTE: models with streaming=True will send tokens as they are generated
    # if the /stream endpoint is called with stream_tokens=True (the default)

    return GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
