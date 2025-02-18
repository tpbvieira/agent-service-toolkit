import logging
import os
from datetime import UTC, datetime

from langchain_postgres import PGVector
from sqlalchemy import TIMESTAMP, Column, Integer, create_engine
from sqlalchemy.dialects.postgresql import TEXT
from sqlalchemy.orm import declarative_base, sessionmaker

from core.embedding import get_embedding_model

logging.basicConfig(level=logging.INFO)

Base = declarative_base()

class AnalysisHistory(Base):

    __tablename__ = "analysis_history"

    id = Column(Integer, primary_key=True)
    code_snippet = Column(TEXT)
    suggestions = Column(TEXT)
    created_at = Column(TIMESTAMP)


class DatabaseManager:

    def __init__(self) -> None:

        AGENT_PGVECTOR_USER = os.environ["AGENT_PGVECTOR_USER"]
        AGENT_PGVECTOR_PWD = os.environ["AGENT_PGVECTOR_PWD"]
        AGENT_PGVECTOR_HOST = os.environ["AGENT_PGVECTOR_HOST"]
        AGENT_PGVECTOR_DB = os.environ["AGENT_PGVECTOR_DB"]

        self.db_url = f"postgresql+psycopg://{AGENT_PGVECTOR_USER}:{AGENT_PGVECTOR_PWD}@{AGENT_PGVECTOR_HOST}/{AGENT_PGVECTOR_DB}"
        self.engine = create_engine(self.db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)


    def add_record(self, code_snippet: str, suggestions: str) -> None:
        session = self.Session()

        new_record = AnalysisHistory(
            code_snippet=code_snippet,
            suggestions=suggestions,
            created_at=datetime.now(UTC)
        )

        session.add(new_record)
        session.commit()
        session.close()


    def get_db_url(self) -> str:
        return self.db_url
    

    def get_vector_store(self, collection_name: str) -> PGVector:
        google_embeddings = get_embedding_model()
        return PGVector(
            embeddings=google_embeddings,
            collection_name= collection_name,
            connection=DatabaseManager().get_db_url(),
            use_jsonb=True,
        )