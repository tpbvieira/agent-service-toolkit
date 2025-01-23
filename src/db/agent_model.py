import logging
import os
from datetime import datetime, timezone

from sqlalchemy import TIMESTAMP, Column, Integer, create_engine
from sqlalchemy.dialects.postgresql import TEXT
from sqlalchemy.orm import declarative_base, sessionmaker

AGENT_PGVECTOR_USER = "agent_db_user"
AGENT_PGVECTOR_PWD = "4g3ntdbus3r"
AGENT_PGVECTOR_HOST = "pgvector"
AGENT_PGVECTOR_DB = "agent_db"

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
        db_url = f"postgresql+psycopg2://{AGENT_PGVECTOR_USER}:{AGENT_PGVECTOR_PWD}@{AGENT_PGVECTOR_HOST}/{AGENT_PGVECTOR_DB}"
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def add_record(self, code_snippet: str, suggestions: str) -> None:
        session = self.Session()

        new_record = AnalysisHistory(
            code_snippet=code_snippet,
            suggestions=suggestions,
            created_at=datetime.now(timezone.utc)
        )

        session.add(new_record)
        session.commit()
        session.close()

