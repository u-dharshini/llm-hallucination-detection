"""
backend/app/models/db_models.py

SQLAlchemy ORM tables. These persist every run of the pipeline so the
frontend (M4) can show history, and so the demo has real data to show.

Note: this is separate from schemas.py (Pydantic, P2's shared Claim object
used for request/response validation). This file is the DB-storage layer.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship

from backend.app.database import Base


class QueryRecord(Base):
    """One row per user question + generated answer (one pipeline run)."""
    __tablename__ = "query_records"

    id = Column(Integer, primary_key=True, index=True)
    question = Column(Text, nullable=False)
    answer_text = Column(Text, nullable=False)
    llm_provider = Column(String, nullable=True)
    llm_model = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    claims = relationship(
        "ClaimRecord", back_populates="query", cascade="all, delete-orphan"
    )


class ClaimRecord(Base):
    """One row per atomic claim extracted from an answer, with its verdict."""
    __tablename__ = "claim_records"

    id = Column(Integer, primary_key=True, index=True)
    query_id = Column(Integer, ForeignKey("query_records.id"), nullable=False)

    claim_text = Column(Text, nullable=False)
    verdict = Column(String, nullable=False)          # "Supported" | "Contradicted" | "Unverifiable"
    confidence = Column(Float, nullable=True)
    source = Column(String, nullable=True)             # e.g. "Wikipedia" | "Semantic Scholar"
    evidence_snippet = Column(Text, nullable=True)

    query = relationship("QueryRecord", back_populates="claims")
