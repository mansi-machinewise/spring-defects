from __future__ import annotations

from datetime import datetime
from pathlib import Path
from sqlalchemy import DateTime, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


class Base(DeclarativeBase):
    pass


class InspectionRecord(Base):
    __tablename__ = "inspection_results"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    inspection_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime)
    spring_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    batch_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    decision: Mapped[str] = mapped_column(String(10))
    payload_json: Mapped[str] = mapped_column(Text)
    report_path: Mapped[str | None] = mapped_column(String(512), nullable=True)


class InspectionRepository:
    def __init__(self, database_path: str = "data/inspection.db") -> None:
        Path(database_path).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{database_path}", connect_args={"check_same_thread": False})
        Base.metadata.create_all(self.engine)

    def save(self, result, payload: str) -> None:
        with Session(self.engine) as session:
            session.add(InspectionRecord(
                inspection_id=result.inspection_id, timestamp=result.timestamp,
                spring_id=result.spring_id, batch_id=result.batch_id,
                decision=result.decision.value, payload_json=payload, report_path=result.report_path,
            ))
            session.commit()

    def latest(self, limit: int = 20) -> list[InspectionRecord]:
        with Session(self.engine) as session:
            return list(session.query(InspectionRecord).order_by(InspectionRecord.id.desc()).limit(limit))
