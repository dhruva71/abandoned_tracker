from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Boolean, Column, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

SQLALCHEMY_DATABASE_URL = "sqlite:///./sql_app.db"

# Database connection
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class VideoEntry(Base):
    __tablename__ = "video_entry"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(String, index=True)
    file_name = Column(String, index=True)
    upload_timestamp = Column(String)
    state = Column(String)
    model_name = Column(String)
    num_frames = Column(Integer)
    # task = relationship("Task", back_populates="video")
    task = Column(String)


# class Task(Base):
#     """
#     Task can be one of the following:
#     Baggage, Fall, Loitering, Fight, Count
#     """
#     id = Column(Integer, primary_key=True, index=True)
#     task_name = Column(String, index=True)
#     video_id = Column(Integer, ForeignKey("video_entry.id"))
#     video = relationship("VideoEntry", back_populates="task")
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
