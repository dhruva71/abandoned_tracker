from sqlalchemy.orm import Session

from . import database, schemas


def get_video(db: Session, video_id: int):
    return db.query(database.VideoEntry).filter(database.VideoEntry.id == video_id).first()


def get_videos(db: Session, skip: int = 0, limit: int = 100):
    return db.query(database.VideoEntry).offset(skip).limit(limit).all()


def create_video(db: Session, video: schemas.VideoEntryCreate):
    db_video = database.VideoEntry(**video.dict())
    db.add(db_video)
    db.commit()
    db.refresh(db_video)
    return db_video


def update_video(db: Session, video: schemas.VideoEntry):
    db_video = db.query(database.VideoEntry).filter(database.VideoEntry.id == video.id).first()
    db_video.file_name = video.file_name
    db_video.state = video.state
    db_video.model_name = video.model_name
    db.commit()
    db.refresh(db_video)
    return db_video


def delete_video(db: Session, video_id: int):
    db_video = db.query(database.VideoEntry).filter(database.VideoEntry.id == video_id).first()
    db.delete(db_video)
    db.commit()
    return db_video


def get_task(db: Session, task_id: int):
    return db.query(database.Task).filter(database.Task.id == task_id).first()


def get_tasks(db: Session, skip: int = 0, limit: int = 100):
    return db.query(database.Task).offset(skip).limit(limit).all()


def create_task(db: Session, task: schemas.TaskCreate):
    db_task = database.Task(**task.dict())
    db.add(db_task)
    db.commit()
    db.refresh(db_task)
    return db_task


def update_task(db: Session, task: schemas.Task):
    db_task = db.query(database.Task).filter(database.Task.id == task.id).first()
    db_task.task_name = task.task_name
    db.commit()
    db.refresh(db_task)
    return db_task


def delete_task(db: Session, task_id: int):
    db_task = db.query(database.Task).filter(database.Task.id == task_id).first()
    db.delete(db_task)
    db.commit()
    return db_task
