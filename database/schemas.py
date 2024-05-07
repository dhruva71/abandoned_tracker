from pydantic import BaseModel

from datatypes import TaskEnum


class VideoEntryBase(BaseModel):
    file_name: str
    state: str
    video_id: str
    model_name: str
    task: str


class VideoEntryCreate(VideoEntryBase):
    upload_timestamp: str
    task: str

    class Config:
        orm_mode = True


class VideoEntry(VideoEntryBase):
    id: int
    num_frames: int
    upload_timestamp: str

    class Config:
        orm_mode = True


class VideoEntryUpdateState(VideoEntryBase):
    state: str

    class Config:
        orm_mode = True


class TaskBase(BaseModel):
    task_name: str

    class Config:
        orm_mode = True


class TaskCreate(TaskBase):
    task_name: TaskEnum
    video_id: int


class Task(TaskBase):
    id: int
    video_id: int
    task_name: TaskEnum

    class Config:
        orm_mode = True
