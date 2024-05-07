import shutil
from enum import Enum
from fastapi import BackgroundTasks
import baggage_processing
import database.database
from database import schemas, crud


class ProcessingState(Enum):
    EMPTY = 0
    PROCESSING = 1
    COMPLETED = 2
    ABORTED = 3


class TaskEnum(Enum):
    Baggage = "Baggage"
    Fall = "Fall"
    Loitering = "Loitering"
    Fight = "Fight"
    Count = "Count"


class ServerStateMachine:
    _state: ProcessingState = ProcessingState.EMPTY
    _model_name = 'rtdetr-x.pt'
    _db = None
    _db_video = None

    @classmethod
    def set_state(cls, db: database.database.SessionLocal, new_state: ProcessingState, task: TaskEnum = TaskEnum.Baggage,
                  **kwargs) -> dict:
        """
        Set the state of the server state machine
        :param new_state:
        :param task: Of type TaskEnum. The task to be performed on the video.
        :param db: The database session. Required.
        :param kwargs: Additional arguments, directly passed to the background task.
        :return:
        """
        if cls._state == ProcessingState.PROCESSING and new_state == ProcessingState.PROCESSING:
            raise ValueError("Cannot set state to PROCESSING when already processing")

        print("Setting state to: ", new_state)
        cls._state = new_state
        cls._db = db

        if cls._state == ProcessingState.EMPTY:
            return {"status": cls._state.name}
        elif cls._state == ProcessingState.PROCESSING:
            if cls._db is None:
                raise ValueError("Database session not provided")
            background_tasks: BackgroundTasks = kwargs.get("background_tasks")
            save_path = kwargs.get("save_path")
            model_name = kwargs.get("model_name")

            # add database entry
            video = schemas.VideoEntryCreate(file_name=save_path, state=cls._state.name, model_name=model_name)
            cls._db_video = crud.create_video(video=video, db=cls._db)

            # run the tracking in the background
            # to avoid blocking the main thread
            # TODO: Add support for other tasks
            if task == TaskEnum.Baggage:
                background_tasks.add_task(baggage_processing.track_objects, save_path, model_name)
            else:
                raise ValueError("Invalid task")
            return {"status": cls._state.name, "save_path": save_path, "model_name": model_name}
        elif cls._state == ProcessingState.COMPLETED:
            # set the state of the video in the database
            # get video id from _db_video
            video_id = cls._db_video.id

            # update the video state in the database
            video = schemas.VideoEntry(id=video_id, file_name=cls._db_video.file_name, state=cls._state.name,
                                       model_name=cls._db_video.model_name)
            cls._db_video = crud.update_video(video=video, db=cls._db)

            return {"status": cls._state.name, "output_dir": baggage_processing.output_dir}
        elif cls._state == ProcessingState.ABORTED:
            baggage_processing.ABORT_FLAG = True
            return {"status": cls._state.name, "output_dir": baggage_processing.output_dir}

    @classmethod
    def get_state(cls):
        return cls._state

    @classmethod
    def is_processing(cls):
        return cls._state == ProcessingState.PROCESSING

    @classmethod
    def is_completed(cls):
        return cls._state == ProcessingState.COMPLETED

    @classmethod
    def is_aborted(cls):
        return cls._state == ProcessingState.ABORTED

    @classmethod
    def abort(cls):
        cls._state = ProcessingState.ABORTED
        return {"status": cls._state.name, "output_dir": baggage_processing.output_dir}

    @classmethod
    def set_model(cls, model_name):
        if cls._state == ProcessingState.PROCESSING:
            raise ValueError("Cannot change the model while processing a video")

        if model_name not in baggage_processing.model_names:
            raise ValueError("Invalid model name")

        cls._model_name = model_name
        return {"status": "accepted", "model": cls._model_name}

    @classmethod
    def get_status(cls):
        """
        Used for /status endpoint
        :return:
        """
        status_dict = {"status": cls._state.name, "model": cls._model_name, }
        if cls._state == ProcessingState.PROCESSING:
            status_dict["frame_count"] = baggage_processing.FRAME_COUNT
            status_dict["frames_to_process"] = baggage_processing.FRAMES_TO_PROCESS
        return status_dict
