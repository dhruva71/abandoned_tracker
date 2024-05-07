import random
import string
from pathlib import Path
from typing import Dict

from fastapi import BackgroundTasks
import baggage_processing
import database.database
import models
from database import schemas, crud
from datatypes import ProcessingState, TaskEnum, GlobalState


class ServerStateMachine:
    _model_name = 'rtdetr-x.pt'
    _db = None
    _db_video = None
    _db_task = None
    _output_dir = None
    _save_path = None

    @classmethod
    def set_state(cls, db: database.database.SessionLocal, new_state: ProcessingState,
                  task: TaskEnum = TaskEnum.Baggage,
                  **kwargs) -> dict:
        """
        Set the state of the server state machine
        :param new_state:
        :param task: Of type TaskEnum. The task to be performed on the video.
        :param db: The database session. Required.
        :param kwargs: Additional arguments, directly passed to the background task.
        :return:
        """
        if GlobalState.get_state() == ProcessingState.PROCESSING and new_state == ProcessingState.PROCESSING:
            raise ValueError("Cannot set state to PROCESSING when already processing")

        print("Setting state to: ", new_state)
        GlobalState.set_state(new_state)
        cls._db = db

        if GlobalState.get_state() == ProcessingState.EMPTY:
            return {"status": GlobalState.get_state().name}
        elif GlobalState.get_state() == ProcessingState.PROCESSING:
            if cls._db is None:
                raise ValueError("Database session not provided")
            background_tasks: BackgroundTasks = kwargs.get("background_tasks")
            save_path = kwargs.get("save_path")
            model_name = kwargs.get("model_name")

            # add database entry
            video = schemas.VideoEntryCreate(file_name=save_path, state=GlobalState.get_state().name, model_name=model_name,
                                             task=task.name)
            cls._db_video = crud.create_video(video=video, db=cls._db)

            # run the tracking in the background
            # to avoid blocking the main thread
            # TODO: Add support for other tasks here
            if task == TaskEnum.Baggage:
                background_tasks.add_task(baggage_processing.track_objects, save_path, model_name)
            else:
                raise ValueError("Invalid task")
            return {"status": GlobalState.get_state().name, "save_path": save_path, "model_name": model_name}
        elif GlobalState.get_state() == ProcessingState.COMPLETED:
            # set the state of the video in the database
            # get video id from _db_video
            video_id = cls._db_video.id

            # update the video state in the database
            video = schemas.VideoEntry(id=video_id, file_name=cls._db_video.file_name, state=GlobalState.get_state().name,
                                       model_name=cls._db_video.model_name, task=task.name)
            cls._db_video = crud.update_video(video=video, db=cls._db)

            return {"status": GlobalState.get_state().name, "output_dir": cls._output_dir}
        elif GlobalState.get_state() == ProcessingState.ABORTED:
            baggage_processing.ABORT_FLAG = True
            return {"status": GlobalState.get_state().name, "output_dir": cls._output_dir}

    @classmethod
    def get_state(cls):
        return GlobalState.get_state()

    @classmethod
    def is_processing(cls):
        return GlobalState.get_state() == ProcessingState.PROCESSING

    @classmethod
    def is_completed(cls):
        return GlobalState.get_state() == ProcessingState.COMPLETED

    @classmethod
    def is_aborted(cls):
        return GlobalState.get_state() == ProcessingState.ABORTED

    @classmethod
    def abort(cls) -> Dict[str, str]:
        GlobalState.set_state(ProcessingState.ABORTED)
        return {"status": GlobalState.get_state().name, "output_dir": baggage_processing.output_dir}

    @classmethod
    def set_model(cls, model_name) -> Dict[str, str]:
        if GlobalState.get_state() == ProcessingState.PROCESSING:
            raise ValueError("Cannot change the model while processing a video")

        if model_name not in models.model_names:
            raise ValueError("Invalid model name")

        cls._model_name = model_name
        return {"status": "accepted", "model": cls._model_name}

    @classmethod
    def get_status(cls) -> Dict[str, str]:
        """
        Used for /status endpoint
        :return:
        """
        status_dict = {"status": GlobalState.get_state().name, "model": cls._model_name, }
        if GlobalState.get_state() == ProcessingState.PROCESSING:
            status_dict["frame_count"] = baggage_processing.FRAME_COUNT
            status_dict["frames_to_process"] = baggage_processing.FRAMES_TO_PROCESS
        return status_dict

    @classmethod
    def get_save_path(cls, file) -> str:
        # create 6 character random folder name
        random_folder_name = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))

        # create folder
        cls._output_dir = Path(f"temp_videos/{random_folder_name}")

        # create the folder if it doesn't exist
        cls._output_dir.mkdir(parents=True, exist_ok=True)

        # save the video to the folder, with the same name as the folder
        cls._save_path = cls._output_dir / f'{random_folder_name}.{file.filename.split(".")[-1]}'

        return str(cls._save_path)
