import datetime
import random
import string
from pathlib import Path
from typing import Dict

from fastapi import BackgroundTasks, Depends

from task_processors import baggage_processing, fight_processing
import database.database
import models
from database import schemas, crud
from database.database import get_db
from datatypes import ProcessingState, TaskEnum
from state.global_state import GlobalState, Observer


class ServerStateMachine(Observer):
    _model_name = 'rtdetr-x.pt'
    _db = None
    _db_video = None
    _db_task = None
    _output_dir = None
    _save_path = None

    @classmethod
    def set_state(cls, db: database.database.SessionLocal = Depends(get_db),
                  new_state: ProcessingState = ProcessingState.EMPTY,
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
        if db is not None:
            cls._db = db

        if GlobalState.get_state() == ProcessingState.EMPTY:
            return {"status": GlobalState.get_state().name}

        elif GlobalState.get_state() == ProcessingState.PROCESSING:
            if cls._db is None:
                raise ValueError("Database session not provided")
            background_tasks: BackgroundTasks = kwargs.get("background_tasks")
            save_path = kwargs.get("save_path").replace('\\\\', "\\")
            model_name = kwargs.get("model_name")

            if background_tasks is None:
                raise ValueError("Background tasks not provided")

            cls._db_task = task

            video_id = GlobalState.get_video_id()
            print("Processing video ID: ", video_id, "Save path: ", save_path, "Model name: ", model_name, "Task: ", task.name)

            # add database entry
            video = schemas.VideoEntryCreate(video_id=video_id, file_name=save_path, state=GlobalState.get_state().name,
                                             model_name=model_name,
                                             task=task.name,
                                             upload_timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                             num_frames=0)
            cls._db_video = crud.create_video(video=video, db=cls._db)

            # run the tracking in the background
            # to avoid blocking the main thread
            # TODO: Add support for other tasks here
            if task == TaskEnum.Baggage:
                print(f'Received task: {task.name}')
                background_tasks.add_task(baggage_processing.track_objects, save_path, model_name)
            elif task == TaskEnum.Fight:
                print(f'Received task: {task.name}')
                background_tasks.add_task(fight_processing.detect_fights, save_path, model_name)
            else:
                raise ValueError("Invalid task")
            return {"status": GlobalState.get_state().name, "save_path": save_path, "model_name": model_name,
                    "video_id": video_id}
        elif GlobalState.get_state() == ProcessingState.COMPLETED:
            # set the state of the video in the database
            # get video id from _db_video
            video_id = cls._db_video.id
            video = schemas.VideoEntryUpdateState(id=video_id, file_name=str(cls._save_path),
                                                  state=GlobalState.get_state().name,
                                                  model_name=cls._model_name,
                                                  task=task.name,
                                                  num_frames=GlobalState.get_frame_count())
            cls._db_video = crud.update_video_state(video=video, db=cls._db)

            return {"status": GlobalState.get_state().name, "output_dir": cls._output_dir}
        elif GlobalState.get_state() == ProcessingState.ABORTED:
            baggage_processing.ABORT_FLAG = True

            # set the state of the video in the database
            # get video id from _db_video
            video_id = cls._db_video.id
            video = schemas.VideoEntryUpdateState(id=video_id, file_name=str(cls._save_path),
                                                  state=GlobalState.get_state().name,
                                                  model_name=cls._model_name,
                                                  task=task.name,
                                                  num_frames=GlobalState.get_frame_count())
            cls._db_video = crud.update_video_state(video=video, db=cls._db)

            return {"status": GlobalState.get_state().name, "output_dir": cls._output_dir,
                    "video_id": GlobalState.get_video_id()}

    @classmethod
    def update(cls, state: ProcessingState, frame_count: int, frames_to_process: int, output_dir: str,
               background_tasks=BackgroundTasks):
        print("State updated to: ", state)
        cls.set_state(db=cls._db, new_state=state, task=TaskEnum.Baggage, background_tasks=background_tasks,
                      save_path=cls._save_path, model_name=cls._model_name)

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
        cls.set_state(db=cls._db, new_state=ProcessingState.ABORTED, task=cls._db_task, )
        return {"status": GlobalState.get_state().name, "output_dir": GlobalState.get_output_dir(), "video_id": GlobalState.get_video_id()}

    @classmethod
    def set_complete(cls):
        GlobalState.set_state(ProcessingState.COMPLETED)
        cls.set_state(db=cls._db, new_state=ProcessingState.COMPLETED, task=cls._db_task)

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
        status_dict = {"status": GlobalState.get_state().name, "model": cls._model_name,
                       }
        if GlobalState.get_state() != ProcessingState.EMPTY:
            status_dict["frame_count"] = GlobalState.get_frame_count()
            status_dict["frames_to_process"] = GlobalState.get_frames_to_process()
            status_dict["output_dir"] = GlobalState.get_output_dir()
            status_dict["video_id"] = GlobalState.get_video_id()
        return status_dict

    @classmethod
    def get_save_path(cls, filename) -> str:
        # create 6 character random folder name
        random_folder_name = ''.join(random.choices(string.ascii_lowercase + string.digits, k=10))

        # create folder
        cls._output_dir = Path(f"temp_videos/{random_folder_name}")

        # create the folder if it doesn't exist
        cls._output_dir.mkdir(parents=True, exist_ok=True)

        # get output directory path as a string
        output_dir_str = str(cls._output_dir)

        GlobalState.set_output_dir(output_dir_str)
        GlobalState.set_video_id(random_folder_name)

        # save the video to the folder, with the same name as the folder
        cls._save_path = cls._output_dir / f'{random_folder_name}.{filename.split(".")[-1]}'

        print(f"Save path: {cls._save_path}")

        return str(cls._save_path)
