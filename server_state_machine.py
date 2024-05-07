import shutil
from enum import Enum
from fastapi import BackgroundTasks
import baggage_processing


class ProcessingState(Enum):
    EMPTY = 0
    PROCESSING = 1
    COMPLETED = 2
    ABORTED = 3


class ServerStateMachine:
    _state: ProcessingState = ProcessingState.EMPTY
    _model_name = 'rtdetr-x.pt'

    @classmethod
    def set_state(cls, new_state, **kwargs) -> dict:
        """
        Set the state of the server state machine
        :param state:
        :param kwargs:
        :return:
        """
        if cls._state == ProcessingState.PROCESSING and new_state == ProcessingState.PROCESSING:
            raise ValueError("Cannot set state to PROCESSING when already processing")

        print("Setting state to: ", new_state)
        cls._state = new_state
        if cls._state == ProcessingState.EMPTY:
            return {"status": cls._state.name}
        elif cls._state == ProcessingState.PROCESSING:
            background_tasks: BackgroundTasks = kwargs.get("background_tasks")
            save_path = kwargs.get("save_path")
            model_name = kwargs.get("model_name")

            # run the tracking in the background
            # to avoid blocking the main thread
            background_tasks.add_task(baggage_processing.track_objects, save_path, model_name)
            return {"status": cls._state.name, "save_path": save_path, "model_name": model_name}
        elif cls._state == ProcessingState.COMPLETED:
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
