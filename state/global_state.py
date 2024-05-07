from datatypes import ProcessingState
from abc import ABC, abstractmethod


class Observer(ABC):
    @abstractmethod
    def update(cls, state: ProcessingState, frame_count: int, frames_to_process: int, output_dir: str):
        pass


class GlobalState:
    """
    A global state to keep track of the current state of the server.
    Used by the server state machine to keep track of the current state of the server.
    """
    _state: ProcessingState = ProcessingState.EMPTY
    _frame_count: int = 0
    _frames_to_process: int = 0
    _output_dir: str = "../output_frames"
    _video_id: str = ""

    _observers: list[Observer] = []

    @classmethod
    def get_state(cls):
        return cls._state

    @classmethod
    def add_observer(cls, observer: Observer):
        cls._observers.append(observer)

    @classmethod
    def notify_observers(cls):
        for observer in cls._observers:
            observer.update(cls._state)

    @classmethod
    def remove_observer(cls, observer: Observer):
        cls._observers.remove(observer)

    @classmethod
    def clear_observers(cls):
        cls._observers.clear()

    @classmethod
    def set_state(cls, state: ProcessingState):
        cls._state = state
        cls.notify_observers()

    @classmethod
    def reset_state(cls):
        cls._state = ProcessingState.EMPTY
        cls.notify_observers()

    @classmethod
    def is_processing(cls):
        return cls._state == ProcessingState.PROCESSING

    @classmethod
    def is_completed(cls):
        return cls._state == ProcessingState.COMPLETED

    @classmethod
    def get_frame_count(cls):
        return cls._frame_count

    @classmethod
    def get_frames_to_process(cls):
        return cls._frames_to_process

    @classmethod
    def get_output_dir(cls):
        return cls._output_dir

    @classmethod
    def set_frame_count(cls, frame_count: int):
        cls._frame_count = frame_count

    @classmethod
    def set_frames_to_process(cls, frames_to_process: int):
        cls._frames_to_process = frames_to_process

    @classmethod
    def set_output_dir(cls, output_dir: str):
        cls._output_dir = output_dir

    @classmethod
    def set_video_id(cls, video_id: str):
        cls._video_id = video_id

    @classmethod
    def get_video_id(cls):
        return cls._video_id
