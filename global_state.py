from datatypes import ProcessingState
from abc import ABC, abstractmethod


class Observer(ABC):
    @abstractmethod
    def update(cls, state: ProcessingState):
        pass


class GlobalState:
    """
    A global state to keep track of the current state of the server.
    Used by the server state machine to keep track of the current state of the server.
    """
    _state: ProcessingState = ProcessingState.EMPTY

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
