from enum import Enum


class ProcessingState(Enum):
    EMPTY = 0
    PROCESSING = 1
    COMPLETED = 2
    ABORTED = 3


class GlobalState:
    _state: ProcessingState = ProcessingState.EMPTY

    @classmethod
    def get_state(cls):
        return cls._state

    @classmethod
    def set_state(cls, state: ProcessingState):
        cls._state = state


class TaskEnum(Enum):
    Baggage = "Baggage"
    Fall = "Fall"
    Loitering = "Loitering"
    Fight = "Fight"
    Count = "Count"
