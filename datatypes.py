from enum import Enum


class ProcessingState(Enum):
    EMPTY = 0
    PROCESSING = 1
    COMPLETED = 2
    ABORTED = 3


class TaskEnum(Enum):
    """
    Used to identify the task of a video
    Possible values: Baggage, Fall, Loitering, Fight, Count
    """
    Baggage = "Baggage"
    Fall = "Fall"
    Loitering = "Loitering"
    Fight = "Fight"
    Count = "Count"


