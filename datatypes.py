from enum import Enum


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
