from enum import Enum

class RecorderState(Enum):
    WAITING = 1
    RECORDING = 2
    ENDING = 3