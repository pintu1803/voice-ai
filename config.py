
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SAMPLE_RATE = 16000
CHANNELS = 1
MODEL_PATH = BASE_DIR + "/models/faster-whisper-base"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
DURATION = 10

SPEECH_PROB_THRESHOLD = 0.5
BLOCK_SIZE = 512     # 32ms
SILENCE_TIMEOUT = 800   # milliseconds
PRE_ROLL_CHUNKS = 10

DICTATE_IDLE_TIMEOUT_MS = 10_00