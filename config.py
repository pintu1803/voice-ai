
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SAMPLE_RATE = 16000
CHANNELS = 1
MODEL_PATH = BASE_DIR + "/models/faster-whisper-base"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
DURATION = 10