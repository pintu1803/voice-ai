# audio_utils.py
import numpy as np
from scipy.io.wavfile import write
from common.config import settings

def save_wav(filename, audio):
    """audio: float32 torch tensor or numpy array in [-1, 1]"""
    if hasattr(audio, "numpy"):          # torch tensor
        audio = audio.numpy()
    audio = np.clip(audio, -1.0, 1.0)    # avoid int16 overflow on loud peaks
    write(filename, settings.sample_rate, (audio * 32767).astype(np.int16))