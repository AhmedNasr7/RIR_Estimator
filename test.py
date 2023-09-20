import numpy as np
import torchaudio 
from loguru import logger





if __name__ == "__main__":

    data_path = "./data/"

    audio_file = data_path + "Accordion Solo_0_1sec.wav"
    rir_file = data_path + "Accordion Solo_0.npy"

    audio, sr = torchaudio.load(audio_file)

    logger.info(f"audio: {audio.shape},  {sr}")
    logger.info(f"audio tensor stats: {audio.mean(), audio.max(), audio.min()}")

    rir = np.load(rir_file)

    logger.info(f"rir: {rir.shape}")
    logger.info(f"rir arr stats: {rir.mean(), rir.max(), rir.min()}")



