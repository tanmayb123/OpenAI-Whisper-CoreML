import numpy as np
from whisper import audio

m80 = np.load(os.path.join(os.path.dirname(inspect.getfile(audio)), "assets", "mel_filters.npz"))["mel_80"].flatten()
np.save("m80.npy", m80)
