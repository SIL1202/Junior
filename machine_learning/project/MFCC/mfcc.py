# convert wav file to png file
import matplotlib.pyplot as plt
import librosa.display
import librosa
import numpy as np

print("Loading file...")
y, sr = librosa.load("assets/file.wav", sr=16000)

print("Computing STFT...")
stft = np.abs(librosa.stft(y))

print("Computing mel spectrogram...")
mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)

print("Computing MFCC...")
mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), sr=sr, n_mfcc=13)

print("Computing delta / delta-delta...")
delta = librosa.feature.delta(mfcc)
delta2 = librosa.feature.delta(mfcc, order=2)

print("Saving files...")
np.save("objects/mfcc.npy", mfcc)
np.save("objects/delta.npy", delta)
np.save("objects/delta2.npy", delta2)

print("Done!")

plt.figure(figsize=(10, 4))
librosa.display.specshow(mfcc, x_axis="time")
plt.colorbar()
plt.title("MFCC")
plt.tight_layout()
plt.savefig("assets/mfcc.png")
plt.close()
