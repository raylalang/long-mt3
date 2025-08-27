# Simple calculator to check frame index from time; edit fs/hop to your actual settings
import math

fs = 16000
hop = 320  # 20 ms hop; change to your spectrogram hop
onset_sec = 1.000
frame_idx_expected = int(round(onset_sec * fs / hop))
print("fs:", fs, "hop:", hop, "onset_sec:", onset_sec)
print("Expected frame idx for 1.000s:", frame_idx_expected)
print(
    "[NOTE] Adjust fs/hop to match your feature extractor; expect ±1 frame agreement in labels."
)
