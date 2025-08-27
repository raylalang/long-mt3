# Simple calculator to check frame index from time;
import math

fs = 16000
hop = 320
onset_sec = 1.000
frame_idx_expected = int(round(onset_sec * fs / hop))
print("fs:", fs, "hop:", hop, "onset_sec:", onset_sec)
print("Expected frame idx for 1.000s:", frame_idx_expected)
