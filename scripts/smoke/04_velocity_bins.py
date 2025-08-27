# Round-trip velocity <-> bin mapping; checks top edge case (127)
import os, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from long_mt3.vocabularies import velocity_to_bin, bin_to_velocity

bins = 32
tests = [0, 1, 32, 64, 96, 126, 127]
for v in tests:
    b_raw = velocity_to_bin(v, bins)
    b = min(b_raw, bins - 1)
    v2 = bin_to_velocity(b, bins)
    print(f"v={v:3d} -> bin={b:2d} (raw={b_raw:2d}) -> v'={v2:3d}")

# Top edge should land on the last valid bin after clamping
assert min(velocity_to_bin(127, bins), bins - 1) == bins - 1
print("[OK]")
