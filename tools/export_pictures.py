"""Export the picture field pairs used by the lab's "Pictures" tour into data/lab/pictures.json.

Each pair is two 42-mode LP fields whose near-field intensities both draw a picture and are practically
identical, while their modal compositions differ (the construction of the paper's Figure 1, select_pair.py).
Coefficients multiply the unit-norm LP modes of the lab's 42-mode fibre (core 12.5 um, NA 0.10, 633 nm).
The paper stores pairs for Psi and lambda; tools/pictures/ regenerates pairs for the other pictures.

    ~/venvs/optics/bin/python tools/export_pictures.py \
        ~/github/mode-decomposition-with-optimized-pairwise-mode-products/reporting/figure1/figure/data/pairs tools/pictures/pairs
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ORDER = ["psi", "lambda", "pi", "A", "heart", "star", "yinyang", "smiley", "moon", "arrow", "ring_gap", "two_spots"]
LABELS = {"psi": "Ψ", "lambda": "λ", "pi": "π", "A": "A", "heart": "heart", "star": "star", "yinyang": "yin-yang",
          "smiley": "smiley", "moon": "moon", "arrow": "arrow", "ring_gap": "broken ring", "two_spots": "two spots"}
MIN_TARGET = 0.85  # pairs that draw their picture less faithfully than this are left out


def main():
    folders = [Path(a).expanduser() for a in sys.argv[1:]]
    found = {}
    for folder in folders:
        for path in sorted(folder.glob("*_pair.npz")):
            name = path.name[: -len("_pair.npz")]
            found.setdefault(name, path)  # earlier folders (the paper's own pairs) win

    pictures = []
    for name in sorted(found, key=lambda n: ORDER.index(n) if n in ORDER else len(ORDER)):
        z = np.load(found[name])
        info = json.loads(found[name].with_suffix(".json").read_text())["selected"]
        a, b = z["cA"].astype(np.complex128), z["cB"].astype(np.complex128)
        a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
        b *= np.exp(-1j * np.angle(np.vdot(a, b)))  # global phase of B aligned to A: shortest path between them
        target = min(info["target_A"], info["target_B"])
        keep = target >= MIN_TARGET
        print(f"{name:10s} target {target:.3f} twin {info['pair_similarity']:.6f} modal TV {info['modal_TV']:.2f}"
              f"{'' if keep else '  (left out)'}")
        if keep:
            pictures.append({"name": name, "label": LABELS.get(name, name), "target": target,
                             "twin": info["pair_similarity"], "modalTV": info["modal_TV"],
                             "A": np.stack([a.real, a.imag], -1).ravel().round(7).tolist(),
                             "B": np.stack([b.real, b.imag], -1).ravel().round(7).tolist()})
    out = ROOT / "data" / "lab" / "pictures.json"
    out.write_text(json.dumps({"modes": 42, "pictures": pictures}, separators=(",", ":")))
    print(f"wrote {len(pictures)} pictures to {out}")


if __name__ == "__main__":
    main()
