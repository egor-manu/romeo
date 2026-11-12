"""Export the published results used by the interactive charts on the ROMEO page.

Reads the compact data folders of the paper repository and writes data/paper.json + data/paper.bin:
held-out experimental modal coefficients and scores (SLM and diffuser), the calibrated PMP spectra,
the simulated noise statistics, and the matched mask controls. Fields are displayed in the released
LP synthesis basis (fixed_operator.npz of the data release).

    ~/venvs/optics/bin/python tools/export_paper_data.py \
        --paper ~/github/mode-decomposition-with-optimized-pairwise-mode-products
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

CASES = {"slm": "slm_fig1_20261018", "diffuser": "diffuser_20261001"}
RELEASE_BASIS = "build/paper_release/data/experimental/paper_handoff_632p8nm_full_20261019/cases/native_duo_20261018/fixed_operator.npz"
# The release labels the last four components LP72 and LP81. Their angular and radial structure in the
# released basis is LP81 and LP91, which is also the guided set at V = 12.4 (LP72 is beyond cut-off).
RELABEL = {"LP72-1": "LP81-1", "LP72-2": "LP81-2", "LP81-1": "LP91-1", "LP81-2": "LP91-2"}


def lp_basis(paper):
    modes = np.load(paper / RELEASE_BASIS)["lp_modes"]
    assert np.abs(modes.imag).max() == 0
    return modes.real.astype(np.float32)


def aligned_nmse(u, v):
    """NMSE of u against v after the optimal complex scalar, per row."""
    alpha = (u.conj() * v).sum(1) / (u.conj() * u).sum(1)
    return (np.abs(alpha[:, None] * u - v) ** 2).sum(1) / (np.abs(v) ** 2).sum(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", type=Path, required=True)
    args = parser.parse_args()
    paper = args.paper.expanduser().resolve()
    data = paper / "data"

    blobs, offset = [], 0

    def add(array):
        nonlocal offset
        raw = np.ascontiguousarray(array, dtype=np.float32).tobytes()
        blobs.append(raw)
        entry = {"offset": offset, "length": int(np.asarray(array).size)}
        offset += len(raw)
        return entry

    basis = lp_basis(paper)
    meta = {"basis": {"pixels": int(basis.shape[-1]), "modes": int(basis.shape[0]), "fields": add(basis)}, "experiment": {}}

    for key, folder in CASES.items():
        case = data / "experimental_20261019_632p8nm" / folder
        labels = json.loads((case / "mode_labels.json").read_text())
        labels = labels[:-4] + [RELABEL[name] for name in labels[-4:]]
        assert len(labels) == basis.shape[0], (len(labels), basis.shape)
        coefficients = np.load(case / "modal_coefficients_eligible_heldout.npz")
        holographic = coefficients["holographic_coefficients"]
        recovered = coefficients["intensity_only_coefficients"]
        rows = {int(r["source_index"]): r for r in csv.DictReader((case / "heldout_scores.csv").open())}
        picked = [rows[int(i)] for i in coefficients["source_indices"]]
        assert all(r["eligible_reference"] == "True" for r in picked)
        nmse = np.array([float(r["nmse"]) for r in picked])
        nmse_lp = np.array([float(r["nmse_vs_lp_projection"]) for r in picked])
        check = aligned_nmse(recovered, holographic)
        print(f"{key}: {len(picked)} fields, median NMSE {np.median(nmse):.4f}, "
              f"vs 42-mode projection {np.median(nmse_lp):.4f} (coefficient check {np.median(check):.4f})")
        metrics = json.loads((case / "metrics.json").read_text())
        eigenvalues = np.sort(np.load(case / "pmp_gram_eigenvalues.npy"))[::-1]
        singular = np.sqrt(eigenvalues / eigenvalues[0])
        meta["experiment"][key] = {
            "count": len(picked),
            "labels": labels,
            "kappa": float(metrics["pmp_condition_reported"]),
            "medianNmse": float(metrics["field_nmse"]["median"]),
            "medianCoherence": float(metrics["field_coherence"]["median"]),
            "launch": [int(r["launch_index"]) for r in picked],
            "state": [int(r["mpc_index"]) for r in picked],
            "nmse": add(nmse),
            "nmseLp": add(nmse_lp),
            "coherence": add([float(r["coherence"]) for r in picked]),
            "holographic": add(np.stack([holographic.real, holographic.imag], -1)),
            "recovered": add(np.stack([recovered.real, recovered.imag], -1)),
            "singular": add(singular),
        }

    stats = list(csv.DictReader((data / "direct_statistics_20261022" / "summary.csv").open()))
    meta["simulation"] = [{
        "modes": int(r["modes"]), "radiusUm": float(r["radius_um"]), "snr": float(r["target_snr_db"]),
        "median": float(r["coefficient_nmse_median"]), "p05": float(r["coefficient_nmse_p05"]),
        "p95": float(r["coefficient_nmse_p95"]), "field": float(r["field_nmse_median"]),
    } for r in stats]
    offsets = json.loads((data / "direct_statistics_20261022" / "snr_peak_offset.json").read_text())
    print("snr/peak offset keys:", list(offsets)[:6])

    controls = []
    for r in csv.DictReader((data / "roi_mask_controls_20261015" / "comparison.csv").open()):
        name = r["name"]
        if name == "flat_same_relay":
            family, grain = "flat", None
        elif r["family"] == "optimized":
            family, grain = "optimised", None
        elif r["family"] == "random" and r["grain"]:
            family, grain = "random", int(float(r["grain"]))
        else:
            continue
        controls.append({"family": family, "grain": grain, "capture": float(r["soft_roi_mean"]),
                         "worst": float(r["soft_roi_worst_state"]), "kappa": float(r["full_sensor_condition"]),
                         "peak": float(r["worst_peak_to_mean"])})
    meta["controls"] = controls
    counts = {}
    for c in controls:
        counts[(c["family"], c["grain"])] = counts.get((c["family"], c["grain"]), 0) + 1
    print("controls:", counts)

    (ROOT / "data" / "paper.bin").write_bytes(b"".join(blobs))
    (ROOT / "data" / "paper.json").write_text(json.dumps(meta, separators=(",", ":")))
    print(f"wrote {offset / 1e6:.2f} MB binary, {len(json.dumps(meta)) / 1e3:.0f} kB json")


if __name__ == "__main__":
    main()
