"""Export the simulated fibres used by the interactive lab on the ROMEO page.

Uses the paper's simulated optical system (pmp_metal `studies.run_42_roi_constrained.setup`: 633 nm, NA 0.10,
lens, 1024^2 phase plane at 8 um, camera 68.59 mm behind it at 5.3 um pitch) with the phase mask of Figure 1
(L1-regularised continuation of the seed-12 mask). Fibres with core radii 4.1, 6, 8 and 12.5 um guide 6, 10,
17 and 42 LP modes, as in the paper's simulations.

For each fibre and each measurement (near field, flat-phase SLM, random mask, Figure 1 mask) it writes the
detector-plane mode fields, the pseudo-inverse of the real PMP Gram matrix (a 42-mode inverse is too slow to
build in a browser), the singular spectrum, the condition number and the light capture. The camera is a
128 x 128 grid over the central 4.07 mm, with fields evaluated at pixel centres; the near field is a 64 x 64
image of the core. Arrays are stored as float16 with one scale per array. Output: data/lab/index.json and one
<name>.json + <name>.bin per fibre, loaded by the page on demand.

    ~/venvs/optics/bin/python tools/export_demo_data.py --pmp-metal ~/github/pmp_metal
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "lab"

FIBRES = [("m06", 4.1), ("m10", 6.0), ("m17", 8.0), ("m42", 12.5)]  # (file stem, core radius in um)
MASK = "reports/42_flat_exterior_seed12/native_l1_1/phase_radians.npy"
CAMERA = 128          # camera pixels per side in the browser
CAMERA_SPAN = 768     # native 5.3 um pixels spanned by the browser camera (central 4.07 mm)
NEAR = 64             # near-field image pixels per side
NEAR_HALF_WIDTH = 1.25  # near-field image half-width in core radii
MASK_VIEW = 160       # mask display pixels per side (central 640^2 phase pixels, 5.12 mm)
RANDOM_CELL = 2       # random-mask cell size in phase pixels (paper: random masks of 1-32 px cells)
# The near field and the flat-phase SLM are ill conditioned, so their inverses are stored in float32 and keep
# singular values down to 2.2e-5 of the largest; the masks are well conditioned and their inverses fit float16.
# Every inverse is built from the float16-rounded fields that the page uses, so the two stay consistent.
INVERSE = {"near": (np.float32, 5e-10), "flat": (np.float32, 5e-10),
           "random": (np.float16, 1e-13), "optimised": (np.float16, 1e-13)}  # (dtype, relative Gram cut-off)


def as_stored(array):
    """Round an array exactly as add() stores it in float16 (one scale per array)."""
    scale = float(np.abs(array).max()) or 1.0
    return (array / scale).astype(np.float16).astype(np.float64) * scale


def names_of(handler):
    return [n.replace("\\mathrm{LP}_{", "LP").replace("\\,", "").replace("}^{\\cos}", "c")
            .replace("}^{\\sin}", "s").replace("}", "") for n in handler.mode_names]


def near_image(modes, radius_um, extent_um):
    """Sample the fibre-grid mode fields on a NEAR x NEAR grid spanning +/- NEAR_HALF_WIDTH core radii."""
    half = NEAR_HALF_WIDTH * radius_um / extent_um
    axis = torch.linspace(-half, half, NEAR)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    grid = torch.stack((xx, yy), -1)[None].expand(len(modes), -1, -1, -1)
    out = F.grid_sample(modes.real[:, None].float(), grid.float(), align_corners=True)[:, 0]
    return out / torch.linalg.vector_norm(out, dim=(1, 2), keepdim=True)


def real_rows(fields):
    """Real PMP rows (pixels x M^2): |F_k|^2, then 2Re(F_k F_j*), -2Im(F_k F_j*) for k < j (page order)."""
    m = fields.shape[0]
    flat = fields.reshape(m, -1).to(torch.complex128)
    cols = [flat[k].abs() ** 2 for k in range(m)]
    for k in range(m):
        for j in range(k + 1, m):
            p = flat[k] * flat[j].conj()
            cols += [2 * p.real, -2 * p.imag]
    return torch.stack(cols, dim=1)


def pseudo_inverse(fields, tol):
    rows = real_rows(fields)
    gram = rows.T @ rows
    evals, evecs = torch.linalg.eigh(gram)
    keep = evals > evals[-1] * tol
    inverse = (evecs[:, keep] / evals[keep]) @ evecs[:, keep].T
    full_rank = bool((evals > evals[-1] * 1e-13).all())  # the reported condition number is the exact one
    kappa = float(torch.sqrt(evals[-1] / evals[0])) if full_rank else math.inf
    singular = torch.sqrt(evals.clamp_min(0).flip(0) / evals[-1])
    return inverse, kappa, singular


def mask_view(phase):
    """Block-averaged phasor angle of the central 640^2 phase pixels, as uint8 (0..255 -> 0..2pi)."""
    lo = 512 - 320
    crop = torch.exp(1j * phase[lo:lo + 640, lo:lo + 640].double())
    b = 640 // MASK_VIEW
    mean = crop.reshape(MASK_VIEW, b, MASK_VIEW, b).mean((1, 3))
    levels = np.round((torch.angle(mean).numpy() % (2 * math.pi)) / (2 * math.pi) * 256).astype(np.int64) % 256
    return levels.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pmp-metal", type=Path, required=True)
    args = parser.parse_args()
    pmp = args.pmp_metal.expanduser().resolve()
    sys.path.insert(0, str(pmp))
    from md_classes import MD_data_handler
    from pmp_capacity import CapacityGeometry, propagate_to_mask
    from studies.run_42_roi_constrained import MatchedFresnel

    torch.set_num_threads(10)
    dev = torch.device("cpu")
    geo = CapacityGeometry(camera_pixels=1024, camera_spacing_um=5.3, mask_pixels=1024, mask_spacing_um=8.,
                           slm_pixel_spacing_um=8., lens_to_mask_um=75_000., mask_to_camera_um=1024 * 8. * 5.3 / .633)
    propagate = MatchedFresnel(1024, 8., 5.3, .633, dev)

    figure1 = torch.from_numpy(np.load(pmp / MASK).astype(np.float32))
    gen = torch.Generator().manual_seed(20261112)
    cells = 1024 // RANDOM_CELL
    random = torch.kron(torch.floor(torch.rand(cells, cells, generator=gen) * 256) / 256 * 2 * math.pi - math.pi,
                        torch.ones(RANDOM_CELL, RANDOM_CELL))
    masks = {"flat": torch.zeros(1024, 1024), "random": random, "optimised": figure1}

    axis = (torch.arange(1024) - 511.5) * 5.3 / 1000.
    r2 = axis[:, None] ** 2 + axis[None, :] ** 2
    soft_roi = torch.exp(-math.log(2.) * (r2 / 2. ** 2) ** 4)  # the paper's soft ROI: half weight at 2 mm
    lo = 512 - CAMERA_SPAN // 2
    step = CAMERA_SPAN // CAMERA
    pick = slice(lo + step // 2, lo + CAMERA_SPAN, step)

    OUT.mkdir(parents=True, exist_ok=True)
    index = {"camera": {"pixels": CAMERA, "spanMm": CAMERA_SPAN * 5.3e-3, "pitchUm": step * 5.3},
             "near": {"pixels": NEAR, "halfWidthCoreRadii": NEAR_HALF_WIDTH}, "mask": {"pixels": MASK_VIEW, "spanMm": 5.12},
             "fibres": []}

    for stem, radius in FIBRES:
        handler = MD_data_handler(device=dev)
        handler.calculate_LP_modes(radius, 1.46, 0.10, 0.633, geo.fiber_pixels)
        modes, names = handler.fiber_modes, names_of(handler)
        extent = float(radius + 20 * 0.633)  # MD_data_handler grid spans +/- (r + 20 lambda)
        M = len(modes)
        print(f"{stem}: core radius {radius} um, {M} modes", flush=True)
        at_mask = propagate_to_mask(modes, float(handler.dx), geo, dev, batch_size=3, optical_layout="lens")
        at_mask = at_mask / torch.linalg.vector_norm(at_mask, dim=(1, 2), keepdim=True)  # unit power per mode

        blobs, offset = [], 0

        def add(array, dtype=np.float16):
            nonlocal offset
            array = np.ascontiguousarray(array)
            scale = float(np.abs(array).max()) if dtype == np.float16 else 1.0
            raw = (array / (scale or 1.0)).astype(dtype).tobytes() if dtype == np.float16 else array.astype(dtype).tobytes()
            blobs.append(raw)
            pad = (-len(raw)) % 4
            blobs.append(b"\0" * pad)
            entry = {"offset": offset, "length": int(array.size), "scale": scale, "type": np.dtype(dtype).name}
            offset += len(raw) + pad
            return entry

        near = near_image(modes, radius, extent)
        fibre = {"label": f"{M} modes", "coreRadiusUm": radius, "modes": M, "names": names,
                 "nearField": add(near.numpy()), "measurements": {}, "masks": {}}

        detectors = {"near": (near.to(torch.complex64), 1.0)}
        for key, phase in masks.items():
            screen = torch.exp(1j * phase)
            cam = torch.empty((M, 1024, 1024), dtype=torch.complex64)
            for s in range(0, M, 6):
                # power per pixel scales with pixel area: rescale so sum |cam|^2 over the plane is 1 per mode
                cam[s:s + 6] = propagate(at_mask[s:s + 6] * screen) * (5.3 / 8.)
            capture = float(((cam.abs() ** 2) * soft_roi).sum((1, 2)).mean())
            # each browser pixel stands for step x step native pixels, so sum |F|^2 is the captured power
            detectors[key] = (cam[:, pick, pick] * step, capture)
            if key != "flat":
                fibre["masks"][key] = add(mask_view(phase), np.uint8)

        iu = torch.triu_indices(M * M, M * M)
        for key, (fields, capture) in detectors.items():
            stored = as_stored(torch.view_as_real(fields.contiguous()).numpy())
            dtype, tol = INVERSE[key]
            inverse, kappa, singular = pseudo_inverse(torch.view_as_complex(torch.from_numpy(stored)), tol)
            print(f"  {key:9s} kappa={kappa:.4g} capture={capture:.3f}", flush=True)
            fibre["measurements"][key] = {
                "grid": int(fields.shape[-1]),
                "fields": add(stored),
                "inverse": add(inverse[iu[0], iu[1]].numpy(), dtype),
                "singular": add(singular.numpy(), np.float32),
                "kappa": None if math.isinf(kappa) else kappa,
                "capture": capture,
            }
        (OUT / f"{stem}.bin").write_bytes(b"".join(blobs))
        (OUT / f"{stem}.json").write_text(json.dumps(fibre, separators=(",", ":")))
        index["fibres"].append({"label": fibre["label"], "modes": M, "coreRadiusUm": radius, "file": stem, "bytes": offset})
        print(f"  wrote {offset / 1e6:.1f} MB", flush=True)

    (OUT / "index.json").write_text(json.dumps(index, indent=1))


if __name__ == "__main__":
    main()
