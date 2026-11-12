"""Website copy of the paper's reporting/figure1/figure/select_pair.py (unchanged except that, when no pair
reaches the target-similarity threshold, it keeps the most picture-like twin pair instead of failing).
Run for the lab: python tools/pictures/select_pair.py ~/github/pmp_metal tools/pictures/pairs <picture> 0.99997 40 3 10 0.93

Joint optimization of two 42-mode LP fields whose near-field intensities both match a target picture and each
other, with modal powers pushed apart. Full-resolution 200^2 core window of the pmp_metal fibre grid.
Usage: python pair_joint.py <pmp_metal> <out> <picture> <min_pair_similarity> <independent_starts> <joint_starts> <threads> [min_target_similarity]"""
import itertools, json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
from md_classes import MD_data_handler
sys.path.insert(0, str(Path(__file__).parent)); from pictures import draw
torch.set_default_dtype(torch.float64); NAME = sys.argv[3]; target_pair = float(sys.argv[4]); NS = int(sys.argv[5]); NJ = int(sys.argv[6]); torch.set_num_threads(int(sys.argv[7])); MIN_T = float(sys.argv[8]) if len(sys.argv) > 8 else .94
h = MD_data_handler(device=torch.device('cpu')); h.calculate_LP_modes(12.5, 1.46, .10, .633, 256)
full = h.fiber_modes.numpy().astype(np.complex128)                    # (42, 256, 256), as in the relay
norms = np.linalg.norm(full.reshape(42, -1), axis=1)
W = slice(28, 228); modes = full[:, W, W] / norms[:, None, None]       # unit-norm modes on the full grid, cropped
n = modes.shape[1]; Ut = torch.tensor(modes.reshape(42, -1))
core_px = 12.5 / h.dx
T = draw(NAME, n, scale=core_px * .85); t = torch.tensor(T.ravel()); t = t / t.norm()
def I(c): return (torch.complex(c[:42], c[42:]) @ Ut).abs()**2
def ncos(a, b): return (a @ b) / a.norm() / b.norm()
def powers(c): p = c[:42]**2 + c[42:]**2; return p / p.sum()

def single(seed):
    g = torch.Generator().manual_seed(seed); c = torch.randn(84, generator=g).requires_grad_()
    opt = torch.optim.LBFGS([c], max_iter=500, line_search_fn='strong_wolfe', tolerance_grad=1e-12, tolerance_change=1e-14)
    def loss(): opt.zero_grad(); L = 1 - ncos(I(c), t); L.backward(); return L
    opt.step(loss); return c.detach()
fits = [single(s) for s in range(NS)]
Is = [I(c) for c in fits]
best = max(((float((powers(fits[i]) - powers(fits[j])).abs().sum() / 2), i, j) for i, j in itertools.combinations(range(NS), 2)
            if ncos(Is[i], Is[j]) >= .98), key=lambda z: z[0])
print('independent best pair TV', round(best[0], 3), flush=True)

def joint(cA, cB, mu, w=200.):
    p = torch.cat([cA, cB]).clone().requires_grad_()
    opt = torch.optim.LBFGS([p], max_iter=800, line_search_fn='strong_wolfe', tolerance_grad=1e-12, tolerance_change=1e-15)
    def loss():
        opt.zero_grad(); a, b = p[:84], p[84:]; IA, IB = I(a), I(b)
        overlap = (powers(a) * powers(b)).sqrt().sum()                 # Bhattacharyya coefficient of modal powers
        L = (1 - ncos(IA, t)) + (1 - ncos(IB, t)) + w * (1 - ncos(IA, IB)) + mu * overlap
        L.backward(); return L
    opt.step(loss); return p.detach()[:84], p.detach()[84:]

rows = []
starts = sorted(((float((powers(fits[i]) - powers(fits[j])).abs().sum() / 2), i, j) for i, j in itertools.combinations(range(NS), 2)
                 if ncos(Is[i], Is[j]) >= .97), key=lambda z: -z[0])[:NJ]
for _, i, j in starts:
    for mu in (0.02, 0.05, 0.1, 0.2, 0.4):
        a, b = joint(fits[i], fits[j], mu)
        IA, IB = I(a), I(b)
        row = dict(start=(i, j), mu=mu, pair_similarity=float(ncos(IA, IB)), target_A=float(ncos(IA, t)), target_B=float(ncos(IB, t)),
                   modal_TV=float((powers(a) - powers(b)).abs().sum() / 2), max_power=float(max(powers(a).max(), powers(b).max())),
                   field_overlap=float(abs(np.vdot((a[:42] + 1j * a[42:]).numpy(), (b[:42] + 1j * b[42:]).numpy())) / a.norm() / b.norm()))
        rows.append((row, a, b)); print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
ok = [r for r in rows if r[0]['pair_similarity'] >= target_pair and min(r[0]['target_A'], r[0]['target_B']) >= MIN_T]
if ok:
    row, a, b = max(ok, key=lambda r: r[0]['modal_TV'])
else:  # website copy: no pair reaches MIN_T, so keep the most picture-like pair among the twins
    twins = [r for r in rows if r[0]['pair_similarity'] >= target_pair] or rows
    row, a, b = max(twins, key=lambda r: min(r[0]['target_A'], r[0]['target_B']))
cA = (a[:42] + 1j * a[42:]).numpy(); cB = (b[:42] + 1j * b[42:]).numpy()
cA /= np.linalg.norm(cA); cB /= np.linalg.norm(cB)
np.savez(OUT / f'{NAME}_pair.npz', cA=cA, cB=cB, T=T, window=np.array([28, 228]), mode_norms=norms)
(OUT / f'{NAME}_pair.json').write_text(json.dumps(dict(selected=row, independent_best_TV=best[0], all=[r[0] for r in rows],
    picture=NAME, mode_names=h.mode_names, note='coefficients multiply unit-norm LP modes (fiber_modes / ||fiber_modes|| over the 256^2 grid)'), indent=1))
print('selected', row)
