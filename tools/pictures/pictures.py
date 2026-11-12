"""Target pictures drawn at 4x and downsampled; `scale` is the outer radius in output pixels."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os, matplotlib
FONT = os.path.join(os.path.dirname(matplotlib.__file__), 'mpl-data', 'fonts', 'ttf', 'DejaVuSans-Bold.ttf')  # the font the paper used

def draw(kind, n, scale):
    s = 4 * n; im = Image.new('L', (s, s)); d = ImageDraw.Draw(im); c = s / 2; k = 4 * scale / 2.4   # shapes span ~2.4k
    if kind == 'heart':
        t = np.linspace(0, 2 * np.pi, 400); hx = 16 * np.sin(t)**3; hy = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
        d.polygon([(c + k * .135 * a, c - k * .135 * b - .15 * k) for a, b in zip(hx, hy)], fill=255)
    elif kind == 'star':
        d.polygon([(c + k * (2.3 if i % 2 == 0 else .95) * np.sin(np.pi * i / 5), c + .15 * k - k * (2.3 if i % 2 == 0 else .95) * np.cos(np.pi * i / 5)) for i in range(10)], fill=255)
    elif kind == 'yinyang':
        d.pieslice([c - 2.3 * k, c - 2.3 * k, c + 2.3 * k, c + 2.3 * k], 90, 270, fill=255)
        d.ellipse([c - 1.15 * k, c - 2.3 * k, c + 1.15 * k, c], fill=255); d.ellipse([c - 1.15 * k, c, c + 1.15 * k, c + 2.3 * k], fill=0)
        d.ellipse([c - .35 * k, c - 1.5 * k, c + .35 * k, c - .8 * k], fill=0); d.ellipse([c - .35 * k, c + .8 * k, c + .35 * k, c + 1.5 * k], fill=255)
    elif kind == 'ring_gap':
        d.ellipse([c - 2.2 * k, c - 2.2 * k, c + 2.2 * k, c + 2.2 * k], fill=255); d.ellipse([c - 1.2 * k, c - 1.2 * k, c + 1.2 * k, c + 1.2 * k], fill=0)
        d.rectangle([c + .5 * k, c - .45 * k, c + 2.5 * k, c + .45 * k], fill=0)
    elif kind in ('A', 'lambda', 'pi', 'psi'):
        txt = {'A': 'A', 'lambda': 'λ', 'pi': 'π', 'psi': 'Ψ'}[kind]
        f = ImageFont.truetype(FONT, int(4.4 * k)); bb = d.textbbox((0, 0), txt, font=f)
        d.text((c - (bb[0] + bb[2]) / 2, c - (bb[1] + bb[3]) / 2), txt, fill=255, font=f)
    elif kind == 'smiley':
        d.ellipse([c - 1.3 * k, c - 1.5 * k, c - .5 * k, c - .7 * k], fill=255); d.ellipse([c + .5 * k, c - 1.5 * k, c + 1.3 * k, c - .7 * k], fill=255)
        d.arc([c - 1.9 * k, c - 1.6 * k, c + 1.9 * k, c + 1.9 * k], 20, 160, fill=255, width=int(.55 * k))
    elif kind == 'two_spots':
        d.ellipse([c - 2.2 * k, c - .6 * k, c - 1.0 * k, c + .6 * k], fill=255); d.ellipse([c + .6 * k, c - 1.6 * k, c + 2.2 * k, c + 0. * k], fill=255)
    elif kind == 'arrow':
        d.polygon([(c - 2.1 * k, c - .4 * k), (c + .5 * k, c - .4 * k), (c + .5 * k, c - 1.2 * k), (c + 2.2 * k, c), (c + .5 * k, c + 1.2 * k), (c + .5 * k, c + .4 * k), (c - 2.1 * k, c + .4 * k)], fill=255)
    elif kind == 'moon':
        d.ellipse([c - 2.2 * k, c - 2.2 * k, c + 2.2 * k, c + 2.2 * k], fill=255); d.ellipse([c - 1.1 * k, c - 2.4 * k, c + 2.9 * k, c + 1.6 * k], fill=0)
    return np.asarray(im.resize((n, n), Image.LANCZOS), float) / 255
