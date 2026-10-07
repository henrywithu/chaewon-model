"""Photographic finishing for the raw Cycles render.

Keeps the raw render untouched and writes a graded version that reads like a
studio fashion photograph: gentle warm/soft-contrast grade, highlight
roll-off, faint lateral chromatic aberration toward the frame edges, a soft
vignette and fine luminance film grain (sized for a ~13MP frame).

Usage: python3 postprocess_photo.py <raw.png> <out_basename>
Writes <out_basename>.jpg (q=95) and <out_basename>.png
"""
import sys
import numpy as np
import cv2

src, out = sys.argv[1], sys.argv[2]
img = cv2.imread(src, cv2.IMREAD_UNCHANGED)
scale = 65535.0 if img.dtype == np.uint16 else 255.0
x = img[..., :3][..., ::-1].astype(np.float32) / scale            # RGB, display-referred (AgX already applied)
H, W = x.shape[:2]
rng = np.random.default_rng(85)

# 1) grade: slight warmth in mids, cool-neutral shadows, soft S-curve, highlight roll-off
lum = x @ np.array([0.2126, 0.7152, 0.0722], np.float32)
mids = np.exp(-((lum - 0.55) / 0.25) ** 2)[..., None]
x = x * (1 + mids * np.array([0.018, 0.004, -0.014], np.float32))
x = np.clip(x, 0, 1)
s = x * x * (3 - 2 * x)                                   # smoothstep S-curve
x = 0.82 * x + 0.18 * s
x = np.where(x > 0.92, 0.92 + (x - 0.92) * 0.6, x)        # roll off the top end
x = x + 0.012 * (1 - x) ** 3                               # lift the deepest blacks a hair (print feel)

# 2) lateral chromatic aberration (scale R up / B down by a tiny amount, edges only)
def rescale(ch, k):
    M = cv2.getRotationMatrix2D((W / 2, H / 2), 0, k)
    return cv2.warpAffine(ch, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
x[..., 0] = rescale(x[..., 0], 1.0004)
x[..., 2] = rescale(x[..., 2], 0.9996)

# 3) vignette (soft, ~0.3 stop at the corners)
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
r2 = ((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2
vig = 1 - 0.16 * np.clip(r2 / 2, 0, 1) ** 1.4
x = x * vig[..., None]

# 4) fine film grain: luminance-weighted, slightly stronger in mids, ~1.2px grain
g = rng.standard_normal((H, W)).astype(np.float32)
g = cv2.GaussianBlur(g, (0, 0), 0.65)
g /= g.std() + 1e-6
lum = x @ np.array([0.2126, 0.7152, 0.0722], np.float32)
amp = 0.008 + 0.010 * np.exp(-((lum - 0.45) / 0.3) ** 2)
x = x + (g * amp)[..., None]
x = np.clip(x, 0, 1)

bgr = (x[..., ::-1] * 255 + 0.5).astype(np.uint8)
cv2.imwrite(out + ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
cv2.imwrite(out + ".png", bgr)
print("wrote", out + ".jpg", out + ".png", W, H)
