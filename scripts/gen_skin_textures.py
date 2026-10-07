"""Generate high-resolution skin texture maps for the Mira character.

Runs outside Blender (system python with numpy + opencv + pillow).
Input:  body_mesh.npz exported from Blender (rest-pose, shape-keyed basemesh:
        vertex coords, loop UVs, triangles, material indices, eye + brow points)
        and the CC0 MakeHuman young_asian_female diffuse texture.
Output: 4K maps in textures/:
        T_Mira_Skin_BaseColor.png   sRGB albedo with natural tone variation + makeup
        T_Mira_Skin_Roughness.png   linear roughness (dewy T-zone, glossy lips, lid shimmer)
        T_Mira_Skin_Height.png      16-bit height (pores, fine lines, micro relief)
        T_Mira_Skin_Mask.png        R=lips, G=eye-makeup shimmer, B=face (for shader tweaks)

Usage: python3 gen_skin_textures.py <body_mesh.npz> <mh_diffuse.png> <out_dir> [size]
"""
import sys
import os
import numpy as np
import cv2

NPZ, DIFFUSE, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
S = int(sys.argv[4]) if len(sys.argv) > 4 else 4096
rng = np.random.default_rng(20261007)

d = np.load(NPZ)
co, uv, tl, tv = d["co"].astype(np.float64), d["uv"].astype(np.float64), d["tri_loops"], d["tri_verts"]
tri_mat, inbody, vn = d["tri_mat"], d["inbody"], d["vn"].astype(np.float64)
eyes_pts, brow_pts = d["eyes"], d["brows"]
mat_names = [str(m) for m in d["mat_names"]]
LIPS = [i for i, m in enumerate(mat_names) if m.endswith(".lips")][0]

keep = np.where(inbody[tv].all(1))[0]
tl, tv, tri_mat = tl[keep], tv[keep], tri_mat[keep]


# ----------------------------------------------------------------- rasterize
def to_px(u):
    return np.stack([u[..., 0] * S, (1.0 - u[..., 1]) * S], -1)


tri_uv_px = to_px(uv[tl])                     # (T,3,2)
tid = np.full((S, S), -1, np.int32)
pts = np.round(tri_uv_px * 16).astype(np.int32)  # 4 bits subpixel precision
for t in range(len(tl)):
    cv2.fillPoly(tid, [pts[t]], int(t), lineType=cv2.LINE_8, shift=4)
valid = tid >= 0
yy, xx = np.nonzero(valid)
T = tid[yy, xx]
P = np.stack([xx + 0.5, yy + 0.5], -1)
A, B, C = tri_uv_px[T, 0], tri_uv_px[T, 1], tri_uv_px[T, 2]
v0, v1, v2 = B - A, C - A, P - A
den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
den[np.abs(den) < 1e-12] = 1e-12
b1 = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / den
b2 = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / den
b0 = 1 - b1 - b2
bary = np.clip(np.stack([b0, b1, b2], -1), 0, 1)
bary /= bary.sum(1, keepdims=True)
pos = (co[tv[T]] * bary[..., None]).sum(1)
nrm = (vn[tv[T]] * bary[..., None]).sum(1)
nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9

# texel size in mm per triangle
a3 = 0.5 * np.linalg.norm(np.cross(co[tv[:, 1]] - co[tv[:, 0]], co[tv[:, 2]] - co[tv[:, 0]]), axis=1)
e1, e2 = tri_uv_px[:, 1] - tri_uv_px[:, 0], tri_uv_px[:, 2] - tri_uv_px[:, 0]
a2 = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]) + 1e-9
mm_per_px_tri = np.sqrt(a3 / a2) * 1000.0
mmpp = mm_per_px_tri[T]


def img(vals, fill=0.0):
    ch = 1 if vals.ndim == 1 else vals.shape[1]
    out = np.full((S, S, ch), fill, np.float32)
    out[yy, xx] = vals.reshape(len(yy), ch)
    return out[..., 0] if ch == 1 else out


def push_pull(im, mask):
    """Fill pixels outside `mask` by pyramid push-pull (padding across UV seams)."""
    im = im.astype(np.float32)
    if im.ndim == 2:
        im = im[..., None]
    w = mask.astype(np.float32)[..., None]
    levels = [(im * w, w)]
    while levels[-1][0].shape[0] > 4:
        c, ww = levels[-1]
        h = c.shape[0] // 2
        c2 = c.reshape(h, 2, h, 2, -1).sum((1, 3))
        w2 = ww.reshape(h, 2, h, 2, -1).sum((1, 3))
        levels.append((c2, w2))
    filled = levels[-1][0] / np.maximum(levels[-1][1], 1e-6)
    for c, ww in reversed(levels[:-1]):
        up = cv2.resize(filled, (c.shape[1], c.shape[0]), interpolation=cv2.INTER_LINEAR)
        if up.ndim == 2:
            up = up[..., None]
        cur = c / np.maximum(ww, 1e-6)
        a = np.clip(ww, 0, 1)
        filled = cur * a + up * (1 - a)
    return filled[..., 0] if filled.shape[-1] == 1 else filled


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


# ----------------------------------------------------------------- landmarks
X, Y, Z = pos[:, 0], pos[:, 1], pos[:, 2]
# eyelid rims: body vertices hugging the eyeball sphere, most-forward per angular bin
eyeL_pts = eyes_pts[eyes_pts[:, 0] > 0]
eyeR_pts = eyes_pts[eyes_pts[:, 0] < 0]
lm = {}
for side, ep in (("L", eyeL_pts), ("R", eyeR_pts)):
    c = ep.mean(0)
    rad = np.median(np.linalg.norm(ep - c, axis=1))
    dist = np.abs(np.linalg.norm(co - c, axis=1) - rad)
    cand = np.where((dist < 0.0025) & (co[:, 1] < c[1] - 0.25 * rad))[0]
    rel = co[cand] - c
    phi = np.arctan2(rel[:, 2], rel[:, 0])
    bins = np.linspace(-np.pi, np.pi, 49)
    rim = []
    for i in range(48):
        m = (phi >= bins[i]) & (phi < bins[i + 1])
        if m.any():
            j = cand[m][np.argmin(co[cand[m], 1])]
            rim.append(co[j])
    rim = np.array(rim)
    s = 1 if side == "L" else -1
    inner = rim[np.argmin(s * rim[:, 0])]
    outer = rim[np.argmax(s * rim[:, 0])]
    lm["eye" + side] = dict(center=c, inner=inner, outer=outer, rim=rim,
                            top=rim[:, 2].max(), bot=rim[:, 2].min(), apex=ep[np.argmin(ep[:, 1])])

lip_tris = np.where(tri_mat == LIPS)[0]
lip_v = np.unique(tv[lip_tris])
lip_co = co[lip_v]
lip_co = lip_co[lip_co[:, 2] > 1.0]
mid = lip_co[np.abs(lip_co[:, 0]) < 0.003]
zr = (mid[:, 2].min(), mid[:, 2].max())
zb = np.linspace(zr[0] + 0.2 * (zr[1] - zr[0]), zr[1] - 0.2 * (zr[1] - zr[0]), 24)
depth = [mid[np.abs(mid[:, 2] - z) < 0.0012][:, 1].max() if np.any(np.abs(mid[:, 2] - z) < 0.0012) else -1 for z in zb]
slit_z = float(zb[int(np.argmax(depth))])
lm["mouth"] = dict(cornerL=lip_co[np.argmax(lip_co[:, 0])], cornerR=lip_co[np.argmin(lip_co[:, 0])],
                   top=lip_co[:, 2].max(), bot=lip_co[:, 2].min(), slit_z=slit_z, front_y=lip_co[:, 1].min())
eye_z = 0.5 * (lm["eyeL"]["center"][2] + lm["eyeR"]["center"][2])
mid_face = (np.abs(co[:, 0]) < 0.004) & (co[:, 2] < eye_z) & (co[:, 2] > lm["mouth"]["top"])
nose_tip = co[mid_face][np.argmin(co[mid_face][:, 1])]
lm["nose_tip"] = nose_tip
head_y = 0.5 * (lm["eyeL"]["center"][1] + lm["eyeR"]["center"][1]) + 0.04
face = (Z > lm["mouth"]["bot"] - 0.06) & (Y < head_y) & (Z < eye_z + 0.09) & (np.abs(X) < 0.085)
for k, v in lm.items():
    if isinstance(v, dict):
        print(k, {kk: (np.round(vv, 4).tolist() if np.ndim(vv) == 1 else (round(float(vv), 4) if np.ndim(vv) == 0 else "...")) for kk, vv in v.items()})
    else:
        print(k, np.round(v, 4))


def g2(x0, z0, sx, sz, rot=0.0):
    """2D gaussian on the frontal (x,z) projection, optional rotation (radians)."""
    dx, dz = X - x0, Z - z0
    cr, sr = np.cos(rot), np.sin(rot)
    u = cr * dx + sr * dz
    v = -sr * dx + cr * dz
    return np.exp(-0.5 * ((u / sx) ** 2 + (v / sz) ** 2))


front = smoothstep(0.0, 0.35, -nrm[:, 1])     # facing camera-ish
facef = face.astype(np.float64)

# ----------------------------------------------------------------- base albedo
diff = cv2.imread(DIFFUSE, cv2.IMREAD_COLOR)[..., ::-1].astype(np.float32) / 255.0
diff = np.clip(cv2.resize(diff, (S, S), interpolation=cv2.INTER_CUBIC), 0, 1)
lin = np.where(diff <= 0.04045, diff / 12.92, ((diff + 0.055) / 1.055) ** 2.4)
alb = lin[yy, xx].astype(np.float64)

# tone: even, luminous fair skin with warm-rosy undertone
target = np.array([0.63, 0.425, 0.345])
mean_face = alb[face & (front > 0.5)].mean(0)
gain = target / mean_face
alb = alb * gain
lum = alb @ np.array([0.2126, 0.7152, 0.0722])
lum_s = cv2.GaussianBlur(push_pull(img(lum), valid), (0, 0), S / 128)[yy, xx]
# reduce blotchy low-frequency contrast on the face (makeup base) while keeping fine detail
even = 0.8 * facef + 0.3
ratio = np.clip(np.median(lum_s[face]) / np.maximum(lum_s, 1e-4), 0.85, 1.15)
alb = alb * (1 - even[:, None]) + alb * ratio[:, None] * even[:, None]

# soften freckles / small blotches on the face (concealer): pull toward a mid-scale blur
mid_blur = cv2.GaussianBlur(push_pull(img(alb), valid), (0, 0), S / 520)[yy, xx]
conceal = (0.55 * facef)[:, None]
alb = alb * (1 - conceal) + mid_blur * conceal

# scalp: darken toward dark-brown hair roots so the scalp never reads as bald skin.
# Hairline height varies around the head: forehead high, temples lower, nape lowest.
hv = co[(co[:, 2] > eye_z) & (np.abs(co[:, 0]) < 0.09)]
hc = np.array([0.0, 0.5 * (hv[:, 1].min() + hv[:, 1].max()), eye_z + 0.01])   # skull centre (bbox)
theta = np.arctan2(np.abs(X - hc[0]), -(Y - hc[1]))          # 0 = front, pi = back
hair_z = np.interp(theta, [0.0, 0.55, 1.15, 1.6, 2.3, np.pi],
                   [eye_z + 0.078, eye_z + 0.068, eye_z + 0.035, eye_z + 0.005, eye_z - 0.07, eye_z - 0.095])
scalp = smoothstep(hair_z - 0.004, hair_z + 0.012, Z) * smoothstep(eye_z - 0.13, eye_z - 0.11, Z)
# keep ears clean
ear_zone = smoothstep(0.055, 0.065, np.abs(X)) * smoothstep(0.045, 0.03, np.abs(Y - hc[1] - 0.005)) * smoothstep(eye_z - 0.05, eye_z - 0.04, Z) * smoothstep(eye_z + 0.04, eye_z + 0.03, Z)
scalp = scalp * (1 - ear_zone)
root_col = np.array([0.035, 0.022, 0.016])
alb = alb * (1 - 0.8 * scalp[:, None]) + root_col * 0.8 * scalp[:, None]


def tint(mask, col, k=1.0, mode="mul"):
    global alb
    m = np.clip(mask * k, 0, 1)[:, None]
    col = np.asarray(col)
    if mode == "mul":
        alb = alb * (1 - m) + (alb * col) * m
    else:
        alb = alb * (1 - m) + col * m


# natural redness (vascular): cheeks, nose, ears, knees, elbows, knuckles, toes
eL, eR = lm["eyeL"], lm["eyeR"]
red = np.zeros(len(X))
for e, s in ((eL, 1), (eR, -1)):
    red += g2(e["center"][0] + s * 0.012, e["center"][2] - 0.03, 0.017, 0.013) * facef
red += g2(nose_tip[0], nose_tip[2], 0.009, 0.012) * facef * 0.8
ears = (np.abs(X) > 0.065) & (Z > eye_z - 0.05) & (Z < eye_z + 0.03) & (Y > head_y - 0.06)
red += ears * 0.5
for zc, xr in ((0.48, (0.04, 0.16)),):          # knees (rest pose)
    red += np.exp(-0.5 * ((Z - zc) / 0.035) ** 2) * ((np.abs(X) > xr[0]) & (np.abs(X) < xr[1])) * (Y < -0.0) * 0.45
red = np.clip(red, 0, 1)
tint(red, [1.04, 0.86, 0.88], 0.55)

# ----------------------------------------------------------------- makeup
mo = lm["mouth"]
mouth_w = mo["cornerL"][0] - mo["cornerR"][0]
# blush: peach-pink on cheek apples, swept slightly up/out, plus a whisper over the nose bridge
blush = np.zeros(len(X))
for e, s in ((eL, 1), (eR, -1)):
    cx = e["center"][0] + s * 0.010
    cz = e["center"][2] - 0.026
    blush += g2(cx, cz, 0.019, 0.012, rot=s * 0.25)
blush += g2(0.0, eye_z - 0.022, 0.012, 0.006) * 0.35
blush = np.clip(blush, 0, 1) * facef * front
tint(blush, [1.03, 0.72, 0.70], 0.42)

# eyes: lid shadow, liner with a short lifted wing, aegyo-sal highlight + shadow
shim = np.zeros(len(X))
liner = np.zeros(len(X))
lid = np.zeros(len(X))
aegyo = np.zeros(len(X))
aeg_sh = np.zeros(len(X))
for side, s in (("L", 1), ("R", -1)):
    e = lm["eye" + side]
    rim = e["rim"]
    # upper/lower lid margin height as function of x (from rim points)
    a_, b_ = e["inner"], e["outer"]
    line_z = a_[2] + (rim[:, 0] - a_[0]) * (b_[2] - a_[2]) / (b_[0] - a_[0])
    upper = rim[rim[:, 2] >= line_z]
    lower = rim[rim[:, 2] < line_z]
    upper = upper[np.argsort(upper[:, 0])]
    lower = lower[np.argsort(lower[:, 0])]
    near = (np.abs(X - e["center"][0]) < 0.03) & (Y < e["center"][1] + 0.01) & facef.astype(bool)
    up_z = np.interp(X, upper[:, 0], upper[:, 2])
    lo_z = np.interp(X, lower[:, 0], lower[:, 2]) if len(lower) > 2 else e["bot"]
    t_out = np.clip((X - e["inner"][0]) / (e["outer"][0] - e["inner"][0]), -0.3, 1.6)   # 0 inner -> 1 outer
    inside_x = (t_out > -0.05) & (t_out < 1.05)
    dz_up = Z - up_z
    # lid shadow: soft gradient from lash line upward ~9mm, deeper at outer half
    lid += near * inside_x * smoothstep(-0.0005, 0.001, dz_up) * np.exp(-dz_up / 0.006) * (0.55 + 0.45 * np.clip(t_out, 0, 1))
    shim += near * inside_x * smoothstep(-0.0005, 0.001, dz_up) * np.exp(-dz_up / 0.005) * np.exp(-0.5 * ((t_out - 0.45) / 0.3) ** 2)
    # liner: thin along upper lash line from 30% outward, then a 4mm wing lifting ~12deg
    thick = 0.0006 + 0.0011 * smoothstep(0.3, 1.0, t_out)
    on_line = near * smoothstep(0.25, 0.4, t_out) * (t_out <= 1.0) * np.exp(-0.5 * (np.maximum(dz_up, 0) / thick) ** 2) * (dz_up > -0.0008)
    wing_len = 0.0045
    ox, oz = e["outer"][0], e["outer"][2]
    wx = (X - ox) * s
    wing_z = oz + 0.0012 + wx * 0.35
    wing = near * (wx > -0.001) * (wx < wing_len) * np.exp(-0.5 * ((Z - wing_z) / (0.0009 * (1 - wx / wing_len) + 0.0002)) ** 2)
    liner += np.clip(on_line + wing, 0, 1)
    # lower lash line soft smudge, outer third
    dz_lo = lo_z - Z
    liner += 0.35 * near * smoothstep(0.55, 0.85, t_out) * (t_out < 1.05) * np.exp(-0.5 * (np.maximum(dz_lo, 0) / 0.0012) ** 2) * (dz_lo > -0.0005)
    # aegyo-sal: highlight band 1-5mm under lower lid, shadow crease at ~6mm
    band = near * inside_x * np.exp(-0.5 * ((dz_lo - 0.0028) / 0.0016) ** 2) * (0.4 + 0.6 * np.exp(-0.5 * ((t_out - 0.45) / 0.3) ** 2))
    aegyo += band
    aeg_sh += near * inside_x * np.exp(-0.5 * ((dz_lo - 0.0062) / 0.0011) ** 2) * np.exp(-0.5 * ((t_out - 0.5) / 0.28) ** 2)

lid = np.clip(lid, 0, 1); liner = np.clip(liner, 0, 1); shim = np.clip(shim, 0, 1)
aegyo = np.clip(aegyo, 0, 1); aeg_sh = np.clip(aeg_sh, 0, 1)
tint(lid, [0.86, 0.66, 0.56], 0.55)               # soft peach-brown lid
tint(aeg_sh, [0.86, 0.74, 0.68], 0.45)            # subtle aegyo-sal shadow
tint(aegyo, [1.10, 1.04, 1.0], 0.55)              # luminous under-eye band
tint(liner, [0.045, 0.028, 0.022], 0.85, mode="set")  # dark brown liner

# the MakeHuman source paints a pale band under its brow cards: re-base that band to the
# surrounding skin tone (keep fine detail, replace low-frequency colour)
bp_all = brow_pts
band = np.zeros(len(X))
for bb in (bp_all[bp_all[:, 0] > 0], bp_all[bp_all[:, 0] < 0]):
    zc_mean = np.median(bb[:, 2])
    xmin, xmax = np.abs(bb[:, 0]).min(), np.abs(bb[:, 0]).max()
    sx = np.sign(bb[0, 0])
    inside = smoothstep(xmin - 0.012, xmin - 0.002, X * sx) * (1 - smoothstep(xmax + 0.002, xmax + 0.012, X * sx))
    band += inside * np.exp(-0.5 * ((Z - zc_mean - 0.001) / 0.0075) ** 2)
band = np.clip(band, 0, 1) * facef * (Y < head_y)
alb_lo = cv2.GaussianBlur(push_pull(img(alb), valid), (0, 0), S / 400)[yy, xx]
ring_ref = np.median(alb[(facef > 0) & (band < 0.05) & (np.abs(Z - eye_z - 0.02) < 0.02) & (front > 0.5)], axis=0)
alb = alb * (1 - band[:, None]) + (alb / np.maximum(alb_lo, 1e-4) * ring_ref) * band[:, None]

# brows: soft powder fill beneath the brow hair cards
bp = brow_pts
bL, bR = bp[bp[:, 0] > 0], bp[bp[:, 0] < 0]
brow = np.zeros(len(X))
for bb in (bL, bR):
    if len(bb) < 3:
        continue
    o = np.argsort(bb[:, 0]); bb = bb[o]
    xs_ = np.linspace(bb[:, 0].min(), bb[:, 0].max(), 40)
    zc = np.array([bb[np.abs(bb[:, 0] - x) < 0.002][:, 2].mean() if np.any(np.abs(bb[:, 0] - x) < 0.002) else np.nan for x in xs_])
    ok = ~np.isnan(zc)
    zline = np.interp(X, xs_[ok], zc[ok])
    ax = np.abs(X)
    x0, x1 = np.abs(xs_).min(), np.abs(xs_).max()
    t = np.clip((ax - x0) / (x1 - x0), 0, 1)                 # 0 = brow head, 1 = tail
    fade = smoothstep(x0 - 0.001, x0 + 0.006, ax) * (1 - smoothstep(x1 - 0.004, x1 + 0.002, ax))
    sig = 0.0027 * (1 - t) + 0.0011 * t
    side_ok = (np.sign(X) == np.sign(bb[0, 0]))
    brow += side_ok * fade * np.exp(-0.5 * ((Z - zline + 0.0005) / sig) ** 2) * facef * (Y < head_y)
brow = cv2.GaussianBlur(img(np.clip(brow, 0, 1)), (0, 0), 3)[yy, xx]
tint(brow, [0.36, 0.27, 0.23], 0.45)

# lips: MLBB gradient (deeper rose-coral at the inner lip, diffused outward), soft edge
lipmask = img(np.isin(T, lip_tris).astype(np.float32))
lipmask = cv2.GaussianBlur(lipmask, (0, 0), 2.2)[yy, xx]
lipmask = np.clip(lipmask * (Z > 1.0) * facef, 0, 1)
lip_h = max(mo["top"] - mo["slit_z"], 1e-3), max(mo["slit_z"] - mo["bot"], 1e-3)
rel = np.where(Z > mo["slit_z"], (Z - mo["slit_z"]) / lip_h[0], (mo["slit_z"] - Z) / lip_h[1])
relx = np.abs(X) / (mouth_w * 0.5)
inner = np.exp(-0.5 * (rel / 0.55) ** 2) * np.exp(-0.5 * (relx / 0.75) ** 2)
tint(lipmask, [1.0, 0.70, 0.70], 0.55)
tint(lipmask * inner, [0.62, 0.16, 0.17], 0.85, mode="set")
tint(lipmask * (1 - inner) * 0.5, [0.92, 0.48, 0.47], 0.6, mode="set")

# tiny moles for natural asymmetry (neck right side, upper chest left, near left jaw)
for (mx, mz, my_front, r) in ((-0.032, 1.40, True, 0.0011), (0.06, 1.25, True, 0.0012)):
    dd = np.sqrt((X - mx) ** 2 + (Z - mz) ** 2)
    mm = np.exp(-0.5 * (dd / r) ** 2) * (Y < 0.02) * (nrm[:, 1] < -0.2)
    tint(mm, [0.25, 0.15, 0.12], 0.85, mode="set")

# subtle highlight/glow (K-beauty "glass skin") is handled in roughness; keep albedo clean
alb = np.clip(alb, 0, 1)

# ----------------------------------------------------------------- roughness
rough = np.full(len(X), 0.52)
rough -= 0.08 * facef
tzone = (g2(0.0, eye_z + 0.04, 0.02, 0.025) + g2(0.0, eye_z - 0.02, 0.006, 0.025) + g2(nose_tip[0], nose_tip[2], 0.008, 0.008)) * facef
glow = np.zeros(len(X))
for e, s in ((eL, 1), (eR, -1)):
    glow += g2(e["center"][0] + s * 0.016, e["center"][2] - 0.022, 0.010, 0.008, rot=s * 0.5) * facef
rough -= 0.12 * np.clip(tzone, 0, 1) + 0.10 * np.clip(glow, 0, 1)
rough -= 0.10 * shim + 0.08 * aegyo
rough = rough * (1 - lipmask) + 0.17 * lipmask
rough += 0.06 * scalp

# ----------------------------------------------------------------- height (pores, lines, micro relief)
mm_img = push_pull(img(mmpp), valid)
H = np.zeros((S, S), np.float32)
# micro relief: band-limited noise, sized in mm via the local texel scale
for sigma_mm, amp in ((0.08, 0.35), (0.25, 0.35), (0.9, 0.25)):
    n = rng.standard_normal((S, S)).astype(np.float32)
    sig_px = np.clip(sigma_mm / np.median(mmpp), 0.5, 40)
    n = cv2.GaussianBlur(n, (0, 0), sig_px)
    n /= n.std() + 1e-6
    H += amp * n
H *= 0.15
# pores: random depressions, density higher on nose/cheeks/forehead
pore_density = np.full(len(X), 0.25)
pore_density += 1.4 * facef * (g2(nose_tip[0], nose_tip[2] + 0.005, 0.012, 0.02) +
                               g2(eL["center"][0] + 0.01, eL["center"][2] - 0.035, 0.02, 0.018) +
                               g2(eR["center"][0] - 0.01, eR["center"][2] - 0.035, 0.02, 0.018) +
                               0.5 * g2(0.0, eye_z + 0.045, 0.03, 0.02))
pore_density *= (1 - lipmask) * (1 - 0.7 * scalp)
dens_img = img(pore_density)
mm_full = mm_img
# expected pores per texel = density(per mm^2) * mm_per_px^2
p_exp = dens_img * mm_full ** 2 * 1.2
pore_seed = (rng.random((S, S)) < np.clip(p_exp, 0, 0.5)).astype(np.float32)
pore_seed *= rng.uniform(0.5, 1.0, (S, S)).astype(np.float32)
pores = cv2.GaussianBlur(pore_seed, (0, 0), 0.9)
pores /= pores.max() + 1e-6
H -= 1.1 * pores
# lip vertical lines
lipm_img = img(lipmask)
lines = rng.standard_normal((S, S)).astype(np.float32)
lines = cv2.GaussianBlur(lines, (0, 0), sigmaX=0.7, sigmaY=6.0)   # UV head island is rotated: lines across v
lines /= lines.std() + 1e-6
H += 0.35 * lines * lipm_img
H = cv2.GaussianBlur(H, (0, 0), 0.6)
H = H * valid
H = push_pull(H, valid)
H = (H - np.percentile(H[valid], 0.5)) / (np.percentile(H[valid], 99.5) - np.percentile(H[valid], 0.5))
H = np.clip(H, 0, 1)

# ----------------------------------------------------------------- write
os.makedirs(OUT, exist_ok=True)
alb_img = push_pull(img(alb), valid)
srgb = np.where(alb_img <= 0.0031308, alb_img * 12.92, 1.055 * np.power(np.clip(alb_img, 0, 1), 1 / 2.4) - 0.055)
cv2.imwrite(os.path.join(OUT, "T_Mira_Skin_BaseColor.png"), (np.clip(srgb, 0, 1)[..., ::-1] * 255 + 0.5).astype(np.uint8))
r_img = push_pull(img(np.clip(rough, 0.05, 1)), valid)
cv2.imwrite(os.path.join(OUT, "T_Mira_Skin_Roughness.png"), (np.clip(r_img, 0, 1) * 255 + 0.5).astype(np.uint8))
cv2.imwrite(os.path.join(OUT, "T_Mira_Skin_Height.png"), (H * 65535 + 0.5).astype(np.uint16))
mask = np.stack([push_pull(img(lipmask), valid), push_pull(img(np.clip(shim + aegyo * 0.6, 0, 1)), valid),
                 push_pull(img(facef), valid)], -1)
cv2.imwrite(os.path.join(OUT, "T_Mira_Skin_Mask.png"), (np.clip(mask, 0, 1)[..., ::-1] * 255 + 0.5).astype(np.uint8))
print("done", S, "texels covered:", int(valid.sum()))
