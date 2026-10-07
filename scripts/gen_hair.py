"""Procedural strand hair for Mira (run inside Blender after posing).

Long, layered dark-brown hair with wispy see-through bangs, face-framing
layers and a soft side part. Guides are grown root->tip in posed world space
with gravity, stiffness and collision against the posed body; children are
interpolated from same-category guides, clumped, waved, given flyaways and
C-curled tips. Output: Curves object "Mira_Hair" parented to the head bone.

Usage (Blender):  ns = {}; exec(open(path).read(), ns); ns["build"]()
"""
import bpy
import math
import numpy as np
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree

RNG = np.random.default_rng(7)
BODY = "Mira_Body"
RIG = "Mira_Body.rig"
CFG = dict(
    n_guides=2200,
    n_children=95000,
    segs=40,
    part_x=0.013,
    main_len=(0.40, 0.47),
    frame_len=(0.21, 0.32),
    bang_density=0.8,
    radius_root=0.000045,
    radius_tip=0.000018,
)


# ------------------------------------------------------------------ mesh helpers
def mesh_arrays(obj, armature=True):
    mods = {m.name: m.show_viewport for m in obj.modifiers}
    for m in obj.modifiers:
        if m.type == 'SUBSURF':
            m.show_viewport = False
        if m.type == 'ARMATURE':
            m.show_viewport = armature
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(dg)
    me = ev.to_mesh()
    me.calc_loop_triangles()
    n = len(me.vertices)
    co = np.empty(n * 3, np.float64); me.vertices.foreach_get("co", co); co = co.reshape(-1, 3)
    nr = np.empty(n * 3, np.float64); me.vertices.foreach_get("normal", nr); nr = nr.reshape(-1, 3)
    nt = len(me.loop_triangles)
    tri = np.empty(nt * 3, np.int32); me.loop_triangles.foreach_get("vertices", tri); tri = tri.reshape(-1, 3)
    ev.to_mesh_clear()
    for m in obj.modifiers:
        m.show_viewport = mods[m.name]
    bpy.context.view_layer.update()
    return co, nr, tri


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def build(cfg=CFG):
    body = bpy.data.objects[BODY]
    rig = bpy.data.objects[RIG]
    rest_co, rest_n, tri = mesh_arrays(body, armature=False)
    pose_co_l, pose_n_l, tri2 = mesh_arrays(body, armature=True)
    assert len(tri) == len(tri2)
    MW = np.array(body.matrix_world)
    pose_co = (MW[:3, :3] @ pose_co_l.T).T + MW[:3, 3]
    pose_n = (MW[:3, :3] @ pose_n_l.T).T
    pose_n /= np.linalg.norm(pose_n, axis=1, keepdims=True) + 1e-12

    # posed BVH for collisions (body only, helpers masked by modifier)
    bvh = BVHTree.FromPolygons([Vector(v) for v in pose_co], [tuple(t) for t in tri], all_triangles=True)

    # head frames
    hb = rig.pose.bones["head"]
    R_head = (rig.matrix_world @ hb.matrix).to_3x3() @ hb.bone.matrix_local.to_3x3().inverted()
    R_head_np = np.array(R_head)
    chest = rig.pose.bones["spine01"]
    R_chest = np.array((rig.matrix_world @ chest.matrix).to_3x3() @ chest.bone.matrix_local.to_3x3().inverted())
    fwd_chest = R_chest @ np.array([0, -1, 0.0])
    up = np.array([0, 0, 1.0])

    # rest-space landmarks
    eyes = bpy.data.objects[[o.name for o in bpy.data.objects if "high-poly" in o.name][0]]
    # eye centre in rest: eyes are deformed by armature too -> use rest bone heads
    eyeL = Vector(rig.data.bones["eye.L"].head_local)
    eye_z = eyeL.z
    head_mask = rest_co[:, 2] > eye_z - 0.02
    hv = rest_co[head_mask & (rest_co[:, 2] > eye_z) & (np.abs(rest_co[:, 0]) < 0.09)]
    hc = np.array([0.0, 0.5 * (hv[:, 1].min() + hv[:, 1].max()), eye_z + 0.01])   # skull centre (bbox)
    brow_z = eye_z + 0.022
    top_z = rest_co[:, 2].max()
    front_y = rest_co[(np.abs(rest_co[:, 0]) < 0.01) & (np.abs(rest_co[:, 2] - (eye_z + 0.05)) < 0.01)][:, 1].min()

    # hairline (same function as the texture generator, pulled 4mm inside)
    def hairline_z(X, Y, Z):
        theta = np.arctan2(np.abs(X - hc[0]), -(Y - hc[1]))
        hz = np.interp(theta, [0.0, 0.55, 1.15, 1.6, 2.3, np.pi],
                       [eye_z + 0.078, eye_z + 0.068, eye_z + 0.035, eye_z + 0.005, eye_z - 0.07, eye_z - 0.095])
        return hz, theta

    # candidate scalp triangles
    tc = rest_co[tri].mean(1)
    hz, th = hairline_z(tc[:, 0], tc[:, 1], tc[:, 2])
    ear = (np.abs(tc[:, 0]) > 0.055) & (np.abs(tc[:, 1] - hc[1] - 0.005) < 0.035) & (tc[:, 2] > eye_z - 0.05) & (tc[:, 2] < eye_z + 0.035)
    scalp_t = np.where((tc[:, 2] > hz + 0.004) & (tc[:, 2] > eye_z - 0.11) & ~ear)[0]
    a = rest_co[tri[scalp_t, 1]] - rest_co[tri[scalp_t, 0]]
    b = rest_co[tri[scalp_t, 2]] - rest_co[tri[scalp_t, 0]]
    area = 0.5 * np.linalg.norm(np.cross(a, b), axis=1)

    def sample_roots(n, density=None):
        w = area if density is None else area * density
        ti = RNG.choice(len(scalp_t), size=n, p=w / w.sum())
        r1, r2 = RNG.random(n), RNG.random(n)
        s = np.sqrt(r1)
        bc = np.stack([1 - s, s * (1 - r2), s * r2], 1)
        t = tri[scalp_t[ti]]
        rest = (rest_co[t] * bc[..., None]).sum(1)
        pos = (pose_co[t] * bc[..., None]).sum(1)
        nrm = (pose_n[t] * bc[..., None]).sum(1)
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
        rn = (rest_n[t] * bc[..., None]).sum(1)
        rn /= np.linalg.norm(rn, axis=1, keepdims=True)
        return rest, pos, nrm, rn

    def classify(rest):
        X, Y, Z = rest[:, 0], rest[:, 1], rest[:, 2]
        hzz, theta = hairline_z(X, Y, Z)
        depth = Z - hzz                       # distance above hairline (front: behind it)
        bang = (theta < 0.62) & (Y < hc[1] - 0.035) & (Z < hzz + 0.045)
        frame = (~bang) & (theta >= 0.5) & (theta < 1.35) & (Z < hzz + 0.03) & (Y < hc[1] + 0.01)
        cat = np.zeros(len(X), np.int32)       # 0 main, 1 bangs, 2 face-framing
        cat[bang] = 1
        cat[frame] = 2
        return cat, theta

    # bangs are see-through: thin the root density there
    rest_tc = tc[scalp_t]
    cat_t, _ = classify(rest_tc)
    dens = np.where(cat_t == 1, cfg["bang_density"], 1.0)

    def flow_dir(rest, rn, cat):
        """Design hair direction at the root in rest space (tangent to scalp)."""
        X, Y, Z = rest[:, 0], rest[:, 1], rest[:, 2]
        px = cfg["part_x"]
        F = np.array([px, front_y + 0.01, eye_z + 0.085])
        K = np.array([px, hc[1] + 0.035, top_z - 0.005])
        seg = K - F
        tpar = np.clip(((rest - F) @ seg) / (seg @ seg), 0, 1)
        near = F + tpar[:, None] * seg
        away = rest - near
        away[:, 2] *= 0.3
        away /= np.linalg.norm(away, axis=1, keepdims=True) + 1e-9
        d = away + np.array([0, 0.25, -0.55])
        # bangs: forward/down over the forehead, gently fanned
        bang = cat == 1
        d[bang] = np.stack([(X[bang] - px) * 9.0, -np.ones(bang.sum()) * 1.0, -np.ones(bang.sum()) * 0.9], 1)
        # face framing: down and slightly forward
        fr = cat == 2
        d[fr] = np.stack([np.sign(X[fr]) * 0.25, -np.ones(fr.sum()) * 0.25, -np.ones(fr.sum())], 1)
        d -= (d * rn).sum(1, keepdims=True) * rn
        d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
        return d

    # ------------------------------------------------------------------ guides
    g_rest, g_pos, g_nrm, g_rn = sample_roots(cfg["n_guides"], dens)
    g_cat, g_theta = classify(g_rest)
    g_dir_rest = flow_dir(g_rest, g_rn, g_cat)
    g_dir = (R_head_np @ g_dir_rest.T).T
    S = cfg["segs"]
    G = np.zeros((len(g_rest), S + 1, 3))
    shoulder_z = float((rig.matrix_world @ rig.pose.bones["clavicle.L"].head).z)
    neck_pt = np.array(rig.matrix_world @ rig.pose.bones["neck02"].head)
    head_M = rig.matrix_world @ hb.matrix
    head_o_pose = np.array(head_M.translation)
    head_o_rest = np.array(hb.bone.head_local)
    Rt = R_head_np.T

    def to_rest(q):
        return Rt @ (q - head_o_pose) + head_o_rest

    def from_rest(r):
        return R_head_np @ (r - head_o_rest) + head_o_pose

    chin_z = eye_z - 0.085
    for i in range(len(g_rest)):
        c = g_cat[i]
        X, Y, Z = g_rest[i]
        if c == 1:     # bang: length so the tip lands around the brows
            L = (Z - brow_z) * 1.55 + 0.012 + RNG.normal(0, 0.003)
            L = max(L, 0.03)
        elif c == 2:
            L = RNG.uniform(*cfg["frame_len"]) + max(0.0, 0.1 * (g_theta[i] - 0.5))
        else:
            back = smoothstep(0.8, 2.6, g_theta[i])
            L = RNG.uniform(*cfg["main_len"]) + 0.03 * back
        layer = RNG.random()
        if c == 1:
            off_head, off_body = 0.0015 + 0.004 * layer, 0.003
        else:
            off_head = 0.0015 + 0.006 * layer ** 0.9
            off_body = 0.003 + 0.012 * layer ** 0.8
        side = 1 if X > cfg["part_x"] else -1
        split = 1.95 if side > 0 else 1.25
        front = (g_theta[i] < split + RNG.normal(0, 0.12)) and c != 1
        ds = L / S
        p = g_pos[i].copy()
        d = g_dir[i] + g_nrm[i] * (0.12 if c != 1 else 0.08)
        d /= np.linalg.norm(d)
        G[i, 0] = p
        for k in range(1, S + 1):
            t = k / S
            if c == 1:
                grav = 0.03 + 0.10 * t
            else:
                grav = 0.16 + 0.30 * t
            d = d + np.array([0, 0, -grav])
            pr_ = to_rest(p)
            on_head = pr_[2] > chin_z - 0.01
            if c != 1 and p[2] < shoulder_z + 0.06:
                d = d + fwd_chest * (0.22 if front else -0.22) * smoothstep(shoulder_z + 0.06, shoulder_z - 0.04, p[2])
                to_neck = neck_pt - p
                to_neck[2] = 0
                d = d - 0.08 * to_neck / (np.linalg.norm(to_neck) + 1e-6)
            if t > 0.8:                     # C-curl at the tips (inward)
                loc, nrm_hit, idx, dist = bvh.find_nearest(Vector(p))
                if loc is not None:
                    d = d - np.array(nrm_hit) * 0.3 * (t - 0.8) / 0.2
            d /= np.linalg.norm(d)
            q = p + d * ds
            # face keep-out (rest head space): nothing but bangs in front of the face
            if c != 1:
                qr = to_rest(q)
                if qr[2] < brow_z + 0.01 and qr[2] > chin_z - 0.04 and qr[1] < hc[1] - 0.02:
                    lim = 0.062 + 0.01 * smoothstep(brow_z, chin_z - 0.04, qr[2])
                    if abs(qr[0]) < lim:
                        qr[0] = np.sign(qr[0] if abs(qr[0]) > 1e-6 else side) * lim
                        q = from_rest(qr)
            loc, nrm_hit, idx, dist = bvh.find_nearest(Vector(q))
            if loc is not None:
                loc = np.array(loc); nh = np.array(nrm_hit)
                off = off_head if on_head else off_body
                h = np.dot(q - loc, nh)
                if h < off:
                    q = loc + nh * off
                elif on_head and dist < 0.05 and h > off + 0.004:
                    q = loc + nh * (off + 0.004)       # hair lies on the scalp, no flaring
            d = (q - p) / (np.linalg.norm(q - p) + 1e-12)
            p = q
            G[i, k] = p
    # light smoothing (keep root)
    for _ in range(2):
        G[:, 1:-1] = 0.5 * G[:, 1:-1] + 0.25 * (G[:, :-2] + G[:, 2:])

    # ------------------------------------------------------------------ children
    c_rest, c_pos, c_nrm, c_rn = sample_roots(cfg["n_children"], dens)
    c_cat, c_theta = classify(c_rest)
    N = len(c_rest)
    C = np.zeros((N, S + 1, 3))
    t = np.linspace(0, 1, S + 1)
    G_off = G - G[:, :1]
    nearest_guide = np.zeros(N, np.int64)
    for cat in (0, 1, 2):
        gi = np.where(g_cat == cat)[0]
        ci = np.where(c_cat == cat)[0]
        if len(gi) == 0 or len(ci) == 0:
            continue
        for s0 in range(0, len(ci), 4000):
            cc = ci[s0:s0 + 4000]
            d2 = ((c_rest[cc, None, :] - g_rest[None, gi, :]) ** 2).sum(-1)
            k = 4
            nn = np.argpartition(d2, k, axis=1)[:, :k]
            dk = np.take_along_axis(d2, nn, 1)
            w = 1.0 / (np.sqrt(dk) + 0.002) ** 2
            w /= w.sum(1, keepdims=True)
            off = (G_off[gi[nn]] * w[..., None, None]).sum(1)
            C[cc] = c_pos[cc, None, :] + off
            nearest_guide[cc] = gi[nn[np.arange(len(cc)), np.argmin(dk, 1)]]
    # clumping toward the nearest guide
    clump = np.where(c_cat == 1, 0.78, 0.6)[:, None] * (t ** np.where(c_cat == 1, 0.5, 0.9)[:, None])
    target = G[nearest_guide] + (c_pos - G[nearest_guide, 0])[:, None, :] * 0.25
    C = C * (1 - clump[..., None]) + target * clump[..., None]
    # waves / strand noise: smooth random offsets growing toward the tips
    ctrl = RNG.normal(0, 1, (N, 5, 3))
    tt = np.linspace(0, 1, 5)
    basis = np.array([np.interp(t, tt, np.eye(5)[j]) for j in range(5)])  # (5, S+1) linear hat functions
    noise = np.einsum('njk,jt->ntk', ctrl, basis)
    amp = np.where(c_cat == 1, 0.0018, 0.0042)[:, None] * (t ** 1.2)[None, :]
    C += noise * amp[..., None]
    # flyaways: a small fraction of strands drift out from the surface
    fly = RNG.random(N) < 0.008
    fdir = RNG.normal(0, 1, (fly.sum(), 1, 3))
    fdir /= np.linalg.norm(fdir, axis=-1, keepdims=True)
    C[fly] += fdir * (0.004 + 0.012 * RNG.random((fly.sum(), 1, 1))) * (t ** 1.5)[None, :, None]
    # length variation (layering / broken ends): trim by resampling
    lf = np.where(c_cat == 1, RNG.uniform(0.85, 1.05, N), RNG.uniform(0.82, 1.0, N))
    lf[fly] *= RNG.uniform(0.4, 0.9, fly.sum())
    tq = t[None, :] * lf[:, None]
    seglen = np.linalg.norm(np.diff(C, axis=1), axis=-1)
    cum = np.concatenate([np.zeros((N, 1)), np.cumsum(seglen, 1)], 1)
    cum /= cum[:, -1:] + 1e-12
    out = np.empty_like(C)
    for a in range(3):
        for i0 in range(0, N, 20000):
            sl = slice(i0, i0 + 20000)
            out[sl, :, a] = np.array([np.interp(tq[j], cum[j], C[j, :, a]) for j in range(sl.start, min(sl.stop, N))])
    C = out
    # collision clean-up for children below the head
    below = C[:, :, 2] < float(np.array(rig.matrix_world @ rig.pose.bones["head"].head)[2]) + 0.02
    idx = np.argwhere(below)
    for (i, k) in idx:
        if k == 0:
            continue
        q = C[i, k]
        loc, nh, _, _ = bvh.find_nearest(Vector(q))
        if loc is None:
            continue
        loc = np.array(loc); nh = np.array(nh)
        if np.dot(q - loc, nh) < 0.002:
            C[i, k] = loc + nh * 0.002

    # drop broken strands (sudden jumps / runaway points)
    head_c = np.array(rig.matrix_world @ rig.pose.bones["head"].head)
    seg = np.linalg.norm(np.diff(C, axis=1), axis=-1)
    dist = np.linalg.norm(C - head_c[None, None, :], axis=-1)
    good = np.isfinite(C).all(axis=(1, 2)) & (seg.max(1) < 0.06) & (dist.max(1) < 0.75) & (C[:, :, 2].max(1) < head_c[2] + 0.2)
    C = C[good]
    N = len(C)

    # ------------------------------------------------------------------ write curves
    name = "Mira_Hair"
    old = bpy.data.objects.get(name)
    if old:
        bpy.data.objects.remove(old, do_unlink=True)
    if name in bpy.data.hair_curves:
        bpy.data.hair_curves.remove(bpy.data.hair_curves[name])
    cv = bpy.data.hair_curves.new(name)
    cv.add_curves([S + 1] * N)
    cv.points.foreach_set("position", C.reshape(-1).astype(np.float32))
    rad = (cfg["radius_root"] * (1 - t) + cfg["radius_tip"] * t)
    rad = np.tile(rad, N).astype(np.float32)
    cv.points.foreach_set("radius", rad)
    obj = bpy.data.objects.new(name, cv)
    coll = bpy.data.collections.get("CHAR_Mira")
    (coll or bpy.context.scene.collection).objects.link(obj)
    # parent to head bone, keep world placement
    obj.parent = rig
    obj.parent_type = 'BONE'
    obj.parent_bone = "head"
    bpy.context.view_layer.update()
    hbm = rig.matrix_world @ hb.matrix @ Matrix.Translation((0, hb.length, 0))
    obj.matrix_parent_inverse = hbm.inverted()
    obj.matrix_world = Matrix.Identity(4)
    # store guide categories for inspection
    return dict(guides=len(G), children=N, cats=[int((c_cat == k).sum()) for k in range(3)])
