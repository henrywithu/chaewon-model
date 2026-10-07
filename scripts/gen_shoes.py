"""Black patent stiletto sandals fitted to Mira's posed feet (run inside Blender).

Per foot: the posed foot is sliced along its length to recover the plantar
profile and width; a sole slab follows it (forefoot outsole on the floor),
a tapered stiletto heel drops from the heel seat to the floor, and straps
(toe band, ankle strap + buckle, back strap) hug the skin 1 mm off.

Usage: ns = {}; exec(open(path).read(), ns); ns["build"]()
"""
import bpy
import bmesh
import math
import numpy as np
from mathutils import Vector, Matrix

BODY = "Mira_Body"
RIG = "Mira_Body.rig"
COLL = "CHAR_Mira"


def posed_world():
    body = bpy.data.objects[BODY]
    st = {m.name: m.show_viewport for m in body.modifiers}
    for m in body.modifiers:
        if m.type in ('SUBSURF', 'COLLISION'):
            m.show_viewport = False
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    ev = body.evaluated_get(dg)
    me = ev.to_mesh()
    co = np.empty(len(me.vertices) * 3); me.vertices.foreach_get("co", co); co = co.reshape(-1, 3)
    # vertex group weights survive the mask modifier via original indices
    idx = np.empty(len(me.vertices), np.int64)
    try:
        orig = me.attributes.get("index")
    except Exception:
        orig = None
    ev.to_mesh_clear()
    for m in body.modifiers:
        m.show_viewport = st[m.name]
    bpy.context.view_layer.update()
    M = np.array(body.matrix_world)
    return (M[:3, :3] @ co.T).T + M[:3, 3]


def posed_world_full():
    """Posed coords for ALL basemesh vertices (mask disabled) so indices match vertex groups."""
    body = bpy.data.objects[BODY]
    st = {m.name: m.show_viewport for m in body.modifiers}
    for m in body.modifiers:
        if m.type in ('SUBSURF', 'COLLISION', 'MASK'):
            m.show_viewport = False
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    ev = body.evaluated_get(dg)
    me = ev.to_mesh()
    co = np.empty(len(me.vertices) * 3); me.vertices.foreach_get("co", co); co = co.reshape(-1, 3)
    ev.to_mesh_clear()
    for m in body.modifiers:
        m.show_viewport = st[m.name]
    bpy.context.view_layer.update()
    M = np.array(body.matrix_world)
    return (M[:3, :3] @ co.T).T + M[:3, 3]


def group_mask(prefixes, side, thr=0.25):
    body = bpy.data.objects[BODY]
    gidx = {g.index for g in body.vertex_groups if g.name.endswith("." + side) and g.name.split(".")[0].startswith(prefixes)}
    body_g = body.vertex_groups["body"].index
    n = len(body.data.vertices)
    m = np.zeros(n, bool)
    inb = np.zeros(n, bool)
    for v in body.data.vertices:
        for g in v.groups:
            if g.group in gidx and g.weight > thr:
                m[v.index] = True
            if g.group == body_g and g.weight > 0.5:
                inb[v.index] = True
    return m & inb


def make_obj(name, bm, mat=None, parent_bone=None):
    me = bpy.data.meshes.get(name)
    old = bpy.data.objects.get(name)
    if old:
        bpy.data.objects.remove(old, do_unlink=True)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = True
    ob = bpy.data.objects.new(name, me)
    bpy.data.collections[COLL].objects.link(ob)
    if mat:
        me.materials.append(mat)
    if parent_bone:
        rig = bpy.data.objects[RIG]
        mw = ob.matrix_world.copy()
        ob.parent = rig
        ob.parent_type = 'BONE'
        ob.parent_bone = parent_bone
        pb = rig.pose.bones[parent_bone]
        pm = rig.matrix_world @ pb.matrix @ Matrix.Translation((0, pb.length, 0))
        ob.matrix_parent_inverse = pm.inverted()
        ob.matrix_world = mw
    return ob


def loft(bm, rings, closed=True, cap=False):
    """rings: list of (n,3) arrays with equal n; quads between consecutive rings."""
    vs = [[bm.verts.new(tuple(p)) for p in r] for r in rings]
    n = len(rings[0])
    for a, b in zip(vs[:-1], vs[1:]):
        for j in range(n if closed else n - 1):
            j2 = (j + 1) % n
            bm.faces.new((a[j], a[j2], b[j2], b[j]))
    if cap:
        bm.faces.new(vs[0][::-1])
        bm.faces.new(vs[-1])
    return vs


def materials():
    pat = bpy.data.materials.get("MAT_Shoe_BlackPatent") or bpy.data.materials.new("MAT_Shoe_BlackPatent")
    pat.use_nodes = True
    b = pat.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (0.008, 0.008, 0.009, 1)
    b.inputs["Roughness"].default_value = 0.18
    b.inputs["Coat Weight"].default_value = 1.0
    b.inputs["Coat Roughness"].default_value = 0.03
    b.inputs["Specular IOR Level"].default_value = 0.6
    sole = bpy.data.materials.get("MAT_Shoe_Sole") or bpy.data.materials.new("MAT_Shoe_Sole")
    sole.use_nodes = True
    b = sole.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (0.02, 0.018, 0.017, 1)
    b.inputs["Roughness"].default_value = 0.55
    metal = bpy.data.materials.get("MAT_Metal_Silver") or bpy.data.materials.new("MAT_Metal_Silver")
    metal.use_nodes = True
    b = metal.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (0.91, 0.92, 0.92, 1)
    b.inputs["Metallic"].default_value = 1.0
    b.inputs["Roughness"].default_value = 0.12
    return pat, sole, metal


def build_foot(side, Wc, pat, sole_m, metal, heel_h=0.086):
    foot = Wc[group_mask(("foot", "toe"), side)]
    leg = Wc[group_mask(("lowerleg02",), side, 0.4)]
    # foot axes
    back = foot[np.argmin(foot @ np.array([0, 1, 0]) * 0)]  # placeholder
    xy = foot[:, :2]
    # principal horizontal direction
    c = xy.mean(0)
    u, s, vt = np.linalg.svd(xy - c, full_matrices=False)
    f2 = vt[0]
    # orient forward = toward the toes (lowest region of the foot is the forefoot)
    proj = (xy - c) @ f2
    low = foot[:, 2] < np.percentile(foot[:, 2], 25)
    if proj[low].mean() < 0:
        f2 = -f2
    f = np.array([f2[0], f2[1], 0.0]); f /= np.linalg.norm(f)
    up = np.array([0, 0, 1.0])
    lat = np.cross(up, f)  # points to the foot's left
    t = (foot - np.r_[c, 0]) @ f
    l = (foot - np.r_[c, 0]) @ lat
    t0, t1 = t.min() - 0.004, t.max() + 0.006
    NS, NW = 56, 10
    ts = np.linspace(t0, t1, NS)
    bot, wl, wr = np.zeros(NS), np.zeros(NS), np.zeros(NS)
    for i, tv in enumerate(ts):
        sel = np.abs(t - tv) < 0.006
        if sel.sum() < 3:
            sel = np.abs(t - tv) < 0.012
        if sel.sum() < 3:
            bot[i] = np.nan; continue
        zb = foot[sel, 2].min()
        bot[i] = zb
        low = sel & (foot[:, 2] < zb + 0.022)
        wl[i], wr[i] = l[low].max(), l[low].min()
    ok = ~np.isnan(bot)
    bot = np.interp(ts, ts[ok], bot[ok]); wl = np.interp(ts, ts[ok], wl[ok]); wr = np.interp(ts, ts[ok], wr[ok])
    # smooth profile
    for _ in range(4):
        bot[1:-1] = 0.5 * bot[1:-1] + 0.25 * (bot[:-2] + bot[2:])
        wl[1:-1] = 0.5 * wl[1:-1] + 0.25 * (wl[:-2] + wl[2:])
        wr[1:-1] = 0.5 * wr[1:-1] + 0.25 * (wr[:-2] + wr[2:])
    # rounded toe / heel outline
    tt = (ts - t0) / (t1 - t0)
    round_ = np.sqrt(np.clip(1 - ((tt - 0.5) / 0.5) ** 8, 0, 1))
    mid = 0.5 * (wl + wr)
    half = 0.5 * (wl - wr) * round_ + 0.003 * round_
    top_z = bot - 0.0008
    # forefoot outsole touches the floor; heel seat sits at heel height
    thick = np.maximum(0.0035, np.minimum(top_z, 0.006))
    floor_contact = top_z < 0.012
    bot_z = np.where(floor_contact, 0.0, top_z - thick)
    bm = bmesh.new()
    base = np.r_[c, 0]
    top_rings, bot_rings = [], []
    for i in range(NS):
        ws = np.linspace(-1, 1, NW)
        cup = 0.0012 * ws ** 4
        ptsT = [base + f * ts[i] + lat * (mid[i] + w * half[i]) + up * (top_z[i] + cup[k]) for k, w in enumerate(ws)]
        ptsB = [base + f * ts[i] + lat * (mid[i] + w * (half[i] + 0.0006)) + up * bot_z[i] for w in ws]
        top_rings.append(np.array(ptsT)); bot_rings.append(np.array(ptsB))
    # closed loop per station: top (left->right) then bottom (right->left)
    rings = [np.vstack([tr, br[::-1]]) for tr, br in zip(top_rings, bot_rings)]
    loft(bm, rings, closed=True, cap=True)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    sole = make_obj("Mira_Shoe_%s_Sole" % side, bm, sole_m)
    sole.data.materials.append(pat)
    # patent finish on the upper rim/top surface: assign by face normal (top faces -> sole_m stays underside)
    for p in sole.data.polygons:
        p.material_index = 1 if p.normal.z > -0.3 else 0
    # ---- stiletto heel under the heel seat
    hi = int(NS * 0.16)
    hc = base + f * ts[hi] + lat * mid[hi]
    seat_z = bot_z[hi]
    rings = []
    NH = 22
    for k in range(NH + 1):
        a = k / NH                        # 0 top -> 1 floor
        z = seat_z * (1 - a) + 0.0 * a
        rx = 0.0125 * (1 - a) ** 1.6 + 0.0042 * a
        ry = 0.0105 * (1 - a) ** 1.6 + 0.0042 * a
        # stiletto leans slightly forward toward the floor contact under the heel bone
        cen = hc + f * (-0.004 + 0.010 * a ** 1.5)
        ang = np.linspace(0, 2 * math.pi, 18, endpoint=False)
        ring = np.array([cen + f * (ry * math.cos(q)) + lat * (rx * math.sin(q)) + up * (z + 0.0008) for q in ang])
        ring[:, 2] = z if k else z + 0.001
        rings.append(ring)
    bm = bmesh.new()
    loft(bm, rings, closed=True, cap=True)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    heel = make_obj("Mira_Shoe_%s_Heel" % side, bm, pat)
    # heel tip (top-lift)
    bm = bmesh.new()
    tipc = hc + f * 0.006
    ang = np.linspace(0, 2 * math.pi, 16, endpoint=False)
    r0 = np.array([tipc + f * 0.0047 * math.cos(q) + lat * 0.0047 * math.sin(q) for q in ang]); r0[:, 2] = 0.0
    r1 = r0.copy(); r1[:, 2] = 0.006
    loft(bm, [r0, r1], closed=True, cap=True)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    make_obj("Mira_Shoe_%s_HeelTip" % side, bm, sole_m)

    # ---- pump upper: convex foot sections lofted heel->pointed toe, then cut along the topline
    allp = np.vstack([foot, leg])
    ta = (allp - base) @ f
    la = (allp - base) @ lat
    za = allp[:, 2]
    NR = 44
    stations = np.linspace(t0 + 0.002, t.max() - 0.002, 64)
    ext = np.linspace(t.max(), t.max() + 0.032, 9)[1:]          # pointed toe extension
    rings_out, meta = [], []
    angs = np.linspace(-math.pi, math.pi, NR, endpoint=False)
    def section(tv):
        sel = np.abs(ta - tv) < 0.004
        if sel.sum() < 6:
            sel = np.abs(ta - tv) < 0.008
        Q = np.stack([la[sel], za[sel]], 1)
        Q = Q[Q[:, 1] < np.interp(tv, ts, top_z) + 0.075]
        H = None
        try:
            import itertools
            idx = np.lexsort((Q[:, 1], Q[:, 0])); S_ = Q[idx]
            def cr(o, a, b):
                return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
            lo_, up_ = [], []
            for p_ in S_:
                while len(lo_) >= 2 and cr(lo_[-2], lo_[-1], p_) <= 0: lo_.pop()
                lo_.append(p_)
            for p_ in S_[::-1]:
                while len(up_) >= 2 and cr(up_[-2], up_[-1], p_) <= 0: up_.pop()
                up_.append(p_)
            H = np.array(lo_[:-1] + up_[:-1])
        except Exception:
            H = Q
        cen = H.mean(0)
        # exact ray / convex-polygon intersection from the centroid
        Hn = np.roll(H, -1, 0)
        rr = np.zeros(len(angs))
        for k_, a_ in enumerate(angs):
            d_ = np.array([math.cos(a_), math.sin(a_)])
            best = 0.0
            for A_, B_ in zip(H - cen, Hn - cen):
                e_ = B_ - A_
                den = d_[0] * (-e_[1]) - d_[1] * (-e_[0])
                if abs(den) < 1e-12:
                    continue
                tt_ = (A_[0] * (-e_[1]) - A_[1] * (-e_[0])) / den
                u_ = (d_[0] * A_[1] - d_[1] * A_[0]) / den
                if tt_ > 0 and -1e-6 <= u_ <= 1 + 1e-6:
                    best = max(best, tt_)
            rr[k_] = best
        return cen, rr + 0.0013
    last_c, last_r = None, None
    for tv in stations:
        c2, rr = section(tv)
        last_c, last_r = c2, rr
        rings_out.append((tv, c2, rr))
    # smooth envelope along the shoe: absolute section points, max-filter then smooth
    T_ = np.array([r[0] for r in rings_out])
    C_ = np.array([r[1] for r in rings_out])
    R_ = np.array([r[2] for r in rings_out])
    PL = C_[:, None, 0] + np.cos(angs)[None, :] * R_       # lateral coord per (station, angle)
    PZ = C_[:, None, 1] + np.sin(angs)[None, :] * R_
    for _ in range(3):
        C_[1:-1] = 0.5 * C_[1:-1] + 0.25 * (C_[:-2] + C_[2:])
    rad = (PL - C_[:, None, 0]) * np.cos(angs)[None, :] + (PZ - C_[:, None, 1]) * np.sin(angs)[None, :]
    env = rad.copy()
    W_ = 4
    for i_ in range(len(env)):
        env[i_] = rad[max(0, i_ - W_):i_ + W_ + 1].max(0)
    for _ in range(10):
        env[1:-1] = 0.5 * env[1:-1] + 0.25 * (env[:-2] + env[2:])
        env = 0.5 * env + 0.25 * (np.roll(env, 1, 1) + np.roll(env, -1, 1))
    env = np.maximum(env, rad)
    for _ in range(2):
        env = 0.5 * env + 0.25 * (np.roll(env, 1, 1) + np.roll(env, -1, 1))
    rings_out = [(T_[i_], C_[i_], env[i_]) for i_ in range(len(T_))]
    last_c, last_r = C_[-1], env[-1]
    sole_tip_z = np.interp(t.max(), ts, top_z)
    for k, tv in enumerate(ext):
        a = (k + 1) / len(ext)
        c2 = last_c.copy()
        c2[1] = last_c[1] * (1 - a) + (sole_tip_z + 0.004) * a
        c2[0] = last_c[0] * (1 - a) + (last_c[0] + (0.004 if side == "R" else -0.004)) * a
        rr = last_r * (1 - a ** 0.85) + 0.0006
        rings_out.append((tv, c2, rr))
    bm = bmesh.new()
    vrows = []
    for (tv, c2, rr) in rings_out:
        row = []
        for a_, r_ in zip(angs, rr):
            lv = c2[0] + math.cos(a_) * r_
            zv = c2[1] + math.sin(a_) * r_
            zv = max(zv, np.interp(tv, ts, top_z, right=sole_tip_z) - 0.0005)   # never below the insole
            row.append(bm.verts.new(tuple(base + f * tv + lat * lv + up * zv)))
        vrows.append(row)
    for a_row, b_row in zip(vrows[:-1], vrows[1:]):
        for j in range(NR):
            j2 = (j + 1) % NR
            bm.faces.new((a_row[j], a_row[j2], b_row[j2], b_row[j]))
    # close the toe point
    tipv = bm.verts.new(tuple(np.mean([np.array(v.co) for v in vrows[-1]], axis=0)))
    for j in range(NR):
        bm.faces.new((vrows[-1][j], vrows[-1][(j + 1) % NR], tipv))
    # topline cut: heel counter ~5cm, low sides, curved vamp over the toe cleavage
    L_ = (t.max() - t0)
    kill = []
    for i, row in enumerate(vrows):
        tv = rings_out[i][0]
        tn = (tv - t0) / L_
        for v in row:
            p = np.array(v.co)
            h = p[2] - np.interp(tv, ts, top_z, right=sole_tip_z)
            lv = (p - base) @ lat
            half_w = max(0.5 * (np.interp(tv, ts, wl) - np.interp(tv, ts, wr)), 0.01)
            ln = (lv - np.interp(tv, ts, mid)) / half_w
            h_cut = np.interp(tn, [0.0, 0.10, 0.30, 0.50, 0.70], [0.050, 0.046, 0.026, 0.020, 0.020])
            t_vamp = 0.70 - 0.07 * min(ln * ln, 1.0)
            if h > h_cut and tn < t_vamp:
                kill.append(v)
    bmesh.ops.delete(bm, geom=list(set(kill)), context='VERTS')
    # relax the cut edge
    for _ in range(6):
        bnd = [v for v in bm.verts if v.is_boundary]
        newp = {}
        for v in bnd:
            nb = [e.other_vert(v) for e in v.link_edges if e.is_boundary]
            if len(nb) == 2:
                newp[v] = (v.co * 2 + nb[0].co + nb[1].co) / 4
        for v, pco in newp.items():
            v.co = pco
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    up_ob = make_obj("Mira_Shoe_%s_Upper" % side, bm, pat)
    so = up_ob.modifiers.new("Thickness", 'SOLIDIFY'); so.thickness = 0.0012; so.offset = 1.0; so.use_even_offset = True; so.use_rim = True
    sd = up_ob.modifiers.new("Subdivision", 'SUBSURF'); sd.levels = 1; sd.render_levels = 2
    # outsole extended to the pointed toe (rebuild sole outline from the upper footprint)
    sole_ob = bpy.data.objects["Mira_Shoe_%s_Sole" % side]
    bm = bmesh.new(); bm.from_mesh(sole_ob.data)
    tmax = t.max()
    for v in bm.verts:
        p = np.array(v.co)
        tv = (p - base) @ f
        if tv > tmax - 0.03:
            a = np.clip((tv - (tmax - 0.03)) / 0.03, 0, 1)
            v.co = Vector(tuple(p + f * (0.032 * a)))
            lv = (np.array(v.co) - base) @ lat - np.interp(tmax, ts, mid)
            shrink = 1 - 0.85 * a ** 1.3
            v.co = Vector(tuple(np.array(v.co) - lat * lv * (1 - shrink)))
    bm.to_mesh(sole_ob.data); bm.free()
    return dict(t_len=float(t1 - t0), heel_seat=float(seat_z), fore_top=float(top_z[-10]))


def build():
    for o in list(bpy.data.objects):
        if o.name.startswith("Mira_Shoe_"):
            me = o.data
            bpy.data.objects.remove(o, do_unlink=True)
            if me and me.users == 0:
                bpy.data.meshes.remove(me)
    pat, sole_m, metal = materials()
    Wc = posed_world_full()
    out = {}
    for side in ("L", "R"):
        out[side] = build_foot(side, Wc, pat, sole_m, metal)
    # group under a parent empty for tidiness
    rig = bpy.data.objects[RIG]
    for o in list(bpy.data.objects):
        if o.name.startswith("Mira_Shoe_"):
            side = o.name.split("_")[2]
            bone = "foot." + side
            mw = o.matrix_world.copy()
            o.parent = rig
            o.parent_type = 'BONE'
            o.parent_bone = bone
            pb = rig.pose.bones[bone]
            pm = rig.matrix_world @ pb.matrix @ Matrix.Translation((0, pb.length, 0))
            o.matrix_parent_inverse = pm.inverted()
            o.matrix_world = mw
    return out
