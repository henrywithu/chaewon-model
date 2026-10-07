"""Garments for Mira: strapless satin corset-bustier + A-line satin mini skirt.

Garments are built in REST space as clean quad tubes by casting rays from the
torso axis to the body, convexifying every horizontal slice (a structured
bodice bridges the cleavage / spine groove like real boning), then offset for
ease. Weights are transferred from the body so the garments follow the rig.

Usage (Blender): ns = {}; exec(open(path).read(), ns); ns["build_top"](); ns["build_skirt"]()
"""
import bpy
import bmesh
import math
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

BODY = "Mira_Body"
RIG = "Mira_Body.rig"
COLL = "CHAR_Mira"


def rest_arrays():
    body = bpy.data.objects[BODY]
    st = {m.name: m.show_viewport for m in body.modifiers}
    for m in body.modifiers:
        if m.type in ('ARMATURE', 'SUBSURF'):
            m.show_viewport = False
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    ev = body.evaluated_get(dg)
    me = ev.to_mesh()
    me.calc_loop_triangles()
    co = np.empty(len(me.vertices) * 3); me.vertices.foreach_get("co", co); co = co.reshape(-1, 3)
    tri = np.empty(len(me.loop_triangles) * 3, np.int32); me.loop_triangles.foreach_get("vertices", tri); tri = tri.reshape(-1, 3)
    ev.to_mesh_clear()
    for m in body.modifiers:
        m.show_viewport = st[m.name]
    bpy.context.view_layer.update()
    return co, tri


def landmarks(co):
    torso = (np.abs(co[:, 0]) < 0.16)
    # bust apex: most forward point in chest band
    chest = torso & (co[:, 2] > 1.05) & (co[:, 2] < 1.3) & (np.abs(co[:, 0]) > 0.03) & (np.abs(co[:, 0]) < 0.12)
    apex = co[chest][np.argmin(co[chest][:, 1])]
    # waist: minimum width slice between 0.9 and apex-0.1
    zs = np.linspace(apex[2] - 0.25, apex[2] - 0.08, 40)
    widths = []
    for z in zs:
        sl = co[torso & (np.abs(co[:, 2] - z) < 0.004) & (np.abs(co[:, 0]) < 0.14)]
        widths.append(sl[:, 0].max() - sl[:, 0].min() if len(sl) > 5 else 9)
    z_w = float(zs[int(np.argmin(widths))])
    # armpit: lowest z where the arm separates from torso (arm verts with |x|>0.14 close to torso)
    rig = bpy.data.objects[RIG]
    z_ap = float(rig.data.bones["upperarm01.L"].head_local.z) - 0.075
    # hip: max width below waist
    zs2 = np.linspace(z_w - 0.25, z_w - 0.05, 40)
    w2 = []
    for z in zs2:
        sl = co[(np.abs(co[:, 2] - z) < 0.004) & (np.abs(co[:, 0]) < 0.25)]
        w2.append(sl[:, 0].max() - sl[:, 0].min() if len(sl) > 5 else 0)
    z_hip = float(zs2[int(np.argmax(w2))])
    return dict(apex=apex, z_w=z_w, z_ap=z_ap, z_hip=z_hip)


def torso_axis_y(co, z):
    for dz, dx in ((0.008, 0.03), (0.015, 0.05), (0.03, 0.07)):
        sl = co[(np.abs(co[:, 2] - z) < dz) & (np.abs(co[:, 0]) < dx)]
        if len(sl) > 4 and (sl[:, 1].max() - sl[:, 1].min()) > 0.08:
            return 0.5 * (sl[:, 1].min() + sl[:, 1].max())
    return -0.02


def convex_ring(pts2d):
    """Convex hull envelope radius per input angle (pts in order of angle)."""
    import itertools
    P = pts2d
    # monotone chain hull
    idx = np.lexsort((P[:, 1], P[:, 0]))
    S = P[idx]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in S:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in S[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def ray_ring(bvh, axis, z, thetas, hull=True, max_r=0.4):
    """Body cross-section radius at angles thetas (0=front(-Y), +=toward +X)."""
    o = Vector((axis[0], axis[1], z))
    pts = []
    for th in thetas:
        d = Vector((math.sin(th), -math.cos(th), 0.0))
        hit = bvh.ray_cast(o, d, max_r)
        if hit[0] is None:
            pts.append(None)
        else:
            pts.append(np.array(hit[0][:2]))
    # fill misses by neighbours
    arr = np.array([p if p is not None else [np.nan, np.nan] for p in pts])
    r = np.linalg.norm(arr - np.array(axis[:2]), axis=1)
    if np.isnan(r).any():
        good = ~np.isnan(r)
        r[~good] = np.interp(np.where(~good)[0], np.where(good)[0], r[good])
    if not hull:
        return r
    P = np.stack([np.sin(thetas) * r, -np.cos(thetas) * r], 1)
    H = convex_ring(P)
    # radius of the hull polygon along each theta direction
    rr = np.empty_like(r)
    Hn = np.roll(H, -1, 0)
    for i, th in enumerate(thetas):
        d = np.array([math.sin(th), -math.cos(th)])
        best = 0
        for a, b in zip(H, Hn):
            # ray-segment intersection from origin
            e = b - a
            den = d[0] * (-e[1]) - d[1] * (-e[0])
            if abs(den) < 1e-12:
                continue
            tt = (a[0] * (-e[1]) - a[1] * (-e[0])) / den
            u = (d[0] * a[1] - d[1] * a[0]) / den
            if tt > 0 and -1e-6 <= u <= 1 + 1e-6:
                best = max(best, tt)
        rr[i] = best if best > 0 else r[i]
    return rr


def make_tube(name, rings, thetas, zs, axis_y):
    """rings: (nz, nt) radii. Returns new mesh object with UV (u around, v up)."""
    nz, nt = rings.shape
    bm = bmesh.new()
    vs = []
    for i in range(nz):
        row = []
        for j in range(nt):
            th = thetas[j]
            r = rings[i, j]
            row.append(bm.verts.new((math.sin(th) * r, axis_y[i] - math.cos(th) * r, zs[i, j])))
        vs.append(row)
    uv_l = bm.loops.layers.uv.new("UVMap")
    for i in range(nz - 1):
        for j in range(nt):
            j2 = (j + 1) % nt
            f = bm.faces.new((vs[i][j], vs[i][j2], vs[i + 1][j2], vs[i + 1][j]))
            uvs = [(j / nt, i / (nz - 1)), ((j + 1) / nt, i / (nz - 1)), ((j + 1) / nt, (i + 1) / (nz - 1)), (j / nt, (i + 1) / (nz - 1))]
            for lp, uv in zip(f.loops, uvs):
                lp[uv_l].uv = uv
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = True
    ob = bpy.data.objects.get(name)
    if ob:
        bpy.data.objects.remove(ob, do_unlink=True)
    ob = bpy.data.objects.new(name, me)
    bpy.data.collections[COLL].objects.link(ob)
    return ob


def rig_it(ob):
    """Transfer body weights (rest) and add armature; parent to rig."""
    body = bpy.data.objects[BODY]
    rig = bpy.data.objects[RIG]
    for vg in body.vertex_groups:
        if vg.name.startswith(("helper", "joint", "HelperGeometry", "JointCubes")):
            continue
        if vg.name not in ob.vertex_groups:
            ob.vertex_groups.new(name=vg.name)
    dt = ob.modifiers.new("WeightsFromBody", 'DATA_TRANSFER')
    dt.object = body
    dt.use_vert_data = True
    dt.data_types_verts = {'VGROUP_WEIGHTS'}
    dt.vert_mapping = 'POLYINTERP_NEAREST'
    dt.layers_vgroup_select_src = 'ALL'
    dt.layers_vgroup_select_dst = 'NAME'
    dt.use_object_transform = False
    # evaluate transfer in rest pose
    pp = rig.data.pose_position
    rig.data.pose_position = 'REST'
    bpy.context.view_layer.update()
    bpy.context.view_layer.objects.active = ob
    for o in bpy.context.selected_objects:
        o.select_set(False)
    ob.select_set(True)
    with bpy.context.temp_override(object=ob, active_object=ob, selected_objects=[ob]):
        bpy.ops.object.modifier_apply(modifier=dt.name)
    rig.data.pose_position = pp
    ob.parent = rig
    from mathutils import Matrix
    ob.matrix_parent_inverse = Matrix.Identity(4)
    ob.matrix_basis = Matrix.Identity(4)
    am = ob.modifiers.new("Armature", 'ARMATURE')
    am.object = rig
    am.use_deform_preserve_volume = False
    bpy.context.view_layer.update()


# ---------------------------------------------------------------------------- top
def build_top(nt=160, nz=44, ease=0.0022):
    co, tri = rest_arrays()
    lm = landmarks(co)
    bvh = BVHTree.FromPolygons([Vector(v) for v in co], [tuple(t) for t in tri], all_triangles=True)
    thetas = np.linspace(-math.pi, math.pi, nt, endpoint=False)
    z_b, z_w, z_ap = lm["apex"][2], lm["z_w"], lm["z_ap"]
    bx = abs(lm["apex"][0])

    def top_edge(th):
        a = np.abs(th)
        # sweetheart: dip at centre, rounded over each breast, straight under the arm and across the back
        return np.interp(a, [0.0, 0.10, 0.36, 0.62, 1.05, 1.6, math.pi],
                         [z_b + 0.006, z_b + 0.030, z_b + 0.052, z_b + 0.050, z_ap - 0.030, z_ap - 0.045, z_ap - 0.060])

    def bottom_edge(th):
        a = np.abs(th)
        return np.interp(a, [0.0, 0.35, 1.0, math.pi], [z_w - 0.022, z_w - 0.018, z_w - 0.014, z_w - 0.012])

    zt, zb = top_edge(thetas), bottom_edge(thetas)
    v = np.linspace(0, 1, nz)
    Z = zb[None, :] + (zt - zb)[None, :] * v[:, None]
    zrow = Z.mean(1)
    axis_y = np.array([torso_axis_y(co, z) for z in zrow])
    # hull radius table H(z, theta) on a fine z grid (rows are sloped, so sample per vertex height)
    zg = np.arange(Z.min() - 0.003, Z.max() + 0.006, 0.003)
    ay_g = np.array([torso_axis_y(co, z) for z in zg])
    ay_g = np.convolve(np.pad(ay_g, 3, mode='edge'), np.ones(7) / 7, mode='valid')
    Hg = np.array([ray_ring(bvh, (0.0, ay_g[k]), float(zg[k]), thetas) for k in range(len(zg))])
    axis_y = np.interp(zrow, zg, ay_g)
    rings = np.zeros((nz, nt))
    for j in range(nt):
        r_col = np.interp(Z[:, j], zg, Hg[:, j])
        # convert from the grid axis to the row axis (small shift along the theta direction)
        ay_v = np.interp(Z[:, j], zg, ay_g)
        rings[:, j] = r_col - (axis_y - ay_v) * (-math.cos(thetas[j]))
    rings += ease
    # structured silhouette: smooth vertically and around (never shrinking inside the body)
    base = rings.copy()
    for _ in range(4):
        rings[1:-1] = 0.5 * rings[1:-1] + 0.25 * (rings[:-2] + rings[2:])
        rings = 0.5 * rings + 0.25 * (np.roll(rings, 1, 1) + np.roll(rings, -1, 1))
        rings = np.maximum(rings, base - 0.0015)
    # piping bead along the neckline and the waist edge
    vv = np.linspace(0, 1, nz)[:, None]
    rings += 0.0012 * np.exp(-0.5 * ((vv - 1.0) / 0.012) ** 2) + 0.0009 * np.exp(-0.5 * (vv / 0.012) ** 2)
    ob = make_tube("Mira_Top_Bustier", rings, thetas, Z, axis_y)
    # boning / princess seams: shallow grooves at seam angles, slight ridges at boning
    me = ob.data
    seams = [0.30, -0.30, 0.78, -0.78, 1.45, -1.45, 2.3, -2.3, math.pi]
    for vtx in me.vertices:
        i, j = divmod(vtx.index, nt)
        th = thetas[j]
        g = 0.0
        for s in seams:
            dth = math.atan2(math.sin(th - s), math.cos(th - s))
            g += -0.0007 * math.exp(-0.5 * (dth / 0.012) ** 2)
            g += 0.00035 * math.exp(-0.5 * ((abs(dth) - 0.035) / 0.012) ** 2)
        n = Vector((math.sin(th), -math.cos(th), 0.0))
        vtx.co += n * g
    ob["lm"] = [float(z_b), float(z_w), float(z_ap)]
    sol = ob.modifiers.new("Thickness", 'SOLIDIFY')
    sol.thickness = 0.0028
    sol.offset = 1.0
    sol.use_even_offset = True
    sol.use_rim = True
    sol.use_quality_normals = True
    rig_it(ob)
    sub = ob.modifiers.new("Subdivision", 'SUBSURF')
    sub.levels = 1
    sub.render_levels = 2
    # order: Armature, Thickness, Subdivision
    _order(ob, ["Armature", "Thickness", "Subdivision"])
    return dict(lm=dict(z_b=float(z_b), z_w=float(z_w), z_ap=float(z_ap), z_hip=float(lm["z_hip"]), apex=lm["apex"].tolist()))


def _order(ob, names):
    for idx, n in enumerate(names):
        if n in ob.modifiers:
            with bpy.context.temp_override(object=ob):
                bpy.ops.object.modifier_move_to_index(modifier=n, index=idx)


# ---------------------------------------------------------------------------- skirt
def build_skirt(nt=192, nz=46, length=0.335, flare=0.42, ease=0.004):
    co, tri = rest_arrays()
    lm = landmarks(co)
    bvh = BVHTree.FromPolygons([Vector(v) for v in co], [tuple(t) for t in tri], all_triangles=True)
    thetas = np.linspace(-math.pi, math.pi, nt, endpoint=False)
    z_w, z_hip = lm["z_w"], lm["z_hip"]
    z_top = z_w + 0.022
    z_hem = z_top - length
    zs = np.linspace(z_top, z_hem, nz)
    Z = np.repeat(zs[:, None], nt, 1)
    axis_y = np.array([torso_axis_y(co, min(max(z, z_hip - 0.02), z_top)) for z in zs])
    rings = np.zeros((nz, nt))
    hip_ring = None
    for i, z in enumerate(zs):
        if z >= z_hip:
            # waistband sits over the tucked-in bustier (bustier ease + thickness + gap)
            over = 0.0052 * float(np.clip((z - (z_w - 0.03)) / 0.02, 0, 1))
            rings[i] = ray_ring(bvh, (0.0, axis_y[i]), float(z), thetas) + ease + over
        else:
            if hip_ring is None:
                hip_ring = ray_ring(bvh, (0.0, axis_y[i]), float(z_hip), thetas) + ease
            # A-line: grow from the hip ring toward a rounder, flared hem
            f = (z_hip - z) / (z_hip - z_hem)
            round_r = hip_ring.mean()
            target = hip_ring * (1 - 0.5 * f) + round_r * 0.5 * f
            rings[i] = target * (1 + flare * f ** 1.1)
            axis_y[i] = axis_y[i - 1]
    ob = make_tube("Mira_Skirt", rings, thetas, Z, axis_y)
    # pin group: waistband rows
    pin = ob.vertex_groups.new(name="pin")
    shrink = ob.vertex_groups.new(name="waistband")
    for vtx in ob.data.vertices:
        z = vtx.co.z
        w = float(np.clip((z - (z_top - 0.05)) / 0.015, 0, 1))
        if w > 0:
            pin.add([vtx.index], w, 'REPLACE')
        if z > z_top - 0.035:
            shrink.add([vtx.index], 1.0, 'REPLACE')
    rig_it(ob)
    return dict(z_top=float(z_top), z_hem=float(z_hem), z_hip=float(z_hip))
