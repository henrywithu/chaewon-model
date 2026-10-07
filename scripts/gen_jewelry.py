"""Fine silver chain necklace with a small crystal drop, draped on the posed body.

The chain path is designed around the posed neck base (higher at the back,
dropping to just below the collarbones in front), projected onto the skin
(0.9 mm off) and built from real alternating oval links; the pendant hangs
from the lowest link. Run inside Blender after posing.
"""
import bpy
import bmesh
import math
import numpy as np
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree

BODY = "Mira_Body"
RIG = "Mira_Body.rig"
COLL = "CHAR_Mira"


def posed_bvh():
    body = bpy.data.objects[BODY]
    st = {m.name: m.show_viewport for m in body.modifiers}
    for m in body.modifiers:
        if m.type in ('SUBSURF',):
            m.show_viewport = False
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    ev = body.evaluated_get(dg)
    me = ev.to_mesh()
    me.calc_loop_triangles()
    M = body.matrix_world
    verts = [M @ v.co for v in me.vertices]
    tris = [tuple(t.vertices) for t in me.loop_triangles]
    ev.to_mesh_clear()
    for m in body.modifiers:
        m.show_viewport = st[m.name]
    bpy.context.view_layer.update()
    return BVHTree.FromPolygons(verts, tris, all_triangles=True)


def metal_mat():
    m = bpy.data.materials.get("MAT_Metal_Silver") or bpy.data.materials.new("MAT_Metal_Silver")
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (0.91, 0.92, 0.92, 1)
    b.inputs["Metallic"].default_value = 1.0
    b.inputs["Roughness"].default_value = 0.12
    return m


def crystal_mat():
    m = bpy.data.materials.get("MAT_Crystal") or bpy.data.materials.new("MAT_Crystal")
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (1, 1, 1, 1)
    b.inputs["Transmission Weight"].default_value = 1.0
    b.inputs["IOR"].default_value = 2.0
    b.inputs["Roughness"].default_value = 0.0
    return m


def build(drop=0.055, link_len=0.0032, wire=0.00045):
    rig = bpy.data.objects[RIG]
    mw = rig.matrix_world
    bvh = posed_bvh()
    neck = mw @ rig.pose.bones["neck01"].head
    sp = rig.pose.bones["spine01"]
    R = (mw @ sp.matrix).to_3x3() @ sp.bone.matrix_local.to_3x3().inverted()
    fwd = (R @ Vector((0, -1, 0))).normalized()
    up = (R @ Vector((0, 0, 1))).normalized()
    lat = fwd.cross(up).normalized()
    # design path
    N = 400
    pts = []
    for i in range(N):
        t = 2 * math.pi * i / N
        d = fwd * math.cos(t) + lat * math.sin(t)
        front = ((1 + math.cos(t)) / 2) ** 2.2
        p = neck + d * (0.055 + 0.03 * front) + up * (0.012 - drop * front)
        pts.append(p)
    # project onto skin and relax (a few iterations keep spacing even)
    P = [Vector(p) for p in pts]
    for it in range(6):
        Q = []
        for p in P:
            loc, nrm, _, _ = bvh.find_nearest(p)
            Q.append(loc + nrm * (0.0009 + wire))
        # gravity sag: nudge down slightly along the surface then re-project
        P = [q + Vector((0, 0, -0.0015)) for q in Q] if it < 5 else Q
        # smooth
        P = [(P[i - 1] + P[i] * 2 + P[(i + 1) % N]) / 4 for i in range(N)]
    P = []
    for p in Q:
        P.append(p)
    # resample by arc length into links
    seg = [(P[(i + 1) % N] - P[i]).length for i in range(N)]
    total = sum(seg)
    nlinks = int(total / (link_len * 0.82))
    cum = np.concatenate([[0], np.cumsum(seg)])
    samples = []
    for k in range(nlinks):
        s = k * total / nlinks
        i = int(np.searchsorted(cum, s) - 1)
        i = max(0, min(i, N - 1))
        a = (s - cum[i]) / max(seg[i], 1e-9)
        samples.append(P[i].lerp(P[(i + 1) % N], a))
    bm = bmesh.new()
    lowest = min(range(len(samples)), key=lambda k: samples[k].z)
    for k, c in enumerate(samples):
        tdir = (samples[(k + 1) % len(samples)] - samples[k - 1]).normalized()
        loc, nrm, _, _ = bvh.find_nearest(c)
        n = nrm.normalized()
        b = tdir.cross(n).normalized()
        # alternate link planes (curb chain links lie nearly flat, twisted)
        ang = (math.radians(70) if k % 2 else math.radians(-20))
        nn = n * math.cos(ang) + b * math.sin(ang)
        bb = tdir.cross(nn).normalized()
        ring_major_a = link_len * 0.5
        ring_major_b = link_len * 0.32
        segs, rsegs = 12, 4
        verts = []
        for i in range(segs):
            th = 2 * math.pi * i / segs
            center = c + tdir * (ring_major_a * math.cos(th)) + bb * (ring_major_b * math.sin(th))
            tang = (-tdir * ring_major_a * math.sin(th) + bb * ring_major_b * math.cos(th)).normalized()
            radial = tang.cross(nn).normalized()
            row = []
            for j in range(rsegs):
                ph = 2 * math.pi * j / rsegs
                row.append(bm.verts.new(center + (radial * math.cos(ph) + nn * math.sin(ph)) * wire))
            verts.append(row)
        for i in range(segs):
            i2 = (i + 1) % segs
            for j in range(rsegs):
                j2 = (j + 1) % rsegs
                bm.faces.new((verts[i][j], verts[i2][j], verts[i2][j2], verts[i][j2]))
    me = bpy.data.meshes.new("Mira_Necklace_Chain")
    bm.to_mesh(me); bm.free()
    for p in me.polygons:
        p.use_smooth = True
    old = bpy.data.objects.get("Mira_Necklace_Chain")
    if old:
        bpy.data.objects.remove(old, do_unlink=True)
    ob = bpy.data.objects.new("Mira_Necklace_Chain", me)
    bpy.data.collections[COLL].objects.link(ob)
    me.materials.append(metal_mat())
    # pendant: bail + faceted crystal drop hanging from the lowest link
    c = samples[lowest]
    loc, nrm, _, _ = bvh.find_nearest(c)
    down = (Vector((0, 0, -1)) - nrm * Vector((0, 0, -1)).dot(nrm)).normalized()
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=1, radius=1.0)
    for v in bm.verts:
        x, y, z = v.co
        zz = z * 0.0042 - 0.0045 if z < 0 else z * 0.0028 - 0.0045
        v.co = Vector((x * 0.0028, y * 0.0028, zz))
    # orient: local -Z = down along the chest, local Y = surface normal
    M = Matrix((down.cross(nrm).normalized(), nrm, -down)).transposed().to_4x4()
    bmesh.ops.transform(bm, matrix=Matrix.Translation(c + nrm * 0.0022) @ M, verts=bm.verts)
    me2 = bpy.data.meshes.new("Mira_Necklace_Pendant")
    bm.to_mesh(me2); bm.free()
    old = bpy.data.objects.get("Mira_Necklace_Pendant")
    if old:
        bpy.data.objects.remove(old, do_unlink=True)
    ob2 = bpy.data.objects.new("Mira_Necklace_Pendant", me2)
    bpy.data.collections[COLL].objects.link(ob2)
    me2.materials.append(crystal_mat())
    # small silver bail ring
    bm = bmesh.new()
    bmesh.ops.create_circle(bm, segments=16, radius=0.0012)
    me3 = bpy.data.meshes.new("Mira_Necklace_Bail")
    bm.to_mesh(me3); bm.free()
    old = bpy.data.objects.get("Mira_Necklace_Bail")
    if old:
        bpy.data.objects.remove(old, do_unlink=True)
    ob3 = bpy.data.objects.new("Mira_Necklace_Bail", me3)
    bpy.data.collections[COLL].objects.link(ob3)
    ob3.matrix_world = Matrix.Translation(c + nrm * 0.0012 - down * 0.0003) @ Matrix((down, nrm.cross(down), nrm)).transposed().to_4x4()
    ob3.data.materials.append(metal_mat())
    wf = ob3.modifiers.new("Wire", 'WIREFRAME'); wf.thickness = 0.0006
    # parent jewelry to the chest bone (keep placement)
    for o in (ob, ob2, ob3):
        mwo = o.matrix_world.copy()
        o.parent = rig
        o.parent_type = 'BONE'
        o.parent_bone = "spine01"
        pm = mw @ sp.matrix @ Matrix.Translation((0, sp.length, 0))
        o.matrix_parent_inverse = pm.inverted()
        o.matrix_world = mwo
    return {"links": len(samples), "length_cm": total * 100}
