"""Stage 3 pipeline: proportions, pose, expression, hair, garments.

Applies everything built after milestone 2 onto mira_idol.blend so the scene
can be rebuilt from scratch in an interactive or background Blender:

    blender -b mira_idol.blend -P scripts/build_stage3.py

Each step is idempotent where practical (re-running resets pose and rebuilds
hair/garments).
"""
import bpy
import os
import sys
import math
import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(PROJ, "scripts")

from bl_ext.user_default.mpfb.services.targetservice import TargetService
from bl_ext.user_default.mpfb.services.humanservice import HumanService
from bl_ext.user_default.mpfb.entities.objectproperties import HumanObjectProperties

BODY = bpy.data.objects["Mira_Body"]
RIG = bpy.data.objects["Mira_Body.rig"]


def set_t(name, val):
    p = TargetService.target_full_path(name)
    sk = TargetService.filename_to_shapekey_name(p)
    kb = BODY.data.shape_keys.key_blocks.get(sk)
    if kb:
        kb.value = val
    else:
        TargetService.load_target(BODY, p, weight=val)


EXPR_DIR = os.path.join(os.path.dirname(os.path.dirname(TargetService.target_full_path("head-oval"))),
                        "expression", "units", "asian")


def expr(name, val, key=None, vgroup=None, smin=0.0):
    kn = key or ("EXPR_" + name)
    kb = BODY.data.shape_keys.key_blocks.get(kn)
    if kb is None:
        kb = TargetService.load_target(BODY, os.path.join(EXPR_DIR, name + ".target.gz"), weight=val, name=kn)
    kb.slider_min = smin
    kb.value = val
    if vgroup:
        kb.vertex_group = vgroup


def step_face_and_height():
    face = {
        "mouth-upperlip-volume-incr": 0.12, "mouth-lowerlip-volume-incr": 0.2, "mouth-scale-depth-decr": 0.0,
        "mouth-cupidsbow-incr": 0.3, "head-fat-decr": 0.18, "l-cheek-volume-incr": 0.0, "r-cheek-volume-incr": 0.0,
        "l-cheek-volume-decr": 0.1, "r-cheek-volume-decr": 0.12, "mouth-trans-backward": 0.45,
        "chin-prognathism-incr": 0.2,
    }
    for k, v in face.items():
        set_t(k, v)
    # petite: ~1.62 m barefoot, ~1.70 m in heels
    HumanObjectProperties.set_value("height", 0.55, entity_reference=BODY)
    TargetService.reapply_macro_details(BODY)
    HumanService.refit(BODY)
    # ground the feet (local mesh min z -> world 0)
    RIG.location = (0, 0, 0)
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    st = {m.name: m.show_viewport for m in BODY.modifiers}
    for m in BODY.modifiers:
        if m.type in ('ARMATURE', 'SUBSURF'):
            m.show_viewport = False
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    me = BODY.evaluated_get(dg).to_mesh()
    co = np.empty(len(me.vertices) * 3); me.vertices.foreach_get("co", co)
    BODY.evaluated_get(dg).to_mesh_clear()
    for m in BODY.modifiers:
        m.show_viewport = st[m.name]
    RIG.location.z = -float(co.reshape(-1, 3)[:, 2].min())
    bpy.context.view_layer.update()


def step_pose():
    ns = {}
    src = open(os.path.join(SCRIPTS, "pose_mira.py")).read().replace('if __name__ == "__main__" or True:\n    pose()', '')
    exec(src, ns)
    pr = dict(ns["PARAMS"])
    pr.update(dict(pelvis_shift=(-0.065, 0.0), pelvis_tilt=10.0, spine_tilt=-17.0, ankle_R=(-0.035, 0.02),
                   ankle_L=(-0.005, -0.165), knee_pole_L=(-0.8, -1.0, 0.0), footyaw_L=-12.0,
                   elbow_L=(0.10, 0.06, -0.24), wrist_L=(0.175, -0.07, 0.82), head_tilt=-8.0))
    ns["pose"](pr)
    # relaxed finger curl, increasing toward the pinky
    from mathutils import Quaternion
    curl = {"finger2": (12, 18, 10), "finger3": (16, 24, 14), "finger4": (20, 28, 16),
            "finger5": (24, 32, 18), "finger1": (6, 12, 8)}
    for side, scale in (("L", 1.0), ("R", 1.2)):
        for f, angs in curl.items():
            for i, a in enumerate(angs):
                n = "%s-%d.%s" % (f, i + 1, side)
                if n in RIG.pose.bones:
                    pb = RIG.pose.bones[n]
                    pb.rotation_mode = 'QUATERNION'
                    pb.rotation_quaternion = Quaternion((1, 0, 0), math.radians(a * scale))
    # eyes engage the camera
    cam = bpy.data.objects["CAM_Main_85mm"]
    for side in ("L", "R"):
        pb = RIG.pose.bones["eye." + side]
        for c in list(pb.constraints):
            pb.constraints.remove(c)
        c = pb.constraints.new('DAMPED_TRACK')
        c.name = "LookAtCamera"
        c.target = cam
        c.track_axis = 'TRACK_Y'
    bpy.context.view_layer.update()


def step_expression():
    for k, v in {"l-eye-height2-incr": 0.32, "r-eye-height2-incr": 0.3, "eyebrows-trans-up": 0.5,
                 "eyebrows-angle-up": 0.2}.items():
        set_t(k, v)
    expr("mouth-corner-puller", 0.5)
    expr("eye-left-slit", 0.26)
    expr("eye-right-slit", 0.2)
    expr("eyebrows-left-extern-up", 0.06)
    expr("mouth-compression", 0.0)
    expr("mouth-protusion", -0.1, smin=-1.0)
    expr("mouth-corner-puller", 0.0, key="EXPR_smile_Left", vgroup="Left")
    expr("mouth-corner-puller", 0.0, key="EXPR_smile_Right", vgroup="Right")
    HumanService.refit(BODY)


def step_skin_tweaks():
    m = bpy.data.materials["MAT_Mira_Skin"]
    N = m.node_tree.nodes
    N["Math"].inputs[1].default_value = 0.3
    N["Principled BSDF"].inputs["Coat Roughness"].default_value = 0.14
    for b in [n for n in N if n.type == 'BUMP']:
        if b.inputs["Distance"].default_value > 0.0001:
            b.inputs["Strength"].default_value = 0.55
            b.inputs["Distance"].default_value = 0.0004
        else:
            b.inputs["Strength"].default_value = 0.12


def hair_material():
    m = bpy.data.materials.get("MAT_Mira_Hair") or bpy.data.materials.new("MAT_Mira_Hair")
    m.use_nodes = True
    N = m.node_tree.nodes
    L = m.node_tree.links
    N.clear()
    out = N.new("ShaderNodeOutputMaterial"); out.location = (600, 0)
    hb = N.new("ShaderNodeBsdfHairPrincipled"); hb.location = (300, 0)
    hb.parametrization = 'MELANIN'
    info = N.new("ShaderNodeHairInfo"); info.location = (-500, 0)
    mr = N.new("ShaderNodeMapRange"); mr.location = (-200, 150)
    mr.inputs["To Min"].default_value = 0.97
    mr.inputs["To Max"].default_value = 0.93
    L.new(info.outputs["Intercept"], mr.inputs["Value"])
    L.new(mr.outputs[0], hb.inputs["Melanin"])
    hb.inputs["Melanin Redness"].default_value = 0.22
    hb.inputs["Tint"].default_value = (1.0, 0.96, 0.93, 1)
    hb.inputs["Roughness"].default_value = 0.24
    hb.inputs["Radial Roughness"].default_value = 0.32
    hb.inputs["Coat"].default_value = 0.12
    hb.inputs["IOR"].default_value = 1.55
    hb.inputs["Offset"].default_value = 0.035
    hb.inputs["Random Color"].default_value = 0.1
    hb.inputs["Random Roughness"].default_value = 0.15
    L.new(info.outputs["Random"], hb.inputs["Random"])
    L.new(hb.outputs[0], out.inputs[0])
    return m


def step_hair():
    ns = {}
    exec(open(os.path.join(SCRIPTS, "gen_hair.py")).read(), ns)
    r = ns["build"]()
    hair = bpy.data.objects["Mira_Hair"]
    hair.data.materials.clear()
    hair.data.materials.append(hair_material())
    scn = bpy.context.scene
    scn.cycles_curves.shape = 'THICK'
    scn.cycles_curves.subdivisions = 3
    return r


def satin_material(name, base, rough, aniso, sheen):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    N = m.node_tree.nodes
    L = m.node_tree.links
    N.clear()
    out = N.new("ShaderNodeOutputMaterial"); out.location = (700, 0)
    b = N.new("ShaderNodeBsdfPrincipled"); b.location = (350, 0)
    b.inputs["Base Color"].default_value = base
    b.inputs["Roughness"].default_value = rough
    b.inputs["Anisotropic"].default_value = aniso
    b.inputs["Sheen Weight"].default_value = sheen
    b.inputs["Sheen Roughness"].default_value = 0.3
    b.inputs["Sheen Tint"].default_value = (1.0, 0.97, 0.92, 1)
    b.inputs["Specular IOR Level"].default_value = 0.55
    tan = N.new("ShaderNodeTangent"); tan.location = (0, -300)
    tan.direction_type = 'UV_MAP'; tan.uv_map = "UVMap"
    L.new(tan.outputs[0], b.inputs["Tangent"])
    tc = N.new("ShaderNodeTexCoord"); tc.location = (-600, -500)
    wave = N.new("ShaderNodeTexWave"); wave.location = (-350, -500)
    wave.wave_type = 'BANDS'; wave.bands_direction = 'Z'
    wave.inputs["Scale"].default_value = 2600.0
    wave.inputs["Distortion"].default_value = 0.6
    L.new(tc.outputs["Object"], wave.inputs["Vector"])
    bump = N.new("ShaderNodeBump"); bump.location = (0, -550)
    bump.inputs["Strength"].default_value = 0.04
    bump.inputs["Distance"].default_value = 0.00005
    L.new(wave.outputs["Fac"], bump.inputs["Height"])
    L.new(bump.outputs[0], b.inputs["Normal"])
    L.new(b.outputs[0], out.inputs[0])
    return m


def step_top():
    ns = {}
    exec(open(os.path.join(SCRIPTS, "gen_clothes.py")).read(), ns)
    r = ns["build_top"]()
    top = bpy.data.objects["Mira_Top_Bustier"]
    top.data.materials.clear()
    top.data.materials.append(satin_material("MAT_Bustier_IvorySatin", (0.80, 0.755, 0.68, 1), 0.28, 0.6, 0.35))
    return r


def cleanup_temp_cameras():
    for n in ("CAM_Tmp", "CAM_HandCheck", "CAM_TorsoCheck", "CAM_FaceCheck34"):
        o = bpy.data.objects.get(n)
        if o:
            bpy.data.objects.remove(o, do_unlink=True)


if __name__ == "__main__":
    step_face_and_height()
    step_pose()
    step_expression()
    step_skin_tweaks()
    print("hair", step_hair())
    print("top", step_top())
    bpy.ops.wm.save_mainfile(compress=True)


# ---------------------------------------------------------------------------- skirt (cloth)
SIM_START, SIM_POSE, SIM_END = 1, 30, 70


def _key_pose():
    """Animate the rig from rest (frame 1) to the final pose (frame SIM_POSE)."""
    rig = RIG
    final = {pb.name: (pb.location.copy(), pb.rotation_quaternion.copy(), pb.rotation_mode) for pb in rig.pose.bones}
    if rig.animation_data:
        rig.animation_data_clear()
    for pb in rig.pose.bones:
        loc, q, mode = final[pb.name]
        pb.rotation_mode = 'QUATERNION'
        pb.location = loc
        pb.rotation_quaternion = q
        pb.keyframe_insert("location", frame=SIM_POSE)
        pb.keyframe_insert("rotation_quaternion", frame=SIM_POSE)
        pb.location = (0, 0, 0)
        pb.rotation_quaternion = (1, 0, 0, 0)
        pb.keyframe_insert("location", frame=SIM_START)
        pb.keyframe_insert("rotation_quaternion", frame=SIM_START)
    return final


def step_skirt(length=0.41):
    ns = {}
    exec(open(os.path.join(SCRIPTS, "gen_clothes.py")).read(), ns)
    r = ns["build_skirt"](length=length)
    sk = bpy.data.objects["Mira_Skirt"]
    sk.data.materials.clear()
    sk.data.materials.append(satin_material("MAT_Skirt_BlackSatin", (0.012, 0.012, 0.013, 1), 0.42, 0.45, 0.35))
    # colliders
    for name, thick in (("Mira_Body", 0.0025), ("Mira_Top_Bustier", 0.0015)):
        ob = bpy.data.objects[name]
        col = ob.modifiers.get("Collision") or ob.modifiers.new("Collision", 'COLLISION')
        ob.collision.thickness_outer = thick
        ob.collision.cloth_friction = 8.0
        # collide with the deformed (posed) mesh before subdivision
        names = [m.name for m in ob.modifiers]
        if "Subdivision" in names:
            with bpy.context.temp_override(object=ob):
                bpy.ops.object.modifier_move_to_index(modifier="Collision", index=names.index("Subdivision"))
    csm = sk.modifiers.new("CorrectiveSmooth", 'CORRECTIVE_SMOOTH')
    csm.factor = 0.8; csm.iterations = 15; csm.rest_source = 'ORCO'
    cl = sk.modifiers.new("Cloth", 'CLOTH')
    s = cl.settings
    s.quality = 10
    s.mass = 0.12
    s.air_damping = 1.5
    s.tension_stiffness = 20; s.compression_stiffness = 20; s.shear_stiffness = 8
    s.bending_stiffness = 0.6
    s.tension_damping = 5; s.compression_damping = 5; s.bending_damping = 0.5
    s.vertex_group_mass = "pin"
    s.pin_stiffness = 1.0
    cs = cl.collision_settings
    cs.use_collision = True
    cs.distance_min = 0.0025
    cs.collision_quality = 4
    cs.use_self_collision = True
    cs.self_distance_min = 0.0015
    cl.point_cache.frame_start = SIM_START
    cl.point_cache.frame_end = SIM_END
    # order: Armature -> Cloth
    with bpy.context.temp_override(object=sk):
        bpy.ops.object.modifier_move_to_index(modifier="Armature", index=0)
        bpy.ops.object.modifier_move_to_index(modifier="CorrectiveSmooth", index=1)
        bpy.ops.object.modifier_move_to_index(modifier="Cloth", index=2)
    scn = bpy.context.scene
    scn.frame_start = SIM_START
    scn.frame_end = SIM_END
    _key_pose()
    scn.frame_set(SIM_START)
    return r


def step_freeze_skirt():
    """Bake the draped skirt at SIM_END into a static mesh; drop the sim animation."""
    scn = bpy.context.scene
    scn.frame_set(SIM_END)
    sk = bpy.data.objects["Mira_Skirt"]
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(sk.evaluated_get(dg), preserve_all_data_layers=True, depsgraph=dg)
    old = sk.data
    sk.modifiers.clear()
    sk.data = me
    me.name = "Mira_Skirt"
    if old.users == 0:
        bpy.data.meshes.remove(old)
    for p in me.polygons:
        p.use_smooth = True
    so = sk.modifiers.new("Thickness", 'SOLIDIFY')
    so.thickness = 0.0012; so.offset = 1.0; so.use_even_offset = True; so.use_rim = True
    sd = sk.modifiers.new("Subdivision", 'SUBSURF'); sd.levels = 1; sd.render_levels = 2
    # pose stays at the final values once the animation is removed
    pose = {pb.name: (pb.location.copy(), pb.rotation_quaternion.copy()) for pb in RIG.pose.bones}
    RIG.animation_data_clear()
    for pb in RIG.pose.bones:
        pb.location, pb.rotation_quaternion = pose[pb.name]
    for name in ("Mira_Body", "Mira_Top_Bustier"):
        ob = bpy.data.objects[name]
        m = ob.modifiers.get("Collision")
        if m:
            ob.modifiers.remove(m)
    scn.frame_start = scn.frame_end = 1
    scn.frame_set(1)
    bpy.context.view_layer.update()
    return {"verts": len(me.vertices)}


def step_shoes():
    ns = {}
    exec(open(os.path.join(SCRIPTS, "gen_shoes.py")).read(), ns)
    return ns["build"]()


def step_jewelry():
    ns = {}
    exec(open(os.path.join(SCRIPTS, "gen_jewelry.py")).read(), ns)
    return ns["build"]()


def step_export_and_textures(size=4096):
    """Re-export the rest body and regenerate the 4K skin maps (system python + OpenCV)."""
    import subprocess
    import tempfile
    body = BODY
    st = {m.name: m.show_viewport for m in body.modifiers}
    for m in body.modifiers:
        m.show_viewport = False
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    em = body.evaluated_get(dg).to_mesh()
    n = len(em.vertices)
    co = np.empty(n * 3, np.float32); em.vertices.foreach_get("co", co)
    em.calc_loop_triangles(); nt = len(em.loop_triangles)
    tl = np.empty(nt * 3, np.int32); em.loop_triangles.foreach_get("loops", tl)
    tv = np.empty(nt * 3, np.int32); em.loop_triangles.foreach_get("vertices", tv)
    tm = np.empty(nt, np.int32); em.loop_triangles.foreach_get("material_index", tm)
    uv = np.empty(len(em.loops) * 2, np.float32); em.uv_layers["UVMap"].data.foreach_get("uv", uv)
    vn = np.empty(n * 3, np.float32); em.vertices.foreach_get("normal", vn)
    body.evaluated_get(dg).to_mesh_clear()
    for m in body.modifiers:
        m.show_viewport = st[m.name]
    gi = body.vertex_groups["body"].index
    inbody = np.zeros(n, bool)
    for v in body.data.vertices:
        for g in v.groups:
            if g.group == gi and g.weight > 0.5:
                inbody[v.index] = True
    # eyes / brows in rest pose (armature at rest)
    pp = RIG.data.pose_position
    RIG.data.pose_position = 'REST'
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()

    def pts(name):
        o = bpy.data.objects[name]
        m = o.evaluated_get(dg).to_mesh(); c = np.array([v.co[:] for v in m.vertices]); o.evaluated_get(dg).to_mesh_clear()
        W = np.array(RIG.matrix_world.inverted() @ o.matrix_world)       # into rig-local (= body-local) space
        return (W[:3, :3] @ c.T).T + W[:3, 3]
    eyes = pts([o.name for o in bpy.data.objects if "high-poly" in o.name][0])
    brows = pts([o.name for o in bpy.data.objects if "eyebrow" in o.name][0])
    RIG.data.pose_position = pp
    bpy.context.view_layer.update()
    npz = os.path.join(tempfile.gettempdir(), "mira_body_mesh.npz")
    np.savez_compressed(npz, co=co.reshape(-1, 3), tri_loops=tl.reshape(-1, 3), tri_verts=tv.reshape(-1, 3), tri_mat=tm,
                        uv=uv.reshape(-1, 2), vn=vn.reshape(-1, 3), inbody=inbody, eyes=eyes, brows=brows,
                        mat_names=np.array(["Mira_Body.body", "Mira_Body.nipple", "Mira_Body.lips", "Mira_Body.fingernails",
                                            "Mira_Body.toenails", "Mira_Body.ears", "Mira_Body.genitals"]))
    diffuse = os.path.join(bpy.utils.user_resource('EXTENSIONS'), ".user", "user_default", "mpfb", "data", "skins",
                           "young_asian_female", "young_lightskinned_female_diffuse3.png")
    py = "/opt/homebrew/bin/python3"
    r = subprocess.run([py, os.path.join(SCRIPTS, "gen_skin_textures.py"), npz, diffuse, os.path.join(PROJ, "textures"), str(size)],
                       capture_output=True, text=True)
    for img in bpy.data.images:
        if img.name.startswith("T_Mira_Skin"):
            img.reload()
    return {"rc": r.returncode, "out": r.stdout[-400:], "err": r.stderr[-400:]}
