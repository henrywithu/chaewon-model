"""Fashion contrapposto pose for the Mira rig (MPFB default 163-bone rig).

Run inside Blender:  exec(open(".../scripts/pose_mira.py").read())
Works in armature space; the rig object is rotated/placed at the end.
Weight on her right leg, left leg relaxed and crossing slightly forward,
right hand on hip, left arm relaxed, S-curve through hips/ribcage/head.
"""
import bpy
import math
from mathutils import Vector, Matrix, Quaternion

RIG = bpy.data.objects["Mira_Body.rig"]
P = RIG.pose.bones
UP = Vector((0, 0, 1))

PARAMS = dict(
    rig_yaw=20.0,                 # whole body turned toward camera-right (deg)
    pelvis_shift=(-0.040, 0.0),   # hips over the weight (right) foot
    pelvis_tilt=7.0,              # left hip drops (deg, about +Y)
    pelvis_yaw=6.0,
    pelvis_anterior=6.0,
    spine_tilt=-13.0,             # counter tilt: left shoulder rises
    spine_yaw=-13.0,
    spine_arch=-7.0,
    neck_tilt=-3.0,
    head_tilt=-7.0,               # playful tilt toward her right shoulder
    head_yaw=-9.0,
    head_nod=5.0,                 # chin slightly down
    plantar=33.0,                 # foot plantarflexion for ~8.5cm stilettos
    toe_dorsi=-30.0,
    ankle_z=0.123,                # ankle height above floor in heels (world)
    ankle_R=(-0.070, 0.015),      # weight foot (x, y) armature space
    ankle_L=(0.010, -0.150),      # free foot: forward and crossing
    footyaw_R=-14.0,              # weight foot turned out
    footyaw_L=-6.0,
    knee_pole_R=(-0.15, -1.0, 0.0),
    knee_pole_L=(-0.55, -1.0, 0.0),   # free knee turns inward across the body
    elbow_R=(-0.24, 0.10, -0.17),     # right elbow offset from shoulder (hand on hip)
    wrist_R=(-0.155, -0.005, 1.03),   # right wrist target (armature space, z relative to floor)
    elbow_L=(0.075, 0.035, -0.255),
    wrist_L=(0.145, -0.045, 0.80),
)


def upd():
    bpy.context.view_layer.update()


def zoff():
    return RIG.location.z


def rot_about(pb, R3, pivot):
    T = Matrix.Translation(pivot) @ R3.to_4x4() @ Matrix.Translation(-pivot)
    pb.matrix = T @ pb.matrix
    upd()


def rot_world(name, axis, deg, pivot=None):
    pb = P[name]
    R3 = Matrix.Rotation(math.radians(deg), 3, Vector(axis))
    rot_about(pb, R3, pb.head.copy() if pivot is None else pivot)


def delta(pb):
    return pb.matrix.to_3x3() @ pb.bone.matrix_local.to_3x3().inverted()


def frame(p, q):
    a = p.normalized()
    b = (q - q.dot(a) * a).normalized()
    c = a.cross(b)
    return Matrix((a, b, c)).transposed()


def aim(name, start, end_now, end_tgt, pole_now, pole_tgt):
    F0 = frame(end_now - start, pole_now)
    F1 = frame(end_tgt - start, pole_tgt)
    rot_about(P[name], F1 @ F0.transposed(), start)


def two_bone(upper, lower, end_bone, target, pole, ref_axis):
    """Analytic IK: upper chain root=P[upper].head, mid=P[lower].head, end=P[end_bone].head."""
    hip = P[upper].head.copy()
    knee = P[lower].head.copy()
    ank = P[end_bone].head.copy()
    l1, l2 = (knee - hip).length, (ank - knee).length
    d = target - hip
    dist = min(d.length, l1 + l2 - 1e-4)
    dn = d.normalized()
    cosA = (l1 * l1 + dist * dist - l2 * l2) / (2 * l1 * dist)
    A = math.acos(max(-1, min(1, cosA)))
    pole = Vector(pole).normalized()
    side = (pole - pole.dot(dn) * dn).normalized()
    knee_t = hip + dn * (l1 * math.cos(A)) + side * (l1 * math.sin(A))
    # upper segment: aim hip->knee, carry the reference axis (knee/elbow direction) to the pole side
    ref_now = delta(P[upper]) @ Vector(ref_axis)
    aim(upper, hip, knee, knee_t, ref_now, side)
    knee = P[lower].head.copy()
    ank = P[end_bone].head.copy()
    ref_now = delta(P[lower]) @ Vector(ref_axis)
    aim(lower, knee, ank, hip + dn * dist, ref_now, side)


def reset():
    for pb in P:
        pb.rotation_mode = 'QUATERNION'
        pb.rotation_quaternion = (1, 0, 0, 0)
        pb.location = (0, 0, 0)
        pb.scale = (1, 1, 1)
    RIG.rotation_euler = (0, 0, 0)
    upd()


def pose(pr=PARAMS):
    reset()
    z0 = zoff()
    # ---- pelvis / root
    root = P["root"]
    rot_world("root", (0, 1, 0), pr["pelvis_tilt"])
    rot_world("root", (0, 0, 1), pr["pelvis_yaw"])
    rot_world("root", (1, 0, 0), pr["pelvis_anterior"])
    # ---- spine counter-rotation (distributed)
    chain = ["spine05", "spine04", "spine03", "spine02", "spine01"]
    w = [0.15, 0.2, 0.25, 0.25, 0.15]
    for n, k in zip(chain, w):
        rot_world(n, (0, 1, 0), (pr["spine_tilt"] - pr["pelvis_tilt"] * 0.0) * k)
        rot_world(n, (0, 0, 1), pr["spine_yaw"] * k)
        rot_world(n, (1, 0, 0), (pr["spine_arch"] - pr["pelvis_anterior"]) * k)
    for n in ("neck01", "neck02", "neck03"):
        rot_world(n, (0, 1, 0), pr["neck_tilt"] / 3)
        rot_world(n, (0, 0, 1), pr["head_yaw"] * 0.4 / 3)
    rot_world("head", (0, 1, 0), pr["head_tilt"])
    rot_world("head", (0, 0, 1), pr["head_yaw"] * 0.6)
    rot_world("head", (1, 0, 0), pr["head_nod"])
    # ---- place pelvis so the weight leg is almost straight
    hipR = P["upperleg01.R"].head.copy()
    ankR = Vector((pr["ankle_R"][0], pr["ankle_R"][1], pr["ankle_z"] - z0))
    l1 = (P["lowerleg01.R"].head - P["upperleg01.R"].head).length
    l2 = (P["foot.R"].head - P["lowerleg01.R"].head).length
    reach = (l1 + l2) * 0.993
    dx = pr["pelvis_shift"][0]
    dy = pr["pelvis_shift"][1]
    hip_xy = Vector((hipR.x + dx, hipR.y + dy))
    horiz = (Vector((ankR.x, ankR.y)) - hip_xy).length
    hip_z = ankR.z + math.sqrt(max(reach * reach - horiz * horiz, 0))
    root.location = (0, 0, 0)
    upd()
    shift = Vector((dx, dy, hip_z - hipR.z))
    # root bone location is in bone-local space: convert armature-space shift
    M = root.bone.matrix_local.to_3x3()
    root.location = M.inverted() @ shift
    upd()
    # ---- legs
    for side, tgt, pole in (("R", ankR, pr["knee_pole_R"]),
                            ("L", Vector((pr["ankle_L"][0], pr["ankle_L"][1], pr["ankle_z"] - z0)), pr["knee_pole_L"])):
        two_bone("upperleg01." + side, "lowerleg01." + side, "foot." + side, tgt, pole, (0, -1, 0))
    # ---- feet: yaw + plantarflexion (ankle->ball direction)
    for side, yaw in (("R", pr["footyaw_R"]), ("L", pr["footyaw_L"])):
        fb = P["foot." + side]
        ank = fb.head.copy()
        ball = P["toe3-1." + side].head.copy()
        rest_vec = (RIG.data.bones["toe3-1." + side].head_local - RIG.data.bones["foot." + side].head_local)
        rest_pitch = math.degrees(math.atan2(-rest_vec.z, Vector((rest_vec.x, rest_vec.y)).length))
        f = Matrix.Rotation(math.radians(yaw), 3, UP) @ Vector((0, -1, 0))
        pitch = math.radians(rest_pitch + pr["plantar"])
        p_t = f * math.cos(pitch) + Vector((0, 0, -1)) * math.sin(pitch)
        lat_t = UP.cross(f)
        lat_now = delta(fb) @ Vector((1, 0, 0))
        aim("foot." + side, ank, ball, ank + p_t, lat_now, lat_t)
        lat = UP.cross(f).normalized()
        for t in range(1, 6):
            nm = "toe%d-1.%s" % (t, side)
            if nm in P:
                rot_world(nm, lat, pr["toe_dorsi"])
    # ---- arms: put the elbow on target with the rest bend plane mapped onto
    # the (shoulder, elbow, wrist) plane, then hinge the forearm onto the wrist.
    for side in ("R", "L"):
        sh = P["upperarm01." + side].head.copy()
        el_t = sh + Vector(pr["elbow_" + side])
        wr = Vector(pr["wrist_" + side])
        wr.z -= z0
        el_now = P["lowerarm01." + side].head.copy()
        wr_now = P["wrist." + side].head.copy()
        aim("upperarm01." + side, sh, el_now, el_t, wr_now - el_now, wr - el_t)
        el_now = P["lowerarm01." + side].head.copy()
        wr_now = P["wrist." + side].head.copy()
        hinge = (el_now - sh).cross(wr_now - el_now)
        aim("lowerarm01." + side, el_now, wr_now, wr, hinge, hinge)
    RIG.rotation_euler = (0, 0, math.radians(pr["rig_yaw"]))
    upd()


if __name__ == "__main__" or True:
    pose()
