import numpy as np


def validate_motion(data, joint_names, body_names=None, robot=None, require_metadata=False):
    """Reject incompatible motion orderings before using precomputed kinematics."""
    names = tuple(joint_names)
    keys = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
            "body_lin_vel_w", "body_ang_vel_w")
    for key in keys:
        if key not in data:
            raise ValueError(f"Motion missing {key}")
    pos = np.asarray(data["joint_pos"])
    if pos.ndim != 2 or pos.shape[0] < 3 or pos.shape[1] != len(names):
        raise ValueError(f"Motion joint_pos must be [frames>=3, {len(names)}], got {pos.shape}")
    frames = pos.shape[0]
    bodies = np.asarray(data["body_pos_w"])
    if bodies.ndim != 3 or bodies.shape[0] != frames or bodies.shape[2] != 3:
        raise ValueError("Motion body_pos_w must be [frames, bodies, 3]")
    count = bodies.shape[1]
    for key in keys:
        array = np.asarray(data[key])
        width = 4 if key == "body_quat_w" else 3
        expected = pos.shape if key.startswith("joint_") else (frames, count, width)
        if array.shape != expected or not np.isfinite(array).all():
            raise ValueError(f"Invalid motion {key}: expected finite {expected}, got {array.shape}")
    if "fps" in data:
        fps = np.asarray(data["fps"])
        if fps.size != 1 or not np.isfinite(fps).all() or fps.item() <= 0:
            raise ValueError("Motion fps must be a finite positive scalar")
    elif require_metadata:
        raise ValueError("OLI motion requires fps")
    for key in ("robot", "joint_names", "body_names"):
        if require_metadata and key not in data:
            raise ValueError(f"OLI motion requires {key}; use the OLI converter")
    if "robot" in data and robot is not None and str(np.asarray(data["robot"]).item()) != robot:
        raise ValueError(f"Motion robot does not match {robot}")
    if "joint_names" in data and tuple(np.asarray(data["joint_names"]).tolist()) != names:
        raise ValueError("Motion joint order does not match robot joint order")
    if "body_names" in data:
        motion_bodies = tuple(np.asarray(data["body_names"]).tolist())
        if len(motion_bodies) != count or len(set(motion_bodies)) != count:
            raise ValueError("Motion body_names must uniquely name each stored body")
        if body_names is not None and motion_bodies != tuple(body_names):
            raise ValueError("Motion body order does not match robot body order")
    if not np.allclose(np.linalg.norm(data["body_quat_w"], axis=-1), 1, atol=1e-3):
        raise ValueError("Motion body quaternions must be unit wxyz quaternions")
