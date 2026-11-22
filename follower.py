"""SO-101 follower I/O. Arm joints use degrees and the gripper uses 0–100."""
import math
import time
from pathlib import Path

import numpy as np

from utils import forward_kinematics

import draccus
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode


def load_calibration(ROBOT_NAME, directory="calibration_files"):
    """Read the named follower calibration from its JSON file."""
    path = Path(directory) / f"{ROBOT_NAME}.json"
    with path.open() as stream, draccus.config_type("json"):
        return draccus.load(dict[str, MotorCalibration], stream)


def setup_motors(calibration, PORT_ID):
    norm_mode_body = MotorNormMode.DEGREES
    bus = FeetechMotorsBus(
                port=PORT_ID,
                motors={
                    "shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                    "shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                    "elbow_flex": Motor(3, "sts3215", norm_mode_body),
                    "wrist_flex": Motor(4, "sts3215", norm_mode_body),
                    "wrist_roll": Motor(5, "sts3215", norm_mode_body),
                    "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
                },
                calibration=calibration,
            )
    bus.connect(True)

    with bus.torque_disabled():
        bus.configure_motors()
        for motor in bus.motors:
            bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
            # Set P_Coefficient to lower value to avoid shakiness (Default is 32)
            bus.write("P_Coefficient", motor, 16)
            # Set I_Coefficient and D_Coefficient to default value 0 and 32
            bus.write("I_Coefficient", motor, 0)
            bus.write("D_Coefficient", motor, 32) 
    return bus


def read_joint_positions(bus):
    """Read normalized positions keyed by motor name."""
    return bus.sync_read("Present_Position")


def command_joint_positions(bus, joint_positions):
    """Send a full or partial joint-position command."""
    if not all(math.isfinite(value) for value in joint_positions.values()):
        raise ValueError("Joint targets must be finite")
    bus.sync_write("Goal_Position", joint_positions, normalize=True)


def move_to_joint_positions(bus, target_joint_positions, duration):
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be positive and finite")
    start_joint_positions = read_joint_positions(bus)
    start_time = time.perf_counter()

    while True:
        elapsed = time.perf_counter() - start_time
        u = min(elapsed / duration, 1.0)
        alpha = 3*u**2 - 2*u**3

        command = {
            joint: (1-alpha)*start_joint_positions[joint] + alpha*target
            for joint, target in target_joint_positions.items()
        }

        command_joint_positions(bus, command)

        if u >= 1.0:
            break

        time.sleep(0.02)


def hold_joint_positions(bus, duration):
    """Wait while the connected servos hold their last commanded positions."""
    time.sleep(max(0, duration))


# Compatibility with earlier lab scripts.
read_position = read_joint_positions
hold_position = hold_joint_positions


def command_position(bus, position):
    return command_joint_positions(bus, position)


def move_to_pose(bus, desired_position, duration, *, target_rotation=None):
    """Follow a world-frame XYZ line in meters, with optional world-frame tool rotation.

    XYZ alone holds the measured orientation. Plans the entire path before moving.
    Requires SciPy. Joint-dictionary callers should use move_to_joint_positions().
    """
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be positive and finite")
    target = np.asarray(desired_position, dtype=float)
    if target.shape != (3,) or not np.isfinite(target).all():
        raise ValueError("desired_position must be a finite XYZ vector")
    measured = read_joint_positions(bus)
    names = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
    angles = np.array([measured[name] for name in names])
    start, initial_matrix = forward_kinematics(measured)
    initial = Rotation.from_matrix(initial_matrix)
    rotation_delta = np.zeros(3)
    if target_rotation is not None:
        matrix = np.asarray(target_rotation, dtype=float)
        if (matrix.shape != (3, 3) or not np.allclose(matrix.T @ matrix, np.eye(3))
                or not np.isclose(np.linalg.det(matrix), 1)):
            raise ValueError("target_rotation must be a proper 3x3 rotation matrix")
        rotation_delta = (initial.inv() * Rotation.from_matrix(matrix)).as_rotvec()

    count = max(1, math.ceil(duration * 50))
    commands = []
    for i in range(1, count + 1):
        u = i / count
        s = u**2 * (3 - 2*u)
        position = start + s * (target - start)
        rotation = initial * Rotation.from_rotvec(s * rotation_delta)

        def residual(candidate):
            xyz, matrix = forward_kinematics(dict(zip(names, candidate)))
            angle_error = (rotation.inv() * Rotation.from_matrix(matrix)).as_rotvec()
            return np.r_[xyz - position, 0.1 * angle_error]

        solution = least_squares(residual, angles, gtol=1e-11)
        if (not solution.success or np.linalg.norm(solution.fun[:3]) > 0.001
                or np.linalg.norm(solution.fun[3:]) > 0.1 * np.deg2rad(1)):
            raise ValueError(f"Waypoint {i}/{count} exceeds the 1 mm or 1 degree tolerance")
        angles = solution.x
        commands.append(dict(zip(names, angles)))

    start_time = time.perf_counter()
    for i, command in enumerate(commands, 1):
        deadline = start_time + duration * i / count
        time.sleep(max(0, deadline - time.perf_counter()))
        command_joint_positions(bus, command)
