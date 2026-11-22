"""SO-101 follower I/O. Arm joints use degrees and the gripper uses 0–100."""
import math
import time
from pathlib import Path

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


def read_position(bus):
    """Read normalized positions keyed by motor name."""
    return bus.sync_read("Present_Position")


def command_position(bus, position):
    """Send a full or partial joint-position command."""
    if not all(math.isfinite(value) for value in position.values()):
        raise ValueError("Joint targets must be finite")
    bus.sync_write("Goal_Position", position, normalize=True)


def move_to_pose(bus, desired_position, duration):
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be positive and finite")
    starting_pose = read_position(bus)
    start_time = time.perf_counter()

    while True:
        elapsed = time.perf_counter() - start_time
        u = min(elapsed / duration, 1.0)
        alpha = 3*u**2 - 2*u**3

        command = {
            joint: (1-alpha)*starting_pose[joint] + alpha*target
            for joint, target in desired_position.items()
        }

        command_position(bus, command)

        if u >= 1.0:
            break

        time.sleep(0.02)


def hold_position(bus, duration):
    """Wait while the connected servos hold their last commanded positions."""
    time.sleep(max(0, duration))
