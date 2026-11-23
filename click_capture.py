"""Click pick then place. Enter runs the pair, R clears it, S parks, Q exits."""
import argparse
import json
from contextlib import ExitStack
from pathlib import Path

import cv2
import numpy as np

import time
import xml.etree.ElementTree as ET


SLEEP_POSE = {
    "shoulder_pan": -8.131868131868131,
    "shoulder_lift": -103.38461538461539,
    "elbow_flex": 96.3956043956044,
    "wrist_flex": 68.87912087912088,
    "wrist_roll": 0.04395604395604396,
    "gripper": 0.7498295841854125,
}


def park_physical(bus, duration):
    from follower import move_to_joint_positions, read_joint_positions
    move_to_joint_positions(bus, SLEEP_POSE, duration)
    # Allow the servos to settle before shutdown can release torque.
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        time.sleep(0.15)
        measured = read_joint_positions(bus)
        if all(abs(measured[j] - SLEEP_POSE[j]) <= 3 for j in SLEEP_POSE):
            return
    raise RuntimeError(f"Sleep pose unconfirmed. Support the arm before power-off. Readback: {measured}")


def park_and_disconnect(bus, duration):
    park_physical(bus, duration)
    bus.disconnect()


def rigid_transform(value):
    T = np.asarray(value, dtype=float)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError("Fill in base_from_board with the measured 4x4 transform.")
    R = T[:3, :3]
    if not (np.allclose(T[3], [0, 0, 0, 1]) and
            np.allclose(R.T @ R, np.eye(3), atol=1e-4) and
            np.isclose(np.linalg.det(R), 1, atol=1e-4)):
        raise ValueError("Expected a rigid transform with a proper rotation.")
    return T


def click_target(pixel, K, distortion, board_from_camera, base_from_board, height):
    from aruco import Aruco
    board = Aruco.pixel_to_board(pixel, K, distortion, board_from_camera, z_board_m=0)
    base = base_from_board[:3, :3] @ board + base_from_board[:3, 3]
    return np.array([base[0], base[1], height])


def pick_place(move, grip, pick, place, cfg):
    lift = np.array([0, 0, cfg["lift_m"]])
    move(pick + lift)
    grip(cfg["gripper_open_pct"])
    move(pick)
    grip(cfg["gripper_closed_pct"])
    move(pick + lift)
    move(place + lift)
    move(place)
    grip(cfg["gripper_open_pct"])
    move(place + lift)


class EndpointPlanner:
    joints = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")

    def __init__(self, scene_path):
        import mujoco
        self.mj = mujoco
        scene = ET.parse(scene_path).getroot()
        robot_path = scene_path.with_name("so101_new_calib_camera.xml")
        robot = ET.parse(robot_path).getroot()
        compiler = robot.find("compiler")
        compiler.set("meshdir", str((robot_path.parent / compiler.get("meshdir", ".")).resolve()))
        include = scene.find("include")
        if include is None or include.get("file") not in ("so101_new_calib.xml", robot_path.name):
            raise ValueError("Expected the SO-101 robot include in scene.xml")
        scene.remove(include)
        scene[:0] = list(robot)
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(scene, encoding="unicode"))
        self.ik_data = mujoco.MjData(self.model)
        self.arm = [self.model.actuator(n).id for n in self.joints]
        joint_ids = self.model.actuator_trnid[self.arm, 0]
        self.qadr = self.model.jnt_qposadr[joint_ids]

    def solve(self, point, start_rad):
        from scipy.optimize import least_squares
        from scipy.spatial.transform import Rotation
        # The project's TCP is 103.4 mm along gripper-local -Z.
        def residual(q):
            self.ik_data.qpos[self.qadr] = q
            self.mj.mj_forward(self.model, self.ik_data)
            body = self.ik_data.body("gripper")
            rotation = body.xmat.reshape(3, 3)
            xyz = body.xpos - 0.1034 * rotation[:, 2]
            return np.r_[xyz - point, 0.1 * Rotation.from_matrix(rotation.copy()).as_rotvec()]

        low, high = self.model.actuator_ctrlrange[self.arm].T
        q = np.clip(start_rad, low, high)
        fit = least_squares(residual, q, bounds=(low, high), gtol=1e-11)
        if np.linalg.norm(fit.fun[:3]) > .001 or np.linalg.norm(fit.fun[3:]) > .1*np.deg2rad(1):
            raise ValueError(f"Cannot reach {point.round(3)} m with identity tool orientation")
        return fit.x

class Simulation(EndpointPlanner):
    def __init__(self, scene_path):
        import mujoco.viewer
        super().__init__(scene_path)
        self.data = mujoco.MjData(self.model)
        self.jaw = self.model.actuator("gripper").id
        mujoco.mj_forward(self.model, self.data)
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        with self.viewer.lock():
            self.viewer.cam.lookat[:] = [0.18, 0, 0.12]
            self.viewer.cam.distance = 0.85
            self.viewer.cam.elevation = -25
        self.viewer.sync()

    def animate(self, ids, goal, duration, window):
        start = self.data.ctrl[ids].copy()
        begin = time.monotonic()
        steps = max(1, round(duration / self.model.opt.timestep))
        stride = max(1, round(1 / (30 * self.model.opt.timestep)))
        for i in range(1, steps + 1):
            u = i / steps
            self.data.ctrl[ids] = start + u*u*(3-2*u)*(goal-start)
            self.mj.mj_step(self.model, self.data)
            if i % stride == 0 or i == steps:
                if not self.viewer.is_running():
                    raise KeyboardInterrupt
                self.viewer.sync()
                if cv2.waitKey(1) & 255 in (27, ord("q")) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    raise KeyboardInterrupt
                time.sleep(max(0, begin + duration*u - time.monotonic()))

    def move(self, point, duration, window):
        goal = self.solve(point, self.data.qpos[self.qadr])
        self.animate(self.arm, goal, duration, window)

    def park(self, duration, window):
        lo, hi = self.model.actuator_ctrlrange[self.jaw]
        jaw = lo + SLEEP_POSE["gripper"] / 100 * (hi - lo)
        ids = self.arm + [self.jaw]
        goal = np.r_[np.deg2rad([SLEEP_POSE[j] for j in self.joints]), jaw]
        self.animate(ids, np.clip(goal, *self.model.actuator_ctrlrange[ids].T), duration, window)

    def grip(self, percent, window):
        lo, hi = self.model.actuator_ctrlrange[self.jaw]
        self.animate(self.jaw, lo + percent/100*(hi-lo), 1, window)


def move_physical(bus, planner, point, duration):
    from follower import read_joint_positions, move_to_joint_positions
    measured = read_joint_positions(bus)
    start = np.deg2rad([measured[name] for name in planner.joints])
    goal = planner.solve(point, start)
    move_to_joint_positions(bus, dict(zip(planner.joints, np.rad2deg(goal))), duration)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--mode", choices=("simulation", "physical"), default="simulation")
    parser.add_argument("--scene", type=Path, help="defaults to model/scene.xml beside the config")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    folder = args.config.parent
    height = float(cfg["capture_z_base_m"])
    if not np.isfinite([height, cfg["lift_m"], cfg["move_seconds"]]).all() or min(
        cfg["lift_m"], cfg["move_seconds"]
    ) <= 0:
        raise ValueError("Use finite heights and positive lift and duration.")
    if not 0 <= cfg["gripper_closed_pct"] < cfg["gripper_open_pct"] <= 100:
        raise ValueError("Expected 0 <= closed < open <= 100.")
    points = []
    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            try:
                target = click_target((x, y), K, dist, board_from_camera, T, height)
            except ValueError as error:
                print(error)
                return
            points.append(((x, y), target))
            print(("Pick" if len(points) == 1 else "Place"), "base XYZ (m):", target.round(4))

    window = "Pick then place | Enter: run | R: reset | S: sleep | Q: quit"
    with ExitStack() as stack:
        from camera import Camera
        T = rigid_transform(cfg["base_from_board"])
        with np.load(folder / cfg["extrinsics"]) as data:
            K, dist = data["camera_matrix"], data["distortion"]
            size = tuple(map(int, data["image_size"]))
            board_from_camera = rigid_transform(data["board_from_camera"])
        cam = stack.enter_context(Camera(cfg["camera_index"], size))
        scene = args.scene or folder / "model/scene.xml"
        sim = None
        if args.mode == "simulation":
            sim = Simulation(scene)
            stack.callback(sim.viewer.close)
        else:
            planner = EndpointPlanner(scene)
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        stack.callback(cv2.destroyAllWindows)
        cv2.setMouseCallback(window, click)
        park_duration = max(3.0, cfg["move_seconds"])
        if sim is None:
            from follower import load_calibration, setup_motors, move_to_joint_positions
            calibration = load_calibration(cfg["robot_id"], folder / cfg["calibration_directory"])
            bus = setup_motors(calibration, cfg["port"])
            stack.callback(park_and_disconnect, bus, park_duration)
        while sim is None or sim.viewer.is_running():
            if sim is not None:
                sim.viewer.sync()
            frame = cam.capture()
            for label, (pixel, _) in zip(("Pick", "Place"), points):
                cv2.circle(frame, pixel, 5, (0, 255, 255), 2)
                cv2.putText(frame, label, pixel, cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
            cv2.imshow(window, frame)
            key = cv2.waitKey(10) & 255
            if key in (27, ord("q")) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord("s"), ord("S")):
                points.clear()
                if sim is None:
                    park_physical(bus, park_duration)
                else:
                    sim.park(park_duration, window)
            if key == ord("r"):
                points.clear()
            if key in (10, 13) and len(points) == 2:
                if sim is not None:
                    try:
                        pick_place(lambda p: sim.move(p, cfg["move_seconds"], window),
                                   lambda pct: sim.grip(pct, window), points[0][1], points[1][1], cfg)
                    except ValueError as error:
                        print(error)
                    except KeyboardInterrupt:
                        break
                else:
                    pick_place(lambda p: move_physical(bus, planner, p, cfg["move_seconds"]),
                               lambda pct: move_to_joint_positions(bus, {"gripper": pct}, duration=1.0),
                               points[0][1], points[1][1], cfg)
                points.clear()


if __name__ == "__main__":
    main()
