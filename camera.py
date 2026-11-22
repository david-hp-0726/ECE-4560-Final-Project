"""RGB capture and ChArUco calibration. All board dimensions are in meters."""
from pathlib import Path
import sys
from uuid import uuid4

import cv2
import numpy as np


class Camera:
    def __init__(self, index=0, resolution=(1280, 720)):
        self.index = index
        self.resolution = tuple(resolution)
        self.stream = None
        self.camera_matrix = self.distortion = None
        self.board = cv2.aruco.CharucoBoard(
            (7, 5), 0.030, 0.022,
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
        )
        self.detector = cv2.aruco.CharucoDetector(self.board)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        if self.stream is not None:
            self.stream.release()
            self.stream = None

    def capture(self, path=None):
        """Read a BGR frame, optionally saving a copy."""
        if self.stream is None:
            backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
            self.stream = cv2.VideoCapture(self.index, backend)
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        ok, frame = self.stream.read()
        if not ok:
            self.close()
            raise RuntimeError("Could not read camera frame. Check camera index and access.")
        if frame.shape[1::-1] != self.resolution:
            raise ValueError(f"Requested {self.resolution}, received {frame.shape[1::-1]}")
        if path is not None and not cv2.imwrite(str(path), frame):
            raise OSError(f"Could not save {path}")
        return frame

    def generate_charuco_pdf(self, path="charuco.pdf"):
        """Landscape US Letter, 7 x 5 squares. Print at 100% and verify 30 mm squares."""
        from PIL import Image
        from reportlab.lib.pagesizes import landscape, letter
        from reportlab.lib.units import mm
        from reportlab.lib.utils import ImageReader
        from reportlab.pdfgen.canvas import Canvas

        width, height = landscape(letter)
        image = self.board.generateImage((2100, 1500))
        pdf = Canvas(str(path), pagesize=(width, height))
        pdf.drawImage(ImageReader(Image.fromarray(image)), (width-210*mm)/2,
                      (height-150*mm)/2, 210*mm, 150*mm)
        pdf.save()
        return Path(path)

    def capture_charuco_images(self, directory="calibration_images"):
        """Space saves the raw frame when corners are visible. Q or Esc exits."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        title = "ChArUco | Space: save | Q: finish"
        cv2.namedWindow(title)
        try:
            while cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) >= 1:
                frame = self.capture()
                corners, ids, _, _ = self.detector.detectBoard(frame)
                preview = frame.copy()
                usable = ids is not None and len(ids) >= 6 and not self.board.checkCharucoCornersCollinear(ids)
                if ids is not None:
                    cv2.aruco.drawDetectedCornersCharuco(preview, corners, ids)
                cv2.imshow(title, preview)
                key = cv2.waitKey(10) & 255
                if key in (27, ord("q"), ord("Q")):
                    break
                if key == 32 and usable:
                    path = directory / f"charuco_{uuid4().hex}.png"
                    if not cv2.imwrite(str(path), frame):
                        raise OSError(f"Could not save {path}")
                    print(f"Saved {path}")
        finally:
            if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) >= 1:
                cv2.destroyWindow(title)

    def calculate_intrinsics(self, directory="calibration_images", path="intrinsics.npz"):
        """Fit intrinsics from collected PNGs. Return RMS reprojection error in pixels."""
        objects, pixels = [], []
        for filename in sorted(Path(directory).glob("charuco_*.png")):
            image = cv2.imread(str(filename), cv2.IMREAD_GRAYSCALE)
            if image is None or image.shape[::-1] != self.resolution:
                raise ValueError(f"Unreadable or wrong resolution: {filename}")
            corners, ids, _, _ = self.detector.detectBoard(image)
            if ids is None or len(ids) < 6 or self.board.checkCharucoCornersCollinear(ids):
                continue
            obj, img = self.board.matchImagePoints(corners, ids)
            objects.append(obj)
            pixels.append(img)
        if len(objects) < 3:
            raise ValueError("Need at least three usable views with different tilts.")
        rms, K, distortion, _, _ = cv2.calibrateCamera(objects, pixels, self.resolution, None, None)
        with open(path, "wb") as output:
            np.savez(output, camera_matrix=K, distortion=distortion, image_size=self.resolution, rms=rms)
        self.camera_matrix, self.distortion = K, distortion
        return rms

    def load_intrinsics(self, path="intrinsics.npz"):
        with np.load(path) as data:
            if tuple(data["image_size"]) != self.resolution:
                raise ValueError("Calibration and capture resolutions differ.")
            self.camera_matrix = data["camera_matrix"]
            self.distortion = data["distortion"]

    def calculate_extrinsics(self, path="extrinsics.npz", *, board=None, preview=False):
        """Capture the workspace GridBoard and return the 4x4 camera-to-board transform.

        Pass a custom GridBoard for measured marker sizes. Use path=None to skip
        saving. Returns None if the board pose cannot be estimated.
        """
        from aruco import Aruco

        if self.camera_matrix is None:
            raise ValueError("Calculate or load camera intrinsics first.")
        return Aruco(board).estimate_board_from_camera(
            self.capture(), self.camera_matrix, self.distortion,
            preview=preview, save_path=path,
        )