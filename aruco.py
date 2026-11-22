"""Make a printable GridBoard and map RGB pixels to planes in its frame.

Install with: python -m pip install -r requirements.txt
Generate a US Letter PDF with: python aruco.py [output.pdf]
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

MARKER_LENGTH_M = 0.050
MARKER_GAP_M = 0.010


class Aruco:
    def __init__(self, board=None):
        self.board = self.make_board() if board is None else board

    @staticmethod
    def make_board(marker_length_m=MARKER_LENGTH_M, marker_gap_m=MARKER_GAP_M):
        """Create a 2x2 DICT_4X4_50 GridBoard with IDs 0 through 3."""
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        return cv2.aruco.GridBoard((2, 2), marker_length_m, marker_gap_m, dictionary)


    @staticmethod
    def load_board(path):
        """Load a GridBoard layout from JSON. Lengths are in meters."""
        config = json.loads(Path(path).read_text(encoding="utf-8"))
        dictionary_id = getattr(cv2.aruco, config["dictionary"], None)
        if dictionary_id is None or not config["dictionary"].startswith("DICT_"):
            raise ValueError(f"Unknown ArUco dictionary: {config['dictionary']}")
        dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        ids = np.asarray(config["ids"], dtype=np.int32)
        return cv2.aruco.GridBoard(
            tuple(config["grid_size"]), config["marker_length_m"],
            config["marker_gap_m"], dictionary, ids,
        )


    def generate_board_pdf(self, path="so101_aruco_board.pdf", *, board=None):
        """Write a US Letter PDF with vector markers at their exact dimensions."""
        import reportlab
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.units import mm
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.pdfgen.canvas import Canvas

        if board is None:
            board = self.board
        path = Path(path)
        width, height = letter
        board_width, board_height, _ = np.asarray(board.getRightBottomCorner()) * 1000 * mm
        if board_width > width - 20 * mm or board_height > height - 55 * mm:
            raise ValueError("Board is too large for the printable US Letter area.")

        left = (width - board_width) / 2
        top = (height + board_height) / 2
        pdf = Canvas(str(path), pagesize=letter)
        pdf.setTitle("ArUco GridBoard")
        dictionary = board.getDictionary()
        cells = dictionary.markerSize + 2  # One black border cell on each side.
        marker_mm = board.getMarkerLength() * 1000
        cell_pt = marker_mm * mm / cells

        for marker_id, corners in zip(board.getIds(), board.getObjPoints()):
            x, y, _ = corners[0] * 1000 * mm
            marker = cv2.aruco.generateImageMarker(dictionary, int(marker_id), cells)
            for row, col in np.argwhere(marker == 0):
                pdf.rect(left + x + col * cell_pt, top - y - (row + 1) * cell_pt,
                         cell_pt, cell_pt, stroke=0, fill=1)

        fonts = Path(reportlab.__file__).parent / "fonts"
        pdfmetrics.registerFont(TTFont("Vera", str(fonts / "Vera.ttf")))
        pdfmetrics.registerFont(TTFont("Vera-Bold", str(fonts / "VeraBd.ttf")))
        pdf.setFont("Vera-Bold", 16)
        pdf.drawCentredString(width / 2, top + 19 * mm, "ArUco GridBoard")
        pdf.setFont("Vera", 11)
        pdf.drawCentredString(
            width / 2, top + 11 * mm,
            f"{marker_mm:g} mm markers  |  {board.getMarkerSeparation() * 1000:g} mm gaps",
        )
        pdf.drawCentredString(
            width / 2, top - board_height - 12 * mm,
            "Print at Actual Size (100%). Measure a marker across its outer black edges.",
        )
        pdf.save()
        return path


    def estimate_board_from_camera(
        self, frame, camera_matrix, distortion, *, board=None, preview=False, save_path=None
    ):
        """Return the 4x4 transform mapping camera points into board coordinates.

        Accepts an OpenCV BGR or grayscale image. Returns None if fewer than two
        board markers are matched or pose estimation fails. If preview=True, press
        any key to close the window. If save_path is given, a successful fit is
        saved as a NumPy archive along with the intrinsics and source image size.
        """
        if board is None:
            board = self.board
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        detector = cv2.aruco.ArucoDetector(board.getDictionary(), params)
        corners, ids, _ = detector.detectMarkers(frame)
        if ids is None or len(ids) < 2:
            return None

        is_board_marker = np.isin(ids.ravel(), board.getIds())
        if np.count_nonzero(is_board_marker) < 2:
            return None
        corners = [marker for marker, keep in zip(corners, is_board_marker) if keep]
        ids = ids[is_board_marker]
        points_board, points_image = board.matchImagePoints(corners, ids)

        ok, rvec, tvec = cv2.solvePnP(
            points_board, points_image, camera_matrix, distortion
        )
        if not ok:
            return None

        rotation, _ = cv2.Rodrigues(rvec)
        board_from_camera = np.eye(4)
        board_from_camera[:3, :3] = rotation.T
        board_from_camera[:3, 3] = -rotation.T @ tvec.ravel()

        if save_path is not None:
            height, width = frame.shape[:2]
            with Path(save_path).open("wb") as output:
                np.savez(
                    output, board_from_camera=board_from_camera,
                    camera_matrix=camera_matrix,
                    distortion=distortion if distortion is not None else np.empty(0),
                    image_size=np.array([width, height]),
                )

        if preview:
            image = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame.copy()
            cv2.aruco.drawDetectedMarkers(image, corners, ids)
            cv2.drawFrameAxes(
                image, camera_matrix, distortion, rvec, tvec, board.getMarkerLength()
            )
            window = "ArUco board pose"
            cv2.imshow(window, image)
            cv2.waitKey(0)
            cv2.destroyWindow(window)

        return board_from_camera


    @staticmethod
    def pixel_to_board(pixel, camera_matrix, distortion, board_from_camera, z_board_m=0.0):
        """Intersect a raw RGB pixel ray with a constant-z plane in the board frame.

        With the printed board face up, +x points right, +y points down the page,
        and +z points into the sheet. A block top above the sheet has negative z.
        """
        pixel = np.asarray(pixel, dtype=np.float64).reshape(1, 1, 2)
        x, y = cv2.undistortPoints(pixel, camera_matrix, distortion).ravel()
        origin = board_from_camera[:3, 3]
        direction = board_from_camera[:3, :3] @ np.array([x, y, 1.0])
        if abs(direction[2]) < 1e-9:
            raise ValueError("Pixel ray is parallel to the plane.")
        distance = (z_board_m - origin[2]) / direction[2]
        if distance <= 0:
            raise ValueError("Plane intersection must be in front of the camera.")
        return origin + distance * direction


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a printable ArUco GridBoard.")
    parser.add_argument("output", nargs="?", default="so101_aruco_board.pdf")
    parser.add_argument("--board-config", type=Path, help="JSON board layout")
    args = parser.parse_args()
    board = Aruco.load_board(args.board_config) if args.board_config else None
    print(Aruco(board).generate_board_pdf(args.output))
