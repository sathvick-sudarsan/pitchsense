"""Synthetic checks only: no model inference, weights, videos, or source pickle loads.

Run from the project root: .venv/Scripts/python -m unittest discover -s tests -v
"""

import unittest

import numpy as np
import supervision as sv
import torch
from ultralytics.engine.results import Results

from player_ball_assigner import PlayerBallAssigner
from speed_and_distance_estimator import SpeedAndDistance_Estimator
from trackers import Tracker
from utils import (
    get_bbox_width,
    get_center_of_bbox,
    get_foot_position,
    measure_distance,
    measure_xy_distance,
)
from view_transformer import ViewTransformer


class SyntheticSmokeTests(unittest.TestCase):
    def test_detection_conversion_and_tracking_without_model(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detection = Results(frame, path="synthetic", names={0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"}, boxes=torch.tensor([
            [5, 5, 15, 15, .9, 0], [20, 20, 30, 40, .9, 1],
            [40, 20, 50, 40, .9, 2], [60, 20, 70, 40, .9, 3],
        ]))
        tracker = Tracker.__new__(Tracker)
        tracker.tracker = sv.ByteTrack()
        tracker.detect_frames = lambda frames: [detection for _ in frames]
        tracks = tracker.get_object_tracks([frame, frame])
        self.assertEqual([len(x) for x in tracks["players"]], [2, 2])
        self.assertEqual([len(x) for x in tracks["referees"]], [1, 1])
        self.assertEqual(tracks["ball"][0][1]["bbox"], [5, 5, 15, 15])
        self.assertEqual(set(tracks["players"][0]), set(tracks["players"][1]))

    def test_bbox_geometry(self):
        bbox = [10, 20, 31, 61]
        self.assertEqual(get_center_of_bbox(bbox), (20, 40))
        self.assertEqual(get_foot_position(bbox), (20, 61))
        self.assertEqual(get_bbox_width(bbox), 21)
        self.assertEqual(measure_distance((1, 2), (4, 6)), 5)
        self.assertEqual(measure_xy_distance((1, 2), (4, 6)), (-3, -4))

    def test_ball_assignment_uses_nearest_foot_and_strict_boundary(self):
        assigner = PlayerBallAssigner()
        players = {1: {"bbox": [0, 0, 20, 20]}, 2: {"bbox": [40, 0, 60, 20]}}
        self.assertEqual(assigner.assign_ball_to_player(players, [37, 18, 41, 22]), 2)
        player = {1: players[1]}
        self.assertEqual(assigner.assign_ball_to_player(player, [88, 19, 90, 21]), 1)
        self.assertEqual(assigner.assign_ball_to_player(player, [89, 19, 91, 21]), -1)
        self.assertEqual(assigner.assign_ball_to_player({}, [0, 0, 2, 2]), -1)

    def test_homography_corners_and_outside(self):
        transformer = ViewTransformer()
        for pixel, target in zip(transformer.pixel_vertices, transformer.target_vertices):
            with self.subTest(pixel=pixel.tolist()):
                np.testing.assert_allclose(transformer.transform_point(pixel)[0], target, atol=1e-4)
        self.assertIsNone(transformer.transform_point(np.array([0, 0])))

    def test_speed_and_accumulated_distance_at_24_fps(self):
        # Seven frames avoid the existing zero-length terminal window at 6 frames.
        tracks = {"players": [{7: {"position_transformed": [i * 0.6, i * 0.8]}} for i in range(7)]}
        SpeedAndDistance_Estimator().add_speed_and_distance_to_tracks(tracks)
        for i in range(6):
            self.assertAlmostEqual(tracks["players"][i][7]["speed"], 86.4)
            self.assertAlmostEqual(tracks["players"][i][7]["distance"], 5 if i < 5 else 6)
        self.assertNotIn("speed", tracks["players"][6][7])

    def test_ball_interpolation_without_model_initialization(self):
        tracker = Tracker.__new__(Tracker)
        positions = [{}, {1: {"bbox": [0, 2, 4, 6]}}, {}, {1: {"bbox": [4, 6, 8, 10]}}]
        result = tracker.interpolate_ball_positions(positions)
        self.assertEqual([frame[1]["bbox"] for frame in result], [
            [0, 2, 4, 6], [0, 2, 4, 6], [2, 4, 6, 8], [4, 6, 8, 10],
        ])
        self.assertEqual(positions[0], {})
        self.assertFalse(hasattr(tracker, "model"))


if __name__ == "__main__":
    unittest.main()
