"""Regression: a moving camera must not create rigid-object world motion."""
import sys
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from camera_motion_geometry import estimate_rigid_pose, project_points, propagate_pose


class CameraMotionGeometryTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(11)
        self.points = rng.uniform(-.25, .25, (80, 3))
        self.K = np.array([[334.48, 0., 383.5], [0., 334.48, 287.5], [0., 0., 1.]])
        self.object_world = np.eye(4)
        self.object_world[:3, :3] = Rotation.from_euler('xyz', [8, -12, 20], degrees=True).as_matrix()
        self.object_world[:3, 3] = [.2, -.1, 2.]

    def test_static_object_with_moving_camera_and_outliers(self):
        rng = np.random.default_rng(7)
        previous_camera = np.eye(4)
        previous_pose = self.object_world.copy()
        for i in range(12):
            camera = np.eye(4)
            camera[:3, :3] = Rotation.from_euler('xyz', [i * .7, i * 1.8, i * -.4], degrees=True).as_matrix()
            camera[:3, 3] = [i * .025, -.01 * i, .005 * i]
            expected = np.linalg.inv(camera) @ self.object_world
            pixels, _ = project_points(self.points, expected, self.K)
            pixels += rng.normal(0, .15, pixels.shape)
            pixels[:8] += [70, -40]
            prior = propagate_pose(previous_camera, camera, previous_pose)
            pose, info = estimate_rigid_pose(self.points, pixels, self.K, prior)
            self.assertEqual(info['status'], 'ok')
            reconstructed_world = camera @ pose
            self.assertLess(np.linalg.norm(reconstructed_world[:3, 3] - self.object_world[:3, 3]), .003)
            error = Rotation.from_matrix(self.object_world[:3, :3].T @ reconstructed_world[:3, :3]).magnitude()
            self.assertLess(np.degrees(error), .5)
            previous_camera, previous_pose = camera, pose

    def test_real_object_motion_is_preserved(self):
        world = self.object_world.copy()
        world[:3, :3] = Rotation.from_euler('z', 35, degrees=True).as_matrix() @ world[:3, :3]
        world[:3, 3] += [.15, .02, -.1]
        camera = np.eye(4)
        camera[:3, :3] = Rotation.from_euler('y', 18, degrees=True).as_matrix()
        camera[:3, 3] = [.1, 0, .05]
        pixels, _ = project_points(self.points, np.linalg.inv(camera) @ world, self.K)
        pose, info = estimate_rigid_pose(self.points, pixels, self.K)
        self.assertEqual(info['status'], 'ok')
        np.testing.assert_allclose(camera @ pose, world, atol=1e-5)

    def test_missing_correspondences_never_copy_the_prior(self):
        pose, info = estimate_rigid_pose(self.points[:4], np.zeros((4, 2)), self.K, self.object_world)
        self.assertIsNone(pose)
        self.assertEqual(info['status'], 'insufficient_correspondences')


if __name__ == '__main__':
    unittest.main()
