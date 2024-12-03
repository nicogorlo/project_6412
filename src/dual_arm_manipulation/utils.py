from pydrake.all import (
    RigidTransform,
    RotationMatrix,
    Quaternion
)
from pydrake.systems.framework import Diagram
import numpy as np
import pydot
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from scipy.spatial.transform import Rotation as R
from numpy.typing import ArrayLike

from time import perf_counter_ns
import logging


def interpolate_6dof_poses(start_pose: np.ndarray, goal_pose: np.ndarray, n_steps: int):
    """
    Linearly interpolate between two 6DOF poses.

    Parameters:
        start_pose (np.ndarray): The start pose as (x_quat, y_quat, z_quat, w_quat, x, y, z).
        goal_pose (np.ndarray): The goal pose as (x_quat, y_quat, z_quat, w_quat, x, y, z).
        n_steps (int): Number of interpolation steps (including start and goal).

    Returns:
        list of tuples: Interpolated poses, each as (x_quat, y_quat, z_quat, w_quat, x, y, z).
    """

    start_quat = convert_quat_wxyz_to_xyzw(start_pose[:4])
    goal_quat = convert_quat_wxyz_to_xyzw(goal_pose[:4])
    start_pos = start_pose[4:]
    goal_pos = goal_pose[4:]

    start_quat /= np.linalg.norm(start_quat)
    goal_quat /= np.linalg.norm(goal_quat)

    start_rot = R.from_quat(start_quat)
    goal_rot = R.from_quat(goal_quat)

    fractions = np.linspace(0, 1, n_steps)

    interpolated_poses = []
    for fraction in fractions:
        interp_rot = start_rot.slerp(goal_rot, fraction)
        interp_quat = convert_quat_xyzw_to_wxyz(interp_rot.as_quat())

        interp_pos = (1 - fraction) * start_pos + fraction * goal_pos

        interpolated_poses.append((*interp_quat, *interp_pos))

    return interpolated_poses


def get_free_faces(contact_mode_name: str, contact_modes: dict):
    for axis in ['X', 'Y', 'Z']:
        if contact_mode_name.startswith(axis):
            return {name: free_contact_mode[0] for name, free_contact_mode in contact_modes.items() if not name.startswith(axis)}


def save_diagram(diagram: Diagram, file: str = 'output/system_diagram.svg'):
    pngfile = pydot.graph_from_dot_data(
        diagram.GetGraphvizString(max_depth=2)
        )[0].create_svg()
    
    with open(file,'wb') as png_file:
        png_file.write(pngfile)


def display_diagram(diagram: Diagram, max_depth: int=2):
    dot_data = diagram.GetGraphvizString(max_depth=max_depth)
    
    png_data = pydot.graph_from_dot_data(dot_data)[0].create_png()
    
    image_stream = BytesIO(png_data)
    img = Image.open(image_stream)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.show() 


def convert_quat_wxyz_to_xyzw(q, batch_mode=False):
    if batch_mode:
        return q[:, [1, 2, 3, 0]]
    else:
        return q[[1, 2, 3, 0]]


def convert_quat_xyzw_to_wxyz(q, batch_mode=False):
    if batch_mode:
        return q[:, [3, 0, 1, 2]]
    else:
        return q[[3, 0, 1, 2]]


def pose_vec_to_transform(pose_vec: ArrayLike) -> RigidTransform:
    """
    Compute the rigid transform from 7d pose vec (qw, qx, qy, qz, x, y, z).
    """
    return RigidTransform(RotationMatrix(Quaternion(pose_vec[:4]/np.linalg.norm(pose_vec[:4]))), p=pose_vec[4:])


def rotation_matrix_from_vectors(v1: np.ndarray, v2: np.ndarray) -> RotationMatrix:
    """
    Compute the rotation matrix that rotates vector v1 to align with vector v2.
    
    Parameters:
        v1: np.array
            Source vector
        v2: np.array
            Target vector
    
    Returns:
        np.array
            3x3 rotation matrix
    """
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    
    cross = np.cross(v1, v2)
    cross_norm = np.linalg.norm(cross)
    
    dot = np.dot(v1, v2)
    
    # edge case: vectors exactly opposite
    if np.isclose(dot, -1.0):
        # arbitrary orthogonal vector
        orthogonal = np.array([1, 0, 0]) if not np.allclose(v1, [1, 0, 0]) else np.array([0, 1, 0])
        axis = np.cross(v1, orthogonal)
        axis = axis / np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        return RotationMatrix(np.eye(3) + 2 * K @ K)
    
    # edge case: vectors  exactly aligned
    if np.isclose(dot, 1.0):
        return RotationMatrix(np.eye(3))
    
    K = np.array([[0, -cross[2], cross[1]],
                  [cross[2], 0, -cross[0]],
                  [-cross[1], cross[0], 0]])
    
    rot_mat = np.eye(3) + K + K @ K * ((1 - dot) / (cross_norm ** 2))

    rot_mat = RotationMatrix(rot_mat)
    
    return rot_mat


class performance_measure:
    """
    A class that measures the execution time of a code block.
    Usage:
    with performance_measure("name of code block"):
        # code block

    Avoid usage with parallel or async code (not tested)
    """

    def __init__(self, name) -> None:
        self.name = name

    def __enter__(self):
        self.start_time = perf_counter_ns()

    def __exit__(self, *args):
        self.end_time = perf_counter_ns()
        self.duration = self.end_time - self.start_time

        print(f"{self.name} - execution time: {(self.duration)/1000000:.2f} ms")
        logging.info(f"{self.name} - execution time: {(self.duration)/1000000:.2f} ms")
