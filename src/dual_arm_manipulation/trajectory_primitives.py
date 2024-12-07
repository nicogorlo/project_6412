from pydrake.all import RigidTransform, RotationMatrix, RollPitchYaw

import numpy as np

from mergedeep import merge
from dual_arm_manipulation.contact_mode import ContactMode
from dual_arm_manipulation.utils import pose_vec_to_transform, quaternion_to_cos_sin_yaw

from typing import Optional
import numpy as np
from scipy.interpolate import interp1d
import random

kDefaultConfig = {
    "primitives": [
        "YAW_90_cw",
        "YAW_90_ccw",
        "ROLL_90_cw",
        "ROLL_90_ccw",
        "PITCH_90_cw",
        "PITCH_90_ccw",
    ],
    "goal_pose": [0.0, 0.0, 0.4],
    "lift_height": 0.4,
    "num_steps": 20,
    "num_steps_lift": 15,
    "num_steps_rotate": 15,
    "num_steps_lower": 15,
}


class TrajectoryPrimitives:
    """
    Class to load and store motion primitives.

    TODO:
    * Integrate visualizer to visualize primitives and chain of primitives.
    * Add random perturbations to the primitives to build convex sets.
    """

    def __init__(
        self,
        start_pose: RigidTransform,
        contact_mode: ContactMode,
        config: dict,
        config_override: dict = {},
    ):
        self.config = merge(config, config_override)
        self.start_pose = start_pose
        self.contact_mode = contact_mode
        self.primitives = []

    def load_primitives(self):
        for primitive_name in self.config["primitives"]:
            for n in range(self.config.get("n_samples_per_primitive", 1)):
                primitive = TrajectoryPrimitive(
                    f"{primitive_name}_{n}", self.contact_mode, self.start_pose if n == 0 else sample_tabletop_pose(np.array(self.config.get("tabletop_bounding_box")), self.contact_mode, self.config), None, self.config
                )
                if not self.config.get("augment", False):
                    self.primitives.append(primitive)
                else:
                    trajs_augmented = self.generate_perturbed_trajectories(
                        primitive.trajectory,
                        f"{primitive_name}_{n}",
                        self.config["n_rand_augmentations"],
                    )
                    print(
                        f"Augmented {len(trajs_augmented)} trajectories for primitive: {primitive_name}_{n}"
                    )
                    self.primitives.extend(trajs_augmented)

    def generate_perturbed_trajectories(
        self,
        original_trajectory: list[RigidTransform],
        primitive_name: str,
        N: int = 10,
    ):
        """
        Generates N perturbed trajectories from an original trajectory by applying
        random smooth perturbations in 6DoF space.

        Args:
            original_trajectory (TrajectoryPrimitive): The original trajectory.
            N (int): Number of perturbed trajectories to generate.
            max_translation (float): Maximum translation perturbation magnitude.
            max_rotation (float): Maximum rotation perturbation magnitude in radians.

        Returns:
            list[TrajectoryPrimitive]: A list containing N perturbed trajectories.
        """
        N_samples = []
        num_frames = len(original_trajectory)
        time_steps = np.linspace(0, 1, num_frames)

        max_translation = self.config.get("augment_max_translation", 0.1)
        max_rotation = self.config.get("augment_max_rotation", 0.1)

        for sample_idx in range(N):

            np.random.seed()

            # key pounts for interpolation
            num_key_points = max(4, num_frames // 10)
            key_times = np.linspace(0, 1, num_key_points)

            # pertubations at key points
            random_translations = np.random.uniform(
                -max_translation, max_translation, size=(num_key_points, 3)
            )
            random_rotations = np.random.uniform(
                -max_rotation, max_rotation, size=(num_key_points, 3)
            )

            # Interpolate perturbations
            delta_translation_func = interp1d(
                key_times,
                random_translations,
                axis=0,
                kind="cubic",
                fill_value="extrapolate",
            )
            delta_rotation_func = interp1d(
                key_times,
                random_rotations,
                axis=0,
                kind="cubic",
                fill_value="extrapolate",
            )

            delta_translations = delta_translation_func(time_steps)
            delta_rotations = delta_rotation_func(time_steps)

            perturbed_trajectory = []
            for i in range(num_frames):
                orig_transform = original_trajectory[i]
                delta_translation = delta_translations[i]
                delta_rotation = delta_rotations[i]

                perturb_transform = RigidTransform(
                    RollPitchYaw(delta_rotation), delta_translation
                )

                new_transform = orig_transform.multiply(perturb_transform)
                perturbed_trajectory.append(new_transform)

            perturbed_trajectory_sample = TrajectoryPrimitive(
                f"PERTURBED_{primitive_name}_{sample_idx}",
                self.contact_mode,
                self.start_pose,
                trajectory=perturbed_trajectory,
                config=self.config,
            )

            N_samples.append(perturbed_trajectory_sample)

        return N_samples

    def __iter__(self):
        return iter(self.primitives)
    
    def __len__(self):
        return len(self.primitives)


class TrajectoryPrimitive:
    """
    Class to store and create a single motion primitive.

    Possible motion primitives are:
    ['YAW_90_cw', 'YAW_90_ccw', 'ROLL_90_cw', 'ROLL_90_ccw', 'PITCH_90_cw', 'PITCH_90_ccw', 'TO_GOAL']

    TODO:
    * Add random perturbations to the primitives.
    * More explicity handle start and end configurations of cube (e.g., which face up).
    """

    def __init__(
        self,
        primitive_name: str,
        contact_mode: ContactMode,
        start_pose: RigidTransform,
        trajectory: Optional[list[RigidTransform]] = None,
        config: dict = kDefaultConfig,
        goal_pose: RigidTransform = RigidTransform(),
        config_overwrite: dict = {},
    ):
        self.primitive_name = primitive_name
        self.start_pose = start_pose
        self.end_pose = RigidTransform()
        self.contact_mode = contact_mode
        self.duration = 0.0
        self.args = merge(config, config_overwrite)
        self.goal_pose = goal_pose
        self.trajectory: list[RigidTransform] = self._create_primitive(trajectory)

    def _create_primitive(self, trajectory: Optional[list[RigidTransform]] = None):
        """
        Create a motion primitive trajectory.

        Args:
            trajectory (list[RigidTransform]): The trajector
        Returns:
            list[RigidTransform]: The trajectory of the primitive
        """
        if trajectory is not None:
            return trajectory
        else:
            trajectory = []
        if self.primitive_name.startswith("YAW_90_cw"):
            # Rotate by -pi/2 about the z-axis (clockwise yaw)
            delta_angle = -np.pi / 2
            num_steps = self.args.get("num_steps", 10)
            for i in range(1, num_steps + 1):
                fraction = i / num_steps
                angle = delta_angle * fraction
                rotation = RotationMatrix.MakeZRotation(angle).multiply(
                    self.start_pose.rotation()
                )
                pose = RigidTransform(rotation, self.start_pose.translation())
                trajectory.append(pose)

        elif self.primitive_name.startswith("YAW_90_ccw"):
            # Rotate by +pi/2 about the z-axis (counterclockwise yaw)
            delta_angle = np.pi / 2
            num_steps = self.args.get("num_steps", 10)
            for i in range(1, num_steps + 1):
                fraction = i / num_steps
                angle = delta_angle * fraction
                rotation = RotationMatrix.MakeZRotation(angle).multiply(
                    self.start_pose.rotation()
                )
                pose = RigidTransform(rotation, self.start_pose.translation())
                trajectory.append(pose)

        elif self.primitive_name.startswith("ROLL_90_cw"):
            # Lift up, rotate about x-axis by -pi/2 (clockwise roll), place down
            lift_height = self.args["lift_height"]
            num_steps_lift = self.args.get("num_steps_lift", 15)
            num_steps_rotate = self.args.get("num_steps_rotate", 15)
            num_steps_lower = self.args.get("num_steps_lower", 15)
            # Lift up
            for i in range(1, num_steps_lift + 1):
                fraction = i / num_steps_lift
                translation = self.start_pose.translation() + np.array(
                    [0, 0, lift_height * fraction]
                )
                pose = RigidTransform(self.start_pose.rotation(), translation)
                trajectory.append(pose)
            # Rotate in air
            for i in range(1, num_steps_rotate + 1):
                fraction = i / num_steps_rotate
                angle = -np.pi / 2 * fraction
                rotation = self.start_pose.rotation().multiply(
                    self.contact_mode.rotate_around_contact_axis(angle)
                )
                translation = self.start_pose.translation() + np.array(
                    [0, 0, lift_height]
                )
                pose = RigidTransform(rotation, translation)
                trajectory.append(pose)
            # Lower down
            for i in range(1, num_steps_lower + 1):
                fraction = i / num_steps_lower
                translation = self.start_pose.translation() + np.array(
                    [0, 0, lift_height * (1 - fraction)]
                )
                rotation = self.start_pose.rotation().multiply(
                    self.contact_mode.rotate_around_contact_axis(-np.pi / 2)
                )
                pose = RigidTransform(rotation, translation)
                trajectory.append(pose)

        elif self.primitive_name.startswith("ROLL_90_ccw"):
            # Lift up, rotate about x-axis by +pi/2 (counterclockwise roll), place down
            lift_height = self.args["lift_height"]
            num_steps_lift = self.args.get("num_steps_lift", 15)
            num_steps_rotate = self.args.get("num_steps_rotate", 15)
            num_steps_lower = self.args.get("num_steps_lower", 15)
            # Lift up
            for i in range(1, num_steps_lift + 1):
                fraction = i / num_steps_lift
                translation = self.start_pose.translation() + np.array(
                    [0, 0, lift_height * fraction]
                )
                pose = RigidTransform(self.start_pose.rotation(), translation)
                trajectory.append(pose)
            # Rotate in air
            for i in range(1, num_steps_rotate + 1):
                fraction = i / num_steps_rotate
                angle = np.pi / 2 * fraction
                rotation = self.start_pose.rotation().multiply(
                    self.contact_mode.rotate_around_contact_axis(angle)
                )
                translation = self.start_pose.translation() + np.array(
                    [0, 0, lift_height]
                )
                pose = RigidTransform(rotation, translation)
                trajectory.append(pose)
            # Lower down
            for i in range(1, num_steps_lower + 1):
                fraction = i / num_steps_lower
                translation = self.start_pose.translation() + np.array(
                    [0, 0, lift_height * (1 - fraction)]
                )
                rotation = self.start_pose.rotation().multiply(
                    self.contact_mode.rotate_around_contact_axis(np.pi / 2)
                )
                pose = RigidTransform(rotation, translation)
                trajectory.append(pose)

        elif self.primitive_name.startswith("TO_GOAL"):
            num_steps = self.args.get("num_steps", 10)
            for i in range(0, num_steps + 1):
                fraction = i / num_steps
                translation = (
                    1 - fraction
                ) * self.start_pose.translation() + fraction * self.goal_pose.translation()
                start_rpy = RollPitchYaw(self.start_pose.rotation())
                goal_rpy = RollPitchYaw(self.goal_pose.rotation())
                delta_rpy = goal_rpy.vector() - start_rpy.vector()
                delta_rpy = (delta_rpy + np.pi) % (2 * np.pi) - np.pi
                rpy = start_rpy.vector() + fraction * delta_rpy
                rotation = RotationMatrix(RollPitchYaw(rpy))
                pose = RigidTransform(rotation, translation)
                trajectory.append(pose)
        
        elif self.primitive_name.startswith("TO_GOAL_INCREMENTAL"):
            last_translation_diff = np.inf
            last_rotation_diff = np.inf
            position_flag = False
            rotation_flag = False

            assert ("translation_step_size" in self.args) and ("rotation_step_size" in self.args), "translation_step_size and rotation_step_size must be provided in the config."

            current_pose = self.start_pose
            direction_translation = (
                self.goal_pose.translation() - self.start_pose.translation()
                ) / np.linalg.norm(self.goal_pose.translation() - self.start_pose.translation())
            start_rpy = RollPitchYaw(self.start_pose.rotation())
            goal_rpy = RollPitchYaw(self.goal_pose.rotation())
            delta_rpy = goal_rpy.vector() - start_rpy.vector()
            (delta_rpy + np.pi) % (2 * np.pi) - np.pi
            direction_rpy = (delta_rpy) / np.linalg.norm(delta_rpy)
            while True:
                translation = (
                    current_pose.translation() + (0 if position_flag else self.args.get("translation_step_size") * direction_translation)
                    )
                rpy = start_rpy.vector() + (0 if rotation_flag else self.args.get("rotation_step_size") * direction_rpy)


                if not position_flag and np.linalg.norm(current_pose.translation() - self.goal_pose.translation()) <= last_translation_diff:
                    last_translation_diff = np.linalg.norm(current_pose.translation() - self.goal_pose.translation())
                else:
                    position_flag = True
                
                if not rotation_flag and np.linalg.norm(RollPitchYaw(current_pose.rotation()).vector() - goal_rpy.vector()) <= last_rotation_diff:
                    last_rotation_diff = np.linalg.norm(RollPitchYaw(current_pose.rotation()).vector() - goal_rpy.vector())
                else:
                    rotation_flag = True
                
                if position_flag and rotation_flag:
                    break

                rotation = RotationMatrix(RollPitchYaw(rpy))
                pose = RigidTransform(rotation, translation)
                trajectory.append(pose)


        else:
            raise ValueError(f"Unknown primitive_name: {self.primitive_name}")

        return trajectory

    def project_to_2D(self) -> list[np.ndarray]:
        """
        Project the trajectory to 2D case.
        """
        trajectory_2d = []
        for pose in self.trajectory:
            quaternion = pose.rotation().ToQuaternion()
            cos_yaw, sin_yaw = quaternion_to_cos_sin_yaw(
                quaternion.w(), quaternion.x(), quaternion.y(), quaternion.z()
            )
            pose_2d = np.array(
                [pose.translation()[0], pose.translation()[1], cos_yaw, sin_yaw]
            )
            trajectory_2d.append(pose_2d)

        return trajectory_2d

    def __iter__(self):
        return iter(self.trajectory)

    def __len__(self):
        return len(self.trajectory)

    def __getitem__(self, idx):
        return self.trajectory[idx]

    def __setitem__(self, idx, value):
        self.trajectory[idx] = value
        return self.trajectory[idx]


def sample_tabletop_pose(bounding_box: np.ndarray, contact_mode: ContactMode, config={}):
        """
        Samples a random pose on the tabletop.
        """

        x = random.uniform(bounding_box[0, 0], bounding_box[0, 1])
        y = random.uniform(bounding_box[1, 0], bounding_box[1, 1])
        z = config["box_height"]

        # random upright rotation matrix
        yaw = random.uniform(-np.pi, np.pi)

        rotation = RollPitchYaw(0, 0, yaw)

        if config.get("tabletop_height_variation") is not None:

            z += random.uniform(
                0,
                config["tabletop_height_variation"],
            )

        if config.get("tabletop_orientation_variation") is not None:
            rotation = RollPitchYaw(
                    random.uniform(
                        -config["tabletop_orientation_variation"],
                        config["tabletop_orientation_variation"],
                    ),
                    random.uniform(
                        -config["tabletop_orientation_variation"],
                        config["tabletop_orientation_variation"],
                    ),
                    yaw,
                )
             
            
        return RigidTransform(rotation, [x, y, z]).multiply(
            RigidTransform(contact_mode.default_pose.rotation(), [0, 0, 0])
        )

