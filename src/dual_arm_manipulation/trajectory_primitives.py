from pydrake.all import (
    RigidTransform,
    RotationMatrix,
    RollPitchYaw
)

import numpy as np

from mergedeep import merge
from dual_arm_manipulation.contact_mode import ContactMode
from dual_arm_manipulation.utils import pose_vec_to_transform

kDefaultConfig = {
    'primitives': ['YAW_90_cw', 'YAW_90_ccw', 'ROLL_90_cw', 'ROLL_90_ccw', 'PITCH_90_cw', 'PITCH_90_ccw'],
    'goal_pose': [0.0, 0.0, 0.4],
    'lift_height': 0.4,
    'num_steps': 20,
    'num_steps_lift': 15,
    'num_steps_rotate': 15,
    'num_steps_lower': 15
}

class TrajectoryPrimitives:
    """
    Class to load and store motion primitives.
    
    TODO:
    * Integrate visualizer to visualize primitives and chain of primitives.
    * Add random perturbations to the primitives to build convex sets.
    """
    def __init__(self, start_pose: RigidTransform, contact_mode: ContactMode, config: dict, config_override: dict = {}):
        self.config = merge(config, config_override)
        self.start_pose = start_pose
        self.contact_mode = contact_mode
        self.primitives = []

    def load_primitives(self):
        for primitive in self.config['primitives']:
            self.primitives.append(TrajectoryPrimitive(primitive, self.contact_mode, self.start_pose, self.config))

    def __iter__(self):
        return iter(self.primitives)


class TrajectoryPrimitive:
    """
    Class to store and create a single motion primitive.

    Possible motion primitives are:
    ['YAW_90_cw', 'YAW_90_ccw', 'ROLL_90_cw', 'ROLL_90_ccw', 'PITCH_90_cw', 'PITCH_90_ccw', 'TO_GOAL']

    TODO:
    * Add random perturbations to the primitives. 
    * More explicity handle start and end configurations of cube (e.g., which face up).
    """
    def __init__(self, 
                 primitive_name: str, 
                 contact_mode: ContactMode, 
                 start_pose: RigidTransform, 
                 config: dict = kDefaultConfig,
                 goal_pose: RigidTransform = RigidTransform(),
                 config_overwrite: dict = {}):
        self.primitive_name = primitive_name
        self.start_pose = start_pose
        self.end_pose = RigidTransform()
        self.contact_mode = contact_mode
        self.duration = 0.0
        self.args = merge(config, config_overwrite)
        self.goal_pose = goal_pose
        self.trajectory: list[RigidTransform] = self._create_primitive()

    def _create_primitive(self):
        trajectory = []

        if self.primitive_name == 'YAW_90_cw':
            # Rotate by -pi/2 about the z-axis (clockwise yaw)
            delta_angle = -np.pi / 2
            num_steps = self.args.get('num_steps', 10)
            for i in range(1, num_steps + 1):
                fraction = i / num_steps
                angle = delta_angle * fraction
                rotation = RotationMatrix.MakeZRotation(angle).multiply(self.start_pose.rotation())
                pose = RigidTransform(rotation, self.start_pose.translation())
                trajectory.append(pose)

        elif self.primitive_name == 'YAW_90_ccw':
            # Rotate by +pi/2 about the z-axis (counterclockwise yaw)
            delta_angle = np.pi / 2
            num_steps = self.args.get('num_steps', 10)
            for i in range(1, num_steps + 1):
                fraction = i / num_steps
                angle = delta_angle * fraction
                rotation = RotationMatrix.MakeZRotation(angle).multiply(self.start_pose.rotation())
                pose = RigidTransform(rotation, self.start_pose.translation())
                trajectory.append(pose)

        elif self.primitive_name == 'ROLL_90_cw':
            # Lift up, rotate about x-axis by -pi/2 (clockwise roll), place down
            lift_height = self.args['lift_height']
            num_steps_lift = self.args.get('num_steps_lift', 15)
            num_steps_rotate = self.args.get('num_steps_rotate', 15)
            num_steps_lower = self.args.get('num_steps_lower', 15)
            # Lift up
            for i in range(1, num_steps_lift + 1):
                fraction = i / num_steps_lift
                translation = self.start_pose.translation() + np.array([0, 0, lift_height * fraction])
                pose = RigidTransform(self.start_pose.rotation(), translation)
                trajectory.append(pose)
            # Rotate in air
            for i in range(1, num_steps_rotate + 1):
                fraction = i / num_steps_rotate
                angle = -np.pi / 2 * fraction
                rotation = self.start_pose.rotation().multiply(self.contact_mode.rotate_around_contact_axis(angle))
                translation = self.start_pose.translation() + np.array([0, 0, lift_height])
                pose = RigidTransform(rotation, translation)
                trajectory.append(pose)
            # Lower down
            for i in range(1, num_steps_lower + 1):
                fraction = i / num_steps_lower
                translation = self.start_pose.translation() + np.array([0, 0, lift_height * (1 - fraction)])
                rotation = self.start_pose.rotation().multiply(self.contact_mode.rotate_around_contact_axis(-np.pi / 2))
                pose = RigidTransform(rotation, translation)
                trajectory.append(pose)

        elif self.primitive_name == 'ROLL_90_ccw':
            # Lift up, rotate about x-axis by +pi/2 (counterclockwise roll), place down
            lift_height = self.args['lift_height']
            num_steps_lift = self.args.get('num_steps_lift', 15)
            num_steps_rotate = self.args.get('num_steps_rotate', 15)
            num_steps_lower = self.args.get('num_steps_lower', 15)
            # Lift up
            for i in range(1, num_steps_lift + 1):
                fraction = i / num_steps_lift
                translation = self.start_pose.translation() + np.array([0, 0, lift_height * fraction])
                pose = RigidTransform(self.start_pose.rotation(), translation)
                trajectory.append(pose)
            # Rotate in air
            for i in range(1, num_steps_rotate + 1):
                fraction = i / num_steps_rotate
                angle = np.pi / 2 * fraction
                rotation = self.start_pose.rotation().multiply(self.contact_mode.rotate_around_contact_axis(angle))
                translation = self.start_pose.translation() + np.array([0, 0, lift_height])
                pose = RigidTransform(rotation, translation)
                trajectory.append(pose)
            # Lower down
            for i in range(1, num_steps_lower + 1):
                fraction = i / num_steps_lower
                translation = self.start_pose.translation() + np.array([0, 0, lift_height * (1 - fraction)])
                rotation = self.start_pose.rotation().multiply(self.contact_mode.rotate_around_contact_axis(np.pi / 2))
                pose = RigidTransform(rotation, translation)
                trajectory.append(pose)

        elif self.primitive_name == 'TO_GOAL': 
            num_steps = self.args.get('num_steps', 10)
            for i in range(1, num_steps + 1):
                fraction = i / num_steps
                translation = (1 - fraction) * self.start_pose.translation() + fraction * self.goal_pose.translation()
                start_rpy = RollPitchYaw(self.start_pose.rotation())
                goal_rpy = RollPitchYaw(self.goal_pose.rotation())
                delta_rpy = goal_rpy.vector() - start_rpy.vector()
                rpy = start_rpy.vector() + fraction * delta_rpy
                rotation = RotationMatrix(RollPitchYaw(rpy))
                pose = RigidTransform(rotation, translation)
                trajectory.append(pose)
        else:
            raise ValueError(f"Unknown primitive_name: {self.primitive_name}")
        return trajectory
    
    def __iter__(self):
        return iter(self.trajectory)