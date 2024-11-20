from abc import ABC, abstractmethod
from pydrake.systems.framework import LeafContext

from pydrake.all import (
    RigidTransform,
    RotationMatrix,
    Quaternion,
    RollPitchYaw
)  

from pathlib import Path
from dual_arm_manipulation import ROOT_DIR
from dual_arm_manipulation.utils import pose_vec_to_transform
import numpy as np

class AbstractPlanner(ABC):
    def __init__(self, plant_context: LeafContext, goal_pose: np.ndarray, simulate: bool = False):
        self.simulate: bool = simulate
        self.plant_context: LeafContext = plant_context
        self.goal_pose: np.ndarray = goal_pose
        
        self.config = {}
        
        # loaded at the start (from config instructions)
        self.contact_modes_ = []
        self.trajectory_primitives_ = []

        self.convex_sets_ = []
        

    @abstractmethod
    def plan(self):
        pass

    def _load_config(self, config_path: Path):
        self.config = config_path


class MotionPrimitives:
    """
    Class to load and store motion primitives.
    
    TODO:
    * Integrate visualizer to visualize primitives and chain of primitives.
    * Add random perturbations to the primitives to build convex sets.
    """
    def __init__(self, start_pose: RigidTransform, contact_mode_name, config: dict):
        self.config = config
        self.start_pose = start_pose
        self.contact_mode_name = contact_mode_name
        self.primitives = []
        self._load_primitives()

    def _load_primitives(self):
        for primitive in self.config['planner']['primitives']:
            self.primitives.append(MotionPrimitive(primitive, self.start_pose, self.config))


class MotionPrimitive:
    """
    Class to store and create a single motion primitive.

    Possible motion primitives are:
    ['YAW_90_cw', 'YAW_90_ccw', 'ROLL_90_cw', 'ROLL_90_ccw', 'PITCH_90_cw', 'PITCH_90_ccw', 'TO_GOAL']

    TODO:
    * Add random perturbations to the primitives. 
    * More explicity handle start and end configurations of cube (e.g., which face up).
    """
    def __init__(self, primitive_name: str, start_pose: RigidTransform, args: dict = {'goal_pose': [0.0, 0.0, 0.4], 'lift_height': 0.4, 'num_steps': 20}):
        self.start_pose = start_pose
        self.end_pose = RigidTransform()
        self.duration = 0.0
        self.args = args
        self.trajectory: list[RigidTransform] = self._create_primitive(start_pose, primitive_name)

    def _create_primitive(self, start_pose: RigidTransform, primitive_name: str):
        trajectory = []

        if primitive_name == 'YAW_90_cw':
            # Rotate by -pi/2 about the z-axis (clockwise yaw)
            delta_angle = -np.pi / 2
            num_steps = self.args['planner'].get('num_steps', 10)
            for i in range(1, num_steps + 1):
                fraction = i / num_steps
                angle = delta_angle * fraction
                rotation = start_pose.rotation().multiply(RotationMatrix.MakeZRotation(angle))
                pose = RigidTransform(rotation, start_pose.translation())
                trajectory.append(pose)

        elif primitive_name == 'YAW_90_ccw':
            # Rotate by +pi/2 about the z-axis (counterclockwise yaw)
            delta_angle = np.pi / 2
            num_steps = self.args['planner'].get('num_steps', 10)
            for i in range(1, num_steps + 1):
                fraction = i / num_steps
                angle = delta_angle * fraction
                rotation = start_pose.rotation().multiply(RotationMatrix.MakeZRotation(angle))
                pose = RigidTransform(rotation, start_pose.translation())
                trajectory.append(pose)

        elif primitive_name == 'ROLL_90_cw':
            # Lift up, rotate about x-axis by -pi/2 (clockwise roll), place down
            lift_height = self.args['planner']['lift_height']
            num_steps_lift = self.args['planner'].get('num_steps_lift', 15)
            num_steps_rotate = self.args['planner'].get('num_steps_rotate', 15)
            num_steps_lower = self.args['planner'].get('num_steps_lower', 15)
            # Lift up
            for i in range(1, num_steps_lift + 1):
                fraction = i / num_steps_lift
                translation = start_pose.translation() + np.array([0, 0, lift_height * fraction])
                pose = RigidTransform(start_pose.rotation(), translation)
                trajectory.append(pose)
            # Rotate in air
            for i in range(1, num_steps_rotate + 1):
                fraction = i / num_steps_rotate
                angle = -np.pi / 2 * fraction
                rotation = start_pose.rotation().multiply(RotationMatrix.MakeXRotation(angle))
                translation = start_pose.translation() + np.array([0, 0, lift_height])
                pose = RigidTransform(rotation, translation)
                trajectory.append(pose)
            # Lower down
            for i in range(1, num_steps_lower + 1):
                fraction = i / num_steps_lower
                translation = start_pose.translation() + np.array([0, 0, lift_height * (1 - fraction)])
                rotation = start_pose.rotation().multiply(RotationMatrix.MakeXRotation(-np.pi / 2))
                pose = RigidTransform(rotation, translation)
                trajectory.append(pose)

        elif primitive_name == 'ROLL_90_ccw':
            # Lift up, rotate about x-axis by +pi/2 (counterclockwise roll), place down
            lift_height = self.args['planner']['lift_height']
            num_steps_lift = self.args['planner'].get('num_steps_lift', 15)
            num_steps_rotate = self.args['planner'].get('num_steps_rotate', 15)
            num_steps_lower = self.args['planner'].get('num_steps_lower', 15)
            # Lift up
            for i in range(1, num_steps_lift + 1):
                fraction = i / num_steps_lift
                translation = start_pose.translation() + np.array([0, 0, lift_height * fraction])
                pose = RigidTransform(start_pose.rotation(), translation)
                trajectory.append(pose)
            # Rotate in air
            for i in range(1, num_steps_rotate + 1):
                fraction = i / num_steps_rotate
                angle = np.pi / 2 * fraction
                rotation = start_pose.rotation().multiply(RotationMatrix.MakeXRotation(angle))
                translation = start_pose.translation() + np.array([0, 0, lift_height])
                pose = RigidTransform(rotation, translation)
                trajectory.append(pose)
            # Lower down
            for i in range(1, num_steps_lower + 1):
                fraction = i / num_steps_lower
                translation = start_pose.translation() + np.array([0, 0, lift_height * (1 - fraction)])
                rotation = start_pose.rotation().multiply(RotationMatrix.MakeXRotation(np.pi / 2))
                pose = RigidTransform(rotation, translation)
                trajectory.append(pose)


        elif primitive_name == 'PITCH_90_cw':
            # Lift up, rotate about x-axis by -pi/2 (clockwise PITCH), place down
            lift_height = self.args['planner']['lift_height']
            num_steps_lift = self.args['planner'].get('num_steps_lift', 15)
            num_steps_rotate = self.args['planner'].get('num_steps_rotate', 15)
            num_steps_lower = self.args['planner'].get('num_steps_lower', 15)
            # Lift up
            for i in range(1, num_steps_lift + 1):
                fraction = i / num_steps_lift
                translation = start_pose.translation() + np.array([0, 0, lift_height * fraction])
                pose = RigidTransform(start_pose.rotation(), translation)
                trajectory.append(pose)
            # Rotate in air
            for i in range(1, num_steps_rotate + 1):
                fraction = i / num_steps_rotate
                angle = -np.pi / 2 * fraction
                rotation = start_pose.rotation().multiply(RotationMatrix.MakeYRotation(angle))
                translation = start_pose.translation() + np.array([0, 0, lift_height])
                pose = RigidTransform(rotation, translation)
                trajectory.append(pose)
            # Lower down
            for i in range(1, num_steps_lower + 1):
                fraction = i / num_steps_lower
                translation = start_pose.translation() + np.array([0, 0, lift_height * (1 - fraction)])
                rotation = start_pose.rotation().multiply(RotationMatrix.MakeYRotation(-np.pi / 2))
                pose = RigidTransform(rotation, translation)
                trajectory.append(pose)

        elif primitive_name == 'PITCH_90_ccw':
            # Lift up, rotate about x-axis by +pi/2 (counterclockwise PITCH), place down
            lift_height = self.args['planner']['lift_height']
            num_steps_lift = self.args['planner'].get('num_steps_lift', 15)
            num_steps_rotate = self.args['planner'].get('num_steps_rotate', 15)
            num_steps_lower = self.args['planner'].get('num_steps_lower', 15)
            # Lift up
            for i in range(1, num_steps_lift + 1):
                fraction = i / num_steps_lift
                translation = start_pose.translation() + np.array([0, 0, lift_height * fraction])
                pose = RigidTransform(start_pose.rotation(), translation)
                trajectory.append(pose)
            # Rotate in air
            for i in range(1, num_steps_rotate + 1):
                fraction = i / num_steps_rotate
                angle = np.pi / 2 * fraction
                rotation = start_pose.rotation().multiply(RotationMatrix.MakeYRotation(angle))
                translation = start_pose.translation() + np.array([0, 0, lift_height])
                pose = RigidTransform(rotation, translation)
                trajectory.append(pose)
            # Lower down
            for i in range(1, num_steps_lower + 1):
                fraction = i / num_steps_lower
                translation = start_pose.translation() + np.array([0, 0, lift_height * (1 - fraction)])
                rotation = start_pose.rotation().multiply(RotationMatrix.MakeYRotation(np.pi / 2))
                pose = RigidTransform(rotation, translation)
                trajectory.append(pose)


        elif primitive_name == 'TO_GOAL':
            # Interpolate between start_pose and self.args['goal_pose']
            goal_pose = pose_vec_to_transform(self.args['eval']['goal_pose'])
            num_steps = self.args['planner'].get('num_steps', 10)
            for i in range(1, num_steps + 1):
                fraction = i / num_steps
                translation = (1 - fraction) * start_pose.translation() + fraction * goal_pose.translation()
                start_rpy = RollPitchYaw(start_pose.rotation())
                goal_rpy = RollPitchYaw(goal_pose.rotation())
                delta_rpy = goal_rpy.vector() - start_rpy.vector()
                rpy = start_rpy.vector() + fraction * delta_rpy
                rotation = RotationMatrix(RollPitchYaw(rpy))
                pose = RigidTransform(rotation, translation)
                trajectory.append(pose)
        else:
            raise ValueError(f"Unknown primitive_name: {primitive_name}")
        return trajectory