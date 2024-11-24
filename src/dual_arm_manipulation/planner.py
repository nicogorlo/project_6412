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

from mergedeep import merge

VALID_CONTACT_MODES = ['X_POS', 'X_NEG', 'Y_POS', 'Y_NEG', 'Z_POS', 'Z_NEG']

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


class ContactMode:
    """
    Class to load and store contact modes.
    """
    def __init__(self, name: str = 'X_POS', config: dict = {}):
        assert name in VALID_CONTACT_MODES, f"Invalid contact mode: {name}"
        self.name = name
        if 'planner' in config:
            self.contact_positions = config['planner']['contact_modes'][name]
        else:
            self.contact_positions = [[0.15, 0, 0], [-0.15, 0, 0]] # TODO: define kDefaultConfig
        self.config = config

        self.default_pose = self.get_default_pose_()

    def get_free_faces(self):
        for axis in ['X', 'Y', 'Z']:
            if self.name.startswith(axis):
                return {name: free_contact_mode[0] for name, free_contact_mode in self.config['planner']['contact_modes'].items() if not name.startswith(axis)}
        
    def get_contact_frame_pos(self, plant):
        if self.name == 'X_POS':
            constrained_axis = 0
            theta_bounds = np.array([0.005, np.pi/4+ 0.1, np.pi/4+ 0.1])
            cube_contact_frame_pos = plant.GetFrameByName("X_pos_contact", plant.GetModelInstanceByName("movable_cuboid"))
            cube_contact_frame_neg = plant.GetFrameByName("X_neg_contact", plant.GetModelInstanceByName("movable_cuboid"))
        elif self.name == 'X_NEG':
            constrained_axis = 0
            theta_bounds = np.array([0.005, np.pi/4+ 0.1, np.pi/4+ 0.1])
            cube_contact_frame_pos = plant.GetFrameByName("X_neg_contact", plant.GetModelInstanceByName("movable_cuboid"))
            cube_contact_frame_neg = plant.GetFrameByName("X_pos_contact", plant.GetModelInstanceByName("movable_cuboid"))
        elif self.name == 'Y_POS':
            constrained_axis = 1
            theta_bounds = np.array([np.pi/4+ 0.1, 0.005, np.pi/4+ 0.1])
            cube_contact_frame_pos = plant.GetFrameByName("Y_pos_contact", plant.GetModelInstanceByName("movable_cuboid"))
            cube_contact_frame_neg = plant.GetFrameByName("Y_neg_contact", plant.GetModelInstanceByName("movable_cuboid"))
        elif self.name == 'Y_NEG':
            constrained_axis = 1
            theta_bounds = np.array([0.005, np.pi/4+ 0.1, np.pi/4+ 0.1])
            cube_contact_frame_pos = plant.GetFrameByName("Y_neg_contact", plant.GetModelInstanceByName("movable_cuboid"))
            cube_contact_frame_neg = plant.GetFrameByName("Y_pos_contact", plant.GetModelInstanceByName("movable_cuboid"))
        elif self.name == 'Z_POS':
            constrained_axis = 2
            theta_bounds = np.array([np.pi/4+ 0.1, np.pi/4+ 0.1, 0.005])
            cube_contact_frame_pos = plant.GetFrameByName("Z_pos_contact", plant.GetModelInstanceByName("movable_cuboid"))
            cube_contact_frame_neg = plant.GetFrameByName("Z_neg_contact", plant.GetModelInstanceByName("movable_cuboid"))
        elif self.name == 'Z_NEG':
            constrained_axis = 2
            theta_bounds = np.array([np.pi/4+ 0.1, np.pi/4+ 0.1, 0.005])
            cube_contact_frame_pos = plant.GetFrameByName("Z_neg_contact", plant.GetModelInstanceByName("movable_cuboid"))
            cube_contact_frame_neg = plant.GetFrameByName("Z_pos_contact", plant.GetModelInstanceByName("movable_cuboid"))
        else:
            print("Invalid contact mode.")
            raise ValueError
        
        return cube_contact_frame_pos, cube_contact_frame_neg, constrained_axis, theta_bounds

    def rotate_around_contact_axis(self, angle):
        if self.name == 'X_POS':
            return RotationMatrix.MakeXRotation(angle)
        elif self.name == 'X_NEG':
            return RotationMatrix.MakeXRotation(-angle)
        elif self.name == 'Y_POS':
            return RotationMatrix.MakeYRotation(angle)
        elif self.name == 'Y_NEG':
            return RotationMatrix.MakeYRotation(-angle)
        elif self.name == 'Z_POS':
            return RotationMatrix.MakeZRotation(angle)
        elif self.name == 'Z_NEG':
            return RotationMatrix.MakeZRotation(-angle)
        else:
            raise ValueError(f"Invalid contact mode: {self.name}")
        
    def get_default_pose_(self):
        if self.name == "X_POS":
            return RigidTransform(RotationMatrix(), np.array([0.0, 0.0, 0.2]))
        elif self.name == "X_NEG":
            return RigidTransform(RotationMatrix(RollPitchYaw((0,0, np.pi))), np.array([0.0, 0.0, 0.2]))
        elif self.name == "Y_POS":
            return RigidTransform(RotationMatrix(RollPitchYaw((np.pi/2, np.pi/2, 0))), np.array([0.0, 0.0, 0.2]))
        elif self.name == "Y_NEG":
            return RigidTransform(RotationMatrix(RollPitchYaw((np.pi/2, -np.pi/2, 0))), np.array([0.0, 0.0, 0.2]))
        elif self.name == "Z_POS":
            return RigidTransform(RotationMatrix(RollPitchYaw((0, np.pi/2, 0))), np.array([0.0, 0.0, 0.2]))
        elif self.name == "Z_NEG":
            return RigidTransform(RotationMatrix(RollPitchYaw((0, -np.pi/2, 0))), np.array([0.0, 0.0, 0.2]))
        else:
            raise ValueError(f"Invalid contact mode: {self.name}")




class MotionPrimitives:
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
        self._load_primitives()

    def _load_primitives(self):
        for primitive in self.config['planner']['primitives']:
            self.primitives.append(MotionPrimitive(primitive, self.contact_mode, self.start_pose, self.config))

    def __iter__(self):
        return iter(self.primitives)


class MotionPrimitive:
    """
    Class to store and create a single motion primitive.

    Possible motion primitives are:
    ['YAW_90_cw', 'YAW_90_ccw', 'ROLL_90_cw', 'ROLL_90_ccw', 'PITCH_90_cw', 'PITCH_90_ccw', 'TO_GOAL']

    TODO:
    * Add random perturbations to the primitives. 
    * More explicity handle start and end configurations of cube (e.g., which face up).
    """
    def __init__(self, primitive_name: str, contact_mode: ContactMode, start_pose: RigidTransform, args: dict = {'goal_pose': [0.0, 0.0, 0.4], 'lift_height': 0.4, 'num_steps': 20}):
        self.primitive_name = primitive_name
        self.start_pose = start_pose
        self.end_pose = RigidTransform()
        self.contact_mode = contact_mode
        self.duration = 0.0
        self.args = args
        self.trajectory: list[RigidTransform] = self._create_primitive()

    def _create_primitive(self):
        trajectory = []

        if self.primitive_name == 'YAW_90_cw':
            # Rotate by -pi/2 about the z-axis (clockwise yaw)
            delta_angle = -np.pi / 2
            num_steps = self.args['planner'].get('num_steps', 10)
            for i in range(1, num_steps + 1):
                fraction = i / num_steps
                angle = delta_angle * fraction
                rotation = RotationMatrix.MakeZRotation(angle).multiply(self.start_pose.rotation())
                pose = RigidTransform(rotation, self.start_pose.translation())
                trajectory.append(pose)

        elif self.primitive_name == 'YAW_90_ccw':
            # Rotate by +pi/2 about the z-axis (counterclockwise yaw)
            delta_angle = np.pi / 2
            num_steps = self.args['planner'].get('num_steps', 10)
            for i in range(1, num_steps + 1):
                fraction = i / num_steps
                angle = delta_angle * fraction
                rotation = RotationMatrix.MakeZRotation(angle).multiply(self.start_pose.rotation())
                pose = RigidTransform(rotation, self.start_pose.translation())
                trajectory.append(pose)

        elif self.primitive_name == 'ROLL_90_cw':
            # Lift up, rotate about x-axis by -pi/2 (clockwise roll), place down
            lift_height = self.args['planner']['lift_height']
            num_steps_lift = self.args['planner'].get('num_steps_lift', 15)
            num_steps_rotate = self.args['planner'].get('num_steps_rotate', 15)
            num_steps_lower = self.args['planner'].get('num_steps_lower', 15)
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
            lift_height = self.args['planner']['lift_height']
            num_steps_lift = self.args['planner'].get('num_steps_lift', 15)
            num_steps_rotate = self.args['planner'].get('num_steps_rotate', 15)
            num_steps_lower = self.args['planner'].get('num_steps_lower', 15)
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

        elif self.primitive_name == 'TO_GOAL': # TODO: Not called right now, needs to be adapted for Y and Z contact modes
            # Interpolate between self.start_pose and self.args['goal_pose']
            goal_pose = pose_vec_to_transform(self.args['eval']['goal_pose'])
            num_steps = self.args['planner'].get('num_steps', 10)
            for i in range(1, num_steps + 1):
                fraction = i / num_steps
                translation = (1 - fraction) * self.start_pose.translation() + fraction * goal_pose.translation()
                start_rpy = RollPitchYaw(self.start_pose.rotation())
                goal_rpy = RollPitchYaw(goal_pose.rotation())
                delta_rpy = goal_rpy.vector() - start_rpy.vector()
                rpy = start_rpy.vector() + fraction * delta_rpy
                rotation = RotationMatrix(RollPitchYaw(rpy))
                pose = RigidTransform(rotation, translation)
                trajectory.append(pose)
        else:
            raise ValueError(f"Unknown primitive_name: {self.primitive_name}")
        return trajectory