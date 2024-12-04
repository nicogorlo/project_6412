from pydrake.all import (
    RigidTransform,
    RotationMatrix,
    RollPitchYaw
)

import numpy as np
from typing import Optional


VALID_CONTACT_MODES = ['X_POS', 'X_NEG', 'Y_POS', 'Y_NEG', 'Z_POS', 'Z_NEG']


class ContactMode:
    """
    Class to load and store contact modes.
    """
    def __init__(self, name: str = 'X_POS', config: dict = {}):
        assert name in VALID_CONTACT_MODES, f"Invalid contact mode: {name}"
        self.name = name
        if 'sampler' in config:
            self.contact_positions = config['sampler']['contact_modes'][name]
        else:
            self.contact_positions = [[0.15, 0, 0], [-0.15, 0, 0]] # TODO: define kDefaultConfig
        self.config = config

        self.default_pose = self.get_default_pose_()

    def get_free_faces(self):
        for axis in ['X', 'Y', 'Z']:
            if self.name.startswith(axis):
                return {name: free_contact_mode[0] for name, free_contact_mode in self.config['sampler']['contact_modes'].items() if not name.startswith(axis)}
        
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
            return RigidTransform(RotationMatrix(), np.array([0.0, 0.0, self.config['sampler']['box_height'] if 'sampler' in self.config else 0.15]))
        elif self.name == "X_NEG":
            return RigidTransform(RotationMatrix(RollPitchYaw((0,0, np.pi))), np.array([0.0, 0.0, self.config['sampler']['box_height'] if 'sampler' in self.config else 0.15]))
        elif self.name == "Y_POS":
            return RigidTransform(RotationMatrix(RollPitchYaw((0.0, 0.0, np.pi/2))), np.array([0.0, 0.0, self.config['sampler']['box_height'] if 'sampler' in self.config else 0.15]))
        elif self.name == "Y_NEG":
            return RigidTransform(RotationMatrix(RollPitchYaw((0.0, 0.0, -np.pi/2))), np.array([0.0, 0.0, self.config['sampler']['box_height'] if 'sampler' in self.config else 0.15]))
        elif self.name == "Z_POS":
            return RigidTransform(RotationMatrix(RollPitchYaw((0, np.pi/2, 0))), np.array([0.0, 0.0, self.config['sampler']['box_height'] if 'sampler' in self.config else 0.15]))
        elif self.name == "Z_NEG":
            return RigidTransform(RotationMatrix(RollPitchYaw((0, -np.pi/2, 0))), np.array([0.0, 0.0, self.config['sampler']['box_height'] if 'sampler' in self.config else 0.15]))
        else:
            raise ValueError(f"Invalid contact mode: {self.name}")
        
