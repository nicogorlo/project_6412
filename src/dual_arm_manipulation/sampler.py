from abc import ABC, abstractmethod
from typing import Optional
from pydrake.systems.framework import LeafContext

from pydrake.all import (
    RigidTransform,
    RotationMatrix,
    Quaternion,
    RollPitchYaw,
    Diagram,
    MultibodyPlant,
    InverseKinematics,
    Solve
)

from pathlib import Path
from dual_arm_manipulation import ROOT_DIR
from dual_arm_manipulation.contact_mode import ContactMode
from dual_arm_manipulation.trajectory_primitives import (
    TrajectoryPrimitives,
    TrajectoryPrimitive,
    sample_tabletop_pose
)
from dual_arm_manipulation.utils import (
    pose_vec_to_transform,
    rotation_matrix_from_vectors,
)
import numpy as np
import yaml
import logging
import random
from mergedeep import merge
from tqdm import tqdm
import pickle


class AbstractSampler(ABC):
    def __init__(
        self, plant: MultibodyPlant, plant_context: LeafContext, simulate: bool = False
    ):
        self.simulate: bool = simulate
        self.plant: MultibodyPlant = plant
        self.plant_context: LeafContext = plant_context

        self.config = {}

        # loaded at the start (from config instructions)
        self.contact_modes_ = []
        self.trajectory_primitives_ = []

        self.convex_sets_ = []

    @abstractmethod
    def sample(self):
        pass

    def _load_config(self, config_path: Path):
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)["sampler"]


class PrimitiveSampler(AbstractSampler):
    def __init__(
        self,
        plant: MultibodyPlant,
        plant_context: LeafContext,
        start_pose: RigidTransform,
        goal_pose: RigidTransform,
        contact_modes: list[ContactMode],
        simulate: bool = False,
        config_path: Path = ROOT_DIR / "config" / "config.yaml",
    ):
        super().__init__(plant, plant_context, simulate)

        self._load_config(config_path)

        self.start_pose: RigidTransform = start_pose
        self.goal_pose: RigidTransform = goal_pose
        self.contact_modes = contact_modes

        self.trajectory_primitives: dict[str, TrajectoryPrimitives] = {
            mode.name: TrajectoryPrimitives(mode.default_pose, mode, self.config)
            for mode in contact_modes
        }
        self.ik_solutions: dict[
            str, dict[str, np.ndarray[Optional[RigidTransform]]]
        ] = {mode.name: {} for mode in contact_modes}

    def sample(self):
        print("Planning")
        return

    def get_traj_primitives(self) -> None:
        """
        Returns the trajectory primitives. These are sample trajectories that can be used to build convex sets.

        Iterates through all the contact modes and samples IK solutions along trajectory primitives.

        """

        for contact_mode in self.contact_modes:

            default_pose = contact_mode.default_pose

            if self.trajectory_primitives[contact_mode.name] is None:
                self.trajectory_primitives[contact_mode.name] = TrajectoryPrimitives(
                    default_pose, contact_mode, self.config, {"augment": True}
                )
            self.trajectory_primitives[contact_mode.name].load_primitives()

            solutions = {}

            for primitive in tqdm(self.trajectory_primitives[contact_mode.name], "[Trajectory Primitives] Primitive Iteration"):

                if (
                    self.ik_solutions[contact_mode.name].get(primitive.primitive_name)
                    is not None
                ):
                    continue

                solution = self.ik_trajectory(
                    primitive.trajectory, contact_mode=contact_mode
                )

                solutions[primitive.primitive_name] = solution

            self.ik_solutions[contact_mode.name].update(solutions)

    def get_goal_conditioned_tabletop_configurations(self):

        sample_final_contact_modes = {}

        for contact_mode in tqdm(
            self.contact_modes,
            "[Goal Conditioned Tabletop Configurations] Contact Mode Iteration",
        ):

            contact_mode_name = contact_mode.name
            # determine if IK solution exists for contact mode in goal configuration
            solution = self.sample_ik(self.goal_pose, contact_mode=contact_mode)

            tabletop_sample_poses = []
            tabletop_sample_solutions = []

            if solution is not None:

                logging.info(
                    f"[goal_conditioned_tabletop_configurations] viable contact mode: {contact_mode_name} for goal pose: {self.goal_pose}"
                )

                # get tabletop configurations that work for the contact mode
                # sample poses in an ellipsoid between the goal pose and tabletop configurations to get a convex set
                for face_name, face_pos in contact_mode.get_free_faces().items():
                    face_normal = face_pos / np.linalg.norm(face_pos)

                    # get poses s.t. face is in contact with the tabletop surface
                    tabletop_rotation = rotation_matrix_from_vectors(
                        np.array([0, 0, 1]), np.array(-face_normal)
                    )
                    tabletop_translation = np.array([0, 0, self.config["box_height"]])
                    for angle in np.linspace(
                        0,
                        2 * np.pi,
                        self.config["tabletop_configurations"]["n_rotations"],
                    ):
                        rotation_sample_sol = RotationMatrix.MakeZRotation(angle)
                        rotation_sample_sol = rotation_sample_sol.multiply(
                            tabletop_rotation
                        )
                        tabletop_sample_pose = RigidTransform(
                            rotation_sample_sol, tabletop_translation
                        )
                        sample_solution = self.sample_ik(
                            tabletop_sample_pose, contact_mode=contact_mode
                        )

                        if sample_solution is not None:
                            tabletop_sample_poses.append(tabletop_sample_pose)
                            tabletop_sample_solutions.append(sample_solution)

                logging.info(
                    f"[goal_conditioned_tabletop_configurations] Sampled {len(tabletop_sample_poses)} valid tabletop poses for contact mode: {contact_mode_name}"
                )

                sample_final_contact_modes[contact_mode_name] = tabletop_sample_poses

                for idx, (tabletop_pose, solution) in enumerate(
                    zip(tabletop_sample_poses, tabletop_sample_solutions)
                ):
                    for i in range(3):
                        random_translation = np.random.uniform(
                            -0.05, 0.05, size=(3,)
                        )
                        random_rotation = np.random.uniform(
                            -0.05, 0.05, size=(3,)
                        )
                        perturb_transform = RigidTransform(
                            RollPitchYaw(random_rotation), random_translation
                        )
                        trajectory_primitive = TrajectoryPrimitive(
                            f"TO_GOAL_{idx}_{i}",
                            contact_mode,
                            tabletop_pose.multiply(perturb_transform),
                            None,
                            self.config,
                            self.goal_pose,
                        )

                        # direct trajectories
                        self.trajectory_primitives[contact_mode.name].primitives.append(
                                trajectory_primitive
                            )
                        solution = self.ik_trajectory(
                            trajectory_primitive.trajectory, contact_mode=contact_mode
                        )
                        self.ik_solutions[contact_mode.name][
                            trajectory_primitive.primitive_name
                        ] = solution

                        # random perturbations around trajectories
                        if self.config.get("augment", False):
                            randomized_samples = self.trajectory_primitives[
                                contact_mode_name
                            ].generate_perturbed_trajectories(
                                trajectory_primitive, trajectory_primitive.primitive_name, 3
                            )
                            self.trajectory_primitives[contact_mode_name].primitives.extend(
                                randomized_samples
                            )

                            for sample in randomized_samples:
                                solution = self.ik_trajectory(
                                    sample.trajectory, contact_mode=contact_mode
                                )
                                self.ik_solutions[contact_mode.name][
                                    sample.primitive_name
                                ] = solution

        return sample_final_contact_modes

    def tabletop_trajectory_samples(
        self,
        bounding_box: np.ndarray,
        n_samples: int = 10,
        samples_per_trajectory: Optional[int] = None,
    ):
        """
        Samples n_samples trajectories on the table top, constrained to the bounding box and checks kinematic feasibility.
        In:
            bounding_box: np.ndarray (2,2), the bounding box of the sample region
            n_samples: int, the number of samples to generate
        """

        for sample_id in tqdm(range(n_samples)):

            for contact_mode in self.contact_modes:
                start_pose = sample_tabletop_pose(bounding_box, contact_mode, self.config)
                goal_pose = sample_tabletop_pose(bounding_box, contact_mode, self.config)
                trajectory_primitive = TrajectoryPrimitive(
                    f"TO_GOAL_INCREMENTAL_{sample_id}",
                    contact_mode,
                    start_pose,
                    None,
                    self.config,
                    goal_pose,
                    config_overwrite=(
                        {"num_steps": samples_per_trajectory}
                        if samples_per_trajectory is not None
                        else {}
                    ),
                )
                self.trajectory_primitives[contact_mode.name].primitives.append(
                    trajectory_primitive
                )
                solution = self.ik_trajectory(
                    trajectory_primitive.trajectory, contact_mode=contact_mode
                )
                self.ik_solutions[contact_mode.name][
                    trajectory_primitive.primitive_name
                ] = solution

                assert len(
                    self.trajectory_primitives[contact_mode.name].primitives[-1]
                ) == len(
                    self.ik_solutions[contact_mode.name][
                        trajectory_primitive.primitive_name
                    ]
                )


    def ik_trajectory(self, tp_traj, contact_mode=ContactMode()):
        """
        solves for the IK solution for each pose in the trajectory:
        """

        q_space_trajectory = []

        if self.config["verbose"]:
            print(len(q_space_trajectory))

        t = 0
        solutions = []
        n_invalid_solutions = 0

        for i, pose in enumerate(tp_traj):

            if len(solutions) != n_invalid_solutions and solutions[-1] is not None:
                solution = self.sample_ik(
                    pose, contact_mode=contact_mode, initial_guess=solutions[-1]
                )
            else:
                solution = self.sample_ik(pose, contact_mode=contact_mode)
            if solution is not None:
                solutions.append(solution)
                q_space_trajectory.append(solution)
            else:
                solutions.append(None)
                q_space_trajectory.append(None)
                if self.config["verbose"]:
                    print("No solution found for the pose.")
                n_invalid_solutions += 1

        return solutions

    def sample_ik(
        self,
        desired_pose,
        contact_mode: ContactMode = ContactMode(),
        initial_guess=None,
        visualizer=None,
    ):

        iiwa1_model = self.plant.GetModelInstanceByName("iiwa_1")
        iiwa2_model = self.plant.GetModelInstanceByName("iiwa_2")

        constrained_axis = None
        cube_contact_frame_neg = None
        cube_contact_frame_pos = None

        (
            cube_contact_frame_pos,
            cube_contact_frame_neg,
            constrained_axis,
            theta_bounds,
        ) = contact_mode.get_contact_frame_pos(self.plant)

        X_CG1 = cube_contact_frame_pos.CalcPoseInBodyFrame(self.plant_context)
        X_CG2 = cube_contact_frame_neg.CalcPoseInBodyFrame(self.plant_context)
        X_WG1 = desired_pose.multiply(X_CG1)
        X_WG2 = desired_pose.multiply(X_CG2)

        # solve for IK for both:
        # 1. Inverse Kinematics for iiwa_1
        ik_iiwa = InverseKinematics(self.plant, self.plant_context)

        # joint limits: (currently only for the universal joints to avoid collision)
        # Retrieve the positions of the Universal Joint angles
        theta1_iiwa1 = ik_iiwa.q()[
            self.plant.GetJointByName("universal_joint_iiwa1").position_start()
        ]
        theta2_iiwa1 = ik_iiwa.q()[
            self.plant.GetJointByName("universal_joint_iiwa1").position_start() + 1
        ]

        # constraints on the angles
        ik_iiwa.prog().AddBoundingBoxConstraint(-np.pi / 5, np.pi / 5, theta1_iiwa1)
        ik_iiwa.prog().AddBoundingBoxConstraint(-np.pi / 5, np.pi / 5, theta2_iiwa1)

        # iiwa_2
        theta1_iiwa2 = ik_iiwa.q()[
            self.plant.GetJointByName("universal_joint_iiwa2").position_start()
        ]
        theta2_iiwa2 = ik_iiwa.q()[
            self.plant.GetJointByName("universal_joint_iiwa2").position_start() + 1
        ]

        ik_iiwa.prog().AddBoundingBoxConstraint(-np.pi / 5, np.pi / 5, theta1_iiwa2)
        ik_iiwa.prog().AddBoundingBoxConstraint(-np.pi / 5, np.pi / 5, theta2_iiwa2)

        end_effector_frame_iiwa1 = self.plant.GetFrameByName(
            "contact_body_iiwa1", iiwa1_model
        )
        end_effector_frame_iiwa2 = self.plant.GetFrameByName(
            "contact_body_iiwa2", iiwa2_model
        )
        cube_frame = self.plant.GetFrameByName(
            "cuboid_body", self.plant.GetModelInstanceByName("movable_cuboid")
        )

        # Add position and orientation constraints for iiwa_1
        ik_iiwa.AddPositionConstraint(
            frameA=cube_contact_frame_pos,
            frameB=end_effector_frame_iiwa1,
            p_BQ=np.zeros(3),
            p_AQ_lower=-0.005 * np.ones(3),
            p_AQ_upper=0.005 * np.ones(3),
        )

        ik_iiwa.AddOrientationConstraint(
            frameAbar=self.plant.world_frame(),
            R_AbarA=X_WG1.rotation(),
            frameBbar=end_effector_frame_iiwa1,
            R_BbarB=RigidTransform().rotation(),
            theta_bound=0.05,
        )

        # Add position and orientation constraints for iiwa_2
        ik_iiwa.AddPositionConstraint(
            frameA=cube_contact_frame_neg,
            frameB=end_effector_frame_iiwa2,
            p_BQ=np.zeros(3),
            p_AQ_lower=-0.005 * np.ones(3),
            p_AQ_upper=0.005 * np.ones(3),
        )

        ik_iiwa.AddOrientationConstraint(
            frameAbar=self.plant.world_frame(),
            R_AbarA=X_WG2.rotation(),
            frameBbar=end_effector_frame_iiwa2,
            R_BbarB=RigidTransform().rotation(),
            theta_bound=0.05,
        )

        # Add a constraint to ensure dynamic feasbility with the cube
        ik_iiwa.AddPositionConstraint(
            frameB=cube_frame,
            p_BQ=np.array([0, 0, 0]),
            frameA=self.plant.world_frame(),
            p_AQ_lower=desired_pose.translation() - 0.005,
            p_AQ_upper=desired_pose.translation() + 0.005,
        )

        ik_iiwa.AddOrientationConstraint(
            frameAbar=self.plant.world_frame(),
            R_AbarA=desired_pose.rotation(),
            frameBbar=cube_frame,
            R_BbarB=RigidTransform().rotation(),
            theta_bound=0.005,
        )

        # add collision constraint:
        ik_iiwa.AddMinimumDistanceLowerBoundConstraint(0.001, 0.01)

        # add initial guess:
        if initial_guess is not None:
            ik_iiwa.prog().AddQuadraticErrorCost(
                0.1 * np.identity(len(ik_iiwa.q())), initial_guess, ik_iiwa.q()
            )
            ik_iiwa.prog().SetInitialGuess(ik_iiwa.q(), initial_guess)

        result_iiwa = Solve(ik_iiwa.prog())
        if result_iiwa.is_success():
            q_sol_iiwa = result_iiwa.GetSolution(ik_iiwa.q())
            if self.config["verbose"]:
                print("Solution found for iiwa:", q_sol_iiwa)
        else:
            if self.config["verbose"]:
                print("No solution found for iiwa.")
            return None

        return q_sol_iiwa

    def save_to_file(self, filename: str):
        out_structure = {
            mode.name: {
                primitive.primitive_name: (
                    primitive.trajectory,
                    self.ik_solutions[mode.name][primitive.primitive_name],
                )
                for primitive in self.trajectory_primitives[mode.name].primitives
            }
            for mode in self.contact_modes
        }
        with open(ROOT_DIR / "output" / filename, "wb") as f:
            pickle.dump(out_structure, f)

    def load_from_file(self, filename: str):
        with open(ROOT_DIR / "output" / filename, "rb") as f:
            out_structure = pickle.load(f)
            for mode in self.contact_modes:
                for primitive_name, (trajectory, solution) in out_structure[
                    mode.name
                ].items():
                    self.trajectory_primitives[mode.name].primitives.append(
                        TrajectoryPrimitive(
                            primitive_name,
                            mode,
                            trajectory[0],
                            trajectory,
                            self.config,
                            self.goal_pose,
                        )
                    )
                    self.ik_solutions[mode.name][primitive_name] = solution
