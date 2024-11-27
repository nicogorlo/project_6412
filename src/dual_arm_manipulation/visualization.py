
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    Box,
    ConstantVectorSource,
    ContactVisualizer,
    ContactVisualizerParams,
    DiagramBuilder,
    Diagram,
    Context,
    RpyFloatingJoint,
    DiscreteContactApproximation,
    FixedOffsetFrame,
    InverseDynamicsController,
    InverseKinematics,
    CollisionFilterDeclaration,
    JointSliders,
    Integrator,
    JacobianWrtVariable,
    LeafSystem,
    LogVectorOutput,
    AffineBall,
    MathematicalProgram,
    MeshcatVisualizer,
    HPolyhedron,
    Hyperellipsoid,
    IrisInConfigurationSpace,
    IrisOptions,
    MeshcatVisualizerParams,
    MultibodyPlant,
    UniversalJoint,
    MultibodyPositionToGeometryPose,
    Multiplexer,
    OsqpSolver,
    Parser,
    GeometrySet,
    PiecewisePolynomial,
    PlanarJoint,
    PrismaticJoint,
    RevoluteJoint,
    Rgba,
    RigidTransform,
    RotationMatrix,
    RollPitchYaw,
    Quaternion,
    SceneGraph,
    Simulator,
    SpatialInertia,
    Sphere,
    StartMeshcat,
    TrajectorySource,
    UnitInertia,   
    SpatialVelocity,
    RollPitchYaw,
    PiecewisePose,
    ProximityProperties,
    CoulombFriction,
    ModelInstanceIndex,
    LoadModelDirectivesFromString, 
    ProcessModelDirectives,
    StartMeshcat, 
    Solve,
    AddContactMaterial, 
    yaml_load_typed
)

from scipy.spatial import ConvexHull

from manipulation.utils import RenderDiagram
from dual_arm_manipulation.utils import save_diagram, display_diagram
from manipulation.station import LoadScenario, MakeHardwareStation
from manipulation.scenarios import AddShape
from manipulation.meshcat_utils import AddMeshcatTriad
from typing import NamedTuple
import numpy as np
import time
import logging
from dual_arm_manipulation import ROOT_DIR
from dual_arm_manipulation.environment import dual_arm_environment
from dual_arm_manipulation.contact_mode import ContactMode
from dual_arm_manipulation.trajectory_primitives import TrajectoryPrimitives, TrajectoryPrimitive
from dual_arm_manipulation.planner import GCSPlanner
from dual_arm_manipulation.utils import interpolate_6dof_poses, get_free_faces, pose_vec_to_transform, rotation_matrix_from_vectors
import yaml


def visualise_trajectory_poses(visualizer: MeshcatVisualizer, poses: list[RigidTransform]):
    """
    Visualize the trajectory poses of a trajectory primitive.

    Parameters:
        visualizer (MeshcatVisualizer): The meshcat visualizer.
        tp_traj (TrajectoryPrimitive): The trajectory primitive.
    """
    for i, pose in enumerate(poses):
        AddMeshcatTriad(
            visualizer, "box_traj/" + "frame_{}".format(i), length=0.1, radius=0.006, X_PT=pose
        )


def visualize_sample_trajectories(plant: MultibodyPlant, plant_context: Context, root_diagram: Diagram, root_context: Context, contact_mode: ContactMode, planner: GCSPlanner, simulator: Simulator, visualizer: MeshcatVisualizer):
    

    for tp_traj, solutions in zip(planner.trajectory_primitives[contact_mode.name], planner.ik_solutions[contact_mode.name]):

        visualise_trajectory_poses(visualizer, tp_traj.trajectory)
            
        for i, (tp_pose, q) in enumerate(zip(tp_traj, solutions)):
            if q is None:
                print(f"Skipping {i}, no solution here.")
                continue
            
            plant.SetPositions(plant_context, q)
            root_diagram.ForcedPublish(root_context)

            time.sleep(0.05)