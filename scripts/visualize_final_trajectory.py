
'''
Run GCS to compute a collision-free trajectory in the configuration space of the two robots
- set up the environment with two iiwa robots
- set up the GCS problem
- solve the GCS problem
- visualize the trajectory
'''

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

import pickle

from pydrake.geometry.optimization import GraphOfConvexSetsOptions, HPolyhedron, Point, ConvexHull # type: ignore
from pydrake.planning import GcsTrajectoryOptimization
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
from dual_arm_manipulation.sampler import PrimitiveSampler
from dual_arm_manipulation.set_creation import SetGen, Node
from dual_arm_manipulation.visualization import visualize_sample_trajectories, visualise_trajectory_poses, visualize_result, animate_sets
from dual_arm_manipulation.utils import interpolate_6dof_poses, pose_vec_to_transform, rotation_matrix_from_vectors
import yaml


def main():

    scenario = "primitives_large_scale" # tabletop, primitives, full, primitives_high_coverage

    with open(ROOT_DIR / "config" / "config.yaml", "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    builder, plant, scene_graph, visualizer = dual_arm_environment(cube=True)

    diagram = builder.Build()
    plant.Finalize()

    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    plant_context = plant.GetMyMutableContextFromRoot(context)

    diagram_context = diagram.CreateDefaultContext()
    scene_graph_context = scene_graph.GetMyMutableContextFromRoot(diagram_context)

    contact_mode_names = cfg["sampler"]["contact_modes"].keys()
    contact_modes = [ContactMode(name, cfg) for name in contact_mode_names]
    n_rotations = cfg["sampler"]["tabletop_configurations"]["n_rotations"]

    with open(ROOT_DIR / "output" / "linear_traj5.pkl", "rb") as f:
        linear_traj_7d = pickle.load(f)

    linear_traj = [pose_vec_to_transform(pose) for pose in linear_traj_7d]
    
    # TODO: LOAD REAL contact_mode_per_sample
    contact_mode_per_sample = len(linear_traj) * ['Y_NEG']
    contact_mode_per_sample[10:] = ['X_POS'] * len(contact_mode_per_sample[10:])

    AddMeshcatTriad(
            visualizer, "goal_pose", length=0.1, radius=0.02, X_PT=linear_traj[-1]
        )
    
    AddMeshcatTriad(
            visualizer, "start_pose", length=0.1, radius=0.02, X_PT=linear_traj[0]
        )

    visualize_result(plant, plant_context, diagram, context, contact_modes, linear_traj, contact_mode_per_sample, visualizer)


    print("Done.")

if __name__ == "__main__":
    main()