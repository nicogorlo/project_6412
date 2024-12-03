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
from dual_arm_manipulation.visualization import visualize_sample_trajectories, visualise_trajectory_poses
from dual_arm_manipulation.utils import interpolate_6dof_poses, get_free_faces, pose_vec_to_transform, rotation_matrix_from_vectors
import yaml


def main():

    t=0

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

    try:
        simulator.AdvanceTo(0)
    except:
        print("Simulation failed.")

    num_positions = plant.num_positions(plant.GetModelInstanceByName("movable_cuboid"))
    print(f"Number of positions: {num_positions}")
    start_pose = pose_vec_to_transform(cfg["eval"]["start_pose"])
    goal_pose = pose_vec_to_transform(cfg["eval"]["goal_pose"])

    AddMeshcatTriad(
            visualizer, "goal_pose", length=0.1, radius=0.006, X_PT=goal_pose
        )
    
    contact_mode_names = cfg["sampler"]["contact_modes"].keys()
    contact_modes = [ContactMode(name, cfg) for name in contact_mode_names]
    n_rotations = cfg["sampler"]["tabletop_configurations"]["n_rotations"]

    sampler = PrimitiveSampler(plant, plant_context, start_pose, goal_pose, contact_modes, simulate=False, config_path=ROOT_DIR / "config" / "config.yaml")
    sampler.tabletop_trajectory_samples(np.array([[-0.3, 0.3],[-0.5, 0.5]]), 100, 30)
    
    sampler.save_to_file("trajectories_tabletop.pkl")
    sampler.load_from_file("trajectories_tabletop.pkl")
        
    sampler.get_traj_primitives()
    sampler.get_goal_conditioned_tabletop_configurations()

    sampler.save_to_file("trajectories_primitives.pkl")

    for contact_mode in contact_modes:
        assert len(sampler.trajectory_primitives[contact_mode.name].primitives) == len(sampler.ik_solutions[contact_mode.name])
        visualize_sample_trajectories(plant, plant_context, diagram, context, contact_mode, sampler, simulator, visualizer)

    print("done.")

if __name__ == "__main__":
    main()