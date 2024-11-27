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
from dual_arm_manipulation.trajectory_primitives import TrajectoryPrimitives, TrajectoryPrimitive
from dual_arm_manipulation.planner import GCSPlanner
from dual_arm_manipulation.visualization import visualize_sample_trajectories, visualise_trajectory_poses
from dual_arm_manipulation.utils import interpolate_6dof_poses, get_free_faces, pose_vec_to_transform, rotation_matrix_from_vectors
import yaml

def AnimateIris(
    root_diagram: Diagram,
    root_context: Context,
    plant: MultibodyPlant,
    region: HPolyhedron,
    speed: float,
    meshcat: MeshcatVisualizer,
):
    """
    A simple hit-and-run-style idea for visualizing the IRIS regions:
    1. Start at the center. Pick a random direction and run to the boundary.
    2. Pick a new random direction; project it onto the current boundary, and run along it. Repeat
    """

    plant_context = plant.GetMyContextFromRoot(root_context)

    q = region.ChebyshevCenter()
    plant.SetPositions(plant_context, q)
    root_diagram.ForcedPublish(root_context)

    print("Press the 'Stop Animation' button in Meshcat to continue.")
    meshcat.AddButton("Stop Animation", "Escape")

    rng = np.random.default_rng()
    nq = plant.num_positions()
    prog = MathematicalProgram()
    qvar = prog.NewContinuousVariables(nq, "q")
    prog.AddLinearConstraint(region.A(), 0 * region.b() - np.inf, region.b(), qvar)
    cost = prog.AddLinearCost(np.ones((nq, 1)), qvar)

    while meshcat.GetButtonClicks("Stop Animation") < 1:
        direction = rng.standard_normal(nq)
        cost.evaluator().UpdateCoefficients(direction)

        result = Solve(prog)
        assert result.is_success()

        q_next = result.GetSolution(qvar)

        # Animate between q and q_next (at speed):
        # TODO: normalize step size to speed... e.g. something like
        # 20 * np.linalg.norm(q_next - q) / speed)
        for t in np.append(np.arange(0, 1, 0.05), 1):
            qs = t * q_next + (1 - t) * q

            
            plant.SetPositions(plant_context, qs)
            root_diagram.ForcedPublish(root_context)
            time.sleep(0.05)

        q = q_next

    meshcat.DeleteButton("Stop Animation")


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
    
    contact_mode_names = cfg["planner"]["contact_modes"].keys()
    contact_modes = [ContactMode(name, cfg) for name in contact_mode_names]
    n_rotations = cfg["planner"]["tabletop_configurations"]["n_rotations"]

    planner = GCSPlanner(plant, plant_context, start_pose, goal_pose, contact_modes, simulate=False, config_path=ROOT_DIR / "config" / "config.yaml")
    planner.get_traj_primitives()
    planner.get_goal_conditioned_tabletop_configurations()
    
    for contact_mode in contact_modes:
        visualize_sample_trajectories(plant, plant_context, diagram, context, contact_mode, simulator, visualizer)

    print("done.")

if __name__ == "__main__":
    main()