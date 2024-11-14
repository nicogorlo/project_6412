''' TODO:
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

from pydrake.geometry.optimization import GraphOfConvexSetsOptions, HPolyhedron, Point # type: ignore
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

from dual_arm_manipulation import ROOT_DIR
from dual_arm_manipulation.environment import dual_arm_environment
from dual_arm_manipulation.utils import interpolate_6dof_poses, get_free_faces, pose_vec_to_transform, rotation_matrix_from_vectors
import yaml


def task_space_trajectory(radius = 0.1, p0 = [0., 0., 0.6], rpy = [np.pi/4, 0., 0.], N=100):

    # Create a RollPitchYaw object
    rpy = RollPitchYaw(rpy[0], rpy[1], rpy[2])

    # Get the rotation matrix
    R0 = rpy.ToRotationMatrix()


    # Get the centre pose
    X_WCenter = RigidTransform(R0, p0)

    # Create a circular trajectory
    thetas = np.linspace(-np.pi,  np.pi, N)

    key_frame_poses_in_world = []
    for theta in thetas:
        R = RotationMatrix.MakeXRotation(theta) 
        pf = np.array([0., 0., radius])
        X_CC = RigidTransform(R, R@pf)
        X_WCC = X_WCenter.multiply(X_CC)
        key_frame_poses_in_world.append(X_WCC)

    return key_frame_poses_in_world

def visualise_trajectory(visualizer, poses):
    for i, pose in enumerate(poses):
        AddMeshcatTriad(
            visualizer, "box_traj/" + "frame_{}".format(i), length=0.1, radius=0.006, X_PT=pose
        )

def sample_ik(plant, plant_context, desired_pose, contact_mode = [[0.22, 0, 0], [-0.22, 0, 0]], initial_guess=None):
    iiwa1_model = plant.GetModelInstanceByName("iiwa_1")
    iiwa2_model = plant.GetModelInstanceByName("iiwa_2")

    # Define the transform between the object pose and the desired gripper pose:
    p_CG1 = contact_mode[0]
    p_CG2 = contact_mode[1]
    R_CG1 = rotation_matrix_from_vectors(np.array(p_CG1), np.array([0,0,1]))
    R_CG2 = rotation_matrix_from_vectors(np.array(p_CG2), np.array([0,0,1]))

    X_CG1 = RigidTransform(R_CG1, np.array(p_CG1))
    X_CG2 = RigidTransform(R_CG2, np.array(p_CG2))
    X_WG1 = desired_pose.multiply(X_CG1)
    X_WG2 = desired_pose.multiply(X_CG2)

    # solve for IK for both:
    # 1. Inverse Kinematics for iiwa_1
    ik_iiwa = InverseKinematics(plant, plant_context)
    end_effector_frame_iiwa1 = plant.GetFrameByName("iiwa_link_7", iiwa1_model)  # Adjust to the correct end-effector frame
    end_effector_frame_iiwa2 = plant.GetFrameByName("iiwa_link_7", iiwa2_model)  # Adjust to the correct end-effector frame
    cube_frame = plant.GetFrameByName("cuboid_body", plant.GetModelInstanceByName("movable_cuboid"))

    # Add position and orientation constraints for iiwa_1
    ik_iiwa.AddPositionConstraint(
        frameB=end_effector_frame_iiwa1,
        p_BQ=np.array([0, 0, 0]),
        frameA=plant.world_frame(),
        p_AQ_lower=X_WG1.translation() - 0.005,
        p_AQ_upper=X_WG1.translation() + 0.005
    )

    ik_iiwa.AddOrientationConstraint(
        frameAbar=plant.world_frame(),
        R_AbarA=X_WG1.rotation(),
        frameBbar=end_effector_frame_iiwa1,
        R_BbarB=RigidTransform().rotation(),
        theta_bound=0.005
    )
    ik_iiwa.AddPositionConstraint(
        frameB=end_effector_frame_iiwa2,
        p_BQ=np.array([0, 0, 0]),
        frameA=plant.world_frame(),
        p_AQ_lower=X_WG2.translation() - 0.005,
        p_AQ_upper=X_WG2.translation() + 0.005
    )
    ik_iiwa.AddOrientationConstraint(
        frameAbar=plant.world_frame(),
        R_AbarA=X_WG2.rotation(),
        frameBbar=end_effector_frame_iiwa2,
        R_BbarB=RigidTransform().rotation(),
        theta_bound=0.005
    )

    # Add a constraint to ensure dynamic feasbility with the cube
    ik_iiwa.AddPositionConstraint(
        frameB=cube_frame,
        p_BQ=np.array([0, 0, 0]),
        frameA=plant.world_frame(),
        p_AQ_lower=desired_pose.translation() - 0.005,
        p_AQ_upper=desired_pose.translation() + 0.005
    )

    ik_iiwa.AddOrientationConstraint(
        frameAbar=plant.world_frame(),
        R_AbarA=desired_pose.rotation(),
        frameBbar=cube_frame,
        R_BbarB=RigidTransform().rotation(),
        theta_bound=0.005
    )

    # add collision constraint:
    ik_iiwa.AddMinimumDistanceLowerBoundConstraint(0.001, 0.01)

    # add initial guess:
    if initial_guess is not None:
        ik_iiwa.prog().AddQuadraticErrorCost(0.1*np.identity(len(ik_iiwa.q())), initial_guess, ik_iiwa.q())
        ik_iiwa.prog().SetInitialGuess(ik_iiwa.q(), initial_guess)

    result_iiwa = Solve(ik_iiwa.prog())
    if result_iiwa.is_success():
        q_sol_iiwa = result_iiwa.GetSolution(ik_iiwa.q())
        print("Solution found for iiwa:", q_sol_iiwa)
    else:
        print("No solution found for iiwa.")
        return None

    return q_sol_iiwa


def get_geometry_set(plant, name):
    model_instance = plant.GetModelInstanceByName(name)
    # Initialize an empty list to collect geometry IDs
    frame_ids = []

    # Loop through each body in the model instance
    for body in plant.GetBodyIndices(model_instance):
        body = plant.get_body(body)  # Retrieve the body object
        # Get the frame ID associated with the body
        frame_id = plant.GetBodyFrameIdOrThrow(body.index())
        # Add the geometry IDs to the list
        frame_ids.append(frame_id)

    # Optionally, create a GeometrySet with these IDs
    frame_set = GeometrySet(frame_ids)
    return frame_set

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

def AnimateBall(root_diagram: Diagram,
    root_context: Context,
    plant: MultibodyPlant,
    E: HPolyhedron,
    speed: float,
    meshcat: MeshcatVisualizer):
    
    plant_context = plant.GetMyContextFromRoot(root_context)
    q = E.center()
    plant.SetPositions(plant_context, plant.GetModelInstanceByName("movable_cuboid"),  q)
    root_diagram.ForcedPublish(root_context)

    print("Press the 'Stop Animation' button in Meshcat to continue.")
    meshcat.AddButton("Stop Animation", "Escape")

    solutions = []
    while meshcat.GetButtonClicks("Stop Animation") < 1:
        
        # compute the next point by sammpling witin the affine ball:
        u = np.random.randn(7)
        if np.linalg.norm(u) > 1:
            u = u / np.linalg.norm(u)
        q_next = E.center() + E.B() @ u

        # Animate between q and q_next (at speed):
        # TODO: normalize step size to speed... e.g. something like
        # 20 * np.linalg.norm(q_next - q) / speed)
        solutions = []
        for t in np.append(np.arange(0, 1, 0.05), 1):
            qs = t * q_next + (1 - t) * q

            pose = pose_vec_to_transform(qs)

            if len(solutions) != 0:
                solution = sample_ik(plant, plant_context, pose, initial_guess=solutions[-1])
            else:
                solution = sample_ik(plant, plant_context, pose)
            
            if solution is None:
                continue
            
            plant.SetPositions(plant_context, solution)
            # plant.SetPositions(plant_context, plant.GetModelInstanceByName("movable_cuboid"),  qs)
            root_diagram.ForcedPublish(root_context)
            time.sleep(0.05)

        q = q_next

    meshcat.DeleteButton("Stop Animation")


def get_convex_set(tp_traj, plant, plant_context, simulator, t):
    # Now solve for the IK for each pose in the trajectory:
    q_space_trajectory = []
    print(len(q_space_trajectory))
    t = 0
    valid_solutions = []
    for i, pose in enumerate(tp_traj):
        if len(valid_solutions) != 0:
            solution = sample_ik(plant, plant_context, pose, initial_guess=valid_solutions[-1])
        else:
            solution = sample_ik(plant, plant_context, pose)
        if solution is not None:
            valid_solutions.append(solution)
            q_space_trajectory.append(solution)
            # plant.SetPositions(plant_context, plant.GetModelInstanceByName("iiwa_1"), solution[0:7])
            # plant.SetPositions(plant_context, plant.GetModelInstanceByName("iiwa_2"), solution[7:14])
            plant.SetPositions(plant_context, solution)
            try:
                simulator.AdvanceTo(0.01*(t+1))
            except:
                print("Simulation failed at t = ", t)
            t += 1
        else:
            print("No solution found for the pose.")

    # MinimumVolumeCircumscribedEllipsoid
    valid_solutions = np.array(valid_solutions).T
    
    task_space_valid_solutions = valid_solutions[14:, :]
    print(task_space_valid_solutions.shape)
    E = AffineBall.MinimumVolumeCircumscribedEllipsoid(task_space_valid_solutions)

    print(E.center())
    print(E.B())
    # Assuming `model_instance` is a valid model instance for the cube or robot.
    plant.SetPositions(plant_context, plant.GetModelInstanceByName("movable_cuboid"),  E.center())

    return E

def get_goal_conditioned_tabletop_configurations(goal_pose, contact_modes, n_tabletop_config_per_side, plant, plant_context, simulator, scene_graph, diagram, visualizer, t):

    sample_final_contact_modes = {}

    AddMeshcatTriad(
            visualizer, "goal_pose", length=0.1, radius=0.006, X_PT=goal_pose
        )
    
    t=100

    for contact_mode_name, contact_mode in contact_modes.items():
        # determine if IK solution exists for contact mode in goal configuration
        solution = sample_ik(plant, plant_context, goal_pose, contact_mode=contact_mode)

        tabletop_sample_poses = []
        

        if solution is not None:
            plant.SetPositions(plant_context, solution)
            # get tabletop configurations that work for the contact mode
            # sample poses in an ellipsoid between the goal pose and tabletop configurations to get a convex set
            for face_name, face_pos in get_free_faces(contact_mode_name, contact_modes).items():
                face_normal = face_pos / np.linalg.norm(face_pos)

                # get poses s.t. face is in contact with the tabletop surface 
                tabletop_rotation = rotation_matrix_from_vectors(np.array([0, 0, 1]), np.array(-face_normal))
                tabletop_translation = np.array([0, 0, 1]) * np.linalg.norm(face_pos)
                for angle in np.linspace(0, 2*np.pi, n_tabletop_config_per_side):
                    rotation_sample_sol = RotationMatrix.MakeZRotation(angle)
                    rotation_sample_sol = rotation_sample_sol.multiply(tabletop_rotation)
                    tabletop_sample_pose = RigidTransform(rotation_sample_sol, tabletop_translation)

                    AddMeshcatTriad(
                        visualizer, "goal_pose", length=0.1, radius=0.006, X_PT=tabletop_sample_pose
                    )
                    
                    sample_solution = sample_ik(plant, plant_context, tabletop_sample_pose, contact_mode=contact_mode)

                    if sample_solution is None:
                        continue
                    plant.SetPositions(plant_context, solution)
                    try:
                        simulator.AdvanceTo(0.01*(t+1))
                    except:
                        print("Simulation failed at t = ", t)

                    tabletop_sample_poses.append(sample_solution)
                    t+=1

                # sample poses in an ellipsoid between the goal pose and valid tabletop configurations to get a convex set
                tp_traj = None

            sample_final_contact_modes[contact_mode_name] = tabletop_sample_poses

    return sample_final_contact_modes


def main():

    t=0

    with open(ROOT_DIR + "/config/config.yaml", "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    builder, plant, scene_graph, visualizer = dual_arm_environment(cube=True)

    diagram = builder.Build()
    plant.Finalize()

    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    plant_context = plant.GetMyMutableContextFromRoot(context)

    diagram_context = diagram.CreateDefaultContext()
    scene_graph_context = scene_graph.GetMyMutableContextFromRoot(diagram_context)

    iiwa1_set = get_geometry_set(plant, "iiwa_1")
    iiwa2_set = get_geometry_set(plant, "iiwa_2")
    cuboid_set = get_geometry_set(plant, "movable_cuboid")

    num_positions = plant.num_positions(plant.GetModelInstanceByName("movable_cuboid"))
    print(f"Number of positions: {num_positions}")

    goal_pose = pose_vec_to_transform(cfg["eval"]["goal_pose"])
    contact_modes = cfg["planner"]["contact_modes"]
    n_rotations = cfg["planner"]["tabletop_configurations"]["n_rotations"]
    sample_final_contact_modes = get_goal_conditioned_tabletop_configurations(goal_pose, contact_modes, n_rotations, plant, plant_context, simulator, scene_graph, diagram, visualizer, t)

    # Define desired end-effector poses
    tp_traj = task_space_trajectory(radius=0.2, p0=[0,0, 0.2], rpy=[0, 0., 0], N=100)

    # visualise task-space trajectory
    visualise_trajectory(visualizer, tp_traj)

    visualizer.StartRecording()
    E = get_convex_set(tp_traj, plant, plant_context, simulator, t)
    diagram.ForcedPublish(context)
    # plant.SetPositions(plant_context, plant.GetModelInstanceByName("movable_cuboid"), E.center())
    AnimateBall(diagram, context, plant, E, 0.1, visualizer)


    # neccessary for visualisation:
    visualizer.PublishRecording()
    while True:
        pass

if __name__ == "__main__":
    main()