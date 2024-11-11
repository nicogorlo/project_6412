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
    MathematicalProgram,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    MultibodyPlant,
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

from manipulation.utils import RenderDiagram
from dual_arm_manipulation.utils import save_diagram, display_diagram
from manipulation.station import LoadScenario, MakeHardwareStation
from dual_arm_manipulation import ROOT_DIR
from manipulation.scenarios import AddShape
from manipulation.meshcat_utils import AddMeshcatTriad
import numpy as np
import time

def dual_arm_environment(cube=True):
    meshcat = StartMeshcat()
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    parser = Parser(plant)
    model_directive = f"""
    directives:
    - add_model:
        name: iiwa_1
        file: package://drake_models/iiwa_description/sdf/iiwa7_with_box_collision.sdf
        default_joint_positions:
            iiwa_joint_1: [0]
            iiwa_joint_2: [0]
            iiwa_joint_3: [0]
            iiwa_joint_4: [0]
            iiwa_joint_5: [0]
            iiwa_joint_6: [0]
            iiwa_joint_7: [0]
    - add_model:
        name: iiwa_2
        file: package://drake_models/iiwa_description/sdf/iiwa7_with_box_collision.sdf
        default_joint_positions:
            iiwa_joint_1: [0]
            iiwa_joint_2: [0]
            iiwa_joint_3: [0]
            iiwa_joint_4: [0]
            iiwa_joint_5: [0]
            iiwa_joint_6: [0]
            iiwa_joint_7: [0]
    - add_model:
        name: table_top
        file: "file://{ROOT_DIR}/assets/table_top.sdf"
    - add_model:
        name: movable_cuboid
        file: "file://{ROOT_DIR}/assets/cuboid.sdf"
    """
    directives = LoadModelDirectivesFromString(model_directive)
    ProcessModelDirectives(directives, plant, parser)

    iiwa1_model = plant.GetModelInstanceByName("iiwa_1")
    iiwa2_model = plant.GetModelInstanceByName("iiwa_2")
    table_top_model = plant.GetModelInstanceByName("table_top")
    cuboid_model = plant.GetModelInstanceByName("movable_cuboid")


    X_iiwa1 = RigidTransform([0.7, 0.0, 0.0])
    X_iiwa2 = RigidTransform([-0.7, 0.0, 0.0])

    # Weld the iiwa robots to the world at the specified positions
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("iiwa_link_0", plant.GetModelInstanceByName("iiwa_1")), X_iiwa1)
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("iiwa_link_0", plant.GetModelInstanceByName("iiwa_2")), X_iiwa2)

    table_top_body = plant.GetBodyByName("table_top_body", table_top_model)
    cuboid_body = plant.GetBodyByName("cuboid_body", cuboid_model)
    if not cube:
        plant.WeldFrames(
            plant.world_frame(),
            cuboid_body.body_frame(),
            RigidTransform([1e6, 0, 0.])
        )
    plant.WeldFrames(
        plant.world_frame(),
        table_top_body.body_frame(),
        RigidTransform([0, 0, -0.05])
    )

    visualizer = MeshcatVisualizer.AddToBuilder(
        builder, scene_graph, meshcat
    )
    print(meshcat.web_url())

    return builder, plant, scene_graph, meshcat

def task_space_trajectory(radius = 0.2, p0 = [0., 0., 0.6], rpy = [np.pi/4, 0., 0.], N=100):

    # Create a RollPitchYaw object
    rpy = RollPitchYaw(rpy[0], rpy[1], rpy[2])

    # Get the rotation matrix
    R0 = rpy.ToRotationMatrix()


    # Get the centre pose
    X_WCenter = RigidTransform(R0, p0)

    # Create a circular trajectory
    thetas = np.linspace(-np.pi/2,  np.pi/2, N)

    key_frame_poses_in_world = []
    for theta in thetas:
        R = RotationMatrix.MakeZRotation(theta) 
        pf = np.array([radius, 0., 0])
        X_CC = RigidTransform(R, R@pf)
        X_WCC = X_WCenter.multiply(X_CC)
        key_frame_poses_in_world.append(X_WCC)

    return key_frame_poses_in_world

def visualise_trajectory(visualizer, poses):
    for i, pose in enumerate(poses):
        AddMeshcatTriad(
            visualizer, "box_traj/" + "frame_{}".format(i), length=0.1, radius=0.006, X_PT=pose
        )

def sample_ik(plant, plant_context, desired_pose, initial_guess=None):
    iiwa1_model = plant.GetModelInstanceByName("iiwa_1")
    iiwa2_model = plant.GetModelInstanceByName("iiwa_2")

    

    # Define the transform between the object pose and the desired gripper pose:
    p_CG1 = [0.2, 0, 0]
    p_CG2 = [-0.2, 0, 0]
    R_CG1 = RollPitchYaw(0., -np.pi/2, 0).ToRotationMatrix()
    R_CG2 = RollPitchYaw(0, np.pi/2, 0.).ToRotationMatrix()

    X_CG1 = RigidTransform(R_CG1, p_CG1)
    X_CG2 = RigidTransform(R_CG2, p_CG2)
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
        ik_iiwa.prog().AddQuadraticErrorCost(0.01*np.identity(len(ik_iiwa.q())), initial_guess, ik_iiwa.q())
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


def main():
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


    # Define desired end-effector poses
    tp_traj = task_space_trajectory(radius=0., p0=[0,0, 0.3], rpy=[0, np.pi/6, 0], N=100)

    # visualise task-space trajectory
    visualise_trajectory(visualizer, tp_traj)

    visualizer.StartRecording()
    # Now solve for the IK for each pose in the trajectory:
    q_space_trajectory = []
    print(len(q_space_trajectory))
    t = 0
    valid_solutions = []
    for i, pose in enumerate(tp_traj):
        if len(valid_solutions) != 0:
            solution = sample_ik(plant, plant_context, pose, valid_solutions[-1])
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

    # neccessary for visualisation:
    visualizer.PublishRecording()
    while True:
        pass

if __name__ == "__main__":
    main()