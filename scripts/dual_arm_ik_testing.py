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
    JointSliders,
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
import numpy as np
import time



def main():
    meshcat = StartMeshcat()
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    parser = Parser(plant)
    model_directive = f"""
    directives:
    - add_model:
        name: iiwa_1
        file: package://drake_models/iiwa_description/sdf/iiwa7_no_collision.sdf
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
        file: package://drake_models/iiwa_description/sdf/iiwa7_no_collision.sdf
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

    plant.WeldFrames(
        plant.world_frame(),
        table_top_body.body_frame(),
        RigidTransform([0, 0, -0.05])
    )

    visualizer = MeshcatVisualizer.AddToBuilder(
        builder, scene_graph, meshcat
    )
    print(meshcat.web_url())

    diagram = builder.Build()

    plant.Finalize()

    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    plant_context = plant.GetMyMutableContextFromRoot(context)

    q0 = plant.GetPositions(plant_context, plant.GetModelInstanceByName("iiwa_1"))
    q1 = plant.GetPositions(plant_context, plant.GetModelInstanceByName("iiwa_2"))

    print(q0)
    print(q1)

    # Confirm that simulation works:

    # Get model instances
    iiwa1_model = plant.GetModelInstanceByName("iiwa_1")
    iiwa2_model = plant.GetModelInstanceByName("iiwa_2")
    context = plant.CreateDefaultContext()  # Create a context for the plant

    # Define desired end-effector poses
    target_pose_iiwa1 = RigidTransform(
        RollPitchYaw(0., 0., 0.),  # Set desired orientation for iiwa_1
        [0., 0., 1.0]                 # Set desired position for iiwa_1
    )
    
    target_pose_iiwa2 = RigidTransform(
        RollPitchYaw(0., 0., 0.), # Set desired orientation for iiwa_2
        [0., -0., 1.]               # Set desired position for iiwa_2
    )

    # 1. Inverse Kinematics for iiwa_1
    ik_iiwa1 = InverseKinematics(plant)
    end_effector_frame_iiwa1 = plant.GetFrameByName("iiwa_link_7", iiwa1_model)  # Adjust to the correct end-effector frame

    # Add position and orientation constraints for iiwa_1
    ik_iiwa1.AddPositionConstraint(
        frameB=end_effector_frame_iiwa1,
        p_BQ=np.array([0, 0, 0]),
        frameA=plant.world_frame(),
        p_AQ_lower=target_pose_iiwa1.translation() - 0.1,
        p_AQ_upper=target_pose_iiwa1.translation() + 0.1
    )
    ik_iiwa1.AddOrientationConstraint(
        frameAbar=plant.world_frame(),
        R_AbarA=target_pose_iiwa1.rotation(),
        frameBbar=end_effector_frame_iiwa1,
        R_BbarB=RigidTransform().rotation(),
        theta_bound=0.1
    )

    # Solve for iiwa_1
    result_iiwa1 = Solve(ik_iiwa1.prog())
    if result_iiwa1.is_success():
        q_sol_iiwa1 = result_iiwa1.GetSolution(ik_iiwa1.q()[0:7])
        print("Solution found for iiwa_1:", q_sol_iiwa1)
    else:
        print("No solution found for iiwa_1.")

    # 2. Inverse Kinematics for iiwa_2
    ik_iiwa2 = InverseKinematics(plant)
    end_effector_frame_iiwa2 = plant.GetFrameByName("iiwa_link_7", iiwa2_model)  # Adjust to the correct end-effector frame

    # Add position and orientation constraints for iiwa_2
    ik_iiwa2.AddPositionConstraint(
        frameB=end_effector_frame_iiwa2,
        p_BQ=np.array([0, 0, 0]),
        frameA=plant.world_frame(),
        p_AQ_lower=target_pose_iiwa2.translation() - 0.1,
        p_AQ_upper=target_pose_iiwa2.translation() + 0.1
    )
    ik_iiwa2.AddOrientationConstraint(
        frameAbar=plant.world_frame(),
        R_AbarA=target_pose_iiwa2.rotation(),
        frameBbar=end_effector_frame_iiwa2,
        R_BbarB=RigidTransform().rotation(),
        theta_bound=0.1
    )

    # Solve for iiwa_2
    result_iiwa2 = Solve(ik_iiwa2.prog())
    if result_iiwa2.is_success():
        q_sol_iiwa2 = result_iiwa2.GetSolution(ik_iiwa2.q()[7:14])
        print("Solution found for iiwa_2:", q_sol_iiwa2)
    else:
        print("No solution found for iiwa_2.")

    if result_iiwa1.is_success():
        q_sol_iiwa1 = np.array(q_sol_iiwa1).reshape(-1, 1)
    else:
        q_sol_iiwa1 = np.zeros((7, 1))

    if result_iiwa2.is_success():
        q_sol_iiwa2 = np.array(q_sol_iiwa2).reshape(-1, 1)
    else:
        q_sol_iiwa2 = np.zeros((7, 1))
    
    # station.GetInputPort("iiwa_left.position").FixValue(station_context, q_sol_iiwa1)
    # station.GetInputPort("iiwa_right.position").FixValue(station_context, q_sol_iiwa2)

    def show_end_effector_box():
        vspace = Box(0.6, 2.0, 1.0)
        meshcat.SetObject("/end_effector_box", vspace, Rgba(0.1, 0.5, 0.1, 0.2))
        meshcat.SetTransform("/end_effector_box", RigidTransform([0.0, 0.0, 0.5]))
    
    plant.SetPositions(plant_context, iiwa1_model, q_sol_iiwa1)
    plant.SetPositions(plant_context, iiwa2_model, q_sol_iiwa2)
    
    meshcat.StartRecording()
    # show_end_effector_box()
    simulator.AdvanceTo(0.01)
    meshcat.PublishRecording()
    while True:
        pass

if __name__ == "__main__":
    main()
