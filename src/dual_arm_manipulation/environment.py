from pydrake.all import (
    DiagramBuilder, MeshcatVisualizer, MultibodyPlant, Parser,
    RigidTransform, RollPitchYaw, SpatialInertia, UnitInertia,
    CoulombFriction, Box, SceneGraph, ProximityProperties,
    InverseDynamicsController, ModelInstanceIndex,
    LoadModelDirectivesFromString, ProcessModelDirectives,
    StartMeshcat, AddMultibodyPlantSceneGraph,
    AddContactMaterial, yaml_load_typed
)
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
    RigidTransform,
    RotationMatrix,
    SceneGraph,
    Simulator,
    SpatialInertia,
    Sphere,
    StartMeshcat,
    TrajectorySource,
    UnitInertia,    
    UniversalJoint
)
from manipulation.station import LoadScenario, MakeHardwareStation
from dual_arm_manipulation import ROOT_DIR
from manipulation.scenarios import AddShape


def AddPointFinger(plant, name):
    finger = AddShape(plant, Sphere(0.01), name, color=[0.9, 0.5, 0.5, 1.0])
    false_body1 = plant.AddRigidBody(
        "false_body1",
        finger,
        SpatialInertia(0, [0, 0, 0], UnitInertia(0, 0, 0)),
    )
    false_body2 = plant.AddRigidBody(
        "false_body2",
        finger,
        SpatialInertia(0, [0, 0, 0], UnitInertia(0, 0, 0)),
    )

    finger_x = plant.AddJoint(
        PrismaticJoint(
            "finger_x",
            plant.world_frame(),
            plant.GetFrameByName("false_body1", finger),
            [1, 0, 0],
            -0.3,
            0.3,
        )
    )
    plant.AddJointActuator("finger_x", finger_x)

    finger_y = plant.AddJoint(
        PrismaticJoint(
            "finger_y",
            plant.GetFrameByName("false_body1", finger),
            plant.GetFrameByName(name),
            [0, 1, 0],
            -0.3,
            0.3,
        )
    )
    plant.AddJointActuator("finger_y", finger_y)

    finger_z = plant.AddJoint(
        PrismaticJoint(
            "finger_z",
            plant.GetFrameByName(name),
            plant.GetFrameByName("false_body2", finger),
            [0, 0, 1],
            0.0,
            0.5,
        )
    )
    finger_z.set_default_translation(0.25)
    plant.AddJointActuator("finger_z", finger_z)

    return finger



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

    axis_first = [1, 0, 0]  # X-axis
    axis_second = [0, 1, 0]  # Y-axis

    # Weld the iiwa robots to the world at the specified positions
    contact_body_iiwa1 = plant.AddRigidBody(
        name="contact_body_iiwa1",
        M_BBo_B=SpatialInertia.Zero(),  # Zero mass and inertia
        model_instance=iiwa1_model
    )

    # Similarly for iiwa_2
    contact_body_iiwa2 = plant.AddRigidBody(
        name="contact_body_iiwa2",
        M_BBo_B=SpatialInertia.Zero(),
        model_instance=iiwa2_model
    )
    
    end_effector_frame_iiwa1 = plant.GetFrameByName("iiwa_link_7", iiwa1_model)
    end_effector_frame_iiwa2 = plant.GetFrameByName("iiwa_link_7", iiwa2_model)
    
    virtual_contact_frame_iiwa1 = plant.AddFrame(FixedOffsetFrame(
        name="virtual_contact_frame_pos",
        P=end_effector_frame_iiwa1,
        X_PF=RigidTransform(p=[0, 0, 0.09])  # TODO Adjust offset 
    ))

    virtual_contact_frame_iiwa2 = plant.AddFrame(FixedOffsetFrame(
        name="virtual_contact_frame_neg",
        P=end_effector_frame_iiwa2,
        X_PF=RigidTransform(p=[0, 0, 0.09])
    ))

    # Add Virtual Universal Joint for iiwa_1 (to model contact)
    universal_joint_iiwa1 = plant.AddJoint(UniversalJoint(
        name="universal_joint_iiwa1",
        frame_on_parent=virtual_contact_frame_iiwa1,
        frame_on_child=contact_body_iiwa1.body_frame(),
        damping=0
    ))

    # Add Virtual Universal Joint for iiwa_2 (to model contact)
    universal_joint_iiwa2 = plant.AddJoint(UniversalJoint(
        name="universal_joint_iiwa2",
        frame_on_parent=virtual_contact_frame_iiwa2,
        frame_on_child=contact_body_iiwa2.body_frame(),
        damping=0
    ))

    sphere_radius = 0.01 

    # Add sphere visuals of virtual contact bodies
    plant.RegisterVisualGeometry(
        body=contact_body_iiwa1,
        X_BG=RigidTransform.Identity(),
        shape=Sphere(sphere_radius),
        name="contact_body_iiwa1_sphere",
        diffuse_color=[1, 0, 0, 1]  
    )

    plant.RegisterVisualGeometry(
        body=contact_body_iiwa2,
        X_BG=RigidTransform.Identity(),
        shape=Sphere(sphere_radius),
        name="contact_body_iiwa2_sphere",
        diffuse_color=[0, 0, 1, 1]
    )

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
