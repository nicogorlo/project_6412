
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
    SpatialVelocity,
    RollPitchYaw,
    ProximityProperties,
    CoulombFriction,
    ModelInstanceIndex,
    LoadModelDirectivesFromString, 
    ProcessModelDirectives,
    StartMeshcat, 
    AddContactMaterial, 
    yaml_load_typed
)
from dual_arm_manipulation.utils import save_diagram, display_diagram
from manipulation.station import LoadScenario, MakeHardwareStation
from dual_arm_manipulation import ROOT_DIR
from manipulation.scenarios import AddShape
import numpy as np


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

def create_environment():
    # Add Meshcat visualizer
    meshcat = StartMeshcat()
    vis = MeshcatVisualizer(meshcat=meshcat)
    print(meshcat.web_url())
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
    # could add iiwa7_with_box_collision.sdf later if we want to account for that

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

    AddPointFinger(plant, 'index')
    AddPointFinger(plant, 'middle')

    plant.Finalize()

    # export cheat input ports
    builder.ExportInput(
        plant.get_applied_generalized_force_input_port(),
        "applied_generalized_force",
    )
    builder.ExportInput(
        plant.get_applied_spatial_force_input_port(),
        "applied_spatial_force",
    )
    # Export any actuation (non-empty) input ports that are not already
    # connected (e.g. by a driver).
    for i in range(plant.num_model_instances()):
        port = plant.get_actuation_input_port(ModelInstanceIndex(i))
        if port.size() > 0 and not builder.IsConnectedOrExported(port):
            builder.ExportInput(port, port.get_name())
    # Export all MultibodyPlant output ports.
    for i in range(plant.num_output_ports()):
        builder.ExportOutput(
            plant.get_output_port(i),
            plant.get_output_port(i).get_name(),
        )
    # Export the only SceneGraph output port.
    builder.ExportOutput(scene_graph.get_query_output_port(), "query_object")

    vis.AddToBuilder(builder, scene_graph, meshcat)

    return builder, plant, vis



def main():
    builder, plant, meshcat_vis = create_environment()

    # Get model instances
    iiwa1_model = plant.GetModelInstanceByName("iiwa_1")
    iiwa2_model = plant.GetModelInstanceByName("iiwa_2")


    # TODO: replace with using meshcat as visualization (as in interactive_ik example, p_set 6)
    
    meshcat_vis.StartRecording()
    # Finalize the diagram
    full_diagram = builder.Build()
    display_diagram(full_diagram)
    simulator = Simulator(full_diagram)
    simulator.set_target_realtime_rate(1.0)

    # Initialize and run the simulation
    context = simulator.get_mutable_context()
    simulator.Initialize()
    simulator.AdvanceTo(10.0)
    meshcat_vis.PublishRecording()

if __name__ == "__main__":
    main()