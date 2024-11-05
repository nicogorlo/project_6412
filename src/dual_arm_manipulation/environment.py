from pydrake.all import (
    DiagramBuilder, MeshcatVisualizer, MultibodyPlant, Parser,
    RigidTransform, RollPitchYaw, SpatialInertia, UnitInertia,
    CoulombFriction, Box, SceneGraph, ProximityProperties,
    LoadModelDirectivesFromString, ProcessModelDirectives,
    StartMeshcat, AddMultibodyPlantSceneGraph,
    AddContactMaterial
)

from dual_arm_manipulation import ROOT_DIR

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
            file: package://drake_models/iiwa_description/sdf/iiwa7_with_box_collision.sdf
        - add_model:
            name: iiwa_2
            file: package://drake_models/iiwa_description/sdf/iiwa7_with_box_collision.sdf
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

    plant.Finalize()

    vis.AddToBuilder(builder, scene_graph, meshcat)

    return builder, plant, vis