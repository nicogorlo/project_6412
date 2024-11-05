from pydrake.all import Simulator, SpatialVelocity, RotationMatrix
from dual_arm_manipulation.environment import create_environment
import numpy as np

def main():
    builder, plant, meshcat_vis = create_environment()

    # Get model instances
    iiwa1_model = plant.GetModelInstanceByName("iiwa_1")
    iiwa2_model = plant.GetModelInstanceByName("iiwa_2")

    # Create controllers
    controller1 = builder.AddSystem(DifferentialIKController(plant, iiwa1_model))
    controller2 = builder.AddSystem(DifferentialIKController(plant, iiwa2_model))

    # Connect state ports
    builder.Connect(
        plant.get_state_output_port(),
        controller1.get_input_port(0)
    )
    builder.Connect(
        controller1.get_output_port(0),
        plant.get_actuation_input_port(iiwa1_model)
    )

    builder.Connect(
        plant.get_state_output_port(),
        controller2.get_input_port(0)
    )
    builder.Connect(
        controller2.get_output_port(0),
        plant.get_actuation_input_port(iiwa2_model)
    )

    # Create desired velocity sources
    desired_velocity1 = SpatialVelocity(np.zeros(3), [0.1, 0, 0])
    desired_velocity2 = SpatialVelocity(np.zeros(3), [0, -0.1, 0])

    desired_velocity1_sys = builder.AddSystem(DesiredVelocitySource(desired_velocity1))
    desired_velocity2_sys = builder.AddSystem(DesiredVelocitySource(desired_velocity2))

    builder.Connect(
        desired_velocity1_sys.get_output_port(0),
        controller1.get_input_port(1)
    )

    builder.Connect(
        desired_velocity2_sys.get_output_port(0),
        controller2.get_input_port(1)
    )
    meshcat_vis.StartRecording()
    # Finalize the diagram
    full_diagram = builder.Build()
    simulator = Simulator(full_diagram)
    simulator.set_target_realtime_rate(1.0)

    # Initialize and run the simulation
    context = simulator.get_mutable_context()
    simulator.Initialize()
    simulator.AdvanceTo(10.0)
    meshcat_vis.PublishRecording()

if __name__ == "__main__":
    main()