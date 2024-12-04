import pickle
from pydrake.common import RandomGenerator
from scripts.set_creation import SetGen, Node
from pydrake.geometry.optimization import HPolyhedron, VPolytope
import numpy as np
from src.dual_arm_manipulation.planner import GCSPlanner
import time
# start_pose = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.15])
# end_pose = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.6])

# Load up primitive sets
with open("set_gen_primitives_X_POS.pkl", "rb") as f:
    primitive_sets_xpos = pickle.load(f)
print("Primitive sets loaded!")
# For the sets, let's use the HPolygon(VPolytope()) representation.
xpos_sets = []
xpos_sets_static = []


# Look at the last 10 sets only!
for node in primitive_sets_xpos.nodes[-10:]:
    convex_hull = node.set
    # if convex_hull.
    convex_set = HPolyhedron(
        VPolytope(
            np.hstack((convex_hull.points[convex_hull.vertices][:,3:],
                       convex_hull.points[convex_hull.vertices][:,:3])).T
        )
    )
    xpos_sets.append(convex_set)
print("Polytopes created!")

xpos_sets_static = [xpos_sets.pop()]
contact_mode_dynamic = {
    "x_pos": xpos_sets
}
contact_mode_static = {
    "x_pos": xpos_sets_static
}

t = time.time()
gcs_planner = GCSPlanner(contact_mode_dynamic, contact_mode_static)
print("GCS Planner created! Time taken:", time.time() - t)

# Now we can plan a trajectory between the two modes
start_pose = xpos_sets_static[0].UniformSample(RandomGenerator())
end_pose = xpos_sets[1].UniformSample(RandomGenerator())
print("Planning from start:", start_pose)
print("To end:", end_pose)
t = time.time()
trajectory = gcs_planner.solve_plan(start_pose, end_pose)
print("Trajectory:", trajectory)
print("Time taken:", time.time() - t)

linear_paths = []
for i in range(len(trajectory) - 1):
    linear_paths.append(np.linspace(trajectory[i], trajectory[i+1], num=10))
linear_traj = np.concatenate(linear_paths)
print(linear_traj.shape)
with open("linear_traj.pkl", "wb") as f:
    pickle.dump(linear_traj, f)
# Map the solved trajectory from start to goal onto a cube display
# Show cube moving around.


