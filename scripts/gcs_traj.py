import pickle
import sys; sys.path.append("/Users/dashora/Courses/project_6412")
from pydrake.common import RandomGenerator
from scripts.set_creation import SetGen, Node
from pydrake.geometry.optimization import HPolyhedron, VPolytope
import numpy as np
from src.dual_arm_manipulation.planner import GCSPlanner
import time


# Set parameters
NUM_SETS = 5


# Load up primitive and tabletop sets
with open("set_gen_static.pkl", "rb") as f:
    static_sets = pickle.load(f)
with open("set_gen_primitives_high_coverage_better.pkl", 'rb') as f:
    free_space_sets = pickle.load(f)
assert set(static_sets.keys()) == set(free_space_sets.keys()), "Keys don't match!"
print("Using contact modes:", list(free_space_sets.keys()))

# Engineer the set lists
dynamic_dict = {}
static_dict = {}
for contact_mode_name in free_space_sets.keys():
    dynamic_dict[contact_mode_name] = []
    for node in free_space_sets[contact_mode_name].nodes[-NUM_SETS:]:
        convex_hull = node.set
        convex_set = HPolyhedron(
            VPolytope(
                np.hstack((convex_hull.points[convex_hull.vertices][:,3:],
                           convex_hull.points[convex_hull.vertices][:,:3])).T
            )
        )
        dynamic_dict[contact_mode_name].append(convex_set)
    static_dict[contact_mode_name] = []
    for node in static_sets[contact_mode_name].nodes[-NUM_SETS:]:
        convex_hull = node.set
        convex_set = HPolyhedron(
            VPolytope(
                np.hstack((convex_hull.points[convex_hull.vertices][:,3:],
                           convex_hull.points[convex_hull.vertices][:,:3])).T
            )
        )
        static_dict[contact_mode_name].append(convex_set)


t = time.time()
gcs_planner = GCSPlanner(dynamic_dict, static_dict)
print("GCS Planner created! Time taken:", time.time() - t)


# Now we can plan a trajectory between the two modes
start_pose = static_dict['Y_NEG'][3].UniformSample(RandomGenerator())
end_pose = dynamic_dict['X_POS'][0].UniformSample(RandomGenerator())
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

# v39 -> v40 [label="intramode_Z_POS_dynamic_5_Z_POS_static_1"]; no use
# v46 -> v53 [label="intramode_Z_NEG_dynamic_2_Z_NEG_static_4"]; no use

# v2 -> v11 [label="intramode_X_POS_dynamic_1_X_POS_static_5"];
# v11 -> v33 [label="intermode_X_POS_static_5_Y_NEG_static_4"];
# make staer in v33, and end in v2.
