import pickle
import sys; sys.path.append("/Users/dashora/Courses/project_6412/src")
from pydrake.common import RandomGenerator
from dual_arm_manipulation.planner import GCSPlanner
from dual_arm_manipulation.trajopt import GCSTrajOptPlanner
from dual_arm_manipulation.set_creation import SetGen, Node
from pydrake.geometry.optimization import HPolyhedron, VPolytope
import numpy as np
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
    if contact_mode_name not in ['Y_NEG', 'X_POS']:
        continue
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



# gcs_tropt_planner = GCSTrajOptPlanner(dynamic_dict, static_dict, D=7, order=1, continuity_order=0)
gcs_planner = GCSPlanner(dynamic_dict, static_dict)

print("GCS Planner created! Time taken:", time.time() - t)

# Now we can plan a trajectory between the two modes
start_pose = static_dict['Y_NEG'][3].UniformSample(RandomGenerator())
end_pose = dynamic_dict['X_POS'][0].UniformSample(RandomGenerator())


# only ones we need here are yneg and xpos!!!! run with just these two!


print("Planning from start:", start_pose)
print("To end:", end_pose)
t = time.time()

trajectory, edge_path = gcs_planner.solve_plan(start_pose, end_pose)
# trajectory = gcs_tropt_planner.solve_plan(start_pose, end_pose)

print("Trajectory:", trajectory)
print("Time taken:", time.time() - t)

linear_paths = []
for i in range(len(trajectory) - 1):
    linear_paths.append(np.linspace(trajectory[i], trajectory[i+1], num=10))
linear_traj = np.concatenate(linear_paths)

switch_wps = gcs_planner.get_switches(linear_traj, edge_path, 10)

print(linear_traj.shape)
print(switch_wps)

with open("linear_traj.pkl", "wb") as f:
    pickle.dump(linear_traj, f)