import pickle
from pydrake.common import RandomGenerator
from dual_arm_manipulation.planner import GCSPlanner
from dual_arm_manipulation.trajopt import GCSTrajOptPlanner
from dual_arm_manipulation.set_creation import SetGen, Node, special_in_hull
from pydrake.geometry.optimization import HPolyhedron, VPolytope, Point
import numpy as np
import time
from dual_arm_manipulation import ROOT_DIR
import yaml
from sklearn.neighbors import KDTree

# Set parameters
# NUM_SETS = 5

# TODO: use projections of start points onto existing convex sets.

"""
with open(ROOT_DIR / "config" / "config.yaml", "rb") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
# test_start_pose = np.array(config["eval"]["start_pose"])
# test_start_pose[-1] = 0.17
# test_end_pose = np.array(config["eval"]["goal_pose"])
# test_end_pose[-1] = 0.3
# test_end_pose[-2] = 0.0


# NICO START GOAL OVERRIDE
# test_start_pose = np.array(config["eval"]["start_pose"])
test_end_pose = np.array([0, 0, 1, 0, 0.0, 0.2, 0.6])
test_start_pose = np.array([1, 0, 0, 0, 0.0, 0.0, 0.15])

point_in_set = lambda pose, node: special_in_hull(np.array([np.hstack((pose[4:], pose[:4]))]), node.spatial_set.equations, node.orientation_set.equations, 1e-12)
"""

total_pts = []
total_static_pts = []


# Load up primitive and tabletop sets
with open("output/highly_compressed/set_gen_static.pkl", "rb") as f:
    static_sets = pickle.load(f)
with open("output/highly_compressed/set_gen_primitives_large_scale.pkl", 'rb') as f:
    free_space_sets = pickle.load(f)
# assert set(static_sets.keys()) == set(free_space_sets.keys()), "Keys don't match!"
with open(ROOT_DIR / "output" / "highly_compressed" / "set_gen_goal_conditioned.pkl", 'rb') as f:
    goal_conditioned_sets = pickle.load(f)
assert set(static_sets.keys()) == set(free_space_sets.keys()) == set(goal_conditioned_sets.keys()), "Keys don't match!"

print("Using contact modes:", list(free_space_sets.keys()))


# Engineer the set lists
dynamic_dict = {}
static_dict = {}
for contact_mode_name in free_space_sets.keys():
    
    if contact_mode_name not in ['Y_POS', 'X_POS']:
        continue
    
    dynamic_dict[contact_mode_name] = []
    for node in free_space_sets[contact_mode_name].nodes[-7:]: # [-NUM_SETS:]:
        convex_hull = node.set
        total_pts.append(convex_hull.points)
        convex_set = HPolyhedron(
            VPolytope(
                np.hstack((convex_hull.points[convex_hull.vertices][:,3:],
                           convex_hull.points[convex_hull.vertices][:,:3])).T
            )
        )
        dynamic_dict[contact_mode_name].append(convex_set)
        
        """
        test_pt_included = convex_set.PointInSet(test_start_pose)
        # test_pt_included = special_in_hull(np.array([np.hstack((test_start_pose[4:], test_start_pose[:4]))]), node.spatial_set.equations, node.orientation_set.equations, 1e-12)
        if test_pt_included:
            print(f"test start pose in dynamic set: {contact_mode_name}")
        test_pt_included = convex_set.PointInSet(test_end_pose)
        # test_pt_included = special_in_hull(np.array([np.hstack((test_end_pose[4:], test_end_pose[:4]))]), node.spatial_set.equations, node.orientation_set.equations, 1e-12)
        if test_pt_included:
            print(f"test end pose in dynamic set: {contact_mode_name}")
        """
    


    static_dict[contact_mode_name] = []
    for node in static_sets[contact_mode_name].nodes[-7:]: #[-NUM_SETS:]:
        convex_hull = node.set
        total_pts.append(convex_hull.points)
        total_static_pts.append(convex_hull.points)
        convex_set = HPolyhedron(
            VPolytope(
                np.hstack((convex_hull.points[convex_hull.vertices][:,3:],
                           convex_hull.points[convex_hull.vertices][:,:3])).T
            )
        )
        static_dict[contact_mode_name].append(convex_set)
        
        """
        test_pt_included = convex_set.PointInSet(test_start_pose)
        # test_pt_included = special_in_hull(np.array([np.hstack((test_start_pose[4:], test_start_pose[:4]))]), node.spatial_set.equations, node.orientation_set.equations, 1e-12)
        if test_pt_included:
            print(f"test start pose in static set: {contact_mode_name}")
        test_pt_included = convex_set.PointInSet(test_end_pose)
        # test_pt_included = special_in_hull(np.array([np.hstack((test_end_pose[4:], test_end_pose[:4]))]), node.spatial_set.equations, node.orientation_set.equations, 1e-12)
        if test_pt_included:
            print(f"test end pose in static set: {contact_mode_name}")
        """


    for node in goal_conditioned_sets[contact_mode_name].nodes[-7:]: #[-NUM_SETS:]:
        convex_hull = node.set
        total_pts.append(convex_hull.points)
        convex_set = HPolyhedron(
            VPolytope(
                np.hstack((convex_hull.points[convex_hull.vertices][:,3:],
                           convex_hull.points[convex_hull.vertices][:,:3])).T
            )
        )
        dynamic_dict[contact_mode_name].append(convex_set)

        """
        test_pt_included = convex_set.PointInSet(test_start_pose)
        # test_pt_included = special_in_hull(np.array([np.hstack((test_start_pose[4:], test_start_pose[:4]))]), node.spatial_set.equations, node.orientation_set.equations, 1e-12)
        if test_pt_included:
            print(f"test start pose in goal-cond dynamic set: {contact_mode_name}")
        test_pt_included = convex_set.PointInSet(test_end_pose)
        # test_pt_included = special_in_hull(np.array([np.hstack((test_end_pose[4:], test_end_pose[:4]))]), node.spatial_set.equations, node.orientation_set.equations, 1e-12)
        if test_pt_included:
            print(f"test end pose in goal-cond dynamic set: {contact_mode_name}")
        """


total_pts = np.concatenate(total_pts)
total_static_pts = np.concatenate(total_static_pts)
print("Total points", total_pts.shape)
print("Total static points", total_static_pts.shape)
total_pts = np.hstack([total_pts[:, 3:], total_pts[:, :3]])
total_static_pts = np.hstack([total_static_pts[:, 3:], total_static_pts[:, :3]])
tree = KDTree(total_pts)
tree_static = KDTree(total_static_pts)


desired_start = np.array([[1, 0, 0, 0, 0.0, 0.0, 0.15]])
desired_goal = np.array([[0, 0, 1, 0, 0.0, 0.2, 0.6]])

# start at xpos dynamic 5 or 6. 
# end at ypos dynamic 5


dist, ind = tree_static.query(desired_start, k=1)
test_start_pose = total_static_pts[ind]
print("Start pose:", test_start_pose)
print("Distance to desired start:", dist)
dist, ind = tree.query(desired_goal, k=1)
test_end_pose = total_pts[ind]
print("End pose:", test_end_pose)
print("Distance to desired end:", dist)


# start at a dynamic set. and end at another dynamic set. 
# start_pose = test_start_pose.reshape(-1, 1)
# end_pose = test_end_pose.reshape(-1, 1)


start_pose = dynamic_dict['X_POS'][4].UniformSample(RandomGenerator())
end_pose = dynamic_dict['Y_POS'][4].UniformSample(RandomGenerator())

start_pt = Point(start_pose)
end_pt = Point(end_pose)

# loop through every single set and print out ones that it is in
for mode_name, sets in static_dict.items():
    for i, set in enumerate(sets):
        if set.IntersectsWith(start_pt):
            print(f"Start pose in static set {i}: {mode_name}")
        if set.IntersectsWith(end_pt):
            print(f"End pose in static set {i}: {mode_name}")
for mode_name, sets in dynamic_dict.items():
    for i, set in enumerate(sets):
        if set.IntersectsWith(start_pt):
            print(f"Start pose in dynamic set {i}: {mode_name}")
        if set.IntersectsWith(end_pt):
            print(f"End pose in dynamic set {i}: {mode_name}")



t = time.time()
# gcs_tropt_planner = GCSTrajOptPlanner(dynamic_dict, static_dict, D=7, order=1, continuity_order=0)
gcs_planner = GCSPlanner(dynamic_dict, static_dict)

print("GCS Planner created! Time taken:", time.time() - t)

# Now we can plan a trajectory between the two modes
#start_pose = static_dict['Y_NEG'][3].UniformSample(RandomGenerator())
# end_pose = dynamic_dict['X_POS'][0].UniformSample(RandomGenerator())

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