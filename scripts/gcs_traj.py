import pickle
import yaml
from pydrake.common import RandomGenerator
from pydrake.geometry.optimization import HPolyhedron, VPolytope, Hyperellipsoid # type: ignore
import numpy as np
import time
from scipy.spatial import ConvexHull
import cvxpy as cp

from dual_arm_manipulation.planner import GCSPlanner, GCSTrajOptPlanner
from dual_arm_manipulation.set_creation import SetGen, Node, special_in_hull, in_hull
from dual_arm_manipulation import ROOT_DIR
import logging

# Set parameters
NUM_SETS = 5


def check_set_validity(hull: ConvexHull):
    vertices = hull.points[hull.vertices]
    is_bounded = np.all(vertices)
    matrix_rank = np.linalg.matrix_rank(vertices)
    full_dimensional = matrix_rank == vertices.shape[1]

    return is_bounded and full_dimensional


def maximum_volume_ellipsoid(hull: ConvexHull) -> Hyperellipsoid:
    vertices = hull.points[hull.vertices]
    A = cp.Variable((vertices.shape[1], vertices.shape[1]), symmetric=True)
    b = cp.Variable(vertices.shape[1])
    constraints = []
    for vertex in vertices:
        constraints.append(cp.norm(A @ vertex + b) <= 1)
    prob = cp.Problem(cp.Maximize(cp.log_det(A)), constraints)
    prob.solve()

    A_pydrake = np.linalg.cholesky(np.linalg.inv(A.value))  # Decompose P⁻¹ into AᵀA
    b_pydrake = b.value
    assert np.allclose(A_pydrake @ A_pydrake.T, np.linalg.inv(A.value))
    return Hyperellipsoid(A_pydrake, b_pydrake)


with open(ROOT_DIR / "config" / "config.yaml", "rb") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

start_pose = np.array(config["eval"]["start_pose"])
end_pose = np.array(config["eval"]["goal_pose"])

# Load up primitive and tabletop sets
with open(ROOT_DIR / "output" / "set_gen_static.pkl", "rb") as f:
    static_sets = pickle.load(f)
with open(ROOT_DIR / "output" / "set_gen_primitives_large_scale.pkl", 'rb') as f:
    free_space_sets = pickle.load(f)
with open(ROOT_DIR / "output" / "set_gen_goal_conditioned.pkl", 'rb') as f:
    goal_conditioned_sets = pickle.load(f)
assert set(static_sets.keys()) == set(free_space_sets.keys()), "Keys don't match!"
assert set(static_sets.keys()) == set(goal_conditioned_sets.keys()), "Keys don't match!"
print("Using contact modes:", list(free_space_sets.keys()))

# Engineer the set lists
dynamic_dict = {}
static_dict = {}
for contact_mode_name in free_space_sets.keys():
    dynamic_dict[contact_mode_name] = []

    for node in free_space_sets[contact_mode_name].nodes:
        convex_hull = node.set
        assert(check_set_validity(convex_hull)), "Invalid set!"
        convex_set = HPolyhedron(
            VPolytope(
                np.hstack((convex_hull.points[convex_hull.vertices][:,4:],
                           convex_hull.points[convex_hull.vertices][:,:4])).T
            )
        )
        spatial_ellipsoid = maximum_volume_ellipsoid(node.spatial_set)
        orientation_ellipsoid = maximum_volume_ellipsoid(node.orientation_set)
        orientation_convex_set = node.orientation_set
        contains_start_chull = spatial_ellipsoid.PointInSet(start_pose[4:], tol=1e-6) and in_hull(np.array([start_pose[:4]]), orientation_convex_set.equations)
        contains_start_ellipsoid = spatial_ellipsoid.PointInSet(start_pose[4:], tol=1e-6) and orientation_ellipsoid.PointInSet(start_pose[:4]*.3, tol=1e-6)
        print(f"{contact_mode_name} contains start: chull - {contains_start_chull}; ellipsoid - {contains_start_ellipsoid}")
        ellipsoid = maximum_volume_ellipsoid(convex_hull)
        dynamic_dict[contact_mode_name].append(ellipsoid)
        
    print("here.")
    for node in goal_conditioned_sets[contact_mode_name].nodes:
        convex_hull = node.set
        convex_set = HPolyhedron(
            VPolytope(
                np.hstack((convex_hull.points[convex_hull.vertices][:,4:],
                           convex_hull.points[convex_hull.vertices][:,:4])).T
            )
        )
        spatial_ellipsoid = maximum_volume_ellipsoid(node.spatial_set)
        orientation_ellipsoid = maximum_volume_ellipsoid(node.orientation_set)
        orientation_convex_set = node.orientation_set
        contains_start_chull = spatial_ellipsoid.PointInSet(start_pose[4:], tol=1e-6) and in_hull(np.array([start_pose[:4]]), orientation_convex_set.equations)
        contains_start_ellipsoid = spatial_ellipsoid.PointInSet(start_pose[4:], tol=1e-6) and orientation_ellipsoid.PointInSet(start_pose[:4]*.3, tol=1e-6)
        print(f"{contact_mode_name} contains start: chull - {contains_start_chull}; ellipsoid - {contains_start_ellipsoid}")
        ellipsoid = maximum_volume_ellipsoid(convex_hull)
        dynamic_dict[contact_mode_name].append(ellipsoid)

    static_dict[contact_mode_name] = []
    for node in static_sets[contact_mode_name].nodes:
        convex_hull = node.set
        convex_set = HPolyhedron(
            VPolytope(
                np.hstack((convex_hull.points[convex_hull.vertices][:,4:],
                           convex_hull.points[convex_hull.vertices][:,:4])).T
            )
        )
        spatial_ellipsoid = maximum_volume_ellipsoid(node.spatial_set)
        orientation_ellipsoid = maximum_volume_ellipsoid(node.orientation_set)
        orientation_convex_set = node.orientation_set
        contains_start_chull = spatial_ellipsoid.PointInSet(start_pose[4:], tol=1e-6) and in_hull(np.array([start_pose[:4]]), orientation_convex_set.equations)
        contains_start_ellipsoid = spatial_ellipsoid.PointInSet(start_pose[4:], tol=1e-6) and orientation_ellipsoid.PointInSet(start_pose[:4]*.3, tol=1e-6)
        print(f"{contact_mode_name} contains start: chull - {contains_start_chull}; ellipsoid - {contains_start_ellipsoid}")
        ellipsoid = maximum_volume_ellipsoid(convex_hull)
        static_dict[contact_mode_name].append(ellipsoid)


t = time.time()
gcs_planner = GCSTrajOptPlanner(dynamic_dict, static_dict)
print("GCS Planner created! Time taken:", time.time() - t)

start_pose = config["eval"]["start_pose"]
end_pose = config["eval"]["goal_pose"]
# Now we can plan a trajectory between the two modes
# start_pose = static_dict['Y_NEG'][3].UniformSample(RandomGenerator())
# goal_pose = dynamic_dict['X_POS'][0].UniformSample(RandomGenerator())
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
