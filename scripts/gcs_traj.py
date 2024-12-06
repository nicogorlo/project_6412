import pickle
from pydrake.common import RandomGenerator
from pydrake.geometry.optimization import HPolyhedron, VPolytope # type: ignore
import numpy as np

from dual_arm_manipulation import ROOT_DIR
from dual_arm_manipulation.planner import GCSPlanner
from dual_arm_manipulation.set_creation import SetGen, Node, in_hull

start_pose = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3])
goal_pose = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.6])

# Load up primitive sets
with open(ROOT_DIR / "output/set_gen_primitives_high_coverage_X_POS.pkl", "rb") as f:
    primitive_sets_xpos = pickle.load(f)
print("Primitive sets loaded!")

xpos_sets = []
xpos_sets_static = []
for node in primitive_sets_xpos.nodes:
    convex_hull = node.set
    
    convex_set = HPolyhedron(
        VPolytope(
            np.hstack((convex_hull.points[convex_hull.vertices][:,3:],
                       convex_hull.points[convex_hull.vertices][:,:3])).T
        )
    )
    if in_hull(np.array([np.hstack((start_pose[4:], start_pose[:4]))]), convex_hull.equations) and xpos_sets_static == []:
        xpos_sets_static.append(convex_set)
    else:
        xpos_sets.append(convex_set)

print("Polytopes created!")
print("Number of sets:", len(xpos_sets))
print("Number of static sets:", len(xpos_sets_static))
# xpos_sets_static = [xpos_sets.pop()]
contact_mode_dynamic = {
    "x_pos": xpos_sets
}
contact_mode_static = {
    "x_pos": xpos_sets_static
}
gcs_planner = GCSPlanner(contact_mode_dynamic, contact_mode_static)
print("GCS Planner created!")
# print("Graph string:", gcs_planner.gcs.GetGraphvizString())


# Now we can plan a trajectory between the two modes
# start_pose = xpos_sets_static[0].UniformSample(RandomGenerator())
# goal_pose = xpos_sets[0].UniformSample(RandomGenerator())
print("Planning from start:", start_pose)
print("To end:", goal_pose)
trajectory = gcs_planner.solve_plan(start_pose, goal_pose)

print("Trajectory:", trajectory)

# Map the solved trajectory from start to goal onto a cube display
# Show cube moving around.


