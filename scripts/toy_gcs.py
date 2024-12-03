import numpy as np
from matplotlib import pyplot as plt
from src.dual_arm_manipulation.planner import GCSPlanner
from pydrake.geometry.optimization import ConvexSet, Point, ConvexHull

# Define the convex sets for each contact mode
r1, r2 = 9, 9
static_r1, static_r2 = 5, 5
x1, y1, x2, y2 = -10, 10, 10, -10
static_x1, static_y1, static_x2, static_y2 = -1, 1, 1, -1
npts = 50
# Sample from a circle centered at (x, y) with radius r
circle1 = ConvexHull(
    [Point([x1 + r1*np.cos(theta), y1 + r1*np.sin(theta)]) for theta in np.linspace(0, 2*np.pi, npts)]
)
circle2 = ConvexHull(
    [Point([x2 + r2*np.cos(theta), y2 + r2*np.sin(theta)]) for theta in np.linspace(0, 2*np.pi, npts)]
)
# Define the static sets
static_circle1 = ConvexHull(
    [Point([static_x1 + static_r1*np.cos(theta), static_y1 + static_r1*np.sin(theta)]) for theta in np.linspace(0, 2*np.pi, npts)]
)
static_circle2 = ConvexHull(
    [Point([static_x2 + static_r2*np.cos(theta), static_y2 + static_r2*np.sin(theta)]) for theta in np.linspace(0, 2*np.pi, npts)]
)

polygon_edges_circle1 = [pt.x() for pt in circle1.participating_sets()]
polygon_edges_circle2 = [pt.x() for pt in circle2.participating_sets()]
polygon_edges_static_circle1 = [pt.x() for pt in static_circle1.participating_sets()]
polygon_edges_static_circle2 = [pt.x() for pt in static_circle2.participating_sets()]
plt.plot([pt[0] for pt in polygon_edges_circle1], [pt[1] for pt in polygon_edges_circle1], 'r')
plt.plot([pt[0] for pt in polygon_edges_circle2], [pt[1] for pt in polygon_edges_circle2], 'b')
plt.plot([pt[0] for pt in polygon_edges_static_circle1], [pt[1] for pt in polygon_edges_static_circle1], 'r--')
plt.plot([pt[0] for pt in polygon_edges_static_circle2], [pt[1] for pt in polygon_edges_static_circle2], 'b--')
plt.show()

# Make a GCS planner
mode_to_sets = {
    "mode1": {circle1},
    "mode2": {circle2}
}
mode_to_static_sets = {
    "mode1": {static_circle1},
    "mode2": {static_circle2}
}
gcs_planner = GCSPlanner(mode_to_sets, mode_to_static_sets)
print("GCS Planner created!")
# Now we can plan a trajectory between the two modes
start = np.array([0, 0])
goal = np.array([x2, y2])
trajectory = gcs_planner.solve_plan(start, goal)
# Plot the trajectory
plt.plot([pt[0] for pt in trajectory], [pt[1] for pt in trajectory], 'g')