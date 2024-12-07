import numpy as np
from matplotlib import pyplot as plt
from dual_arm_manipulation.planner import GCSPlanner
from pydrake.geometry.optimization import ConvexSet, Point, ConvexHull, HPolyhedron, VPolytope # type: ignore

# Define the convex sets for each contact mode
r1, r2 = 9, 9
static_r1, static_r2 = 5, 5
start_x, start_y = 0, 4.75
end_x, end_y = 10, -10
x1, y1, x2, y2 = -10, 10, 10, -10
static_x1, static_y1, static_x2, static_y2 = -1, 1, 1, -1
npts = 20

# Sample from a circle centered at (x, y) with radius r
circle1_pts = np.array([[x1 + r1*np.cos(theta), y1 + r1*np.sin(theta)] for theta in np.linspace(0, 2*np.pi, npts)]).T
circle1 = HPolyhedron(VPolytope(circle1_pts))

circle2_pts = np.array([[x2 + r2*np.cos(theta), y2 + r2*np.sin(theta)] for theta in np.linspace(0, 2*np.pi, npts)]).T
circle2 = HPolyhedron(VPolytope(circle2_pts))

# Sample static sets
static_circle1_pts = np.array([[static_x1 + static_r1*np.cos(theta), static_y1 + static_r1*np.sin(theta)] for theta in np.linspace(0, 2*np.pi, npts)]).T
static_circle1 = HPolyhedron(VPolytope(static_circle1_pts))

static_circle2_pts = np.array([[static_x2 + static_r2*np.cos(theta), static_y2 + static_r2*np.sin(theta)] for theta in np.linspace(0, 2*np.pi, npts)]).T
static_circle2 = HPolyhedron(VPolytope(static_circle2_pts))

# Plot
polygon_edges_circle1 = [circle1_pts[..., i] for i in range(circle1_pts.shape[1])]
polygon_edges_circle2 = [circle2_pts[..., i] for i in range(circle2_pts.shape[1])]
polygon_edges_static_circle1 = [static_circle1_pts[..., i] for i in range(static_circle1_pts.shape[1])]
polygon_edges_static_circle2 = [static_circle2_pts[..., i] for i in range(static_circle2_pts.shape[1])]
plt.plot([pt[0] for pt in polygon_edges_circle1], [pt[1] for pt in polygon_edges_circle1], 'r')
plt.plot([pt[0] for pt in polygon_edges_circle2], [pt[1] for pt in polygon_edges_circle2], 'b')
plt.plot([pt[0] for pt in polygon_edges_static_circle1], [pt[1] for pt in polygon_edges_static_circle1], 'r--')
plt.plot([pt[0] for pt in polygon_edges_static_circle2], [pt[1] for pt in polygon_edges_static_circle2], 'b--')

# Plot Start and Goal
plt.scatter(start_x, start_y, c='g', label='Start')
plt.scatter(end_x, end_y, c='g', label='End')

# Make a GCS planner
mode_to_sets = {
    "mode1": {circle1},
    "mode2": {circle2},
}
mode_to_static_sets = {
    "mode1": {static_circle1},
    "mode2": {static_circle2}
}
gcs_planner = GCSPlanner(mode_to_sets, mode_to_static_sets)
print("GCS Planner created!")

# Now we can plan a trajectory between the two modes
start_point = np.array([start_x, start_y])
end_point = np.array([end_x, end_y])
trajectory = gcs_planner.solve_plan(start_point, end_point)

print("Trajectory:", trajectory)


plt.plot([pt[0] for pt in trajectory], [pt[1] for pt in trajectory], 'g')
plt.scatter([pt[0] for pt in trajectory], [pt[1] for pt in trajectory], c='g')
plt.legend()
plt.show()

