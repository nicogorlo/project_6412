import numpy as np
from matplotlib import pyplot as plt
from pydrake.geometry.optimization import ConvexSet, Point, ConvexHull, HPolyhedron, VPolytope, GraphOfConvexSetsOptions
from pydrake.planning import GcsTrajectoryOptimization

# Define the convex sets for each contact mode
r1, r2 = 20, 15
static_r1, static_r2 = 5, 5
start_x, start_y = 0, 40
end_x, end_y = 30, -10
x1, y1, x2, y2 = -10, 10, 10, -10
static_x1, static_y1, static_x2, static_y2 = -1, 1, 1, -1
npts = 20

# Sample from a circle centered at (x, y) with radius r
circle1_pts = np.array([[x1 + r1*np.cos(theta), y1 + 2*r1*np.sin(theta)] for theta in np.linspace(0, 2*np.pi, npts)]).T
circle1 = HPolyhedron(VPolytope(circle1_pts))

circle2_pts = np.array([[x2 + 2*r2*np.cos(theta), y2 + r2*np.sin(theta)] for theta in np.linspace(0, 2*np.pi, npts)]).T
circle2 = HPolyhedron(VPolytope(circle2_pts))

# Plot
polygon_edges_circle1 = [circle1_pts[..., i] for i in range(circle1_pts.shape[1])]
polygon_edges_circle2 = [circle2_pts[..., i] for i in range(circle2_pts.shape[1])]

plt.plot([pt[0] for pt in polygon_edges_circle1], [pt[1] for pt in polygon_edges_circle1], 'r')
plt.plot([pt[0] for pt in polygon_edges_circle2], [pt[1] for pt in polygon_edges_circle2], 'b')

# Plot Start and Goal
plt.scatter(start_x, start_y, c='g', label='Start')
plt.scatter(end_x, end_y, c='g', label='End')

# make plot square:
plt.axis('equal')

regions = [circle1, circle2]
x_start = np.array([start_x, start_y])
x_goal = np.array([end_x, end_y])

qdot_min = -10
qdot_max = 10
order = 5
continuity_order = 1   # C2 continuity

trajopt = GcsTrajectoryOptimization(2)
gcs_regions = trajopt.AddRegions(regions, order=order)
source = trajopt.AddRegions([Point(x_start)], order=0)
target = trajopt.AddRegions([Point(x_goal)], order=0)
trajopt.AddEdges(source, gcs_regions)
trajopt.AddEdges(gcs_regions, target)
trajopt.AddTimeCost()
for o in range(1, continuity_order + 1):
    print(f"adding C{o} constraints")
    trajopt.AddContinuityConstraints(o)
trajopt.AddVelocityBounds([qdot_min] * 2, [qdot_max] * 2)
options = GraphOfConvexSetsOptions()
# options.max_rounded_paths = 0
[traj, result] = trajopt.SolvePath(source, target, options)
print(f"result.is_success() = {result.is_success()}")

def plot_trajectory(traj):

    plt.plot(*traj.value(traj.start_time()), "kx")
    plt.plot(*traj.value(traj.end_time()), "kx")
    times = np.linspace(traj.start_time(), traj.end_time(), 1000)
    waypoints = traj.vector_values(times)
    plt.plot(*waypoints, "b", zorder=5)

plot_trajectory(traj)

plt.show()