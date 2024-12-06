
from pydrake.all import (
    Diagram,
    Context,
    MeshcatVisualizer,
    MultibodyPlant,
    RigidTransform,
    Simulator,
    HPolyhedron,
    VPolytope,
    MathematicalProgram,
    Solve
)


from manipulation.meshcat_utils import AddMeshcatTriad
from typing import NamedTuple
import numpy as np
import time
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
from scipy.spatial import ConvexHull, Delaunay
import matplotlib.patches as patches
from shapely.geometry import Point, Polygon
import math

from dual_arm_manipulation import ROOT_DIR
from dual_arm_manipulation.environment import dual_arm_environment
from dual_arm_manipulation.contact_mode import ContactMode
from dual_arm_manipulation.trajectory_primitives import TrajectoryPrimitives, TrajectoryPrimitive
from dual_arm_manipulation.sampler import PrimitiveSampler
from dual_arm_manipulation.set_creation import SetGen, Node
from dual_arm_manipulation.utils import interpolate_6dof_poses, get_free_faces, pose_vec_to_transform_position_first, rotation_matrix_from_vectors

def visualise_trajectory_poses(visualizer: MeshcatVisualizer, poses: list[RigidTransform]):
    """
    Visualize the trajectory poses of a trajectory primitive.

    Parameters:
        visualizer (MeshcatVisualizer): The meshcat visualizer.
        tp_traj (TrajectoryPrimitive): The trajectory primitive.
    """
    for i, pose in enumerate(poses):
        AddMeshcatTriad(
            visualizer, "box_traj/" + "frame_{}".format(i), length=0.1, radius=0.006, X_PT=pose
        )


def visualize_sample_trajectories(plant: MultibodyPlant, plant_context: Context, root_diagram: Diagram, root_context: Context, contact_mode: ContactMode, sampler: PrimitiveSampler, simulator: Simulator, visualizer: MeshcatVisualizer):
    
    for trajectory_sample in sampler.trajectory_primitives[contact_mode.name]:

        solution = sampler.ik_solutions[contact_mode.name][trajectory_sample.primitive_name]

        visualise_trajectory_poses(visualizer, trajectory_sample.trajectory)
            
        for i, (tp_pose, q) in enumerate(zip(trajectory_sample, solution)):
            if q is None:
                print(f"Skipping {i}, no solution here.")
                continue
            
            plant.SetPositions(plant_context, q)
            root_diagram.ForcedPublish(root_context)

            time.sleep(0.05)


def visualize_generated_sets(plant: MultibodyPlant, plant_context: Context, root_diagram: Diagram, root_context: Context, set_gen: SetGen, visualizer: MeshcatVisualizer):
    
    for node in set_gen.nodes:
        
        poses = [pose_vec_to_transform_position_first(pose) for pose in node.points]

        visualise_trajectory_poses(visualizer, poses)

        # q = sample_ik
                
        # plant.SetPositions(plant_context, q)
        # root_diagram.ForcedPublish(root_context)

        time.sleep(1.0)


def animate_sets(
    root_diagram: Diagram,
    root_context: Context,
    plant: MultibodyPlant,
    set_gen: HPolyhedron,
    meshcat: MeshcatVisualizer
):
    """
    A simple hit-and-run-style idea for visualizing the convex regions:
    1. Start at the center. Pick a random direction and run to the boundary.
    2. Pick a new random direction; project it onto the current boundary, and run along it. Repeat
    """

    for node in set_gen.nodes:

        plant_context = plant.GetMyContextFromRoot(root_context)
        convex_hull = node.set
        region = HPolyhedron(VPolytope(np.hstack((convex_hull.points[convex_hull.vertices][:,3:], convex_hull.points[convex_hull.vertices][:,:3])).T))
        q = region.ChebyshevCenter()
        plant.SetPositions(plant_context, plant.GetModelInstanceByName("movable_cuboid"),  q)
        root_diagram.ForcedPublish(root_context)

        print("Press the 'Next Region' button in Meshcat to continue.")
        meshcat.AddButton("Next Region", "Escape")

        rng = np.random.default_rng()
        nq = 7
        prog = MathematicalProgram()
        qvar = prog.NewContinuousVariables(nq, "q")
        prog.AddLinearConstraint(region.A(), 0 * region.b() - np.inf, region.b(), qvar)
        cost = prog.AddLinearCost(np.ones((nq, 1)), qvar)

        while meshcat.GetButtonClicks("Next Region") < 1:
            direction = rng.standard_normal(nq)
            cost.evaluator().UpdateCoefficients(direction)

            result = Solve(prog)
            assert result.is_success()

            q_next = result.GetSolution(qvar)

            for t in np.append(np.arange(0, 1, 0.05), 1):
                qs = t * q_next + (1 - t) * q
                plant.SetPositions(plant_context, plant.GetModelInstanceByName("movable_cuboid"),  qs)
                root_diagram.ForcedPublish(root_context)
                time.sleep(0.05)

            q = q_next


def project_to_2D(vecs: np.ndarray) -> np.ndarray:
    """
    Project the trajectory to the 2D case using vectorized operations.
    """
    quaternions = vecs[:, 3:]
    
    cos_yaw, sin_yaw = quaternion_to_cos_sin_yaw(
        quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]
    )
    

    trajectory_2d = np.column_stack((vecs[:, 0], vecs[:, 1], cos_yaw, sin_yaw))
    
    return trajectory_2d


def visualize_4D_sets(sampler: PrimitiveSampler, sets: dict[SetGen]):
    """
    Visualize the 4D sets of poses and orientations for each contact mode.
    Creates a plot for each contact mode, showing the 2D convex hull of positions as well as arrows indicating the extent of orientations
    
    TODO: extend to take data from SetGen and visualize multiple sets for each contact mode at once.
    """
    contact_modes = list(sampler.config['contact_modes'].keys())
    num_modes = len(contact_modes)

    # Determine grid size for subplots
    ncols = int(np.ceil(np.sqrt(num_modes)))
    nrows = int(np.ceil(num_modes / ncols))

    # Create a figure and a grid of subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*10, nrows*10))
    axes = axes.flatten()

    for idx, contact_mode in enumerate(contact_modes):

        ax = axes[idx]

        contact_mode_sets = sets[contact_mode]
        poses_2d = []
        poses_2d_negative = []
        for trajectory_sample in sampler.trajectory_primitives[contact_mode]:
            trajectory_2d = [i for (i, j) in zip(trajectory_sample.project_to_2D(), sampler.ik_solutions[contact_mode][trajectory_sample.primitive_name]) if j is not None]
            poses_2d.extend(trajectory_2d)
            trajectory_2d_negative = [i for (i, j) in zip(trajectory_sample.project_to_2D(), sampler.ik_solutions[contact_mode][trajectory_sample.primitive_name]) if j is None]
            poses_2d_negative.extend(trajectory_2d_negative)

        if not poses_2d:
            print(f"No data for contact mode {contact_mode}")
            continue

        poses_2d = np.array(poses_2d)
        positions = poses_2d[:, :2]
        orientations = poses_2d[:, 2:]

        # samples
        for pose in poses_2d[::2]:
            x, y, q1, q2 = pose
            ax.plot(x, y, 'go', markersize=2, alpha=0.3)
            ax.arrow(x, y, q1*0.05, q2*0.05, head_width=0.01, head_length=0.01, length_includes_head=True, overhang=0.005, fc='g', ec='g', alpha=0.3)

        for pose in poses_2d_negative[::2]:
            x, y, q1, q2 = pose
            ax.plot(x, y, 'ro', markersize=2, alpha=0.3)
            ax.arrow(x, y, q1*0.05, q2*0.05, head_width=0.01, head_length=0.01, length_includes_head=True, overhang=0.005, fc='r', ec='r', alpha=0.3)

        colors_sets = plt.colormaps['rainbow'](np.linspace(0, 1, len(contact_mode_sets.nodes)))
        for node in contact_mode_sets.nodes:

            poses_2d_node = project_to_2D(node.points)
            positions_node = poses_2d_node[:, :2]
            
            # Compute the convex hull of positions (x, y)
            hull2d = ConvexHull(positions_node)
            hull_points = positions_node[hull2d.vertices]
            hull_polygon = Polygon(hull_points)

            # grid
            x_min, y_min = hull_points.min(axis=0)
            x_max, y_max = hull_points.max(axis=0)
            grid_x, grid_y = np.meshgrid(
                np.linspace(x_min, x_max, 12),
                np.linspace(y_min, y_max, 12)
            )
            grid_points = np.c_[grid_x.ravel(), grid_y.ravel()]

            # grid points inside convex hull
            hull_delaunay = Delaunay(hull_points)
            inside = hull_delaunay.find_simplex(grid_points) >= 0
            grid_points_inside = grid_points[inside]

            # convex hull of positions
            hull_points = np.vstack([hull_points, hull_points[0]])
            ax.plot(hull_points[:, 0], hull_points[:, 1], 'r--', lw=2)
            ax.fill(hull_points[:, 0], hull_points[:, 1], 'lightgrey', alpha=0.5)
            # ax.plot(positions_node[:, 0], positions_node[:, 1], 'o', markersize=2)

            # convex hull inequalities in 4D
            hull4d = ConvexHull(poses_2d_node)
            A = hull4d.equations[:, :4]
            b = -hull4d.equations[:, 4]
            

            # For each grid point, compute extent of orientations
            for point in tqdm(grid_points_inside, desc=f'Processing {contact_mode}'):
                x, y = point

                # inequalities for q1 and q2
                # fixed x and y -> inequalities become linear in q1 and q2
                A_q = A[:, 2:]
                c = b - A[:, 0] * x - A[:, 1] * y 

                theta_samples = np.linspace(-np.pi, np.pi, 36)
                feasible_orientations = []

                for theta in theta_samples:
                    q1 = np.cos(theta)
                    q2 = np.sin(theta)

                    # check convex hull inequalities for point
                    lhs = np.dot(A_q, np.array([q1, q2]))
                    if np.all(lhs <= c + 1e-3):
                        feasible_orientations.append((q1, q2))

                print(f"N Feasible orientations: {len(feasible_orientations)}")

                for feasible_orientation in feasible_orientations:

                    [ax.arrow(x, y, feasible_orientation[0]*0.05, feasible_orientation[1]*0.05, head_width=0.01, head_length=0.01, length_includes_head=True, overhang=0.005, fc='b', ec='b')]

                    # min_theta = min(feasible_orientations)
                    # max_theta = max(feasible_orientations)
                    # mean_theta = (min_theta + max_theta) / 2
                    # angle_extent = (max_theta - min_theta) * 180 / np.pi

                    # Plot the cone
                    # cone = patches.Wedge(
                    #     (x, y), 0.05,
                    #     min_theta * 180 / np.pi,
                    #     max_theta * 180 / np.pi,
                    #     width=0.05,
                    #     color='blue', alpha=0.3
                    # )
                    # ax.add_patch(cone)

                    # # Optionally, plot an arrow indicating the mean orientation
                    # dx = 0.1 * np.cos(mean_theta)
                    # dy = 0.1 * np.sin(mean_theta)
                    # ax.arrow(x, y, dx, dy, head_width=0.02, head_length=0.03, fc='k', ec='k')

        

        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title(f'Contact Mode: {contact_mode}')
        ax.axis('equal')

    for idx in range(num_modes, nrows * ncols):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.savefig(ROOT_DIR / "output" / 'all_contact_modes.png')
    plt.show()
    print("Finished visualizing all contact modes.")


def quaternion_to_cos_sin_yaw(qw, qx, qy, qz):
    """
    Computes cos(yaw) and sin(yaw) directly from quaternion components.

    Parameters:
        qw, qx, qy, qz (float): Components of the quaternion.

    Returns:
        cos_yaw, sin_yaw (float): The cosine and sine of the yaw angle.
    """
    # Compute cos(yaw)
    cos_yaw = 1 - 2 * (qy**2 + qz**2)
    
    # Compute sin(yaw)
    sin_yaw = 2 * (qw * qz + qx * qy)

    # normalize
    cos_yaw, sin_yaw = [cos_yaw, sin_yaw] / np.linalg.norm([cos_yaw, sin_yaw])
    
    return cos_yaw, sin_yaw

def visualize_6d_points(data):
    """
    Visualize 6D points with positions and orientations.

    Parameters:
    - data: numpy array of shape (N, 7), where each row is (x, y, z, qw, qx, qy, qz)
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extract positions and quaternions
    positions = data[:, 0:3]
    quaternions = data[:, 3:7]  # Assuming order is (qw, qx, qy, qz)

    for pos, quat in zip(positions, quaternions):
        # Convert quaternion to rotation object
        # Note: scipy expects quaternions in (qx, qy, qz, qw) order
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])

        # Rotate a unit vector along the x-axis
        vec = r.apply([1, 0, 0])

        # Plot the point
        ax.scatter(pos[0], pos[1], pos[2], color='blue', s=50)

        # Plot the orientation arrow
        ax.quiver(
            pos[0], pos[1], pos[2],    # Starting point of the arrow
            vec[0], vec[1], vec[2],    # Direction components
            length=0.1, normalize=True, color='red'
        )

    # Set labels and show the plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def visualize_convex_hulls(hulls):
    """
    Visualizes a list of convex hulls of 7D poses as 3D polytopes
    using x, y, and yaw components.

    Parameters:
        hulls (list of scipy.spatial.ConvexHull): List of convex hulls in 7D pose space.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # generate colors:
    colors = plt.cm.viridis(np.linspace(0, 1, len(hulls)))

    for idx, hull in enumerate(hulls):
        points_7d = hull.points  # Shape: (N, 7)

        # x, y, cos(yaw), sin(yaw)
        projected_points = []
        for point in points_7d:
            x, y, z, qw, qx, qy, qz = point
            cos_yaw, sin_yaw = quaternion_to_cos_sin_yaw(qw, qx, qy, qz)
            projected_points.append([x, y, cos_yaw, sin_yaw])

        projected_points = np.array(projected_points)  # Shape: (N, 4)

        # convex hull in 4D space (x, y, cos(yaw), sin(yaw))
        hull_4d = ConvexHull(projected_points)

        # x, y, yaw
        vertices = projected_points[hull_4d.vertices]
        x_vals = vertices[:, 0]
        y_vals = vertices[:, 1]
        cos_yaws = vertices[:, 2]
        sin_yaws = vertices[:, 3]
        yaws = np.arctan2(sin_yaws, cos_yaws)

        # faces
        faces = hull_4d.simplices
        for simplex in faces:
            # vertices
            simplex_vertices = projected_points[simplex]
            x_face = simplex_vertices[:, 0]
            y_face = simplex_vertices[:, 1]
            cos_yaw_face = simplex_vertices[:, 2]
            sin_yaw_face = simplex_vertices[:, 3]
            yaw_face = np.arctan2(sin_yaw_face, cos_yaw_face)

            # Handle yaw discontinuity
            yaw_face = np.unwrap(yaw_face)

            face_points = np.column_stack((x_face, y_face, yaw_face))

            # Create the polygon and add to the plot
            poly = Poly3DCollection([face_points], alpha=0.4, color = colors[idx])
            poly.set_edgecolor('k')
            ax.add_collection3d(poly)

        # Optionally plot the vertices
        ax.scatter(x_vals, y_vals, yaws, color='r', s=20)

    # Set labels and limits
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Yaw (radians)')

    # Optionally set yaw limits to [-π, π]
    ax.set_zlim(-np.pi, np.pi)

    plt.show()

def visualize_3D(set_gen: SetGen):
    """
    Visualize the convex hulls projected to the tabletop of the generated sets.
    """
    # Example: Randomly generate 7D poses and create hulls

    hulls = []
    for node in set_gen.nodes:
        positions = node.points[:, :3]
        quaternions = node.points[:, 3:]

        # Combine into 7D poses
        poses_7d = np.column_stack((positions, quaternions))

        # Create convex hull in 7D space
        hull = ConvexHull(poses_7d)
        hulls.append(hull)

    visualize_convex_hulls(hulls)