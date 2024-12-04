import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
from pydrake.all import (
    Hyperellipsoid,
    AffineBall,
    Intersection,
)
from matplotlib.patches import Polygon
import copy


''' RRT utils: 

'''

# Parameters
class RRTParams:
    def __init__(self):
        self.step_size = 0.5       # Step size for expanding the tree
        self.goal_sample_rate = 0.1  # Probability of sampling any goal
        self.max_iterations = 2000  # Max number of iterations
        self.map_bounds = [0, 10]   # Bounds of the map (square)
        self.obstacles = [          # List of circular obstacles
            (3, 3, 1),
            (7, 7, 1.5),
            (5, 5, 1),
        ]
params = RRTParams()

# Utility functions
def sample_free(goals, params):
    """Sample a random point in free space, prioritising goals."""
    if np.random.rand() < params.goal_sample_rate:
        return goals[np.random.randint(len(goals))]
    return np.random.uniform(params.map_bounds[0], params.map_bounds[1], 2)

def is_in_collision(point, obstacles):
    """Check if a point is in collision with any obstacles."""
    for ox, oy, radius in obstacles:
        if np.linalg.norm(point - np.array([ox, oy])) <= radius:
            return True
    return False

def steer(from_point, to_point, step_size):
    """Steer from 'from_point' towards 'to_point'."""
    direction = to_point - from_point
    distance = np.linalg.norm(direction)
    if distance < step_size:
        return to_point
    return from_point + direction / distance * step_size

def plot_paths(paths, params):
    """Visualise the RRT and paths."""
    fig, ax = plt.subplots()
    plt.xlim(params.map_bounds)
    plt.ylim(params.map_bounds)
    plt.gca().set_aspect('equal')

    # Plot obstacles
    for ox, oy, radius in params.obstacles:
        circle = plt.Circle((ox, oy), radius, color='red', fill=True, alpha=0.2)
        plt.gca().add_patch(circle)

    # # Plot RRT tree
    # for parent, child, col in tree:
    #     plt.plot([parent[0], child[0]], [parent[1], child[1]], '-g')

    # Plot paths to goals
    for path in paths:
        if path:
            for i in range(len(path) - 1):
                
                plt.plot([path[i][0][0], path[i + 1][0][0]], [path[i][0][1], path[i + 1][0][1]], '-b', alpha=0.5)
                if path[i][1]:
                    plt.scatter(path[i][0][0], path[i][0][1], color='r', s=20, marker='x')
                else:
                    plt.scatter(path[i][0][0], path[i][0][1], color='b', s=20, marker='o')
            # path = np.array(path)
            # plt.plot(path[:, 0], path[:, 1], '-b', linewidth=2, marker='o')

    # Mark start and goals
    plt.scatter(paths[0][0][0][0], paths[0][0][0][1], color='yellow', label="Start", s=100)
    # for goal in goals:
    #     plt.scatter(goal[0], goal[1], color='green', s=100, label="Goal" if np.array_equal(goal, goals[0]) else None)
    plt.legend()
    plt.show()


# Multi-goal RRT Algorithm
def rrt_multi_goal(start, goals, params):
    nodes = [start]
    tree = []
    paths = [None] * len(goals)
    reached_goals = set()

    for i in range(params.max_iterations):
        # Sample random point
        sample = sample_free(goals, params)

        # Find nearest node
        nearest_index = KDTree(nodes).query(sample)[1]
        nearest_node = nodes[nearest_index]

        # Steer towards sample
        new_node = steer(nearest_node, sample, params.step_size)

        # Check for collisions
        collision =  is_in_collision(new_node, params.obstacles)

        # Add node and edge
        nodes.append(new_node)
        tree.append((nearest_node, new_node, collision))

        # Check if any goal is reached
        for goal_idx, goal in enumerate(goals):
            if goal_idx in reached_goals:
                continue
            if np.linalg.norm(new_node - goal) < params.step_size:
                # Reconstruct path
                path = [(goal, is_in_collision(goal, params.obstacles))]
                current_node = new_node
                while current_node is not None:
                    col =  is_in_collision(current_node, params.obstacles)
                    path.append((current_node, col))
                    current_node = next((parent for parent, child, col in tree if np.array_equal(child, current_node)), None)
                paths[goal_idx] = path[::-1]
                reached_goals.add(goal_idx)

        # Stop if all goals are reached
        if len(reached_goals) == len(goals):
            break

    return tree, paths

''' Convex set generation:

Args:
    mode_samples (per mode): dict[str, list[TrajectoryPrimitive]]
    ik_solutions: dict[str, list[Optional[np.ndarray]]] # None if invalid IK solution else IK solution (1x24)
    -- parse:
    trajs = list[list[(pose, ik_solutions)]]

    save data dict[str, list[(trajectory, ik_solutions)]] (saved as pkl) ik_solution 24 dim
'''

def parse_traj_primitives(mode_samples, ik_solutions):

    for mode, traj_primitives in mode_samples.items():
        assert len(traj_primitives) == len(ik_solutions[mode])
        for i in range(len(traj_primitives)):
            traj = traj_primitives[i].trajectory
            assert len(traj) == len(ik_solutions[mode][i])
            

        ik_solution = ik_solutions[mode]
        traj = primitive.trajectory
        assert len(ik_solution) == len(traj)

        traj = traj_primitives[mode]
        trajs.append([(pose, ik_solutions[mode][i]) for i, pose in enumerate(traj.poses)])
    return trajs

def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull, qhull_options='QJ')

    return hull.find_simplex(p)>=0

def construct_combined_set(node1, node2):
    s1 = node1[1]
    s2 = node2[1]
    combined_set = np.concatenate([s1, s2], axis=0)
    # print(combined_set.shape)
    # compute the convex hull of the two sets
    combined_hull = ConvexHull(combined_set, qhull_options='QJ')
    # compute the set density:
    volume = combined_hull.volume
    density = len(combined_set) / volume
    return combined_hull, combined_set, density 

def construct_set(paths, density_threshold=20, max_iterations=2000):
    ## construct initial sets based on paths:
    Nodes = []
    node_indx = 0
    negative_points = []
    for path in paths:
        if path:
            for i in range(len(path) - 1):
                p1 = path[i]
                p2 = path[i + 1]
                if p1[1]:
                    # invalid IK solution
                    negative_points.append(p1[0])
                    continue
                if p2[1]:
                    negative_points.append(p2[0])
                    continue
                segment_points = np.array([p1[0], p2[0]])
                # compute set from segment
                
                node = [None, segment_points, node_indx]
                Nodes.append(node)
                node_indx += 1
    print('Initial number of nodes: ', len(Nodes))
    ## merge sets:
    negative_points = np.array(negative_points)
    track_nodes = []
    for merge_iteration in range(min(len(Nodes), max_iterations)):
        next_iter = False
        set_centers = []
        for node in Nodes:
            set_centers.append(np.mean(node[1], axis=0))
        kdtree = KDTree(set_centers)
        if len(Nodes) == 1:
            break
        for i in range(len(Nodes)):
            _, closest_indices = kdtree.query(set_centers[i], k=3) # get the top 5 closest
            for j in closest_indices:
                if i == j:
                    continue
                ## compute the merged set density:

                E, union_points, density = construct_combined_set(Nodes[i], Nodes[j])
                
                # print(density)
                # add check that no negative points are in the set:
                ### check
                if density:
                    if in_hull(negative_points, union_points).any():
                        continue
                    if density > density_threshold:
                        node = [E, union_points, node_indx]
                        Nodes.append(node)
                        node_indx += 1
                        if i > j:
                            Nodes.pop(i)
                            Nodes.pop(j)
                        else:
                            Nodes.pop(j)
                            Nodes.pop(i)
                        next_iter=True
                        break
            if next_iter:
                break
        
        
        # check if the previous graph is the same:
        if track_nodes:
            if len(track_nodes[-1]) == len(Nodes):
                break
        track_nodes.append(copy.deepcopy(Nodes))
        
    print(len(track_nodes))
    print(f"Number of nodes after merging: {len(Nodes)}")
    return Nodes, track_nodes

def visualise_sets(paths, params, Nodes):
    """Visualise the RRT and paths."""
    fig, ax = plt.subplots()
    plt.xlim(params.map_bounds)
    plt.ylim(params.map_bounds)
    plt.gca().set_aspect('equal')

    # Plot obstacles
    for ox, oy, radius in params.obstacles:
        circle = plt.Circle((ox, oy), radius, color='red', fill=True, alpha=0.2)
        plt.gca().add_patch(circle)

    # # Plot RRT tree
    # for parent, child, col in tree:
    #     plt.plot([parent[0], child[0]], [parent[1], child[1]], '-g')

    # Plot paths to goals
    # for path in paths:
    #     if path:
    #         for i in range(len(path) - 1):
                
    #             plt.plot([path[i][0][0], path[i + 1][0][0]], [path[i][0][1], path[i + 1][0][1]], '-b', alpha=0.5)
    #             if path[i][1]:
    #                 plt.scatter(path[i][0][0], path[i][0][1], color='r', s=20, marker='x')
    #             else:
    #                 plt.scatter(path[i][0][0], path[i][0][1], color='b', s=20, marker='o')
            # path = np.array(path)
            # plt.plot(path[:, 0], path[:, 1], '-b', linewidth=2, marker='o')

    for node in Nodes:
        if node is not None:
            scv = node[0]
            if scv is not None:
                plt.fill(scv.points[scv.vertices, 0], scv.points[scv.vertices, 1], color='lightblue', alpha=0.5)
            else:
                plt.plot(node[1][:, 0], node[1][:, 1], color='lightblue', alpha=0.5)

    # Mark start and goals
    plt.scatter(paths[0][0][0][0], paths[0][0][0][1], color='yellow', label="Start", s=100)
    # for goal in goals:
    #     plt.scatter(goal[0], goal[1], color='green', s=100, label="Goal" if np.array_equal(goal, goals[0]) else None)
    plt.legend()
    plt.show()

# Main
if __name__ == "__main__":
    np.random.seed(42)
    start = np.array([7, 3])
    # goals = [np.array([9, 9]), np.array([1, 8]), np.array([8, 1])]
    # generate random goals:
    paths = []
    for i in range(50):
        goals = [np.random.uniform(params.map_bounds[0], params.map_bounds[1], 2) for _ in range(1)]
        tree, path = rrt_multi_goal(start, goals, params)
        paths.append(path[0])

    plot_paths(paths, params)

    Nodes, track_nodes  = construct_set(paths)

    visualise_sets(paths, params, Nodes)
    # for i in range(len(track_nodes)):
    #     if i % 30 == 0:
    #         visualise_sets(paths, params, track_nodes[i])