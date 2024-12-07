import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import (
    KDTree,
    ConvexHull,
    Delaunay,
)
import copy
from matplotlib.patches import Polygon
import copy
import pickle
import types
from joblib import Parallel, delayed
import math


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
            (-2, -2, 11),
            (12, 12, 11),
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


def plot_paths(paths, params, path_col):
    """Visualise the RRT and paths."""
    fig, ax = plt.subplots()
    plt.xlim(params.map_bounds)
    plt.ylim(params.map_bounds)
    plt.gca().set_aspect('equal')

    # Plot obstacles
    for ox, oy, radius in params.obstacles:
        circle = plt.Circle((ox, oy), radius, color='red', fill=False, alpha=0.2)
        plt.gca().add_patch(circle)

    # # Plot RRT tree
    # for parent, child, col in tree:
    #     plt.plot([parent[0], child[0]], [parent[1], child[1]], '-g')

    # Plot paths to goals
    for path in paths:
        if path:
            for i in range(len(path) - 1):
                
                plt.plot([path[i][0][0], path[i + 1][0][0]], [path[i][0][1], path[i + 1][0][1]], path_col, alpha=0.5)
                if path[i][1]:
                    plt.scatter(path[i][0][0], path[i][0][1], color='r', s=20, marker='x')
                else:
                    plt.scatter(path[i][0][0], path[i][0][1], color=path_col, s=20, marker='o')
            # path = np.array(path)
            # plt.plot(path[:, 0], path[:, 1], '-b', linewidth=2, marker='o')

    # Mark start and goals
    # plt.scatter(paths[0][0][0][0], paths[0][0][0][1], color='yellow', label="Start", s=100)
    # for goal in goals:
    #     plt.scatter(goal[0], goal[1], color='green', s=100, label="Goal" if np.array_equal(goal, goals[0]) else None)
    # plt.legend()
    # plt.show()


# Multi-goal RRT Algorithm
def rrt_multi_goal(start, goals, params, invert_obstacles=False):
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
                    if invert_obstacles:
                        col = not col
                    path.append((current_node, col))
                    current_node = next((parent for parent, child, col in tree if np.array_equal(child, current_node)), None)
                paths[goal_idx] = path[::-1]
                reached_goals.add(goal_idx)

        # Stop if all goals are reached
        if len(reached_goals) == len(goals):
            break

    return tree, paths

def run_path_gen(params, start= np.array([1,1]), num_paths = 10, path_col = 'b', invert_obstacles=False):
    paths = []
    for i in range(num_paths):
        goals = [np.random.uniform(params.map_bounds[0], params.map_bounds[1], 2) for _ in range(1)]
        tree, path = rrt_multi_goal(start, goals, params, invert_obstacles=invert_obstacles)
        paths.append(path[0])

    plot_paths(paths, params, path_col)
    return paths

''' Set gen stuff:

'''


## the slow step - could be parallelised?
def in_hull(points, equations, tol=1e-12):
    return np.any(
        np.all(
            np.add(np.dot(points, equations[:, :-1].T), equations[:, -1]) <= tol,
            axis=1,
        )
    )
    ## update the in_hull function to use the seperated orientation and translation vectors:



#### define a node struct consisting of [set, points in the set, node indx]
class Node:
    def __init__(self, set, points, node_indx):
        """
        Args:
            set: ConvexHull object
            points: np.array of points in the set
            node_indx: int
        """
        self.set: ConvexHull = set
        self.points = points
        self.node_indx = node_indx
        self.center = np.mean(points, axis=0)
        self.density = None

    def __call__(self):
        pass

    def gen_set(self):
        """generate the convex hull set - if there are less than 8 points, linearly interpolate between them:"""
        if len(self.points) < 4:
            # randomly sample 8 - len(points) points from the first two points of convex hull (just for utility):
            t = np.linspace(0, 1, 4 - len(self.points))
            qt = self.points[0] + t[:, None] * (self.points[1] - self.points[0])
            qt = np.concatenate([qt, self.points], axis=0)
            self.set = ConvexHull(qt, qhull_options="QJ")
        else:
            self.set = ConvexHull(self.points, qhull_options="QJ")

    def __str__(self):
        return f"Node {self.node_indx} with {len(self.points)} points"

    def __repr__(self):
        return f"Node {self.node_indx} with {len(self.points)} points"

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        return self.points[idx]

    def __setitem__(self, idx, val):
        self.points[idx] = val

    def __iter__(self):
        return iter(self.points)

    def __next__(self):
        return next(self.points)

    def __contains__(self, item):
        """Args:
            item: np.array of shape (n, d) where n is the number of points and d is the dimension of the points
        Desc:
            checks if the item is in the convex hull by constructing a Delaunay triangulation
        """
        # compute simplex points:
        # vertex_points = self.set.points[self.set.vertices]
        # print(f'Point reduction: {len(self.points)} -> {len(vertex_points)}')
        # return (Delaunay(vertex_points, qhull_options='QJ').find_simplex(item) >= 0).any()
        # return self.t_contains(item, 1e-12)
        return in_hull(item, self.set.equations, 1e-12)

    def convert_to_drake(self):
        """Converts the scipy convex hull objet to a drake object"""
        pass

    def __or__(self, other):
        combined_set = np.concatenate([self.points, other.points], axis=0)
        # combined_hull = ConvexHull(combined_set, qhull_options='QJ')
        return Node(
            None, combined_set, None
        )  # return a new node object with an empty node_indx and set object

    def volume(self):
        if self.set is None:
            self.gen_set()
        return self.set.volume

    def _density(self):
        if self.density is not None:
            return self.density
        self.density = len(self.points) / self.volume()
        return self.density
        # return self._spatial_density() * self._orientation_density()
    
    def compress(self):
        self.points = np.unique(
            np.round(self.points, decimals=10), axis=0
        )  # roundin avoids precision issues
        if self.set is None:
            self.gen_set()
        return


    def lossy_compression(self, other, negative_points, points_ratio=0.95, max_iterations=20):
        v1 = self.volume()
        v2 = other.volume()
        min_vol = max(v1, v2)
        
        combined_set = np.concatenate([self.points, other.points], axis=0)
        num_of_points = len(combined_set)
        num_of_reduced_points = int(points_ratio*num_of_points)
        
        def evaluate_sample():
            # Sample a random set of points
            sample_points = combined_set[
                np.random.choice(len(combined_set), num_of_reduced_points, replace=False)
            ]
            # Compute the convex hull of the sample points
            # compute the node object for the sample hull:
            sample_node = Node(None, sample_points, None)
            sample_node.gen_set()
            # evaluate whether the sample hull contains the negative points:
            contains_negative = np.array(negative_points) in sample_node
            # Return the sample hull and its volume
            return sample_node, sample_points, contains_negative
        
        # Run RANSAC iterations in parallel
        results = Parallel(n_jobs=-1)(
            delayed(evaluate_sample)() for _ in range(max_iterations)
        )

        # Find the best result
        max_vol, max_node, max_points = None, None, None
        for sample_node, sample_points, contains_negative in results:
            if contains_negative:
                # print('tested set contains negative points')
                continue
            volume = sample_node.volume()
            if volume > min_vol:
                max_vol = volume
                max_node = sample_node
                max_points = sample_points
        if max_vol is None:
            print("Failure to compress")
            return None
        else:
            # # Compute members of the compressed hull
            # in_compressed_hull = np.all(
            #     np.add(
            #         np.dot(combined_set, max_set.equations[:, :-1].T),
            #         max_set.equations[:, -1],
            #     )
            #     <= 1e-12,
            #     axis=1,
            # )
            # merged_points = combined_set[in_compressed_hull]
            print('Nodes merged,set vol ratio: ', max_vol/min_vol, 'points in new set: ', len(max_points))
            return max_node
        

class SetGen:
    def __init__(self):

        self.nodes: list[Node] = []
        self.negative_points: list[np.ndarray] = []
        self.node_indx: int = 0

    def __call__(self, paths):
        self.construct_initial_sets(paths)

    def construct_initial_sets(self, paths):
        """construct the initial set problem from the path data
        Args:
            list[(trajectory, ik_solutions)]
        Desc:

        """
        for i, traj in enumerate(paths):
            for i in range(len(traj) - 1):
                X1 = traj[i][0]  # need to convert to 7D
                X2 = traj[i + 1][0]  # need to convert to 7D
                if traj[i][1] is True:
                    self.negative_points.append(X1)
                    continue
                if traj[i + 1][1] is True:
                    self.negative_points.append(X2)
                    continue
                segment_points = np.array([X1, X2])
                node = Node(None, segment_points, self.node_indx)
                self.nodes.append(node)
                self.node_indx += 1
        print("Initial number of nodes: ", len(self.nodes))
        print("Number of negative points: ", len(self.negative_points))

    def deshatter(
        self,
        max_iterations=5000,
        density_threshold=0,
        k=10,
    ):
        """deshatter the nodes by merging them together"""

        initial_node_count = len(self.nodes)
        check_pairs = []
        for merge_iter in range(min(len(self.nodes), max_iterations)):
            current_node_count = len(self.nodes)
            print(f"Iteration {merge_iter}")
            print("Current Number of nodes: ", len(self.nodes))

            exit_flag = False
            if len(self.nodes) == 1:
                break
            # sort the nodes by density (highest to lowest)
            self.nodes.sort(key=lambda x: x._density(), reverse=True)
            # randomly shuffle the nodes
            # np.random.shuffle(self.nodes)
            # compute KD Tree based on node centers
            kdtree = KDTree([node.center for node in self.nodes])
            # loop through the nodes:
            for i, node in enumerate(self.nodes):
                # find the nearest node to the current node

                _, closest_indices = kdtree.query(
                    node.center, k=min(k, len(self.nodes))
                )  # get the top k closest
                for j in closest_indices:
                    if i == j:
                        continue
                    test_node = self.nodes[j]
                    print(f"Checking for node: {node} and test node: {test_node}")

                    # check if the pair has already been checked
                    if (node.node_indx, test_node.node_indx) in check_pairs or (
                        test_node.node_indx,
                        node.node_indx,
                    ) in check_pairs:
                        print("Already checked this pair" + "-" * 50)
                        continue

                    merged = (
                        node | test_node
                    )  # only compute the convex hull when needed

                    # print("Checking for compression:")
                    merged.compress()

                    check_pairs.append((node.node_indx, test_node.node_indx))
                    check_pairs.append((test_node.node_indx, node.node_indx))

                    # check if the merged node contains a singularity
                    # print("Checking for singularity")
                    # if merged.contains_singularity():
                    #     print("Singularity detected" + '-'*50)
                    #     continue
                    # check if the merged node contains any negative points
                    print("Checking for negative points")
                    if np.array(self.negative_points) in merged:
                        print("Negative points detected" + "-" * 50)
                        continue
                    # check if the density of the merged node is above the threshold
                    print("Checking for density")
                    if (
                        merged._density() > density_threshold
                    ):
                        print(
                            "Density threshold exceeded, merging nodes: ",
                            f"spatial: {merged._spatial_density()}, orientation: {merged._orientation_density()}",
                        )
                        # remove the two nodes and add the merged node
                        # give the merged node an index:
                        merged.node_indx = self.node_indx
                        self.node_indx += 1
                        self.nodes.remove(node)
                        self.nodes.remove(test_node)
                        self.nodes.append(merged)
                        exit_flag = True
                        break
                if exit_flag:
                    break
            updated_node_count = len(self.nodes)
            if current_node_count == updated_node_count:
                print("No more nodes to merge, exiting")
                break

        print("Final number of nodes: ", len(self.nodes))

    def lossy_deshatter(self, max_iterations=5000,
        density_threshold=0,
        points_ratio=0.95,
        k=10):
        """deshatter the nodes by merging them together"""

        original_number_of_positive_points = np.sum([len(n.points) for n in self.nodes])

        initial_node_count = len(self.nodes)
        check_pairs = []
        for merge_iter in range(min(len(self.nodes), max_iterations)):
            current_node_count = len(self.nodes)
            print(f"Iteration {merge_iter}")
            print("Current Number of nodes: ", len(self.nodes))
            current_pos_points = np.sum([len(n.points) for n in self.nodes])
            print(f'Current number of positive points vs original: {original_number_of_positive_points} -> {current_pos_points}')

            exit_flag = False
            if len(self.nodes) == 1:
                break
            # sort the nodes by density (highest to lowest)
            self.nodes.sort(key=lambda x: x._density(), reverse=True)
            # randomly shuffle the nodes
            # np.random.shuffle(self.nodes)
            # compute KD Tree based on node centers
            kdtree = KDTree([node.center for node in self.nodes])
            # loop through the nodes:
            for i, node in enumerate(self.nodes):
                # find the nearest node to the current node

                _, closest_indices = kdtree.query(
                    node.center, k=min(k, len(self.nodes))
                )  # get the top k closest
                for j in closest_indices:
                    if i == j:
                        continue
                    test_node = self.nodes[j]
                    print(f"Checking for node: {node} and test node: {test_node}")

                    # check if the pair has already been checked
                    if (node.node_indx, test_node.node_indx) in check_pairs or (
                        test_node.node_indx,
                        node.node_indx,
                    ) in check_pairs:
                        print("Already checked this pair" + "-" * 50)
                        continue

                    merged = node.lossy_compression(test_node, np.array(self.negative_points), points_ratio=points_ratio, max_iterations=20)
                    if merged is None:
                        continue
                    else:
                        merged.node_indx = self.node_indx
                        self.node_indx += 1
                        self.nodes.remove(node)
                        self.nodes.remove(test_node)
                        self.nodes.append(merged)
                        exit_flag = True
                        break
                if exit_flag:
                    break
            updated_node_count = len(self.nodes)
            if current_node_count == updated_node_count:
                print("No more nodes to merge, exiting")
                break
        print("Final number of nodes: ", len(self.nodes))

if __name__ == "__main__":
    np.random.seed(42)
    dynamic_params_1 = RRTParams()
    dynamic_params_1.obstacles = [
        (12, 12, 12),
    ]
    dynamic_params_1_start = np.array([1,1])
    dynamic_params_2 = RRTParams()
    dynamic_params_2.obstacles = [
        (-2, -2, 12),
    ]
    dynamic_params_2_start = np.array([9,9])

    static_params = RRTParams()
    static_params.obstacles = [
       (5,5,3)
    ]
    static_params_start = np.array([5,5])



    dynamic_1_paths = run_path_gen(dynamic_params_1, dynamic_params_1_start)
    dynamic_2_paths = run_path_gen(dynamic_params_2, dynamic_params_2_start)
    static_paths = run_path_gen(static_params, static_params_start, path_col='g', invert_obstacles=True)
    plt.show()

    # compute the sets for the dynamic and static sets
    contact_mode_paths = {
        'dynamic_1': dynamic_1_paths,
        'dynamic_2': dynamic_2_paths,
        'static':  static_paths
    }

    
    for contact_mode in contact_mode_paths.keys():
        print(f"Contact mode: {contact_mode}")
        set_gen = SetGen()
        set_gen.construct_initial_sets(contact_mode_paths[contact_mode])
        set_gen.deshatter()
        # print set statistics:
        for node in set_gen.nodes:
            print(f"Node {node.node_indx} with {len(node)} points, density: {node._density()}")
        print("+"*50)

