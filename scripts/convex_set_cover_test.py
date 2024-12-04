import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from pydrake.all import (
    Hyperellipsoid,
    AffineBall,
    Intersection,
)
from matplotlib.patches import Ellipse

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

def plot_paths(tree, paths, params, goals, Nodes, Edges):
    """Visualise the RRT and paths."""
    fig, ax = plt.subplots()
    plt.xlim(params.map_bounds)
    plt.ylim(params.map_bounds)
    plt.gca().set_aspect('equal')

    # Plot obstacles
    for ox, oy, radius in params.obstacles:
        circle = plt.Circle((ox, oy), radius, color='red', fill=True)
        plt.gca().add_patch(circle)

    # # Plot RRT tree
    # for parent, child, col in tree:
    #     plt.plot([parent[0], child[0]], [parent[1], child[1]], '-g')

    # Plot paths to goals
    for path in paths:
        if path:
            for i in range(len(path) - 1):
                
                plt.plot([path[i][0][0], path[i + 1][0][0]], [path[i][0][1], path[i + 1][0][1]], '-b')
                if path[i][1]:
                    plt.scatter(path[i][0][0], path[i][0][1], color='b', s=20, marker='x')
                else:
                    plt.scatter(path[i][0][0], path[i][0][1], color='b', s=20, marker='o')
            # path = np.array(path)
            # plt.plot(path[:, 0], path[:, 1], '-b', linewidth=2, marker='o')

    # Mark start and goals
    plt.scatter(paths[0][0][0][0], paths[0][0][0][1], color='yellow', label="Start", s=100)
    # for goal in goals:
    #     plt.scatter(goal[0], goal[1], color='green', s=100, label="Goal" if np.array_equal(goal, goals[0]) else None)

    for node in Nodes:
        visualise_set(ax, node[0])
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


def construct_set(paths, top_k=5, density_threshold=30):
    ''' construct a graph of convex sets from the path
    Args:
        paths: list of paths to goals
    Returns:
        graph: a graph of convex sets where there are few invalid IK solutions
    Desc:
        1. For each path, slide into trajectory segments and compute the convex set of each segment
        2. For each of the sets, add the relevant connections to the graph based on the trajectory
        4. compute KDTree for the graph based on approx set centres
        3. For a fixed number of iterations: (loop over non-intersecting sets) worse case O(N^2)
            a. select a random pair of close sets
            b. find the intersection of the two sets
            c. if the two sets intersect, add the intersection to the connectivity graph
        4. For a fixed number of iterations: (loop over intersecting sets) worst case O(N^2)
            a. select a random pair of close sets
            b. compute the set associated with the intersection of the two sets of points
            c. compute the density of the set (number of points in the set/volume)
            d. if the density is greater than a threshold:
                i. replace the two sets with the intersection set
                ii. add the union of connectivity to the graph
                iii. remove the two sets from the graph
    '''

    # Step 1: Slide into trajectory segments and compute the convex set of each segment
    ## each node is described by a tuple [set, points, index]
    ## each edge is described by a tuple [node1_index, node2_index, intersection=None]
    Nodes = []
    Edges = []

    node_indx = 0
    for path in paths:
        if path:
            for i in range(len(path) - 1):
                p1 = path[i]
                p2 = path[i + 1]
                if p1[1] or p2[1]:
                    # invalid IK solution
                    continue
                segment_points = [p1[0], p2[0]]
                # compute set from segment
                cvx_set = compute_line_segment_affine_ball(p1[0], p2[0])
                
                node = [cvx_set, segment_points, node_indx]
                Nodes.append(node)
                node_indx += 1

                # Step 2: For each of the sets, add the relevant connections to the graph based on the trajectory
                if i > 0:
                    Edges.append([node_indx - 1, node_indx])
    print(len(Nodes), len(Edges))
    # return Nodes, Edges
    ### compute KD trees for the graph based on approx set centres
    for merge_iteration in range(500):
        next_iter = False
        set_centers = [node[0].center() for node in Nodes]
        kdtree = KDTree(set_centers)
        ### For each set, find the closest k sets
        if len(Nodes) == 1:
            print(len(Nodes[0][1]))
            break
        for i in range(len(Nodes)):
            _, closest_indices = kdtree.query(set_centers[i], k=2) # get the top 5 closest
            for j in closest_indices:
                if i == j:
                    continue
                # compute intersection of the two sets
                (E, union_points), density = compute_merged_density(Nodes[i], Nodes[j])
                if density:
                    if density > density_threshold:
                        node = [E, list(union_points), node_indx]
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
                        pass
                # else:
                #     # almost perfect overlap:
                #     Nodes.pop(i)
                #     next_iter=True
                #     break
            if next_iter:
                break
                        


            # if intersection_vol > 0:
            #     # print(intersection_vol)
            #     Edges.append([i, j, intersection_vol])

                ### compute the set density if two sets merged
                

    return Nodes, Edges

def compute_volume(set):
    B = set.B()
    return np.abs(np.linalg.det(B))

def compute_merged_density(node1, node2):
    ''' Computes the set density of two merged sets:
    - take the union of the two set points
    - compute the minimum spanning ellipsoid of the joint points
    - compute the density of the minimm spanning ellipsoid
    '''
    set1_points = node1[1]
    set2_points = node2[1]
    union_points = np.array(set1_points + set2_points)
    E = AffineBall.MinimumVolumeCircumscribedEllipsoid(union_points.T)

    V = compute_volume(E)
    if V ==0.:
        return (E, union_points), None
    density = len(union_points)/V
    return (E, union_points), density

def compute_line_segment_affine_ball(p1, p2, buffer = 0.04):
    D = p1.shape[0]
    c = (p1 + p2) / 2

    u = p2 - c
    u = u / np.linalg.norm(u)

    v = np.zeros(D)
    v[0] = 1.0
    v = u - v

    M = np.eye(D)- 2 * v[:, None]@v[:, None].T/(np.linalg.norm(v)**2) # Householder transformation
    
    # multiply final row by -1:
    M[-1] = -M[-1]

    # compute Det M:
    # print('det M:', np.linalg.det(M))


    S = np.eye(D)*np.linalg.norm(p2/2 - p1/2)**2 /2
    S[0, 0] = np.linalg.norm(p2/2 - p1/2)**2 + buffer
    # print(S)
    return AffineBall(M.T@S@M, c)

def compute_intersection_volume_sampling(set1, set2, samples = 100):
    ## cmpute the intersection volume via sampling:
    c1= set1.center()
    B1 = set1.B()
    c2 = set2.center()
    B2 = set2.B()
    D = c1.shape[0]
    N1 = np.random.randn(samples, D)
    # normalise to lie on the unit ball:
    # N1 = np.where(np.linalg.norm(N1, axis=-1, keepdims=True) > 1, N1/(np.linalg.norm(N1, axis=-1, keepdims=True)), N1)
    N1 = N1/(np.linalg.norm(N1, axis=-1, keepdims=True))
    P1 = np.matmul(B1, N1[..., None]) + c1[..., None] # P = Bu'(Bu1 + c1 - c2) <= 1
    # invert the transform to get to the other ball: (must be full rank)
    N2 = np.linalg.inv(B2)@(P1 - c2[..., None])
    # compute the number of the resltig points wich lie in the unit ball:
    Q = np.mean(np.linalg.norm(N2[..., 0], axis=-1) <= 1.0)
    return Q



def visualise_set(ax, affine_ball):

    # compute the eigenvalues and eigenvectors of B:
    # print(affine_ball.B())
    eigenvalues, eigenvectors = np.linalg.eigh(affine_ball.B())
    # compute the angle of rotation
    c  = affine_ball.center()
    ellipse = Ellipse(c, width=2 * np.sqrt(np.abs(eigenvalues[0])), height=2 * np.sqrt(np.abs(eigenvalues[1])),
                  angle=np.degrees(np.arctan2(*eigenvectors[:, 0][::-1])), color='b', alpha=0.5)
    
    ax.add_patch(ellipse)
    # ax.plot(c[0], c[1], 'ro')  # Red dot for center
    # print(c)
    # print(2 * np.sqrt(eigenvalues[0]))
    # print(2 * np.sqrt(eigenvalues[1]))

def visualise_sets(Nodes, paths):
    fig, ax = plt.subplots()
    for node in Nodes:
        visualise_set(ax, node[0])
    for path in paths:
        if path:
            for i in range(len(path) - 1):
                
                plt.plot([path[i][0][0], path[i + 1][0][0]], [path[i][0][1], path[i + 1][0][1]], '-b')
                if path[i][1]:
                    plt.scatter(path[i][0][0], path[i][0][1], color='b', s=20, marker='x')
                else:
                    plt.scatter(path[i][0][0], path[i][0][1], color='b', s=20, marker='o')
    # set aspect equal
    ax.set_aspect('equal', 'box')
    plt.show()

    


# Main
if __name__ == "__main__":
    np.random.seed(42)
    start = np.array([6, 3])
    # goals = [np.array([9, 9]), np.array([1, 8]), np.array([8, 1])]
    # generate random goals:
    paths = []
    for i in range(10):
        goals = [np.random.uniform(params.map_bounds[0], params.map_bounds[1], 2) for _ in range(1)]
        tree, path = rrt_multi_goal(start, goals, params)
        paths.append(path[0])
    
    print('start')
    Nodes, Edges = construct_set(paths, None)
    print('done')
    print(len(Nodes))
    visualise_sets(Nodes, paths)

    for idx, path in enumerate(paths):
        if path:
            print(f"Path to goal {idx + 1} found!")
        else:
            print(f"Goal {idx + 1} not reachable.")
    plot_paths(tree, [path for path in paths if path], params, goals, Nodes, Edges)
