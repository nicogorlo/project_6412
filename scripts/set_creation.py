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
import pickle
import types
from joblib import Parallel, delayed

## the slow step - could be parallelised?
def in_hull(points, equations):
    return np.any(np.all(np.add(np.dot(points, equations[:,:-1].T), equations[:,-1]) <= 1e-12, axis=1))

def in_hull_parallel(points, equations, tol=1e-12, n_jobs=-1):
    """
    Check if any point lies inside the convex hull defined by the equations using parallel processing.

    Parameters:
    points (np.ndarray): Array of points to check, shape (N, D).
    equations (np.ndarray): Array of convex hull equations, shape (M, D+1).
    tol (float): Tolerance for the inequality check.
    n_jobs (int): Number of parallel jobs. Default is -1 (use all available cores).

    Returns:
    bool: True if any point lies inside the convex hull, False otherwise.
    """
    def check_point(point):
        return np.all(np.dot(point, equations[:, :-1].T) + equations[:, -1] <= tol)

    results = Parallel(n_jobs=n_jobs)(delayed(check_point)(point) for point in points)
    return np.any(results)

def in_hull_parallel_batch(points, equations, tol=1e-12, n_jobs=-1, batch_size=100):
    """
    Check if any point lies inside the convex hull defined by the equations using parallel processing with batch support.

    Parameters:
    points (np.ndarray): Array of points to check, shape (N, D).
    equations (np.ndarray): Array of convex hull equations, shape (M, D+1).
    tol (float): Tolerance for the inequality check.
    n_jobs (int): Number of parallel jobs. Default is -1 (use all available cores).
    batch_size (int): Number of points to process in each batch.

    Returns:
    bool: True if any point lies inside the convex hull, False otherwise.
    """
    def process_batch(batch):
        return np.any(
            np.all(np.dot(batch, equations[:, :-1].T) + equations[:, -1] <= tol, axis=1)
        )
    
    # Split points into batches
    batches = [points[i:i+batch_size] for i in range(0, len(points), batch_size)]

    # Parallel processing of batches
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_batch)(batch) for batch in batches
    )
    
    return np.any(results)

#### define a node struct consisting of [set, points in the set, node indx]
class Node():
    def __init__(self, set, points, node_indx):
        '''
        Args:
            set: ConvexHull object
            points: np.array of points in the set
            node_indx: int
        '''
        self.set = set
        self.points = points
        self.node_indx = node_indx
        self.center = np.mean(points, axis=0)
        self.density = None

    def __call__(self):
        pass

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
        '''Args:
            item: np.array of shape (n, d) where n is the number of points and d is the dimension of the points
        Desc:
            checks if the item is in the convex hull by constructing a Delaunay triangulation
        '''
        # compute simplex points:
        # vertex_points = self.set.points[self.set.vertices]
        # print(f'Point reduction: {len(self.points)} -> {len(vertex_points)}')
        # return (Delaunay(vertex_points, qhull_options='QJ').find_simplex(item) >= 0).any()
        return self.t_contains(item, 1e-12)
    
    def t_contains(self, item, tol):
        ''' Faster contains operation via linear programming:
        Args:
            item: np.array of shape (n, d) where n is the number of points and d is the dimension of the points
            tol: float tolerance
        '''
        hull = self.set
        return in_hull_parallel_batch(item, hull.equations)
    
    def convert_to_drake(self):
        ''' Converts the scipy convex hull objet to a drake object
        '''
        pass

    def __or__(self, other):
        combined_set = np.concatenate([self.points, other.points], axis=0)
        combined_hull = ConvexHull(combined_set, qhull_options='QJ')
        return Node(combined_hull, combined_set, None) # return a new node object with an empty node_indx
    
    def volume(self):
        return self.set.volume
    
    def _density(self):
        if self.density is not None:
            return self.density
        self.density = len(self.points) / self.volume()
        return self.density
    
    def contains_singularity(self):
        ''' check if the node contains a singularity
        Desc:
            - computes the convex hull of only the quaternion portion of the points
            - returns True if the origin is in the convex hull, False otherwise
        '''
        quat_points = self.points[:, 3:]
        return Delaunay(quat_points, qhull_options='QJ').find_simplex(np.array([[0, 0, 0, 0]])) >= 0
        

class SetGen():
    def __init__(self):

        self.nodes = []
        self.negative_points = []
        self.node_indx = 0

        pass
    def __call__(self):
        pass

    def construct_initial_sets(self, paths):
        ''' construct the initial set problem from the path data
        Args:
            list[(trajectory, ik_solutions)]
        Desc:

        '''
        for i, (traj, ik_sols) in enumerate(paths):
            assert len(traj) == len(ik_sols)
            for i in range(len(traj) - 1):
                X1 = self.X_to_7D(traj[i]) # need to convert to 7D
                X2 = self.X_to_7D(traj[i + 1]) # need to convert to 7D
                if ik_sols[i] is None:
                    self.negative_points.append(X1)
                    continue
                if ik_sols[i + 1] is None:
                    self.negative_points.append(X2)
                    continue
                segment_points = np.array([X1, X2])
                set_points = self.linear_interpolation(segment_points)
                E = ConvexHull(set_points, qhull_options='QJ')
                node = Node(E, set_points, self.node_indx)
                self.nodes.append(node)
                self.node_indx += 1
        print('Initial number of nodes: ', len(self.nodes))
        print('Number of negative points: ', len(self.negative_points))

    def X_to_7D(self, X):
        ''' convert the SE(3) RigidBody pose to 7D position (x, y, z, q1, q2, q3, q4)
        '''
        quat = X.rotation().ToQuaternion()
        return np.concatenate([X.translation(), np.array([quat.w(), quat.x(), quat.y(), quat.z()])])
    
    def linear_interpolation(self, segment, points = 8):
        ''' generate the initial sets from the nodes
        Args:
            segments: numpy array of the two points which define the convex hull
            points: interpolation numer of points
        Desc:
            QuickHull algorithms requires additional points to construct an initial simplex
            Performs simple linear interpolation to compute the additional points along the trajectory path:
        '''
        t = np.linspace(0, 1, points) 
        qt = segment[0] + t[:, None] * (segment[1] - segment[0])
        # normalise the quaternion portion of the points:
        qt[:, 3:] = qt[:, 3:] / np.linalg.norm(qt[:, 3:], axis=1)[:, None]
        return qt
    
    def deshatter(self, max_iterations = 2000, density_threshold = 30, k=3):
        ''' deshatter the nodes by merging them together
        '''

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
                
                _, closest_indices = kdtree.query(node.center, k=k) # get the top k closest
                for j in closest_indices:
                    if i == j:
                        continue
                    test_node = self.nodes[j]
                    print(f'Checking for node: {node} and test node: {test_node}')

                    # check if the pair has already been checked
                    if (node.node_indx, test_node.node_indx) in check_pairs or (test_node.node_indx, node.node_indx) in check_pairs:
                        print("Already checked this pair" + '-'*50)
                        continue

                    # check if it's the first merge:
                    if test_node.node_indx < initial_node_count:
                        test_red_node_points = np.array([test_node.points[0], test_node.points[-1]])
                        test_red_node = Node(None, test_red_node_points, None)
                        merged = node | test_red_node
                    else:
                        merged = node | test_node

                    check_pairs.append((node.node_indx, test_node.node_indx))
                    check_pairs.append((test_node.node_indx, node.node_indx))
                    
                    # check if the merged node contains a singularity
                    print("Checking for singularity")
                    if merged.contains_singularity():
                        print("Singularity detected" + '-'*50)
                        continue
                    # check if the merged node contains any negative points
                    print("Checking for negative points")
                    if np.array(self.negative_points) in merged:
                        print("Negative points detected" + '-'*50)
                        continue
                    # check if the density of the merged node is above the threshold
                    print("Checking for density")
                    if merged._density() > density_threshold:
                        print("Density threshold exceeded, merging nodes: ", merged._density())
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
def main():
    # load in the data:
    with open("../output/trajectories_tabletop.pkl", "rb") as f:
        out_structure = pickle.load(f)
    
    print(len(out_structure))
    print(out_structure.keys())

    set_gen = SetGen()
    set_gen.construct_initial_sets(out_structure['X_POS'])
    set_gen.deshatter()

if __name__ == "__main__":
    main()