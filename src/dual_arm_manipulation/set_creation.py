import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import (
    KDTree,
    ConvexHull,
    Delaunay,
)
from scipy.spatial.transform import Rotation as R
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
import math

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from dual_arm_manipulation import ROOT_DIR
from dual_arm_manipulation.utils import performance_measure

"""
TODO:
- Implement the convex hull compression algorithm for reduced dimensionality:

"""


## the slow step - could be parallelised?
def in_hull(points, equations, tol=1e-12):
    return np.any(
        np.all(
            np.add(np.dot(points, equations[:, :-1].T), equations[:, -1]) <= tol,
            axis=1,
        )
    )
    ## update the in_hull function to use the seperated orientation and translation vectors:

def special_in_hull(points, spatial_equation, orientation_equations, tol=1e-12):
    spatial = np.all(np.add(np.dot(points[:, :3], spatial_equation[:, :-1].T), spatial_equation[:, -1]) <= tol, axis=1)
    orientation = np.all(np.add(np.dot(0.1*points[:, 3:], orientation_equations[:, :-1].T), orientation_equations[:, -1]) <= tol, axis=1)
    return np.any(np.all([spatial, orientation], axis=0))




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
    batches = [points[i : i + batch_size] for i in range(0, len(points), batch_size)]

    # Parallel processing of batches
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_batch)(batch) for batch in batches
    )

    return np.any(results)


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
        self.spatial_density = None
        self.orientation_density = None

        self.spatial_set = None
        self.orientation_set = None

    def __call__(self):
        pass

    def gen_set(self):
        """generate the convex hull set - if there are less than 8 points, linearly interpolate between them:"""
        if len(self.points) < 8:
            # randomly sample 8 - len(points) points from the first two points of convex hull (just for utility):
            t = np.linspace(0, 1, 8 - len(self.points))
            qt = self.points[0] + t[:, None] * (self.points[1] - self.points[0])
            # normalise the quaternion portion of the points:
            qt[:, 3:] = qt[:, 3:] / np.linalg.norm(qt[:, 3:], axis=1)[:, None]
            # combine qt with the original points:
            qt = np.concatenate([qt, self.points], axis=0)
            self.set = ConvexHull(qt, qhull_options="QJ")
            self.spatial_set = ConvexHull(
                qt[:, :3], qhull_options="QJ"
            )  # define the spatial set for density computation
            # add the zero quaternion to the orientation set:
            orientation_points = np.concatenate([qt[:, 3:], np.array([[0, 0, 0, 0]])], axis=0)
            self.orientation_set = ConvexHull(
                orientation_points, qhull_options="QJ"
            )  # define the orientation set for density computation
        else:
            self.set = ConvexHull(self.points, qhull_options="QJ")
            self.spatial_set = ConvexHull(self.points[:, :3], qhull_options="QJ")
            # add the zero quaternion to the orientation set:
            orientation_points = np.concatenate(
                [self.points[:, 3:], np.array([[0, 0, 0, 0]])], axis=0
            )
            self.orientation_set = ConvexHull(orientation_points, qhull_options="QJ")

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
        return special_in_hull(item, self.spatial_set.equations, self.orientation_set.equations, 1e-12)

    def t_contains(self, item, tol):
        """Faster contains operation via linear programming:
        Args:
            item: np.array of shape (n, d) where n is the number of points and d is the dimension of the points
            tol: float tolerance
        """
        if self.set is None:
            self.gen_set()
        if len(self.points) < 1000:
            return in_hull(self.points, self.set.equations)
        else:
            return in_hull(item, self.set.equations)

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

    def spatial_volume(self):
        if self.spatial_set is None:
            self.gen_set()
        return self.spatial_set.volume

    def orientation_volume(self):
        if self.orientation_set is None:
            self.gen_set()
        return self.orientation_set.volume

    def _density(self):
        # if self.density is not None:
        #     return self.density
        # self.density = len(self.points) / self.volume()
        # return self.density
        return self._spatial_density() * self._orientation_density()

    def _spatial_density(self):
        if self.spatial_density is not None:
            return self.spatial_density
        self.spatial_density = len(self.points) / self.spatial_volume()
        return self.spatial_density

    def _orientation_density(self):
        if self.orientation_density is not None:
            return self.orientation_density
        self.orientation_density = len(self.points) / self.orientation_volume()
        return self.orientation_density

    def contains_singularity(self):
        """check if the node contains a singularity
        Desc:
            - computes the convex hull of only the quaternion portion of the points
            - returns True if the origin is in the convex hull, False otherwise
        """
        quat_points = self.points[:, 3:]
        return False
        return (
            Delaunay(quat_points, qhull_options="QJ").find_simplex(
                np.array([[0, 0, 0, 0]])
            )
            >= 0
        )

    def compress(
        self, max_iterations=1000, vol_threshold=0.9, points_threshold=100, batch_size=5
    ):
        """compress the points in the node by removing the points that don't contribute to the convex hull volume:
        Desc:
            - compute the convex hull of the points
            - compute the initial volume of the convex hull
            - sample a batch of random points from the convex hull
            - compute the volume of the hull that would result in the removal of the point
            - remove the point that results in the smallest volume if below a specific threshold
        """

        if len(self.points) < points_threshold:
            return
        else:
            init_vol = self.volume()
            print(f"Running set compression: {len(self.points)} points")
            for step in range(max_iterations):
                # shuffle the points
                np.random.shuffle(self.points)
                # track the point that results in the smallest volume
                max_vol = 0
                max_point = None
                # compute the volume of the hull without the point, restrict to batch size
                for i in range(batch_size):
                    test_points = np.delete(self.points, i, axis=0)
                    test_hull = ConvexHull(test_points, qhull_options="QJ")
                    test_vol = test_hull.volume
                    if test_vol > max_vol:
                        max_vol = test_vol
                        max_point = i

                if max_vol / init_vol > vol_threshold:
                    # remove the point
                    print(f"Removing point: {self.points[max_point]}")
                    self.points = np.delete(self.points, max_point, axis=0)
                    # recompute the hull
                    self.set = ConvexHull(self.points, qhull_options="QJ")
                    # recomputes the object:
                    self.center = np.mean(self.points, axis=0)
                    self.density = None

                # check if the volume is above the threshold
                if max_vol / init_vol < vol_threshold:
                    break
                # check if the nuber of points is below the threshold
                if len(self.points) < points_threshold:
                    break

            print(
                f"Final number of points: {len(self.points)}",
                f"Final volume ratio: {max_vol/ init_vol}",
            )

    def ransac_compress(
        self, max_iterations=1000, vol_threshold=0.9, points_threshold=300
    ):
        vertices = self.set.points[self.set.vertices]
        if len(vertices) <= points_threshold:
            return
        print(f"Running set compression: {len(self.points)} points")
        max_vol = 0
        max_points = None
        max_set = None
        init_vol = self.volume()
        solved = False
        for s in range(max_iterations):
            # sample a random set of points
            sample_points = vertices[
                np.random.choice(len(vertices), points_threshold, replace=False)
            ]
            # compute the convex hull of the sample points
            sample_hull = ConvexHull(sample_points, qhull_options="QJ")
            # compute the sample hull volume:
            volume = sample_hull.volume
            if volume > max_vol:
                max_vol = volume
                max_points = sample_points
                max_set = sample_hull
            if volume / init_vol > vol_threshold:
                break
        if max_vol / init_vol < vol_threshold:
            print("Failure to compress, volume ratio: ", max_vol / init_vol)
            return
        else:
            # compute members of the compressed hull:
            in_compressed_hull = np.all(
                np.add(
                    np.dot(self.points, max_set.equations[:, :-1].T),
                    max_set.equations[:, -1],
                )
                <= 1e-12,
                axis=1,
            )
            self.points = self.points[in_compressed_hull]

            self.points = max_points
            self.set = max_set
            self.center = np.mean(self.points, axis=0)
            self.density = None
            print(
                f"Final number of points: {len(self.points)}",
                f"Final volume ratio: {max_vol/ init_vol}",
            )

    def parallel_ransac_compress(
        self, max_iterations=1000, vol_threshold=0.9, points_threshold=300
    ):
        # generate the set if there is none
        # remove duplicate points:
        self.points = np.unique(
            np.round(self.points, decimals=10), axis=0
        )  # roundin avoids precision issues
        if self.set is None:
            self.gen_set()
        vertices = self.set.points[self.set.vertices]
        if len(vertices) <= points_threshold:
            return
        return
        print(f"Running set compression: {len(self.points)} points")
        init_vol = self.volume()

        def evaluate_sample():
            # Sample a random set of points
            sample_points = vertices[
                np.random.choice(len(vertices), points_threshold, replace=False)
            ]
            # Compute the convex hull of the sample points
            sample_hull = ConvexHull(sample_points, qhull_options="QJ")
            # Return the sample hull and its volume
            return sample_hull, sample_hull.volume, sample_points

        # Run RANSAC iterations in parallel
        results = Parallel(n_jobs=-1)(
            delayed(evaluate_sample)() for _ in range(max_iterations)
        )

        # Find the best result
        max_vol, max_set, max_points = 0, None, None
        for sample_hull, volume, sample_points in results:
            if volume > max_vol:
                max_vol = volume
                max_set = sample_hull
                max_points = sample_points
            if max_vol / init_vol > vol_threshold:
                break

        if max_vol / init_vol < vol_threshold:
            print("Failure to compress, volume ratio: ", max_vol / init_vol)
            return
        else:
            # Compute members of the compressed hull
            in_compressed_hull = np.all(
                np.add(
                    np.dot(self.points, max_set.equations[:, :-1].T),
                    max_set.equations[:, -1],
                )
                <= 1e-12,
                axis=1,
            )
            self.points = self.points[in_compressed_hull]

            self.points = max_points
            self.set = max_set
            self.center = np.mean(self.points, axis=0)
            self.density = None
            print(
                f"Final number of points: {len(self.points)}",
                f"Final volume ratio: {max_vol / init_vol}",
            )

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
        for i, (primitive_name, (traj, ik_sols)) in enumerate(paths.items()):
            assert len(traj) == len(ik_sols)
            for i in range(len(traj) - 1):
                X1 = self.X_to_7D(traj[i])  # need to convert to 7D
                X2 = self.X_to_7D(traj[i + 1])  # need to convert to 7D
                if ik_sols[i] is None:
                    self.negative_points.append(X1)
                    continue
                if ik_sols[i + 1] is None:
                    self.negative_points.append(X2)
                    continue
                segment_points = np.array([X1, X2])
                # set_points = self.linear_interpolation(segment_points)
                # E = ConvexHull(set_points, qhull_options='QJ')
                node = Node(None, segment_points, self.node_indx)
                self.nodes.append(node)
                self.node_indx += 1
        print("Initial number of nodes: ", len(self.nodes))
        print("Number of negative points: ", len(self.negative_points))

    def X_to_7D(self, X):
        """convert the SE(3) RigidBody pose to 7D position (x, y, z, q1, q2, q3, q4)"""
        quat = X.rotation().ToQuaternion()
        return np.concatenate(
            [X.translation(), np.array([quat.w(), quat.x(), quat.y(), quat.z()])]
        )

    def X_to_4D(self, X):
        """convert the SE(2) RigidBody pose to 4D quaternion (q1, q2, q3, q4)"""
        quat = X.rotation().ToQuaternion()
        return np.array([quat.w(), quat.x(), quat.y(), quat.z()])

    def linear_interpolation(self, segment, points=8):
        """generate the initial sets from the nodes
        Args:
            segments: numpy array of the two points which define the convex hull
            points: interpolation numer of points
        Desc:
            QuickHull algorithms requires additional points to construct an initial simplex
            Performs simple linear interpolation to compute the additional points along the trajectory path:
        """
        t = np.linspace(0, 1, points)
        qt = segment[0] + t[:, None] * (segment[1] - segment[0])
        # normalise the quaternion portion of the points:
        qt[:, 3:] = qt[:, 3:] / np.linalg.norm(qt[:, 3:], axis=1)[:, None]
        return qt

    def deshatter(
        self,
        max_iterations=5000,
        spatial_density_threshold=0,
        orientation_density_threshold=0,
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
                    merged.parallel_ransac_compress()

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
                        merged._spatial_density() > spatial_density_threshold
                        and merged._orientation_density()
                        > orientation_density_threshold
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
        spatial_density_threshold=0,
        orientation_density_threshold=0,
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

                    merged = node.lossy_compression(test_node, np.array(self.negative_points), points_ratio=0.8, max_iterations=20)
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







        



def main():
    # load in the data:
    scenario = "primitives_high_coverage"  # "tabletop" or "full" or "primitives" or "primitives_high_coverage"

    with open(ROOT_DIR / f"output/trajectories_{scenario}.pkl", "rb") as f:
        out_structure = pickle.load(f)

    sets: dict[SetGen] = dict()

    print(len(out_structure))
    print(out_structure.keys())

    for contact_mode in out_structure.keys():
        
        set_gen = SetGen()
        set_gen.construct_initial_sets(out_structure[contact_mode])

        set_gen.deshatter()
        

        print("-" * 50)
        # compute some stats about the nodes:
        for node in set_gen.nodes:
            print(node)
            print("Volume: ", node.volume())
            print("Density: ", node._density())
            print("Spatial Density: ", node._spatial_density())
            print("Orientation Density: ", node._orientation_density())

        set_gen.lossy_deshatter()

        print("-" * 50)
        # compute some stats about the nodes:
        for node in set_gen.nodes:
            print(node)
            print("Volume: ", node.volume())
            print("Density: ", node._density())
            print("Spatial Density: ", node._spatial_density())
            print("Orientation Density: ", node._orientation_density())

        sets[contact_mode] = set_gen

        break


    with open(ROOT_DIR / f"output/set_gen_{scenario}.pkl", "wb") as f:
            pickle.dump(sets, f)

if __name__ == "__main__":
    main()
