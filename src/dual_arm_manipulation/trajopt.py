from pydrake.geometry.optimization import (
    GraphOfConvexSets,
    GraphOfConvexSetsOptions,
    ConvexSet,
    Point,
    ConvexHull,
    HPolyhedron,
    VPolytope

)
from pydrake.planning import GcsTrajectoryOptimization
from pydrake.solvers import MosekSolver
from itertools import combinations, product
from typing import Dict, Set
import numpy as np

class GCSTrajOptPlanner:
    def __init__(
        self,
        mode_to_sets: Dict[str, Set[ConvexSet]],
        mode_to_static_sets: Dict[str, Set[ConvexSet]],
        D = 2,
        order = 1,
        continuity_order = 0,
    ):  
        self.mode_to_sets = mode_to_sets
        self.start_vertex, self.end_vertex = None, None
        self.modes = list(mode_to_sets.keys())
        self.mode_to_static_sets = mode_to_static_sets
        assert set(mode_to_sets.keys()) == set(mode_to_static_sets.keys()), "All modes must have a static set"
        self.num_contact_modes = len(mode_to_sets.keys())
        self.order = order
        self.continuity_order = continuity_order

        self.gcs_tropt = GcsTrajectoryOptimization(D)
        self.gcs = self.gcs_tropt.graph_of_convex_sets() # grab the underlying graph of convex sets from the trajopt class
        self.build_graph()
        self.solver_options = GraphOfConvexSetsOptions()
        self.solver_options.preprocessing = True
        self.solver_options.convex_relaxation = True

        

    def build_graph(self):
        
        # build subgraphs for each dynamic and static set:
        self.subgraphs = {}
        for mode_name in self.modes:
            dynamic_sets = self.mode_to_sets[mode_name]
            static_sets = self.mode_to_static_sets[mode_name]
            r_dynamic = self.gcs_tropt.AddRegions(list(dynamic_sets), name=f"{mode_name}_dynamic", order=self.order)
            r_static = self.gcs_tropt.AddRegions(list(static_sets), name=f"{mode_name}_static", order=self.order)
            self.subgraphs[mode_name] = (r_dynamic, r_static)
        print("Subgraphs Added!")

        # connect the dynamic and static subgraphs within each mode
        for mode_name in self.modes:
            dynamic_region = self.subgraphs[mode_name][0]
            static_region = self.subgraphs[mode_name][1]
            self.gcs_tropt.AddEdges(dynamic_region, static_region)
            self.gcs_tropt.AddEdges(static_region, dynamic_region)
        print("Intra-Mode Edges Added!")

        # connect all the static subgraphs:
        for mode1, mode2 in combinations(self.modes, 2):
            static_region1 = self.subgraphs[mode1][1]
            static_region2 = self.subgraphs[mode2][1]
            self.gcs_tropt.AddEdges(static_region1, static_region2)
            self.gcs_tropt.AddEdges(static_region2, static_region1)
        print("Inter-Mode Edges Connected!")


    def solve_plan(self, start_pt, end_pt):
        # add start and end regions:
        source = self.gcs_tropt.AddRegions([Point(start_pt)], order=0, name='start')
        target = self.gcs_tropt.AddRegions([Point(end_pt)], order=0, name='end')

        # add an edge from the source with every static subgraph:
        for mode_name in self.modes:
            static_region = self.subgraphs[mode_name][1]
            self.gcs_tropt.AddEdges(source, static_region)
            self.gcs_tropt.AddEdges(static_region, source)

        # add an edge from the target with every subgraph:
        for mode_name in self.modes:
            dynamic_region = self.subgraphs[mode_name][0]
            static_region = self.subgraphs[mode_name][1]
            self.gcs_tropt.AddEdges(static_region, target)
            self.gcs_tropt.AddEdges(target, static_region)
            self.gcs_tropt.AddEdges(dynamic_region, target)
            self.gcs_tropt.AddEdges(target, dynamic_region)
            

        print(self.gcs_tropt.GetGraphvizString())

        # Configure constraints
        # self.gcs_tropt.AddTimeCost()
        self.gcs_tropt.AddPathLengthCost()
        for o in range(1, self.continuity_order + 1):
            print(f"adding C{o} constraints")
            self.gcs_tropt.AddContinuityConstraints(o)
        print("Solving...")
        traj, result = self.gcs_tropt.SolvePath(source, target, self.solver_options)
        print(f"result.is_success() = {result.is_success()}")
        return traj
    
    # TODO: how do we get an edge path?
    def get_waypoints_and_switches(self, composite_traj, edge_path, num_wps=100):
        times = np.linspace(composite_traj.start_time(), composite_traj.end_time(), num_wps).tolist()
        segment_times = [0] + [composite_traj.segment(i).end_time() for i in range(len(edge_path))]
        waypoints = [composite_traj.value(time) for time in times]
        edges_accounted_for = set()
        switch_times = []
        for wp, t in zip(waypoints, times):
            edge_idx = np.digitize(t, segment_times)
            edge = edge_path[edge_idx].name()
            u = edge.u()
            v = edge.v()
            if "static" in u.name() and "static" in v.name() and (not (u.name().split(":")[0] == v.name().split(":")[0])):
                if u.set().PointInSet(wp) and v.set().PointInSet(wp) and edge_idx not in edges_accounted_for:
                    print(f"Switching from {u.name()} to {v.name()} at time {t}")
                    edges_accounted_for.add(edge_idx)
                    switch_times.append(t)

def test():

    # load in the convex sets:
    # Define the convex sets for each contact mode
    import matplotlib.pyplot as plt
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
    gcs_planner = GCSTrajOptPlanner(mode_to_sets, mode_to_static_sets)
    print("GCS Planner created!")

    # Now we can plan a trajectory between the two modes
    start_point = np.array([start_x, start_y])
    end_point = np.array([end_x, end_y])
    trajectory = gcs_planner.solve_plan(start_point, end_point)

    print("Trajectory:", trajectory)

    def plot_trajectory(traj):
        plt.plot(*traj.value(traj.start_time()), "kx")
        plt.plot(*traj.value(traj.end_time()), "kx")
        times = np.linspace(traj.start_time(), traj.end_time(), 1000)
        waypoints = traj.vector_values(times)
        plt.plot(*waypoints, "b", zorder=5)
    
    plot_trajectory(trajectory)

    # square plot:
    plt.axis('equal')

    plt.show()

if __name__ == "__main__":
    print('Running test')
    # test()
    print('Test complete')

