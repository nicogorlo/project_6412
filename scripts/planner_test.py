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

        # connect the dynamic and static subgraphs within each mode
        for mode_name in self.modes:
            dynamic_region = self.subgraphs[mode_name][0]
            static_region = self.subgraphs[mode_name][1]
            self.gcs_tropt.AddEdges(dynamic_region, static_region)
            self.gcs_tropt.AddEdges(static_region, dynamic_region)

        # connect all the static subgraphs:
        for mode1, mode2 in combinations(self.modes, 2):
            static_region1 = self.subgraphs[mode1][1]
            static_region2 = self.subgraphs[mode2][1]
            self.gcs_tropt.AddEdges(static_region1, static_region2)
            self.gcs_tropt.AddEdges(static_region2, static_region1)

        
        # # Make vertices for each contact mode (static and dynamic)
        # self.mode_to_dynamic_vertices = {}
        # self.mode_to_static_vertices = {}
        # for mode_name in self.modes:
        #     convex_sets = self.mode_to_sets[mode_name]
        #     self.mode_to_dynamic_vertices[mode_name] = []
        #     for i, convex_set in enumerate(convex_sets):
        #         v = self.gcs.AddVertex(convex_set, name=f"{mode_name}_dynamic_{i+1}")
        #         self.mode_to_dynamic_vertices[mode_name].append(v)
        #     static_sets = self.mode_to_static_sets[mode_name]
        #     self.mode_to_static_vertices[mode_name] = []
        #     for i, static_set in enumerate(static_sets):
        #         v = self.gcs.AddVertex(static_set, name=f"{mode_name}_static_{i+1}")
        #         self.mode_to_static_vertices[mode_name].append(v)
        # print("Vertices Added!")
        
        # # Densely connect vertices within the same contact mode (intra)
        # for mode_name in self.modes:
        #     dynamic_vertices = self.mode_to_dynamic_vertices[mode_name]
        #     static_vertices = self.mode_to_static_vertices[mode_name]
        #     all_vertices = dynamic_vertices + static_vertices
        #     for u, v in product(all_vertices, repeat=2):
        #         if u.id() == v.id():
        #             continue
        #         if u.set().IntersectsWith(v.set()):
        #             self.gcs.AddEdge(
        #                 u, v, name=f"intramode_{u.name()}_{v.name()}"
        #             )
        # print("Intra-Mode Vertices Connected!")
        
        # # Connect static sets between contact modes (inter)
        # for mode1, mode2 in product(self.modes, repeat=2):
        #     if mode1 == mode2:
        #         continue
        #     static_vertices1 = self.mode_to_static_vertices[mode1]
        #     static_vertices2 = self.mode_to_static_vertices[mode2]
        #     for u, v in product(static_vertices1, static_vertices2):
        #         if u.set().IntersectsWith(v.set()):
        #             self.gcs.AddEdge(
        #                 u, v, name=f"intermode_{u.name()}_{v.name()}"
        #             )
        # print("Inter-Mode Static Vertices Connected!")

    def solve_plan(self, start_pt, end_pt):
        # add start and end regions:
        source = self.gcs_tropt.AddRegions([Point(start_pt)], order=0)
        target = self.gcs_tropt.AddRegions([Point(end_pt)], order=0)   

        # add an edge from the source with every static subgraph:
        for mode_name in self.modes:
            static_region = self.subgraphs[mode_name][1]
            self.gcs_tropt.AddEdges(source, static_region)
            self.gcs_tropt.AddEdges(static_region, source)

        # add an edge from the target with every static subgraph:
        for mode_name in self.modes:
            dynamic_region = self.subgraphs[mode_name][0]
            static_region = self.subgraphs[mode_name][1]
            self.gcs_tropt.AddEdges(static_region, target)
            self.gcs_tropt.AddEdges(target, static_region)
            self.gcs_tropt.AddEdges(dynamic_region, target)
            self.gcs_tropt.AddEdges(target, dynamic_region)
            


        # if self.start_vertex is not None:
        #     self.gcs.RemoveVertex(self.start_vertex)
        # if self.end_vertex is not None:
        #     self.gcs.RemoveVertex(self.end_vertex)
        # self.start_vertex = self.gcs.AddVertex(start_set, name="start")
        # self.end_vertex = self.gcs.AddVertex(end_set, name="end")
        
        # # Connect the start to all static equilibrium vertices
        # for mode_name in self.modes:
        #     for static_vertex in self.mode_to_static_vertices[mode_name]:
        #         if self.start_vertex.set().IntersectsWith(static_vertex.set()):
        #             self.gcs.AddEdge(self.start_vertex, static_vertex, name=f"start_{static_vertex.name()}")
                    
        # # Connect the end to all vertices
        # for v in self.gcs.Vertices():
        #     if self.end_vertex.set().IntersectsWith(v.set()):
        #         if v.id() == self.end_vertex.id():
        #             continue
        #         self.gcs.AddEdge(v, self.end_vertex, name=f"{v.name()}_end")

        print(self.gcs_tropt.GetGraphvizString())
        '''
        This is where we do all the configuration for constraints:
        '''

        # self.gcs_tropt.AddTimeCost()
        self.gcs_tropt.AddPathLengthCost()
        for o in range(1, self.continuity_order + 1):
            print(f"adding C{o} constraints")
            self.gcs_tropt.AddContinuityConstraints(o)
        [traj, result] = self.gcs_tropt.SolvePath(source, target, self.solver_options)
        print(f"result.is_success() = {result.is_success()}")
        return traj
    
    # def get_switches(self, wp_path, edge_path, wp_sampling_rate):
    #     edges_accounted_for = set()
    #     switch_wps = []
    #     for idx, wp in enumerate(wp_path):
    #         edge = edge_path[idx // wp_sampling_rate]
    #         uname = edge.u().name()
    #         vname = edge.v().name()
    #         if "static" in uname and "static" in vname and "intermode" in edge.name():
    #             if idx // wp_sampling_rate not in edges_accounted_for:
    #                 if u.set().PointInSet(wp) and v.set().PointInSet(wp):
    #                     edges_accounted_for.add(idx // wp_sampling_rate)
    #                     static_cm_from = "_".join(uname.split('_')[:-2])
    #                     static_cm_to = "_".join(vname.split('_')[:-2])
    #                     t = (idx, static_cm_from, static_cm_to)
    #                     switch_wps.append(t)
    #     return switch_wps
    
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
    test()
    print('Test complete')

