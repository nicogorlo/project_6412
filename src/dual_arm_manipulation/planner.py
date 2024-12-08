from pydrake.geometry.optimization import (
    GraphOfConvexSets,
    GraphOfConvexSetsOptions,
    ConvexSet,
    Point,
    ConvexHull
)

# from pydrake.symbolic import 
from pydrake.math import le
from pydrake.solvers import MosekSolver
from itertools import combinations, product
from typing import Dict, Set
import numpy as np
from tqdm import tqdm

# TODO: add edge constraint for Xu solution to be in the overlapping set.

class GCSPlanner:
    
    def __init__(
        self,
        mode_to_sets: Dict[str, Set[ConvexSet]],
        mode_to_static_sets: Dict[str, Set[ConvexSet]]
    ):
        """
        mode_to_sets: a mapping between contact mode name to a set of convex sets corresponding
        to valid object poses
        mode_to_static_sets: a mapping between contact mode name to a set of convex sets corresponding
        to possible static equilibrium sets for the object 
        teleport: a boolean for whether we assume the robot can teleport between contact modes
        """
        self.mode_to_sets = mode_to_sets
        self.start_vertex, self.end_vertex = None, None
        self.modes = list(mode_to_sets.keys())
        self.mode_to_static_sets = mode_to_static_sets
        assert set(mode_to_sets.keys()) == set(mode_to_static_sets.keys()), "All modes must have a static set"
        self.num_contact_modes = len(mode_to_sets.keys())
        self.gcs = GraphOfConvexSets()
        self.build_graph()
        print("Graph of Convex Sets Built!")
        self.solver_options = GraphOfConvexSetsOptions()
        self.solver_options.solver = MosekSolver()
        self.solver_options.preprocessing = True
        self.solver_options.convex_relaxation = True
        
            
    
    def build_graph(self):
        
        # Make vertices for each contact mode (static and dynamic)
        self.mode_to_dynamic_vertices = {}
        self.mode_to_static_vertices = {}
        print("Adding vertices!")
        for mode_name in tqdm(self.modes):
            convex_sets = self.mode_to_sets[mode_name]
            self.mode_to_dynamic_vertices[mode_name] = []
            for i, convex_set in enumerate(convex_sets):
                v = self.gcs.AddVertex(convex_set, name=f"{mode_name}_dynamic_{i+1}")
                self.mode_to_dynamic_vertices[mode_name].append(v)
            static_sets = self.mode_to_static_sets[mode_name]
            self.mode_to_static_vertices[mode_name] = []
            for i, static_set in enumerate(static_sets):
                v = self.gcs.AddVertex(static_set, name=f"{mode_name}_static_{i+1}")
                self.mode_to_static_vertices[mode_name].append(v)
        
        
        # Densely connect vertices within the same contact mode (intra)
        print("Adding Intra-Mode Vertices!")
        for mode_name in tqdm(self.modes):
            dynamic_vertices = self.mode_to_dynamic_vertices[mode_name]
            static_vertices = self.mode_to_static_vertices[mode_name]
            all_vertices = dynamic_vertices + static_vertices
            for u, v in combinations(all_vertices, 2):
                if u.set().IntersectsWith(v.set()):
                    self.gcs.AddEdge(
                        u, v, name=f"intramode_{u.name()}_{v.name()}"
                    )
                    self.gcs.AddEdge(
                        v, u, name=f"intramode_{v.name()}_{u.name()}"
                    )
        
        
        # Connect static sets between contact modes (inter)
        print("Adding Inter-Mode Vertices!")
        for mode1, mode2 in tqdm(combinations(self.modes, 2)):
            static_vertices1 = self.mode_to_static_vertices[mode1]
            static_vertices2 = self.mode_to_static_vertices[mode2]
            for u, v in product(static_vertices1, static_vertices2):
                if u.set().IntersectsWith(v.set()):
                    self.gcs.AddEdge(
                        u, v, name=f"intermode_{u.name()}_{v.name()}"
                    )
                    self.gcs.AddEdge(
                        v, u, name=f"intermode_{v.name()}_{u.name()}"
                    )
        
    def add_edge_costs(self):
        for edge in self.gcs.Edges():
            xu = edge.xu()
            xv = edge.xv()
            cost = np.linalg.norm(xu - xv)
            edge.AddCost(cost)
    
    def add_edge_constraints(self):
        for edge in self.gcs.Edges():
            xu = edge.xu()
            A = edge.v().set().A()
            b = edge.v().set().b()
            constraint = le(A @ xu - b, 0)
            for c in constraint:
                edge.AddConstraint(c)

        
    def solve_plan(self, start_pt, end_pt):    
        
        self.add_edge_constraints()

        # Set the start and end vertices
        start_set = Point(start_pt)
        end_set = Point(end_pt)
        if self.start_vertex is not None:
            self.gcs.RemoveVertex(self.start_vertex)
        if self.end_vertex is not None:
            self.gcs.RemoveVertex(self.end_vertex)
        self.start_vertex = self.gcs.AddVertex(start_set, name="start")
        self.end_vertex = self.gcs.AddVertex(end_set, name="end")
        
        """# Connect the start to all static equilibrium vertices
        for mode_name in self.modes:
            for static_vertex in self.mode_to_static_vertices[mode_name]:
                if self.start_vertex.set().IntersectsWith(static_vertex.set()):
                    self.gcs.AddEdge(self.start_vertex, static_vertex, name=f"start_{static_vertex.name()}")"""
                    
        # Connect the start and end to all vertices
        for v in self.gcs.Vertices():
            if self.end_vertex.set().IntersectsWith(v.set()):
                if v.id() == self.end_vertex.id():
                    continue
                self.gcs.AddEdge(v, self.end_vertex, name=f"{v.name()}_end")
            if self.start_vertex.set().IntersectsWith(v.set()):
                if v.id() == self.start_vertex.id():
                    continue
                self.gcs.AddEdge(self.start_vertex, v, name=f"start_{v.name()}")
        
        # Add edge costs
        self.add_edge_costs()

        print("Graph:", self.gcs.GetGraphvizString())
        
        # Solve the plan
        self.prog_result = self.gcs.SolveShortestPath(self.start_vertex, self.end_vertex, options=self.solver_options)
        print("Shortest Path Optimization Solved!")
        edge_path = self.gcs.GetSolutionPath(self.start_vertex, self.end_vertex, self.prog_result, tolerance=0.75)
        print("Greedy DFS Path Found!")
        vertex_path = [e.u() for e in edge_path]
        vertex_path.append(edge_path[-1].v())
        print("Edge Path", [e.name() for e in edge_path])
        print("Vertex Path", [v.name() for v in vertex_path])
        return [v.GetSolution(self.prog_result) for v in vertex_path], edge_path
    
    def get_switches(self, wp_path, edge_path, wp_sampling_rate):
        edges_accounted_for = set()
        switch_wps = []
        for idx, wp in enumerate(wp_path):
            edge = edge_path[idx // wp_sampling_rate]
            uname = edge.u().name()
            vname = edge.v().name()
            if "static" in uname and "static" in vname and "intermode" in edge.name():
                if idx // wp_sampling_rate not in edges_accounted_for:
                    if edge.u().set().PointInSet(wp) and edge.v().set().PointInSet(wp):
                        edges_accounted_for.add(idx // wp_sampling_rate)
                        static_cm_from = "_".join(uname.split('_')[:-2])
                        static_cm_to = "_".join(vname.split('_')[:-2])
                        t = (idx, static_cm_from, static_cm_to)
                        switch_wps.append(t)
        return switch_wps
