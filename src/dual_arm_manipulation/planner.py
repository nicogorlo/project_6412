from pydrake.geometry.optimization import (
    GraphOfConvexSets,
    GraphOfConvexSetsOptions,
    ConvexSet,
    Point,
    ConvexHull
)
from pydrake.solvers import MosekSolver
from itertools import combinations, product
from typing import Dict, Set
import numpy as np

class GCSPlanner:
    
    def __init__(
        self,
        mode_to_sets: Dict[str, Set[ConvexSet]],
        mode_to_static_sets: Dict[str, Set[ConvexSet]]
    ):
        """
        mode_to_sets: a mapping between contact mode name to a set of convex sets corresponding
        to valid object poses
        teleport: a boolean for whether we assume the robot can teleport between contact modes
        """
        self.mode_to_sets = mode_to_sets
        self.modes = list(mode_to_sets.keys())
        self.mode_to_static_sets = mode_to_static_sets
        assert set(mode_to_sets.keys()) == set(mode_to_static_sets.keys()), "All modes must have a static set"
        self.num_contact_modes = len(mode_to_sets.keys())
        self.gcs = GraphOfConvexSets()
        self.build_graph()
        print("Graph of Convex Sets Built!")
        solver_options = GraphOfConvexSetsOptions()
        solver_options.solver = MosekSolver()
    
    def build_graph(self):
        
        # Make vertices for each contact mode (static and dynamic)
        self.mode_to_dynamic_vertices = {}
        self.mode_to_static_vertices = {}
        for mode_name in self.modes:
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
        for mode_name in self.modes:
            dynamic_vertices = self.mode_to_dynamic_vertices[mode_name]
            static_vertices = self.mode_to_static_vertices[mode_name]
            all_vertices = dynamic_vertices + static_vertices
            for u, v in product(all_vertices, repeat=2):
                if u.id() == v.id():
                    continue
                if u.set().IntersectsWith(v.set()):
                    self.gcs.AddEdge(
                        u, v, name=f"intramode_{u.name()}_{v.name()}"
                    )
        
        # Connect static sets between contact modes (inter)
        for mode1, mode2 in product(self.modes, repeat=2):
            if mode1 == mode2:
                continue
            static_vertices1 = self.mode_to_static_vertices[mode1]
            static_vertices2 = self.mode_to_static_vertices[mode2]
            for u, v in product(static_vertices1, static_vertices2):
                if u.set().IntersectsWith(v.set()):
                    self.gcs.AddEdge(
                        u, v, name=f"intermode_{u.name()}_{v.name()}"
                    )
        
    def solve_plan(self, start_pt: np.ndarray, end_pt: np.ndarray):
        start_set = Point(start_pt)
        end_set = Point(end_pt)
        if self.start_vertex is not None:
            self.gcs.RemoveVertex(self.start_vertex)
        if self.end_vertex is not None:
            self.gcs.RemoveVertex(self.end_vertex)
        self.start_vertex = self.gcs.AddVertex(start_set, name="start")
        self.end_vertex = self.gcs.AddVertex(end_set, name="end")
        # Connect the start and end vertices to all static equilibrium vertices
        for mode_name in self.modes:
            for static_vertex in self.mode_to_static_vertices[mode_name]:
                if self.start_vertex.set().IntersectsWith(static_vertex.set()):
                    self.gcs.AddEdge(self.start_vertex, static_vertex, name=f"start_{static_vertex.name()}")
                if self.end_vertex.set().IntersectsWith(static_vertex.set()):
                    self.gcs.AddEdge(static_vertex, self.end_vertex, name=f"{static_vertex.name()}_end")
        # Solve the plan
        self.prog_result = self.gcs.SolveShortestPath(self.start_vertex, self.end_vertex, options=self.solver_options)
        print("Shortest Path Optimization Solved!")
        edge_path = self.gcs.GetSolutionPath(self.start_vertex, self.end_vertex, self.prog_result)
        print("Greedy DFS Path Found!")
        vertex_path = [e.u() for e in edge_path]
        vertex_path.append(edge_path[-1].v())
        return [v.GetSolution(self.prog_result) for v in vertex_path]
    
    
    