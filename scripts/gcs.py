from pydrake.geometry.optimization import GraphOfConvexSets, GraphOfConvexSetsOptions, ConvexSet, Point, ConvexHull
from pydrake.solvers import MosekSolver
from typing import Dict, List, Set
import numpy as np
import matplotlib.pyplot as plt


# TODO: come up with a cost function for distance between 6Dof poses, and add to edges.
# 0 cost for edges between same mode, and some cost for edges between different modes.
# TODO: find out how to translate the edge lists into a feasible trajectory.

# Set solving options
solver_options = GraphOfConvexSetsOptions()
solver_options.solver = MosekSolver()


def construct_convex_hulls(
    name_to_trajectory_set: Dict[str, Set[List[np.ndarray]]]
) -> Dict[str, List[ConvexSet]]:
    """
    name_to_trajectory_set: A dictionary mapping contact mode name to a set of trajectories where
    a trajectory is a list of (x y z r p y) vectors
    returns: A dictionary mapping contact mode name to a convex hull which covers the trajectories.
    """
    name_to_hull = {}
    for name, trajectories in name_to_trajectory_set.items():
        point_set = []
        for trajectory in trajectories:
            for task_pose in trajectory:
                point_set.append(Point(task_pose))
        convex_set = ConvexHull(point_set)
        name_to_hull[name] = [convex_set]
    return name_to_hull


def construct_graph_of_convex_sets(
    name_to_convex_sets: Dict[str, List[ConvexSet]]
) -> GraphOfConvexSets:
    """
    name_to_convex_sets: A dictionary mapping contact mode name to a list of convex sets
    returns: A GraphOfConvexSets object with vertices and edges connecting the vertices
    """
    gcs = GraphOfConvexSets()
    # for each contact mode, create nodes
    for name, convex_sets in name_to_convex_sets.items():
        contact_mode_vertices = []
        for i, convex_set in enumerate(convex_sets):
            added_v = gcs.AddVertex(convex_set, name=f"{name}_{i+1}")
            contact_mode_vertices.append(added_v)
        # Densely connect all vertices created for this mode
        for i in range(len(contact_mode_vertices)):
            for j in range(len(contact_mode_vertices)):
                if i != j:
                    v_i, v_j = contact_mode_vertices[i], contact_mode_vertices[j]
                    gcs.AddEdge(v_i, v_j, name=f"{v_i.name()}_{v_j.name()}")
    # Loop through all vertices in different modes and check for ConvexSet intersections
    for i, vertex_i in enumerate(gcs.Vertices()):
        for j, vertex_j in enumerate(gcs.Vertices()):
            if i != j and vertex_i.name().split("_")[:-1] != vertex_j.name().split("_")[:-1]:
                if vertex_i.set().IntersectsWith(vertex_j.set()):
                    gcs.AddEdge(vertex_i, vertex_j, name=f"swtich_{vertex_i.name()}_{vertex_j.name()}")
    return gcs



def getStartGoalEdges(gcs, start, goal):
    start_vs = []
    goal_vs = []
    for v in gcs.Vertices():
        if v.set().PointInSet(start):
            start_vs.append(v)
        if v.set().PointInSet(goal):
            goal_vs.append(v)
    return start_vs, goal_vs

def generate_path(
    gcs: GraphOfConvexSets,
    start: np.ndarray,
    end: np.ndarray
):
    """
    gcs: A GraphOfConvexSets object
    start_vertex: A start point
    end_vertex: A goal point
    returns: A list of edges connecting the start and end vertices
    """
    #source = gcs.AddVertex(Point(start), "source")
    #target = gcs.AddVertex(Point(end), "target")
    
    start_nodes, goal_nodes = getStartGoalEdges(gcs, start, end)
    assert len(start_nodes) > 0, "Start isn't connected to graph"
    assert len(goal_nodes) > 0, "Goal isn't connected to graph"
    source = gcs.AddVertex(Point(start), "source")
    target = gcs.AddVertex(Point(end), "target")
    for s in start_nodes:
        print("Adding edge from start to", s.name())
        gcs.AddEdge(source, s, name=f"source_{s.name()}")
    for g in goal_nodes:
        print("Adding edge from", g.name(), "to goal")
        gcs.AddEdge(g, target, name=f"target_{g.name()}")
    
    print("Graph Constructed!")
    
    
    prog_result = gcs.SolveShortestPath(source, target, options=solver_options)
    # prog_result = gcs.SolveShortestPath(start_nodes[0], goal_nodes[0], options=solver_options)
    print("Shortest Path Found!")
    
    
    
    # edge_list = gcs.GetSolutionPath(source, target, prog_result)
    edge_list = gcs.GetSolutionPath(start_nodes[0], goal_nodes[0], prog_result)
    print("Path Found!")
    waypoints = [prog_result.GetSolution(e.xv()) for e in edge_list]
    
    gcs.RemoveVertex(source)
    gcs.RemoveVertex(target)
    
    return edge_list, waypoints