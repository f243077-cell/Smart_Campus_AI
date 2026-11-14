"""
search_navigation.py
-----------------------------------------------------------------------
Module 3.6 - Search & Navigation
Computes campus routes on the predefined campus graph.

Project policy:
  * BFS  -> unweighted route problems (fewest stops)
  * A*   -> weighted route problems when a heuristic is available
  * UCS  -> weighted route problems without a suitable heuristic (fallback)
-----------------------------------------------------------------------
"""

import heapq
import math
from collections import deque

import numpy as np

import config


class NavigationError(Exception):
    """Raised when a route cannot be computed (unknown node, no path...)."""


# Building coordinates taken from the "Campus weighted Graph" diagram.
CAMPUS_COORDINATES = {
    "Main_Gate": (0, 4),
    "Parking": (2, 4),
    "Admin_Block": (3, 5),
    "Student_Services": (6, 5),
    "Exam_Hall": (8, 5),
    "Seminar_Room": (10, 4),
    "Library": (6, 2),
    "AI_Lab": (9, 2),
    "Bus_Stop": (0, 1),
    "Medical_Center": (1, 1),
    "Cafeteria": (4, 1),
    "Science_Block": (7, 1),
    "Hostel": (2, 0),
}

# Undirected edges with walking cost, taken from the same diagram.
CAMPUS_EDGES = [
    ("Admin_Block", "Student_Services", 1),
    ("Main_Gate", "Admin_Block", 4),
    ("Admin_Block", "Exam_Hall", 2),
    ("Main_Gate", "Parking", 2),
    ("Exam_Hall", "Seminar_Room", 1),
    ("Student_Services", "Library", 2),
    ("Main_Gate", "Bus_Stop", 1),
    ("Parking", "Bus_Stop", 2),
    ("Main_Gate", "Hostel", 5),
    ("Parking", "Science_Block", 3),
    ("Exam_Hall", "Science_Block", 3),
    ("Seminar_Room", "Science_Block", 2),
    ("Library", "AI_Lab", 3),
    ("AI_Lab", "Science_Block", 1),
    ("Bus_Stop", "Medical_Center", 2),
    ("Medical_Center", "Hostel", 3),
    ("Cafeteria", "Library", 2),
    ("Cafeteria", "Science_Block", 3),
    ("Hostel", "Cafeteria", 2),
]


class CampusGraph:
    """
    Class: CampusGraph
    Purpose: Store the campus map as an adjacency list (weighted and
             unweighted views) and provide BFS, UCS and A* route search.
    """

    def __init__(self, edges=None, coordinates=None):
        """
        Function: __init__
        Purpose : Build the adjacency list from an edge list and compute
                  the admissible heuristic scaling factor for A*.
        Inputs  : edges       - list of (node_a, node_b, cost) tuples
                  coordinates - dict node -> (x, y), may be None
        """
        self.edges = list(edges if edges is not None else CAMPUS_EDGES)
        self.coordinates = dict(coordinates if coordinates is not None else CAMPUS_COORDINATES)
        self.adjacency = {}
        for node_a, node_b, cost in self.edges:
            if cost <= 0:
                raise NavigationError(f"Edge {node_a}-{node_b} must have a positive cost.")
            self.adjacency.setdefault(node_a, {})[node_b] = cost
            self.adjacency.setdefault(node_b, {})[node_a] = cost
        self.heuristic_scale = self._compute_heuristic_scale()

    # ---------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------
    @property
    def nodes(self):
        """
        Function: nodes
        Purpose : Return a sorted list of all building names in the graph.
        """
        return sorted(self.adjacency.keys())

    def has_node(self, node_name):
        """
        Function: has_node
        Purpose : Tell whether a building name exists in the campus graph.
        """
        return node_name in self.adjacency

    def heuristic_available(self):
        """
        Function: heuristic_available
        Purpose : A heuristic is available only if every node has
                  coordinates (needed for the straight-line distance).
        """
        return all(node in self.coordinates for node in self.adjacency)

    def _euclidean_distance(self, node_a, node_b):
        """
        Function: _euclidean_distance
        Purpose : Straight-line distance between two buildings using numpy.
        """
        point_a = np.array(self.coordinates[node_a], dtype=float)
        point_b = np.array(self.coordinates[node_b], dtype=float)
        return float(np.linalg.norm(point_a - point_b))

    def _compute_heuristic_scale(self):
        """
        Function: _compute_heuristic_scale
        Purpose : Plain Euclidean distance OVERESTIMATES some edges on this
                  map (Admin_Block-Student_Services costs 1 but is 3 units
                  apart), which would make A* non-admissible. We scale the
                  Euclidean distance by k = min(edge_cost / edge_length)
                  over all edges. Then h(n) <= true cost for every path,
                  so the heuristic is admissible and consistent.
        Returns : float scaling factor (0 if no heuristic is available)
        """
        if not self.heuristic_available():
            return 0.0
        ratios = []
        for node_a, node_b, cost in self.edges:
            length = self._euclidean_distance(node_a, node_b)
            if length > 0:
                ratios.append(cost / length)
        return min(ratios) if ratios else 0.0

    def heuristic(self, node_name, goal_name):
        """
        Function: heuristic
        Purpose : Admissible A* heuristic = scaled straight-line distance.
        """
        return self.heuristic_scale * self._euclidean_distance(node_name, goal_name)

    def path_cost(self, path):
        """
        Function: path_cost
        Purpose : Sum the weighted cost of consecutive edges along a path.
        """
        return sum(self.adjacency[path[i]][path[i + 1]] for i in range(len(path) - 1))

    def _validate_endpoints(self, start_node, goal_node):
        """
        Function: _validate_endpoints
        Purpose : Raise NavigationError if start or goal is not on the map.
        """
        for node_name in (start_node, goal_node):
            if not self.has_node(node_name):
                raise NavigationError(f"Unknown campus location '{node_name}'. "
                                      f"Valid locations: {', '.join(self.nodes)}")

    @staticmethod
    def _rebuild_path(parent_map, goal_node):
        """
        Function: _rebuild_path
        Purpose : Walk back through parent pointers to rebuild the path.
        """
        path = [goal_node]
        while parent_map[path[-1]] is not None:
            path.append(parent_map[path[-1]])
        return list(reversed(path))

    # ---------------------------------------------------------------
    # Search algorithms
    # ---------------------------------------------------------------
    def breadth_first_search(self, start_node, goal_node):
        """
        Function: breadth_first_search
        Purpose : BFS on the unweighted view of the graph. Finds the route
                  with the fewest stops (edges). Goal test on generation.
        Returns : dict with path, hops, weighted cost, expanded nodes
        """
        self._validate_endpoints(start_node, goal_node)
        parent_map = {start_node: None}
        frontier = deque([start_node])
        expanded_order = []
        if start_node == goal_node:
            return self._result("BFS", [start_node], expanded_order)
        while frontier:
            current_node = frontier.popleft()
            expanded_order.append(current_node)
            for neighbour in sorted(self.adjacency[current_node]):
                if neighbour not in parent_map:
                    parent_map[neighbour] = current_node
                    if neighbour == goal_node:
                        return self._result("BFS", self._rebuild_path(parent_map, goal_node),
                                            expanded_order)
                    frontier.append(neighbour)
        raise NavigationError(f"No route exists from {start_node} to {goal_node}.")

    def uniform_cost_search(self, start_node, goal_node):
        """
        Function: uniform_cost_search
        Purpose : UCS on the weighted graph. Always expands the frontier
                  node with the lowest path cost g(n). Optimal for positive
                  edge costs; used when no heuristic is available.
        """
        return self._best_first_search(start_node, goal_node, use_heuristic=False)

    def a_star_search(self, start_node, goal_node):
        """
        Function: a_star_search
        Purpose : A* on the weighted graph with f(n) = g(n) + h(n) using the
                  admissible scaled-Euclidean heuristic. Falls back to UCS
                  automatically if the heuristic is not available.
        """
        if not self.heuristic_available():
            return self.uniform_cost_search(start_node, goal_node)
        return self._best_first_search(start_node, goal_node, use_heuristic=True)

    def _best_first_search(self, start_node, goal_node, use_heuristic):
        """
        Function: _best_first_search
        Purpose : Shared priority-queue search used by UCS (h = 0) and A*.
                  Goal test happens when a node is POPPED, which keeps the
                  result optimal.
        """
        self._validate_endpoints(start_node, goal_node)
        algorithm_name = "A*" if use_heuristic else "UCS"
        start_h = self.heuristic(start_node, goal_node) if use_heuristic else 0.0
        frontier = [(start_h, 0.0, start_node)]
        best_cost = {start_node: 0.0}
        parent_map = {start_node: None}
        expanded_order = []
        closed_set = set()
        while frontier:
            _, cost_so_far, current_node = heapq.heappop(frontier)
            if current_node in closed_set:
                continue
            closed_set.add(current_node)
            expanded_order.append(current_node)
            if current_node == goal_node:
                return self._result(algorithm_name, self._rebuild_path(parent_map, goal_node),
                                    expanded_order)
            for neighbour, edge_cost in sorted(self.adjacency[current_node].items()):
                new_cost = cost_so_far + edge_cost
                if new_cost < best_cost.get(neighbour, math.inf):
                    best_cost[neighbour] = new_cost
                    parent_map[neighbour] = current_node
                    h_value = self.heuristic(neighbour, goal_node) if use_heuristic else 0.0
                    heapq.heappush(frontier, (new_cost + h_value, new_cost, neighbour))
        raise NavigationError(f"No route exists from {start_node} to {goal_node}.")

    def _result(self, algorithm_name, path, expanded_order):
        """
        Function: _result
        Purpose : Package a search result into a uniform dictionary.
        """
        return {
            "algorithm": algorithm_name,
            "path": path,
            "hops": len(path) - 1,
            "cost": self.path_cost(path),
            "nodes_expanded": len(expanded_order),
            "expanded_order": expanded_order,
        }


# -------------------------------------------------------------------
# Module-level helpers used by the router
# -------------------------------------------------------------------
_DEFAULT_GRAPH = None


def get_campus_graph():
    """
    Function: get_campus_graph
    Purpose : Return one shared CampusGraph instance (built lazily).
    """
    global _DEFAULT_GRAPH
    if _DEFAULT_GRAPH is None:
        _DEFAULT_GRAPH = CampusGraph()
    return _DEFAULT_GRAPH


def find_route(source, destination, route_preference=config.DEFAULT_ROUTE_PREFERENCE,
               campus_graph=None):
    """
    Function: find_route
    Purpose : Apply the project search policy and return the route.
                'stops'                 -> unweighted problem -> BFS
                'distance'              -> weighted + heuristic -> A*
                'distance_no_heuristic' -> weighted, no heuristic -> UCS
    Returns : result dict from the selected algorithm plus 'policy' text
    """
    campus_graph = campus_graph or get_campus_graph()
    if route_preference == "stops":
        result = campus_graph.breadth_first_search(source, destination)
        result["policy"] = "Unweighted problem (fewest stops) -> BFS"
    elif route_preference == "distance_no_heuristic" or not campus_graph.heuristic_available():
        result = campus_graph.uniform_cost_search(source, destination)
        result["policy"] = "Weighted problem without heuristic -> UCS (fallback)"
    elif route_preference == "distance":
        result = campus_graph.a_star_search(source, destination)
        result["policy"] = "Weighted problem with admissible heuristic -> A*"
    else:
        raise NavigationError(f"Unknown route preference '{route_preference}'. "
                              f"Use one of {config.ROUTE_PREFERENCES}.")
    return result


def compare_algorithms(source, destination, campus_graph=None):
    """
    Function: compare_algorithms
    Purpose : Run BFS, UCS and A* on the same query so they can be compared
              side by side (path, hops, cost, nodes expanded).
    Returns : list of result dicts
    """
    campus_graph = campus_graph or get_campus_graph()
    return [campus_graph.breadth_first_search(source, destination),
            campus_graph.uniform_cost_search(source, destination),
            campus_graph.a_star_search(source, destination)]
