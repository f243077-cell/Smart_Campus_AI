"""
search_navigation.py
-----------------------------------------------------------------------
Module 3.6 - Search & Navigation
Computes campus routes on the predefined campus graph.

Two working modes (handout, Search module):
  * Operational mode - one algorithm, one route, chosen by the policy:
        unweighted graph               -> BFS
        weighted graph + heuristic     -> A*
        weighted graph, no heuristic   -> UCS (fallback)
  * Comparison / demo mode - every algorithm runs on the same query:
        unweighted: BFS, DFS, DLS, IDS, Bidirectional BFS
        weighted  : UCS, Greedy Best-First, A*, RBFS
Every result follows the standard search output
{algorithm_used, path, cost, steps} plus optional academic fields
(nodes_expanded, expanded_order, execution_time_ms).
-----------------------------------------------------------------------
"""

import heapq
import math
import time
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
             unweighted views) and provide every search algorithm of the
             lab sequence.
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
    def breadth_first_search(self, start_node, goal_node, trace=None):
        """
        Function: breadth_first_search
        Purpose : BFS on the unweighted view of the graph. Finds the route
                  with the fewest stops (edges). Goal test on generation.
                  If a list is passed as trace, one entry per decision is
                  appended to it (used by the UI's step-by-step view); the
                  search itself is unchanged.
        Returns : dict with path, hops, weighted cost, expanded nodes
        """
        self._validate_endpoints(start_node, goal_node)
        parent_map = {start_node: None}
        frontier = deque([start_node])
        expanded_order = []
        if start_node == goal_node:
            return self._result("BFS", [start_node], expanded_order, "unweighted")
        while frontier:
            current_node = frontier.popleft()
            expanded_order.append(current_node)
            neighbour_decisions = []
            for neighbour in sorted(self.adjacency[current_node]):
                if neighbour not in parent_map:
                    parent_map[neighbour] = current_node
                    if neighbour == goal_node:
                        neighbour_decisions.append(self._decision(neighbour, current_node, None, "goal",
                                                                  "goal found when generated"))
                        self._record_step(trace, "BFS", current_node, None, neighbour_decisions,
                                          list(frontier), parent_map, expanded_order, goal_reached=True)
                        return self._result("BFS", self._rebuild_path(parent_map, goal_node),
                                            expanded_order, "unweighted")
                    frontier.append(neighbour)
                    neighbour_decisions.append(self._decision(neighbour, current_node, None, "added",
                                                              "new node, added to the back of the queue"))
                else:
                    neighbour_decisions.append(self._decision(neighbour, current_node, None, "rejected",
                                                              "already seen, ignored"))
            self._record_step(trace, "BFS", current_node, None, neighbour_decisions,
                              list(frontier), parent_map, expanded_order)
        raise NavigationError(f"No route exists from {start_node} to {goal_node}.")

    def uniform_cost_search(self, start_node, goal_node, trace=None):
        """
        Function: uniform_cost_search
        Purpose : UCS on the weighted graph. Always expands the frontier
                  node with the lowest path cost g(n). Optimal for positive
                  edge costs; used when no heuristic is available.
        """
        return self._best_first_search(start_node, goal_node, use_heuristic=False, trace=trace)

    def a_star_search(self, start_node, goal_node, trace=None):
        """
        Function: a_star_search
        Purpose : A* on the weighted graph with f(n) = g(n) + h(n) using the
                  admissible scaled-Euclidean heuristic. Falls back to UCS
                  automatically if the heuristic is not available.
        """
        if not self.heuristic_available():
            return self.uniform_cost_search(start_node, goal_node, trace=trace)
        return self._best_first_search(start_node, goal_node, use_heuristic=True, trace=trace)

    def _best_first_search(self, start_node, goal_node, use_heuristic, trace=None):
        """
        Function: _best_first_search
        Purpose : Shared priority-queue search used by UCS (h = 0) and A*.
                  Goal test happens when a node is POPPED, which keeps the
                  result optimal. trace works as in breadth_first_search.
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
            priority, cost_so_far, current_node = heapq.heappop(frontier)
            if current_node in closed_set:
                if trace is not None:
                    trace.append({"kind": "stale", "algorithm": algorithm_name, "node": current_node,
                                  "g": cost_so_far, "f": priority})
                continue
            closed_set.add(current_node)
            expanded_order.append(current_node)
            current_h = priority - cost_so_far
            if current_node == goal_node:
                self._record_step(trace, algorithm_name, current_node, (cost_so_far, current_h, priority), [],
                                  sorted(frontier), parent_map, expanded_order, goal_reached=True)
                return self._result(algorithm_name, self._rebuild_path(parent_map, goal_node),
                                    expanded_order)
            neighbour_decisions = []
            for neighbour, edge_cost in sorted(self.adjacency[current_node].items()):
                new_cost = cost_so_far + edge_cost
                h_value = self.heuristic(neighbour, goal_node) if use_heuristic else 0.0
                values = (edge_cost, new_cost, h_value, new_cost + h_value)
                if neighbour in closed_set:
                    neighbour_decisions.append(self._decision(neighbour, current_node, values, "rejected",
                                                              "already expanded (closed)"))
                elif new_cost < best_cost.get(neighbour, math.inf):
                    improved = neighbour in best_cost
                    old_cost = best_cost.get(neighbour)
                    best_cost[neighbour] = new_cost
                    parent_map[neighbour] = current_node
                    heapq.heappush(frontier, (new_cost + h_value, new_cost, neighbour))
                    reason = (f"cheaper path found (g {old_cost:g} -> {new_cost:g}), queue updated" if improved
                              else "new node, pushed onto the priority queue")
                    neighbour_decisions.append(self._decision(neighbour, current_node, values,
                                                              "improved" if improved else "added", reason))
                else:
                    neighbour_decisions.append(self._decision(
                        neighbour, current_node, values, "rejected",
                        f"not cheaper than known g {best_cost[neighbour]:g}, ignored"))
            self._record_step(trace, algorithm_name, current_node, (cost_so_far, current_h, priority),
                              neighbour_decisions, sorted(frontier), parent_map, expanded_order)
        raise NavigationError(f"No route exists from {start_node} to {goal_node}.")

    @staticmethod
    def _decision(neighbour, parent, values, outcome, reason):
        """
        Function: _decision
        Purpose : One neighbour decision for the search trace.
                  values = (edge_cost, g, h, f) for UCS/A*, None for BFS.
        """
        decision = {"node": neighbour, "from": parent, "outcome": outcome, "reason": reason}
        if values is not None:
            decision.update(dict(zip(("edge", "g", "h", "f"), values)))
        return decision

    @staticmethod
    def _record_step(trace, algorithm_name, current_node, current_values, neighbour_decisions,
                     frontier, parent_map, expanded_order, goal_reached=False):
        """
        Function: _record_step
        Purpose : Append one expansion step to the trace (if tracing): the
                  node chosen, why, what happened to each neighbour, and the
                  frontier / parent pointers after the step.
        """
        if trace is None:
            return
        if algorithm_name == "BFS":
            queue = [{"node": node} for node in frontier]
        else:
            queue = [{"node": node, "g": g_value, "h": f_value - g_value, "f": f_value}
                     for f_value, g_value, node in frontier]
        step = {"kind": "expand", "algorithm": algorithm_name, "node": current_node,
                "neighbours": neighbour_decisions, "frontier": queue, "parents": dict(parent_map),
                "expanded": list(expanded_order), "goal_reached": goal_reached}
        if current_values is not None:
            step.update(dict(zip(("g", "h", "f"), current_values)))
        trace.append(step)

    def _result(self, algorithm_name, path, expanded_order, graph_type="weighted"):
        """
        Function: _result
        Purpose : Package a search result into the standard search output
                  (algorithm_used, path, cost, steps) plus academic fields.
                  An empty path means the algorithm found no route (for
                  example Depth-Limited Search hitting its limit).
        """
        found = bool(path)
        return {
            "algorithm_used": algorithm_name,
            "path": path,
            "cost": self.path_cost(path) if found else None,
            "steps": len(path) - 1 if found else None,
            "nodes_expanded": len(expanded_order),
            "expanded_order": expanded_order,
            "graph_type": graph_type,
            "found": found,
        }

    # ---------------------------------------------------------------
    # Comparison algorithms (academic mode)
    # ---------------------------------------------------------------
    def depth_first_search(self, start_node, goal_node):
        """
        Function: depth_first_search
        Purpose : DFS (graph search, LIFO stack). Finds a route quickly but
                  not necessarily the shortest one. Unweighted view.
        """
        self._validate_endpoints(start_node, goal_node)
        stack = [(start_node, [start_node])]
        visited, expanded_order = set(), []
        while stack:
            current_node, path = stack.pop()
            if current_node in visited:
                continue
            visited.add(current_node)
            expanded_order.append(current_node)
            if current_node == goal_node:
                return self._result("DFS", path, expanded_order, "unweighted")
            # push in reverse so the alphabetically first neighbour is explored first
            for neighbour in sorted(self.adjacency[current_node], reverse=True):
                if neighbour not in visited:
                    stack.append((neighbour, path + [neighbour]))
        raise NavigationError(f"No route exists from {start_node} to {goal_node}.")

    def depth_limited_search(self, start_node, goal_node, depth_limit=config.DEPTH_LIMIT,
                             expanded_order=None):
        """
        Function: depth_limited_search
        Purpose : DLS - depth-first search that never goes deeper than
                  depth_limit edges (no cycles along the current path).
                  Returns a result with found=False when the limit cuts
                  the search off before the goal.
        """
        self._validate_endpoints(start_node, goal_node)
        if depth_limit < 0:
            raise NavigationError("Depth limit must be zero or more.")
        expanded_order = [] if expanded_order is None else expanded_order

        def recursive_dls(node, path, remaining):
            """
            Function: recursive_dls
            Purpose : Expand node; return the path to the goal or None.
            """
            expanded_order.append(node)
            if node == goal_node:
                return path
            if remaining == 0:
                return None
            for neighbour in sorted(self.adjacency[node]):
                if neighbour not in path:
                    found_path = recursive_dls(neighbour, path + [neighbour], remaining - 1)
                    if found_path:
                        return found_path
            return None

        path = recursive_dls(start_node, [start_node], depth_limit)
        result = self._result(f"DLS (limit {depth_limit})", path or [], expanded_order, "unweighted")
        result["depth_limit"] = depth_limit
        return result

    def iterative_deepening_search(self, start_node, goal_node):
        """
        Function: iterative_deepening_search
        Purpose : IDS - run DLS with limit 0, 1, 2, ... until the goal is
                  found. Finds the fewest-stops route like BFS with the
                  memory use of DFS; nodes are re-expanded at every depth.
        """
        self._validate_endpoints(start_node, goal_node)
        expanded_order = []
        for depth_limit in range(len(self.adjacency)):
            result = self.depth_limited_search(start_node, goal_node, depth_limit, expanded_order)
            if result["found"]:
                final = self._result("IDS", result["path"], expanded_order, "unweighted")
                final["depth_limit"] = depth_limit
                return final
        raise NavigationError(f"No route exists from {start_node} to {goal_node}.")

    def bidirectional_bfs(self, start_node, goal_node):
        """
        Function: bidirectional_bfs
        Purpose : Bidirectional BFS - grow one BFS from the start and one
                  from the goal, one level at a time, until they meet.
                  Fewest-stops route while expanding fewer nodes.
        """
        self._validate_endpoints(start_node, goal_node)
        if start_node == goal_node:
            return self._result("Bidirectional BFS", [start_node], [], "unweighted")
        parents = {"forward": {start_node: None}, "backward": {goal_node: None}}
        frontiers = {"forward": deque([start_node]), "backward": deque([goal_node])}
        expanded_order = []
        while frontiers["forward"] and frontiers["backward"]:
            for side, other_side in (("forward", "backward"), ("backward", "forward")):
                for _ in range(len(frontiers[side])):            # expand one whole level
                    current_node = frontiers[side].popleft()
                    expanded_order.append(current_node)
                    for neighbour in sorted(self.adjacency[current_node]):
                        if neighbour in parents[side]:
                            continue
                        parents[side][neighbour] = current_node
                        if neighbour in parents[other_side]:
                            path = self._join_paths(parents, neighbour)
                            return self._result("Bidirectional BFS", path, expanded_order, "unweighted")
                        frontiers[side].append(neighbour)
        raise NavigationError(f"No route exists from {start_node} to {goal_node}.")

    @staticmethod
    def _join_paths(parents, meeting_node):
        """
        Function: _join_paths
        Purpose : Join the forward and backward BFS trees at the meeting node.
        """
        forward_path = [meeting_node]
        while parents["forward"][forward_path[-1]] is not None:
            forward_path.append(parents["forward"][forward_path[-1]])
        backward_path = []
        node = parents["backward"][meeting_node]
        while node is not None:
            backward_path.append(node)
            node = parents["backward"][node]
        return list(reversed(forward_path)) + backward_path

    def greedy_best_first_search(self, start_node, goal_node):
        """
        Function: greedy_best_first_search
        Purpose : Greedy Best-First Search - always expand the frontier node
                  with the smallest heuristic h(n). Fast, but ignores the
                  cost already travelled, so the route may not be cheapest.
        """
        self._validate_endpoints(start_node, goal_node)
        if not self.heuristic_available():
            raise NavigationError("Greedy Best-First Search needs a heuristic (building coordinates).")
        frontier = [(self.heuristic(start_node, goal_node), start_node)]
        parent_map = {start_node: None}
        visited, expanded_order = set(), []
        while frontier:
            _, current_node = heapq.heappop(frontier)
            if current_node in visited:
                continue
            visited.add(current_node)
            expanded_order.append(current_node)
            if current_node == goal_node:
                return self._result("Greedy Best-First", self._rebuild_path(parent_map, goal_node),
                                    expanded_order)
            for neighbour in sorted(self.adjacency[current_node]):
                if neighbour not in visited and neighbour not in parent_map:
                    parent_map[neighbour] = current_node
                    heapq.heappush(frontier, (self.heuristic(neighbour, goal_node), neighbour))
        raise NavigationError(f"No route exists from {start_node} to {goal_node}.")

    def recursive_best_first_search(self, start_node, goal_node):
        """
        Function: recursive_best_first_search
        Purpose : RBFS - A*-like search in linear memory. Follows the best
                  child while its f = g + h stays within the best
                  alternative; otherwise it backs up, remembering the
                  child's best f value. Optimal with an admissible h.
        """
        self._validate_endpoints(start_node, goal_node)
        if not self.heuristic_available():
            raise NavigationError("RBFS needs a heuristic (building coordinates).")
        expanded_order = []

        def rbfs(node, path, cost_so_far, node_f, f_limit):
            """
            Function: rbfs
            Purpose : One RBFS call. Returns (path or None, backed-up f).
            """
            expanded_order.append(node)
            if node == goal_node:
                return path, node_f
            successors = []
            for neighbour, edge_cost in sorted(self.adjacency[node].items()):
                if neighbour in path:
                    continue
                child_cost = cost_so_far + edge_cost
                child_f = max(child_cost + self.heuristic(neighbour, goal_node), node_f)
                successors.append([child_f, child_cost, neighbour])
            if not successors:
                return None, math.inf
            while True:
                successors.sort()
                best = successors[0]
                if best[0] > f_limit:
                    return None, best[0]
                alternative = successors[1][0] if len(successors) > 1 else math.inf
                found_path, best[0] = rbfs(best[2], path + [best[2]], best[1], best[0],
                                           min(f_limit, alternative))
                if found_path:
                    return found_path, best[0]

        path, _ = rbfs(start_node, [start_node], 0.0, self.heuristic(start_node, goal_node), math.inf)
        if not path:
            raise NavigationError(f"No route exists from {start_node} to {goal_node}.")
        return self._result("RBFS", path, expanded_order)


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


def choose_algorithm(graph_type=config.DEFAULT_GRAPH_TYPE, use_heuristic=config.USE_HEURISTIC,
                     campus_graph=None):
    """
    Function: choose_algorithm
    Purpose : Official search policy of the project (handout 7.2):
                unweighted                     -> BFS
                weighted and heuristic usable  -> A*
                weighted, no heuristic         -> UCS
    Returns : (algorithm name, policy sentence)
    """
    campus_graph = campus_graph or get_campus_graph()
    if graph_type not in config.GRAPH_TYPES:
        raise NavigationError(f"Unknown graph type '{graph_type}'. Use one of {config.GRAPH_TYPES}.")
    if graph_type == "unweighted":
        return "BFS", "Unweighted graph -> BFS (fewest stops)"
    if use_heuristic and campus_graph.heuristic_available():
        return "A*", "Weighted graph with admissible heuristic -> A*"
    return "UCS", "Weighted graph without heuristic -> UCS (fallback)"


def find_route(source, destination, graph_type=config.DEFAULT_GRAPH_TYPE, use_heuristic=config.USE_HEURISTIC,
               campus_graph=None):
    """
    Function: find_route
    Purpose : Operational mode - run the ONE algorithm chosen by the
              search policy and return the standard search output plus
              the policy text.
    """
    campus_graph = campus_graph or get_campus_graph()
    algorithm_name, policy = choose_algorithm(graph_type, use_heuristic, campus_graph)
    searches = {"BFS": campus_graph.breadth_first_search, "UCS": campus_graph.uniform_cost_search,
                "A*": campus_graph.a_star_search}
    result = searches[algorithm_name](source, destination)
    result["policy"] = policy
    return result


def trace_search(source, destination, algorithm_name, campus_graph=None):
    """
    Function: trace_search
    Purpose : Run one search algorithm and also record every decision it
              makes (which node it expands, why, and what happens to each
              neighbour). Used by the step-by-step view in the web UI.
    Returns : (result dict, list of trace steps)
    """
    campus_graph = campus_graph or get_campus_graph()
    searches = {"BFS": campus_graph.breadth_first_search, "UCS": campus_graph.uniform_cost_search,
                "A*": campus_graph.a_star_search}
    if algorithm_name not in searches:
        raise NavigationError(f"Unknown algorithm '{algorithm_name}'. Use BFS, UCS or A*.")
    trace = []
    result = searches[algorithm_name](source, destination, trace=trace)
    return result, trace


OPERATIONAL_ALGORITHMS = ["BFS", "UCS", "A*"]
COMPARISON_ALGORITHMS = ["BFS", "DFS", "DLS", "IDS", "Bidirectional BFS",
                         "UCS", "Greedy Best-First", "A*", "RBFS"]


def compare_algorithms(source, destination, campus_graph=None, include_all=True):
    """
    Function: compare_algorithms
    Purpose : Comparison / demo mode - run the algorithms on the same
              query so they can be compared on path, cost, steps, nodes
              expanded and execution time. include_all=False runs only
              the three operational algorithms (BFS, UCS, A*).
    Returns : list of result dicts (an algorithm that finds no route,
              e.g. DLS cut off by its limit, has found=False)
    """
    campus_graph = campus_graph or get_campus_graph()
    searches = {
        "BFS": campus_graph.breadth_first_search,
        "DFS": campus_graph.depth_first_search,
        "DLS": campus_graph.depth_limited_search,
        "IDS": campus_graph.iterative_deepening_search,
        "Bidirectional BFS": campus_graph.bidirectional_bfs,
        "UCS": campus_graph.uniform_cost_search,
        "Greedy Best-First": campus_graph.greedy_best_first_search,
        "A*": campus_graph.a_star_search,
        "RBFS": campus_graph.recursive_best_first_search,
    }
    results = []
    for algorithm_name in (COMPARISON_ALGORITHMS if include_all else OPERATIONAL_ALGORITHMS):
        started = time.perf_counter()
        result = searches[algorithm_name](source, destination)
        result["execution_time_ms"] = round((time.perf_counter() - started) * 1000, 3)
        result["operational"] = algorithm_name in OPERATIONAL_ALGORITHMS
        results.append(result)
    return results
