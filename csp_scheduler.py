"""
csp_scheduler.py
-----------------------------------------------------------------------
Module 3.5 - CSP Scheduler
Finds conflict-free assignments of (time slot, room) values to
variables while respecting domains and constraints.

Techniques used:
  * AC-3 arc consistency as preprocessing
  * Backtracking search
  * MRV  (Minimum Remaining Values) variable ordering, degree tie-break
  * LCV  (Least Constraining Value) value ordering
  * Forward checking after every assignment

The CSP module NEVER runs before logical validation; the router calls it
only after the Logic / KB module has approved the request.
-----------------------------------------------------------------------
"""

from collections import deque

import pandas as pd

import config


class SchedulingError(Exception):
    """Raised when the CSP is malformed or a booking request is invalid."""


class ConstraintSatisfactionProblem:
    """
    Class: ConstraintSatisfactionProblem
    Purpose: Generic binary CSP with a backtracking solver.
    """

    def __init__(self, variables, domains):
        """
        Function: __init__
        Purpose : Store variables and their domains; start with no
                  constraints. Every variable must have a domain.
        """
        missing = [variable for variable in variables if variable not in domains]
        if missing:
            raise SchedulingError(f"Variables without a domain: {missing}")
        self.variables = list(variables)
        self.domains = {variable: list(domains[variable]) for variable in variables}
        self.constraints = {}          # (Xi, Xj) -> list of (label, predicate(vi, vj))
        self.neighbours = {variable: set() for variable in variables}
        self.statistics = {"assignments_tried": 0, "backtracks": 0, "ac3_pruned": 0}

    def add_constraint(self, variable_a, variable_b, predicate, label):
        """
        Function: add_constraint
        Purpose : Add a binary constraint predicate(value_a, value_b). The
                  reverse arc is stored automatically.
        """
        if variable_a not in self.domains or variable_b not in self.domains:
            raise SchedulingError(f"Unknown variable in constraint {variable_a}-{variable_b}")
        self.constraints.setdefault((variable_a, variable_b), []).append((label, predicate))
        self.constraints.setdefault((variable_b, variable_a), []).append(
            (label, lambda value_b, value_a, original=predicate: original(value_a, value_b)))
        self.neighbours[variable_a].add(variable_b)
        self.neighbours[variable_b].add(variable_a)

    def is_consistent_pair(self, variable_a, value_a, variable_b, value_b):
        """
        Function: is_consistent_pair
        Purpose : True when every constraint between two variables holds.
        """
        return all(predicate(value_a, value_b)
                   for _, predicate in self.constraints.get((variable_a, variable_b), []))

    # ---------------------------------------------------------------
    # AC-3
    # ---------------------------------------------------------------
    def ac3(self, domains):
        """
        Function: ac3
        Purpose : Enforce arc consistency. Removes values that have no
                  supporting value in a neighbour's domain.
        Returns : False if some domain becomes empty (no solution)
        """
        queue = deque(self.constraints.keys())
        while queue:
            variable_a, variable_b = queue.popleft()
            if self._revise(domains, variable_a, variable_b):
                if not domains[variable_a]:
                    return False
                for neighbour in self.neighbours[variable_a] - {variable_b}:
                    queue.append((neighbour, variable_a))
        return True

    def _revise(self, domains, variable_a, variable_b):
        """
        Function: _revise
        Purpose : Delete values of variable_a with no support in variable_b.
        """
        revised = False
        for value_a in list(domains[variable_a]):
            if not any(self.is_consistent_pair(variable_a, value_a, variable_b, value_b)
                       for value_b in domains[variable_b]):
                domains[variable_a].remove(value_a)
                self.statistics["ac3_pruned"] += 1
                revised = True
        return revised

    # ---------------------------------------------------------------
    # Backtracking with MRV, LCV and forward checking
    # ---------------------------------------------------------------
    def solve(self, value_preference=None):
        """
        Function: solve
        Purpose : Run AC-3 then backtracking search.
        Inputs  : value_preference - optional function(variable, value) ->
                  sortable key used as a tie-break after LCV (e.g. prefer
                  earlier slots for high priority requests)
        Returns : dict variable -> value, or None if no solution exists
        """
        working_domains = {variable: list(values) for variable, values in self.domains.items()}
        if not self.ac3(working_domains):
            return None
        return self._backtrack({}, working_domains, value_preference)

    def _backtrack(self, assignment, domains, value_preference):
        """
        Function: _backtrack
        Purpose : Recursive backtracking search with forward checking.
        """
        if len(assignment) == len(self.variables):
            return dict(assignment)
        variable = self._select_unassigned_variable(assignment, domains)
        for value in self._order_domain_values(variable, assignment, domains, value_preference):
            self.statistics["assignments_tried"] += 1
            if all(self.is_consistent_pair(variable, value, other, assignment[other])
                   for other in self.neighbours[variable] if other in assignment):
                assignment[variable] = value
                reduced_domains = self._forward_check(variable, value, assignment, domains)
                if reduced_domains is not None:
                    result = self._backtrack(assignment, reduced_domains, value_preference)
                    if result is not None:
                        return result
                del assignment[variable]
                self.statistics["backtracks"] += 1
        return None

    def _select_unassigned_variable(self, assignment, domains):
        """
        Function: _select_unassigned_variable
        Purpose : MRV heuristic - choose the variable with the fewest legal
                  values; break ties with the degree heuristic (most
                  constraints on unassigned variables).
        """
        unassigned = [variable for variable in self.variables if variable not in assignment]
        return min(unassigned, key=lambda variable: (
            len(domains[variable]),
            -len([n for n in self.neighbours[variable] if n not in assignment]),
            variable))

    def _order_domain_values(self, variable, assignment, domains, value_preference):
        """
        Function: _order_domain_values
        Purpose : LCV heuristic - try first the value that rules out the
                  fewest values in the neighbours' domains.
        """
        def values_ruled_out(value):
            """
            Function: values_ruled_out
            Purpose : Count neighbour values that would become illegal if
                      this value were assigned (smaller = less constraining).
            """
            count = 0
            for neighbour in self.neighbours[variable]:
                if neighbour in assignment:
                    continue
                count += sum(1 for other_value in domains[neighbour]
                             if not self.is_consistent_pair(variable, value, neighbour, other_value))
            return count

        def sort_key(value):
            """
            Function: sort_key
            Purpose : Order values by LCV count first, then the optional
                      caller preference as a tie-break.
            """
            preference = value_preference(variable, value) if value_preference else 0
            return (values_ruled_out(value), preference)

        return sorted(domains[variable], key=sort_key)

    def _forward_check(self, variable, value, assignment, domains):
        """
        Function: _forward_check
        Purpose : After assigning variable=value, remove inconsistent values
                  from every unassigned neighbour. Returns the new domains
                  or None if a neighbour's domain becomes empty.
        """
        new_domains = {name: list(values) for name, values in domains.items()}
        new_domains[variable] = [value]
        for neighbour in self.neighbours[variable]:
            if neighbour in assignment:
                continue
            new_domains[neighbour] = [other for other in new_domains[neighbour]
                                      if self.is_consistent_pair(variable, value, neighbour, other)]
            if not new_domains[neighbour]:
                return None
        return new_domains

    def violated_constraints(self, assignment):
        """
        Function: violated_constraints
        Purpose : Verification helper - list every constraint broken by a
                  complete assignment (empty list means valid solution).
        """
        violations = []
        for (variable_a, variable_b), constraint_list in self.constraints.items():
            if variable_a < variable_b:
                for label, predicate in constraint_list:
                    if not predicate(assignment[variable_a], assignment[variable_b]):
                        violations.append(f"{variable_a}-{variable_b}: {label}")
        return violations


# -------------------------------------------------------------------
# Value helpers: a value is a (slot, room) tuple
# -------------------------------------------------------------------
def slot_index(value):
    """
    Function: slot_index
    Purpose : Position of a value's time slot in config.TIME_SLOTS.
    """
    return config.TIME_SLOTS.index(value[0])


def different_slot(value_a, value_b):
    """
    Function: different_slot
    Purpose : Constraint - the two values must use different time slots.
    """
    return value_a[0] != value_b[0]


def different_slot_or_room(value_a, value_b):
    """
    Function: different_slot_or_room
    Purpose : Constraint - two activities may not share the same room at
              the same time.
    """
    return value_a != value_b


def comes_before(value_a, value_b):
    """
    Function: comes_before
    Purpose : Precedence constraint - value_a's slot is earlier than value_b's.
    """
    return slot_index(value_a) < slot_index(value_b)


# -------------------------------------------------------------------
# The exam timetabling problem from the CSP diagram (G1..G6)
# -------------------------------------------------------------------
EXAM_GROUPS = {"G1": 45, "G2": 40, "G3": 55, "G4": 30, "G5": 50, "G6": 35}

EXAM_CONSTRAINT_EDGES = [
    ("G1", "G2", "!= (shared students)"),
    ("G1", "G4", "!= (shared students)"),
    ("G2", "G4", "!= (shared students)"),
    ("G3", "G5", "!= (shared students)"),
    ("G3", "G6", "!= (shared students)"),
    ("G5", "G6", "!= (shared students)"),
    ("G1", "G3", "!= examiner clash"),
    ("G2", "G5", "!= supervisor clash"),
]
EXAM_PRECEDENCE = [("G4", "G3", "< precedence")]


def build_exam_csp():
    """
    Function: build_exam_csp
    Purpose : Build the exam timetabling CSP from the project diagram.
              Variables G1-G6, values (slot, room) of exam-capable rooms
              with enough capacity (unary constraint), binary constraints
              from the diagram, and no two groups in the same room/slot.
    Returns : ConstraintSatisfactionProblem
    """
    exam_rooms = [name for name, info in config.ROOMS.items()
                  if info["type"] in ("ExamHall", "SeminarRoom")]
    domains = {}
    for group_name, group_size in EXAM_GROUPS.items():
        domains[group_name] = [(slot, room) for slot in config.TIME_SLOTS for room in exam_rooms
                               if config.ROOMS[room]["capacity"] >= group_size]
    csp = ConstraintSatisfactionProblem(list(EXAM_GROUPS), domains)
    for group_a, group_b, label in EXAM_CONSTRAINT_EDGES:
        csp.add_constraint(group_a, group_b, different_slot, label)
    for group_a, group_b, label in EXAM_PRECEDENCE:
        csp.add_constraint(group_a, group_b, comes_before, label)
    group_names = list(EXAM_GROUPS)
    for index_a, group_a in enumerate(group_names):
        for group_b in group_names[index_a + 1:]:
            csp.add_constraint(group_a, group_b, different_slot_or_room, "no room double-booking")
    return csp


def solve_exam_schedule():
    """
    Function: solve_exam_schedule
    Purpose : Solve the exam CSP and return the schedule as a pandas
              DataFrame plus solver statistics and constraint check.
    """
    csp = build_exam_csp()
    solution = csp.solve()
    if solution is None:
        raise SchedulingError("The exam timetable has no feasible solution.")
    schedule = pd.DataFrame(
        [{"Group": group, "Students": EXAM_GROUPS[group], "Slot": value[0], "Room": value[1],
          "Building": config.ROOMS[value[1]]["building"]}
         for group, value in solution.items()]
    ).sort_values(["Slot", "Group"]).reset_index(drop=True)
    return {"schedule": schedule, "solution": solution, "statistics": dict(csp.statistics),
            "violations": csp.violated_constraints(solution)}


# -------------------------------------------------------------------
# Booking scheduler used by Booking_or_Scheduling / Full_Service requests
# -------------------------------------------------------------------
class CampusScheduler:
    """
    Class: CampusScheduler
    Purpose: Keep track of occupied (slot, room) pairs and assign new
             booking requests with a CSP.
    """

    def __init__(self):
        """
        Function: __init__
        Purpose : Start with the solved exam timetable plus a few fixed
                  weekly bookings already occupying rooms.
        """
        self.occupied = {}             # (slot, room) -> booked_by
        exam_result = solve_exam_schedule()
        for group, value in exam_result["solution"].items():
            self.occupied[value] = f"Exam {group}"
        fixed_bookings = [
            (("09:00-10:30", "Lab1"), "AI Lab class (DrKhan)"),
            (("09:00-10:30", "Study_Room_L1"), "Reading group"),
            (("11:00-12:30", "Meeting_Room_A"), "Admin meeting"),
        ]
        for value, owner in fixed_bookings:
            self.occupied.setdefault(value, owner)

    def booking_table(self):
        """
        Function: booking_table
        Purpose : Return all current bookings as a pandas DataFrame.
        """
        rows = [{"Slot": slot, "Room": room, "Booked_By": owner}
                for (slot, room), owner in self.occupied.items()]
        return pd.DataFrame(rows).sort_values(["Slot", "Room"]).reset_index(drop=True)

    def schedule_request(self, user_id, resource_type, group_size, preferred_slot=None,
                         sessions=1, priority_label=None, commit=True):
        """
        Function: schedule_request
        Purpose : Build and solve a CSP for one booking request.
                  Variables : Session_1 .. Session_n
                  Domain    : (slot, room) with matching room type, enough
                              capacity, free, and not in the user's busy slots
                  Constraints: sessions in different and increasing slots
                  Preference : preferred slot first; Urgent/High priority
                               (from ANN) prefers the earliest slot; LCV
                               keeps larger rooms free.
        Returns : dict with success flag, assignment rows and statistics
        """
        if resource_type not in config.RESOURCE_TYPES:
            raise SchedulingError(f"Unknown resource type '{resource_type}'.")
        if group_size <= 0:
            raise SchedulingError("Group size must be a positive number.")
        if not 1 <= sessions <= len(config.TIME_SLOTS):
            raise SchedulingError(f"Sessions must be between 1 and {len(config.TIME_SLOTS)}.")
        if preferred_slot is not None and preferred_slot not in config.TIME_SLOTS:
            raise SchedulingError(f"Preferred slot must be one of {config.TIME_SLOTS}.")

        busy_slots = set(config.USER_BUSY_SLOTS.get(user_id, []))
        candidate_values = [
            (slot, room)
            for slot in config.TIME_SLOTS for room, info in config.ROOMS.items()
            if info["type"] == resource_type and info["capacity"] >= group_size
            and (slot, room) not in self.occupied and slot not in busy_slots
        ]
        session_names = [f"Session_{number}" for number in range(1, sessions + 1)]
        csp = ConstraintSatisfactionProblem(session_names,
                                            {name: candidate_values for name in session_names})
        for index in range(len(session_names) - 1):
            csp.add_constraint(session_names[index], session_names[index + 1],
                               comes_before, "sessions in increasing slots")

        urgent = priority_label in ("Urgent", "High")

        def value_preference(_, value):
            """
            Function: value_preference
            Purpose : Tie-break key for LCV ordering - preferred slot first,
                      earliest slot for urgent requests, then the smallest
                      room that fits (keeps bigger rooms available).
            """
            preferred_rank = 0 if preferred_slot and value[0] == preferred_slot else 1
            capacity = config.ROOMS[value[1]]["capacity"]
            if urgent:
                return (slot_index(value), preferred_rank, capacity)
            return (preferred_rank, capacity, slot_index(value))

        solution = csp.solve(value_preference) if candidate_values else None
        reasons = []
        if solution is None:
            if not any(info["type"] == resource_type and info["capacity"] >= group_size
                       for info in config.ROOMS.values()):
                reasons.append(f"No {resource_type} has capacity for {group_size} people")
            elif not candidate_values:
                reasons.append(f"Every suitable {resource_type} is already booked in your free slots")
            else:
                reasons.append(f"Not enough free slots for {sessions} separate sessions")
            return {"success": False, "assignments": [], "statistics": dict(csp.statistics),
                    "reasons": reasons, "domain_size": len(candidate_values)}

        assignments = []
        for name in session_names:
            slot, room = solution[name]
            if commit:
                self.occupied[(slot, room)] = f"{user_id} ({resource_type})"
            assignments.append({"session": name, "slot": slot, "room": room,
                                "building": config.ROOMS[room]["building"],
                                "capacity": config.ROOMS[room]["capacity"]})
        if preferred_slot and assignments[0]["slot"] != preferred_slot:
            reasons.append(f"Preferred slot {preferred_slot} was not available")
        return {"success": True, "assignments": assignments, "statistics": dict(csp.statistics),
                "reasons": reasons, "domain_size": len(candidate_values)}
