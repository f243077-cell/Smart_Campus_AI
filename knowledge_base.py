"""
knowledge_base.py
-----------------------------------------------------------------------
Module 3.4 - Logic / Knowledge Base
A small First-Order Logic (definite clause) reasoning engine.

  * Atoms are tuples:          ("Eligible", "Ali", "AI")
  * Variables start with '?':  ("Student", "?x")
  * Rules are Horn clauses:    premise_1 AND premise_2 ... => conclusion

Supports unification, forward chaining (derive every entailed fact)
and backward chaining (prove a query and return a proof trace). It is
the gatekeeper of the platform: if a request is not entailed, the
pipeline stops here and the rejection is explained.
-----------------------------------------------------------------------
"""

import itertools

import config


class KnowledgeBaseError(Exception):
    """Raised for malformed facts, rules or queries."""


class Rule:
    """
    Class: Rule
    Purpose: Hold one definite clause: a list of premises and a conclusion.
    """

    def __init__(self, rule_name, premises, conclusion, description=""):
        """
        Function: __init__
        Purpose : Store the rule parts after validating their shape.
        """
        for atom in list(premises) + [conclusion]:
            validate_atom(atom)
        self.rule_name = rule_name
        self.premises = [tuple(atom) for atom in premises]
        self.conclusion = tuple(conclusion)
        self.description = description

    def __str__(self):
        """
        Function: __str__
        Purpose : Human readable form, e.g. R1: Teaches(?x,?c) => Instructor(?x,?c)
        """
        left_side = " AND ".join(format_atom(atom) for atom in self.premises)
        return f"{self.rule_name}: {left_side} => {format_atom(self.conclusion)}"


# -------------------------------------------------------------------
# Atom helpers and unification
# -------------------------------------------------------------------
def validate_atom(atom):
    """
    Function: validate_atom
    Purpose : Make sure an atom is a non-empty tuple of strings whose first
              element is the predicate name.
    """
    if not isinstance(atom, (tuple, list)) or len(atom) < 2:
        raise KnowledgeBaseError(f"Atom {atom!r} must be (Predicate, arg1, ...).")
    if not all(isinstance(part, str) and part for part in atom):
        raise KnowledgeBaseError(f"Atom {atom!r} must contain only non-empty strings.")


def is_variable(term):
    """
    Function: is_variable
    Purpose : A term is a variable when it starts with '?'.
    """
    return isinstance(term, str) and term.startswith("?")


def format_atom(atom):
    """
    Function: format_atom
    Purpose : Convert ("Eligible","Ali","AI") into the string Eligible(Ali, AI).
    """
    return f"{atom[0]}({', '.join(atom[1:])})"


def parse_atom(text):
    """
    Function: parse_atom
    Purpose : Parse a string like 'Eligible(Ali, AI)' into an atom tuple.
              Lower-case single words like x are treated as variables ?x.
    """
    text = text.strip()
    if "(" not in text or not text.endswith(")"):
        raise KnowledgeBaseError("Query must look like Predicate(arg1, arg2).")
    predicate, argument_text = text[:-1].split("(", 1)
    arguments = [argument.strip() for argument in argument_text.split(",") if argument.strip()]
    if not predicate.strip() or not arguments:
        raise KnowledgeBaseError("Query must have a predicate name and at least one argument.")
    converted = []
    for argument in arguments:
        if argument.startswith("?"):
            converted.append(argument)
        elif len(argument) == 1 and argument.islower():
            converted.append("?" + argument)
        else:
            converted.append(argument)
    return tuple([predicate.strip()] + converted)


def substitute(atom, bindings):
    """
    Function: substitute
    Purpose : Replace every bound variable inside an atom with its value,
              following chains of bindings (?x -> ?y -> Ali).
    """
    result = []
    for term in atom:
        while is_variable(term) and term in bindings:
            term = bindings[term]
        result.append(term)
    return tuple(result)


def unify(atom_a, atom_b, bindings=None):
    """
    Function: unify
    Purpose : Unify two atoms. Returns an extended bindings dict when they
              can be made identical, or None if unification fails.
    """
    bindings = dict(bindings or {})
    if len(atom_a) != len(atom_b) or atom_a[0] != atom_b[0]:
        return None
    for term_a, term_b in zip(atom_a[1:], atom_b[1:]):
        term_a = substitute((term_a,), bindings)[0]
        term_b = substitute((term_b,), bindings)[0]
        if term_a == term_b:
            continue
        if is_variable(term_a):
            bindings[term_a] = term_b
        elif is_variable(term_b):
            bindings[term_b] = term_a
        else:
            return None
    return bindings


class KnowledgeBase:
    """
    Class: KnowledgeBase
    Purpose: Store ground facts and Horn rules and answer queries with
             forward and backward chaining.
    """

    MAX_PROOF_DEPTH = 12

    def __init__(self):
        """
        Function: __init__
        Purpose : Create an empty KB and a counter used to rename rule
                  variables apart on every rule use.
        """
        self.facts = set()
        self.rules = []
        self._rename_counter = itertools.count(1)

    # ---------------------------------------------------------------
    # Building the KB
    # ---------------------------------------------------------------
    def add_fact(self, atom):
        """
        Function: add_fact
        Purpose : Add a ground fact (no variables allowed) to the KB.
        """
        validate_atom(atom)
        if any(is_variable(term) for term in atom[1:]):
            raise KnowledgeBaseError(f"Fact {format_atom(atom)} must not contain variables.")
        self.facts.add(tuple(atom))

    def add_rule(self, rule):
        """
        Function: add_rule
        Purpose : Append a Rule object to the KB.
        """
        if not isinstance(rule, Rule):
            raise KnowledgeBaseError("add_rule expects a Rule object.")
        self.rules.append(rule)

    def _rename_rule(self, rule):
        """
        Function: _rename_rule
        Purpose : Standardize apart - give the rule fresh variable names so
                  variables of different rule uses never clash.
        """
        suffix = f"_{next(self._rename_counter)}"
        renamed = lambda atom: tuple(term + suffix if is_variable(term) else term for term in atom)
        return [renamed(atom) for atom in rule.premises], renamed(rule.conclusion)

    # ---------------------------------------------------------------
    # Forward chaining
    # ---------------------------------------------------------------
    def forward_chain(self):
        """
        Function: forward_chain
        Purpose : Repeatedly fire every rule whose premises are satisfied by
                  known facts until no new fact appears (fixed point).
        Returns : dict derived_fact -> (rule_name, list_of_premise_facts)
        """
        known_facts = set(self.facts)
        derivations = {}
        new_fact_found = True
        while new_fact_found:
            new_fact_found = False
            for rule in self.rules:
                # Match against a snapshot so the fact set is not changed while iterating.
                for bindings in list(self._match_premises(rule.premises, frozenset(known_facts), {})):
                    new_fact = substitute(rule.conclusion, bindings)
                    if new_fact not in known_facts and not any(is_variable(t) for t in new_fact):
                        known_facts.add(new_fact)
                        derivations[new_fact] = (rule.rule_name,
                                                 [substitute(p, bindings) for p in rule.premises])
                        new_fact_found = True
        return derivations

    def _match_premises(self, premises, known_facts, bindings):
        """
        Function: _match_premises
        Purpose : Generator yielding every binding set that satisfies all
                  premises (a join over the known facts).
        """
        if not premises:
            yield bindings
            return
        first_premise = substitute(premises[0], bindings)
        for fact in known_facts:
            extended = unify(first_premise, fact, bindings)
            if extended is not None:
                yield from self._match_premises(premises[1:], known_facts, extended)

    # ---------------------------------------------------------------
    # Backward chaining
    # ---------------------------------------------------------------
    def backward_chain(self, goal, bindings=None, depth=0):
        """
        Function: backward_chain
        Purpose : Generator that proves a goal. A goal holds if it unifies
                  with a fact, or with the conclusion of a rule whose
                  premises can all be proved recursively.
        Yields  : (bindings, proof_steps) for every proof found
        """
        bindings = bindings or {}
        if depth > self.MAX_PROOF_DEPTH:
            return
        goal = substitute(goal, bindings)
        for fact in sorted(self.facts):
            extended = unify(goal, fact, bindings)
            if extended is not None:
                yield extended, [f"{format_atom(fact)} is a known fact"]
        for rule in self.rules:
            premises, conclusion = self._rename_rule(rule)
            extended = unify(goal, conclusion, bindings)
            if extended is None:
                continue
            for final_bindings, premise_steps in self._prove_all(premises, extended, depth + 1):
                proved_goal = substitute(goal, final_bindings)
                step = (f"{format_atom(proved_goal)} by {rule.rule_name} "
                        f"[{' AND '.join(format_atom(substitute(p, final_bindings)) for p in premises)}]")
                yield final_bindings, premise_steps + [step]

    def _prove_all(self, goals, bindings, depth):
        """
        Function: _prove_all
        Purpose : Prove a conjunction of goals left to right, threading the
                  bindings from one goal to the next.
        """
        if not goals:
            yield bindings, []
            return
        for first_bindings, first_steps in self.backward_chain(goals[0], bindings, depth):
            for rest_bindings, rest_steps in self._prove_all(goals[1:], first_bindings, depth):
                yield rest_bindings, first_steps + rest_steps

    def ask(self, query):
        """
        Function: ask
        Purpose : Answer a query. Returns a dict with:
                    entailed    - True/False
                    query       - formatted query
                    answers     - variable bindings (for queries with variables)
                    proof       - proof steps of the first proof
                    missing     - facts that would be needed (on failure)
        """
        validate_atom(query)
        query = tuple(query)
        query_variables = [term for term in query[1:] if is_variable(term)]
        answers, first_proof, seen = [], None, set()
        for bindings, proof_steps in self.backward_chain(query):
            grounded = substitute(query, bindings)
            if grounded in seen:
                continue
            seen.add(grounded)
            if first_proof is None:
                first_proof = proof_steps
            answers.append({variable: substitute((variable,), bindings)[0]
                            for variable in query_variables})
            if not query_variables:
                break
        if first_proof is not None:
            return {"entailed": True, "query": format_atom(query), "answers": answers,
                    "proof": first_proof, "missing": []}
        return {"entailed": False, "query": format_atom(query), "answers": [],
                "proof": [], "missing": self.explain_failure(query)}

    def explain_failure(self, goal, depth=0):
        """
        Function: explain_failure
        Purpose : When a goal cannot be proved, find which premises are
                  missing for each rule that could have concluded it. This
                  gives the user a clear reason for the rejection.
        Returns : list of human readable explanation strings
        """
        if depth > 3:
            return []
        explanations = []
        for rule in self.rules:
            premises, conclusion = self._rename_rule(rule)
            bindings = unify(goal, conclusion, {})
            if bindings is None:
                continue
            missing_premises = self._first_missing_premises(premises, bindings)
            if missing_premises:
                explanations.append(f"{rule.rule_name} needs: "
                                    + ", ".join(format_atom(p) for p in missing_premises))
        if not explanations:
            explanations.append(f"No fact or rule can establish {format_atom(goal)}")
        return explanations

    def _first_missing_premises(self, premises, bindings):
        """
        Function: _first_missing_premises
        Purpose : Walk the premises left to right and return the ones that
                  cannot be proved under the current bindings.
        """
        missing = []
        current_bindings = dict(bindings)
        for premise in premises:
            grounded = substitute(premise, current_bindings)
            proof = next(self.backward_chain(grounded, current_bindings), None)
            if proof is None:
                missing.append(tuple(t if not is_variable(t) else "?" for t in grounded))
            else:
                current_bindings = proof[0]
        return missing

    # ---------------------------------------------------------------
    # Convenience gatekeeper checks used by the router
    # ---------------------------------------------------------------
    def is_registered(self, user_id):
        """
        Function: is_registered
        Purpose : True if the KB knows the user under any role.
        """
        return self.ask(("HasRole", user_id, "?r"))["entailed"]

    def check_role(self, user_id, role):
        """
        Function: check_role
        Purpose : Verify that the role claimed in the request is supported
                  by the KB (prevents a student pretending to be staff).
        """
        result = self.ask(("HasRole", user_id, role))
        if not result["entailed"]:
            known_roles = [a["?r"] for a in self.ask(("HasRole", user_id, "?r"))["answers"]]
            if known_roles:
                result["missing"] = [f"{user_id} is registered as {', '.join(known_roles)}, "
                                     f"not as {role}"]
            else:
                result["missing"] = [f"{user_id} is not a registered campus user"]
        return result


def build_campus_knowledge_base():
    """
    Function: build_campus_knowledge_base
    Purpose : Create the campus KB with the facts and rules from the FOL
              diagram of the project (R1-R4) plus campus authorization
              rules used by the booking, urgent and full-service flows.
    Returns : KnowledgeBase
    """
    knowledge_base = KnowledgeBase()

    # Facts from the FOL diagram
    diagram_facts = [
        ("Teaches", "DrKhan", "AI"),
        ("Enrolled", "Ali", "AI"),
        ("Student", "Sara"),
        ("Student", "Ali"),
        ("Completed", "Ali", "ProgrammingFundamentals"),
    ]
    # Additional campus facts
    campus_facts = [
        ("Student", "Ahmed"),
        ("Student", "Fatima"),
        ("Completed", "Fatima", "ProgrammingFundamentals"),
        ("Completed", "Ahmed", "ProgrammingFundamentals"),
        ("Completed", "Ahmed", "DataStructures"),
        ("Enrolled", "Fatima", "DataStructures"),
        ("Teaches", "DrAyesha", "DataStructures"),
        ("Staff", "Bilal"),
        ("Prerequisite", "AI", "ProgrammingFundamentals"),
        ("Prerequisite", "DataStructures", "ProgrammingFundamentals"),
        ("Prerequisite", "MachineLearning", "AI"),
        ("Prerequisite", "Databases", "DataStructures"),
        ("Course", "AI"), ("Course", "DataStructures"),
        ("Course", "MachineLearning"), ("Course", "Databases"),
        ("Course", "ProgrammingFundamentals"),
        ("Completed", "Ahmed", "AI"),
    ]
    for fact in diagram_facts + campus_facts:
        knowledge_base.add_fact(fact)

    rules = [
        # Rules R1-R4 exactly as in the FOL diagram
        Rule("R1", [("Teaches", "?x", "?c")], ("Instructor", "?x", "?c"),
             "Whoever teaches a course is its instructor"),
        Rule("R2", [("Enrolled", "?x", "AI")], ("UseLab", "?x", "Lab1"),
             "Students enrolled in AI may use Lab1"),
        Rule("R3", [("Student", "?x"), ("Completed", "?x", "ProgrammingFundamentals")],
             ("Eligible", "?x", "AI"),
             "A student who completed Programming Fundamentals is eligible for AI"),
        Rule("R4", [("Instructor", "?x", "AI")], ("UseLab", "?x", "Lab1"),
             "The AI instructor may use Lab1"),
        # General eligibility via prerequisites
        Rule("R5", [("Student", "?x"), ("Prerequisite", "?c", "?p"), ("Completed", "?x", "?p")],
             ("Eligible", "?x", "?c"),
             "A student is eligible for a course after completing its prerequisite"),
        # Roles
        Rule("R6", [("Student", "?x")], ("HasRole", "?x", "Student"), "Student role"),
        Rule("R7", [("Instructor", "?x", "?c")], ("HasRole", "?x", "Instructor"), "Instructor role"),
        Rule("R8", [("Staff", "?x")], ("HasRole", "?x", "Staff"), "Staff role"),
        # Booking authorization
        Rule("R9", [("HasRole", "?x", "Student")], ("CanBook", "?x", "StudyRoom"),
             "Students can book study rooms"),
        Rule("R10", [("HasRole", "?x", "Instructor")], ("CanBook", "?x", "StudyRoom"),
             "Instructors can book study rooms"),
        Rule("R11", [("HasRole", "?x", "Instructor")], ("CanBook", "?x", "SeminarRoom"),
             "Instructors can book seminar rooms"),
        Rule("R12", [("HasRole", "?x", "Instructor")], ("CanBook", "?x", "ExamHall"),
             "Instructors can book exam halls"),
        Rule("R13", [("UseLab", "?x", "Lab1")], ("CanBook", "?x", "ComputerLab"),
             "Anyone allowed to use Lab1 can book a computer lab"),
        Rule("R14", [("HasRole", "?x", "Staff")], ("CanBook", "?x", "MeetingRoom"),
             "Staff can book meeting rooms"),
        Rule("R15", [("HasRole", "?x", "Instructor")], ("CanBook", "?x", "MeetingRoom"),
             "Instructors can book meeting rooms"),
        # Urgent services: any registered campus member may request them
        Rule("R16", [("HasRole", "?x", "?r")], ("CanRequest", "?x", "UrgentService"),
             "Every registered campus member can request urgent services"),
    ]
    for rule in rules:
        knowledge_base.add_rule(rule)
    return knowledge_base


def resource_query_for_target(user_id, target):
    """
    Function: resource_query_for_target
    Purpose : Turn an Eligibility_Check target into the matching FOL query.
                course name        -> Eligible(user, course)
                Lab1 / AI_Lab      -> UseLab(user, Lab1)
                resource type name -> CanBook(user, type)
    """
    if target in ("Lab1", "AI_Lab"):
        return ("UseLab", user_id, "Lab1")
    if target in config.RESOURCE_TYPES:
        return ("CanBook", user_id, target)
    return ("Eligible", user_id, target)
