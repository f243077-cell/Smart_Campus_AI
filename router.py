"""
router.py
-----------------------------------------------------------------------
Module 3.2 - Request Router (control-flow brain of the platform)
Reads the request_type of the processed request and decides WHICH
modules run and in WHAT order (handout 7.1):

  Navigation_Only         -> Search
  Eligibility_Check       -> Logic_KB
  Booking_or_Scheduling   -> Logic_KB -> CSP -> optional Search
  Urgent_Service_Request  -> ANN -> Logic_KB -> CSP -> optional Search
  Full_Service_Request    -> ANN -> Logic_KB -> CSP -> Search

Search is optional when route guidance is not needed (no
current_location given). The router produces the standard control
object {request_id, selected_pipeline, needs_ann, needs_logic,
needs_csp, needs_search}, dispatches the request to the modules and
collects their outputs. Logic_KB is the gatekeeper: if the request is
not allowed, CSP and Search never run ("Eligible / Allowed?" decision).
-----------------------------------------------------------------------
"""

import config
import preprocessing
import response_layer
from ann_module import ModelError, PriorityPredictor
from csp_scheduler import CampusScheduler, SchedulingError
from knowledge_base import KnowledgeBaseError, build_campus_knowledge_base, parse_atom
from search_navigation import NavigationError, find_route, get_campus_graph

# Pipelines for display: optional modules are shown in brackets.
PIPELINES = {request_type: [module if not optional else f"({module})" for module, optional in template]
             for request_type, template in config.PIPELINE_TEMPLATES.items()}


def route_request(standard_request, needs):
    """
    Function: route_request
    Purpose : The router decision (handout 7.1 pseudocode): inspect the
              processed request, read request_type and return the
              standard router output template.
    Inputs  : standard_request - validated request object
              needs            - needs_* flags from preprocessing
    Returns : router control object
    """
    request_type = standard_request["request_type"]
    if request_type not in config.PIPELINE_TEMPLATES:
        raise preprocessing.RequestValidationError(f"Unsupported request type '{request_type}'.")
    selected_pipeline = [module for module, optional in config.PIPELINE_TEMPLATES[request_type]
                         if not optional or needs[f"needs_{module.lower().split('_')[0]}"]]
    return {**config.ROUTER_OUTPUT_TEMPLATE,
            "request_id": standard_request["request_id"],
            "selected_pipeline": selected_pipeline,
            "needs_ann": "ANN" in selected_pipeline,
            "needs_logic": "Logic_KB" in selected_pipeline,
            "needs_csp": "CSP" in selected_pipeline,
            "needs_search": "Search" in selected_pipeline}


class SmartCampusSystem:
    """
    Class: SmartCampusSystem
    Purpose: Owns one instance of every AI module and runs requests
             through the pipeline chosen by the router.
    """

    def __init__(self, verbose=False):
        """
        Function: __init__
        Purpose : Build the campus graph, knowledge base, scheduler and load
                  (or train) the ANN models.
        """
        config.ensure_directories()
        self.campus_graph = get_campus_graph()
        self.knowledge_base = build_campus_knowledge_base()
        self.scheduler = CampusScheduler()
        self.priority_predictor = PriorityPredictor().load_or_train(verbose=verbose)
        self.graph_type = config.DEFAULT_GRAPH_TYPE
        self.use_heuristic = config.USE_HEURISTIC
        self.module_handlers = {
            "ANN": self._run_ann_module,
            "Logic_KB": self._run_logic_module,
            "CSP": self._run_csp_module,
            "Search": self._run_search_module,
        }

    # ---------------------------------------------------------------
    # Main entry point
    # ---------------------------------------------------------------
    def process_request(self, raw_request):
        """
        Function: process_request
        Purpose : Full request lifecycle (handout 5):
                  preprocessing -> router -> selected modules -> final
                  response. Never raises: every failure becomes a
                  'rejected' response that explains where and why.
        Returns : final response dict (see response_layer.build_response)
        """
        try:
            request = preprocessing.validate_request(raw_request, self.campus_graph.nodes)
            needs = preprocessing.prepare_module_inputs(request)
            router_output = route_request(request, needs)
        except preprocessing.RequestValidationError as error:
            return response_layer.build_invalid_response(raw_request, str(error))

        context = {"request": request, "needs": needs, "router": router_output, "modules_run": [],
                   "results": {}, "rejected_at": None, "stop_reason": None}
        for module_name in router_output["selected_pipeline"]:
            try:
                keep_going = self.module_handlers[module_name](context)
            except (NavigationError, SchedulingError, ModelError, KnowledgeBaseError) as error:
                context["rejected_at"] = module_name
                context["stop_reason"] = f"{module_name} module error: {error}"
                keep_going = False
            context["modules_run"].append(module_name)
            if not keep_going:
                break
        return response_layer.build_response(context)

    # ---------------------------------------------------------------
    # Module handlers (each returns True to continue the pipeline)
    # ---------------------------------------------------------------
    def _service_distance(self, request):
        """
        Function: _service_distance
        Purpose : Distance feature for the ANN: weighted route cost from the
                  requester to the nearest building that serves the
                  category (0 when already there).
        """
        buildings = {config.ROOMS[room]["building"] for room in config.CATEGORY_ROOMS[request["category"]]}
        return min(0.0 if building == request["current_location"]
                   else find_route(request["current_location"], building, "weighted", True,
                                   self.campus_graph)["cost"]
                   for building in buildings)

    def _run_ann_module(self, context):
        """
        Function: _run_ann_module
        Purpose : Build the feature vector x and predict PRIORITY only
                  (Perceptron = binary_priority, MLP = final_priority).
                  x7 Eligibility = the request's eligibility_claim when it
                  is true, otherwise whether the KB knows the requester.
                  The ANN never decides permission.
        """
        request = context["request"]
        eligible = request["eligibility_claim"] or self.knowledge_base.is_registered(request["name"])
        feature_vector = preprocessing.build_feature_vector(request, self._service_distance(request), eligible)
        context["results"]["ANN"] = self.priority_predictor.predict(feature_vector)
        return True

    def _run_logic_module(self, context):
        """
        Function: _run_logic_module
        Purpose : Gatekeeper ("Eligible / Allowed?"). Verifies the claimed
                  role, then:
                    Eligibility_Check -> answers the request's FOL query
                    service requests  -> CanRequest(name, category) and,
                                         for a group, RepresentsGroup(name, group)
                  Returns the standard logic output {allowed, entailed,
                  explanation}; stops the pipeline when not allowed.
        """
        request = context["request"]
        name = request["name"]
        checks = [("Role authorization", self.knowledge_base.check_role(name, request["role"].capitalize()))]
        role_ok = checks[0][1]["entailed"]
        if request["request_type"] == "Eligibility_Check":
            if role_ok:
                checks.append(("Query", self.knowledge_base.ask(parse_atom(request["query"]))))
            entailed = role_ok and checks[-1][1]["entailed"]
            allowed = role_ok
        else:
            if role_ok:
                checks.append((f"Permission for {request['category']}",
                               self.knowledge_base.ask(("CanRequest", name, request["category"]))))
                if request["group_id"]:
                    checks.append((f"Request for group {request['group_id']}",
                                   self.knowledge_base.ask(("RepresentsGroup", name, request["group_id"]))))
            allowed = all(result["entailed"] for _, result in checks)
            entailed = allowed
        context["results"]["Logic_KB"] = {
            **config.LOGIC_OUTPUT_TEMPLATE, "allowed": allowed, "entailed": entailed,
            "explanation": explain_checks(checks), "query": checks[-1][1]["query"], "checks": checks}
        if not allowed:
            context["rejected_at"] = "Logic_KB"
            failed = next(result for _, result in checks if not result["entailed"])
            context["stop_reason"] = f"{failed['query']} could not be proved: " + "; ".join(failed["missing"])
        return allowed

    def _run_csp_module(self, context):
        """
        Function: _run_csp_module
        Purpose : Assign a room and slot with the CSP scheduler. The MLP
                  priority (final_priority) is used for the scheduling
                  decision; the Perceptron output is for comparison only.
        """
        request = context["request"]
        final_priority = context["results"].get("ANN", {}).get("final_priority")
        result = self.scheduler.schedule_request(request["name"], request["category"], request["preferred_slot"],
                                                 request["group_id"], final_priority)
        context["results"]["CSP"] = result
        if result["decision"] != "accepted":
            context["rejected_at"] = "CSP"
            context["stop_reason"] = result["notes"]
            return False
        return True

    def _run_search_module(self, context):
        """
        Function: _run_search_module
        Purpose : Operational search: ONE algorithm chosen by the policy
                  (weighted graph + heuristic -> A*). For service requests
                  the destination is the building assigned by the CSP.
        """
        request = context["request"]
        destination = request["destination"] or context["results"]["CSP"]["destination"]
        if request["current_location"] == destination:
            context["results"]["Search"] = {**config.SEARCH_OUTPUT_TEMPLATE, "algorithm_used": "None",
                                            "path": [destination], "nodes_expanded": 0,
                                            "expanded_order": [], "found": True,
                                            "policy": "Already at the destination, no search needed"}
            return True
        context["results"]["Search"] = find_route(request["current_location"], destination, self.graph_type,
                                                  self.use_heuristic, self.campus_graph)
        return True


def explain_checks(checks):
    """
    Function: explain_checks
    Purpose : One readable explanation of the Logic / KB checks, e.g.
              "UsesLab(DrKhan, Lab1) is entailed: Teaches(DrKhan, AI)
              [fact] -> Instructor(DrKhan, AI) [R1] -> ..."
    """
    sentences = []
    for label, result in checks:
        if result["entailed"]:
            chain = []
            for step in result["proof"]:
                if step.endswith(" is a known fact"):
                    chain.append(f"{step.replace(' is a known fact', '')} [fact]")
                elif " by " in step:
                    conclusion, _, rest = step.partition(" by ")
                    chain.append(f"{conclusion} [{rest.split(' ', 1)[0]}]")
            sentences.append(f"{label}: {result['query']} is entailed"
                             + (f" ({' -> '.join(chain)})" if chain else "") + ".")
        else:
            sentences.append(f"{label}: {result['query']} is NOT entailed ({'; '.join(result['missing'])}).")
    return " ".join(sentences)

