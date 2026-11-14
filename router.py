"""
router.py
-----------------------------------------------------------------------
Module 3.2 - Request Router (control-flow brain of the platform)
Reads the request_type and decides WHICH modules run and in WHAT order.
The router does not solve the request itself; it directs the request
through the correct pipeline and collects every module's output:

  Navigation_Only         -> Search
  Eligibility_Check       -> Logic
  Booking_or_Scheduling   -> Logic -> CSP
  Urgent_Service_Request  -> ANN -> Logic -> Search
  Full_Service_Request    -> ANN -> Logic -> CSP -> Search

Logic / KB is a gatekeeper: if it rejects, CSP and Search are skipped.
-----------------------------------------------------------------------
"""

import config
import preprocessing
import response_layer
from ann_module import ModelError, PriorityPredictor
from csp_scheduler import CampusScheduler, SchedulingError
from knowledge_base import build_campus_knowledge_base, resource_query_for_target
from search_navigation import NavigationError, find_route, get_campus_graph

PIPELINES = {
    "Navigation_Only": ["Search"],
    "Eligibility_Check": ["Logic"],
    "Booking_or_Scheduling": ["Logic", "CSP"],
    "Urgent_Service_Request": ["ANN", "Logic", "Search"],
    "Full_Service_Request": ["ANN", "Logic", "CSP", "Search"],
}


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
        self.module_handlers = {
            "ANN": self._run_ann_module,
            "Logic": self._run_logic_module,
            "CSP": self._run_csp_module,
            "Search": self._run_search_module,
        }

    def eligibility_targets(self):
        """
        Function: eligibility_targets
        Purpose : List every valid Eligibility_Check target: courses known
                  to the KB, the AI lab, and bookable resource types.
        """
        courses = sorted(fact[1] for fact in self.knowledge_base.facts if fact[0] == "Course")
        return courses + ["Lab1", "AI_Lab"] + config.RESOURCE_TYPES

    # ---------------------------------------------------------------
    # Main entry point
    # ---------------------------------------------------------------
    def process_request(self, raw_request):
        """
        Function: process_request
        Purpose : Full request lifecycle:
                  preprocessing -> routing -> module execution -> response.
                  Never raises: every failure becomes a response with
                  status INVALID or ERROR.
        Returns : final response dict (see response_layer.build_response)
        """
        try:
            request = preprocessing.validate_request(raw_request, self.campus_graph.nodes,
                                                     self.eligibility_targets())
        except preprocessing.RequestValidationError as error:
            return response_layer.build_invalid_response(raw_request, str(error))

        pipeline = PIPELINES[request["request_type"]]
        context = {"request": request, "pipeline": pipeline, "modules_run": [],
                   "results": {}, "status": None, "stop_reason": None}
        for module_name in pipeline:
            try:
                keep_going = self.module_handlers[module_name](context)
            except (NavigationError, SchedulingError, ModelError) as error:
                context["status"] = "ERROR"
                context["stop_reason"] = f"{module_name} module error: {error}"
                keep_going = False
            context["modules_run"].append(module_name)
            if not keep_going:
                break
        if context["status"] is None:
            context["status"] = self._final_status(request["request_type"])
        return response_layer.build_response(context)

    @staticmethod
    def _final_status(request_type):
        """
        Function: _final_status
        Purpose : Status label when every module of the pipeline succeeded.
        """
        return {"Navigation_Only": "COMPLETED", "Eligibility_Check": "APPROVED",
                "Booking_or_Scheduling": "APPROVED", "Urgent_Service_Request": "DISPATCHED",
                "Full_Service_Request": "APPROVED"}[request_type]

    # ---------------------------------------------------------------
    # Module handlers (each returns True to continue the pipeline)
    # ---------------------------------------------------------------
    def _run_ann_module(self, context):
        """
        Function: _run_ann_module
        Purpose : Build the ANN feature vector and predict PRIORITY only.
                  Distance = weighted route cost to the destination (for
                  full-service: to the nearest building with the resource).
        """
        request = context["request"]
        route_distance = 0.0
        if "destination" in request:
            route_distance = find_route(request["source"], request["destination"], "distance")["cost"]
        elif request.get("resource_type"):
            buildings = {info["building"] for info in config.ROOMS.values()
                         if info["type"] == request["resource_type"]}
            route_distance = min(0.0 if building == request["source"]
                                 else find_route(request["source"], building, "distance")["cost"]
                                 for building in buildings)
        eligibility_flag = self.knowledge_base.is_registered(request["user_id"])
        feature_vector = preprocessing.build_feature_vector(request, route_distance, eligibility_flag)
        context["results"]["ANN"] = self.priority_predictor.predict(feature_vector)
        return True

    def _run_logic_module(self, context):
        """
        Function: _run_logic_module
        Purpose : Gatekeeper. First verify the claimed role, then the
                  request-specific permission / eligibility query. Stops
                  the pipeline with an explanation if anything fails.
        """
        request = context["request"]
        user_id = request["user_id"]
        checks = []
        role_result = self.knowledge_base.check_role(user_id, request["role"])
        checks.append(("Role authorization", role_result))

        if role_result["entailed"]:
            request_type = request["request_type"]
            if request_type == "Eligibility_Check":
                query = resource_query_for_target(user_id, request["target"])
                label = "Eligibility / access"
            elif request_type == "Urgent_Service_Request":
                query = ("CanRequest", user_id, "UrgentService")
                label = "Urgent service permission"
            else:
                query = ("CanBook", user_id, request["resource_type"])
                label = "Booking permission"
            checks.append((label, self.knowledge_base.ask(query)))

        allowed = all(result["entailed"] for _, result in checks)
        context["results"]["Logic"] = {"allowed": allowed, "checks": checks}
        if not allowed:
            context["status"] = "REJECTED"
            failed = [result for _, result in checks if not result["entailed"]][0]
            context["stop_reason"] = (f"{failed['query']} could not be proved: "
                                      + "; ".join(failed["missing"]))
        return allowed

    def _run_csp_module(self, context):
        """
        Function: _run_csp_module
        Purpose : Assign slot(s) and room(s) with the CSP scheduler. The
                  ANN priority (if any) only changes value ordering.
        """
        request = context["request"]
        priority_label = context["results"].get("ANN", {}).get("priority")
        result = self.scheduler.schedule_request(
            request["user_id"], request["resource_type"], request["group_size"],
            request.get("preferred_slot"), request.get("sessions", 1), priority_label)
        context["results"]["CSP"] = result
        if not result["success"]:
            context["status"] = "UNSCHEDULABLE"
            context["stop_reason"] = "; ".join(result["reasons"])
            return False
        return True

    def _run_search_module(self, context):
        """
        Function: _run_search_module
        Purpose : Compute the campus route. For full-service requests the
                  destination is the building of the room assigned by CSP;
                  urgent requests always use the weighted fastest route.
        """
        request = context["request"]
        if request["request_type"] == "Full_Service_Request":
            destination = context["results"]["CSP"]["assignments"][0]["building"]
        else:
            destination = request["destination"]
        preference = "distance" if request["request_type"] == "Urgent_Service_Request" \
            else request["route_preference"]
        if request["source"] == destination:
            context["results"]["Search"] = {"algorithm": "None", "path": [destination], "hops": 0,
                                            "cost": 0, "nodes_expanded": 0,
                                            "policy": "Already at the destination"}
            return True
        context["results"]["Search"] = find_route(request["source"], destination, preference,
                                                  self.campus_graph)
        return True
