"""
response_layer.py
-----------------------------------------------------------------------
Module 3.7 - Final Response Layer (Final Response Generator)
Combines the outputs of the modules that actually ran into ONE
user-facing answer in the standard response format (handout 6, 12.7):

  {request_id, decision, priority, eligibility, assignment, route, message}

decision is one of: completed (route only), answered (eligibility
query), accepted (booking / urgent / full service) or rejected (at
preprocessing, Logic_KB, CSP or Search, with the reason). Fields of
modules that did not run stay empty. For explainability the response
also carries the numbered explanation, the selected pipeline and the
detailed module outputs. Every response is appended to a CSV log.
-----------------------------------------------------------------------
"""

import json
import os
from datetime import datetime

import pandas as pd

import config

LINE_WIDTH = 72
STANDARD_KEYS = list(config.FINAL_RESPONSE_TEMPLATE)
LOG_COLUMNS = ["timestamp", "request_id", "name", "role", "request_type", "category", "selected_pipeline",
               "modules_run", "decision", "rejected_at", "final_priority", "allowed", "assigned_room",
               "assigned_slot", "route", "message"]
REJECTION_MESSAGES = {
    "Preprocessing": "Your request could not be processed because the input is invalid.",
    "Logic_KB": "Your request has been rejected because you are not eligible for this service.",
    "CSP": "Your request has been rejected because no conflict-free assignment exists.",
    "Search": "Your request has been rejected because no valid route could be found.",
    "ANN": "Your request has been rejected because the priority could not be predicted.",
}


def _timestamp():
    """
    Function: _timestamp
    Purpose : Current time as text for the response and the log.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def build_invalid_response(raw_request, error_message):
    """
    Function: build_invalid_response
    Purpose : Response for a request rejected during preprocessing
              (missing field, invalid location, slot out of range,
              unsupported request type). No AI module runs.
    """
    raw_request = raw_request if isinstance(raw_request, dict) else {}
    return {
        **config.FINAL_RESPONSE_TEMPLATE,
        "request_id": "-",
        "decision": "rejected",
        "message": f"{REJECTION_MESSAGES['Preprocessing']} {error_message}",
        "rejected_at": "Preprocessing",
        "timestamp": _timestamp(),
        "name": str(raw_request.get("name", "-")),
        "role": str(raw_request.get("role", "-")),
        "request_type": str(raw_request.get("request_type", "-")),
        "category": str(raw_request.get("category", "")),
        "selected_pipeline": [],
        "modules_run": [],
        "explanation": [f"Preprocessing: input validation failed - {error_message}"],
        "details": {"raw_request": raw_request},
    }


def build_response(context):
    """
    Function: build_response
    Purpose : Combine module results from the router context into the
              standard final response and a step-by-step explanation.
              Only the outputs of executed modules are filled in.
    """
    request, results, router = context["request"], context["results"], context["router"]
    explanation = [f"Preprocessing: request validated and normalized ({request['request_id']}).",
                   f"Router selected pipeline: {' -> '.join(router['selected_pipeline'])}."]
    response = {
        **config.FINAL_RESPONSE_TEMPLATE,
        "request_id": request["request_id"],
        "priority": {}, "eligibility": {}, "assignment": {}, "route": {},
        "rejected_at": context["rejected_at"],
        "timestamp": _timestamp(),
        "name": request["name"],
        "role": request["role"],
        "request_type": request["request_type"],
        "category": request["category"],
        "selected_pipeline": router["selected_pipeline"],
        "modules_run": context["modules_run"],
        "explanation": explanation,
        "details": {"request": request, "router": router, "module_inputs": context["needs"]["module_inputs"]},
    }

    if "ANN" in results:
        ann = results["ANN"]
        response["priority"] = {key: ann[key] for key in config.PRIORITY_OUTPUT_TEMPLATE}
        response["details"]["ann"] = ann
        explanation.append(f"ANN: feature vector x = {[_number(v) for v in ann['features'].values()]}; "
                           f"Perceptron (baseline) says {ann['binary_priority']}, MLP final priority "
                           f"{ann['final_priority']} (confidence {ann['confidence']:.2f}). "
                           "Priority does NOT grant permission.")

    if "Logic_KB" in results:
        logic = results["Logic_KB"]
        response["details"]["logic"] = logic
        if request["request_type"] == "Eligibility_Check":
            response["eligibility"] = {"entailed": logic["entailed"], "explanation": logic["explanation"]}
        else:
            response["eligibility"] = {"allowed": logic["allowed"], "explanation": logic["explanation"]}
        explanation.append(f"Logic/KB: {logic['explanation']}")
        if request["request_type"] != "Eligibility_Check":
            explanation.append(f"Eligible / Allowed? {'Yes - continue to CSP.' if logic['allowed'] else 'No - stop.'}")

    if "CSP" in results:
        csp = results["CSP"]
        response["details"]["csp"] = csp
        if csp["decision"] == "accepted":
            response["assignment"] = {"room": csp["assigned_room"], "slot": csp["assigned_slot"],
                                      "time": csp["slot_time"], "building": csp["destination"]}
            explanation.append(f"CSP: assigned {csp['assigned_room']} in slot {csp['assigned_slot']} "
                               f"({csp['slot_time']}). {csp['notes']} "
                               f"[{csp['statistics']['assignments_tried']} values tried, "
                               f"domain size {csp['domain_size']}]")
        else:
            explanation.append(f"CSP: no conflict-free assignment - {csp['notes']}")

    if "Search" in results:
        search = results["Search"]
        response["route"] = {"algorithm": search["algorithm_used"], "path": search["path"],
                             "cost": search["cost"], "steps": search["steps"]}
        response["details"]["search"] = search
        explanation.append(f"Search: {search['policy']}; path {' -> '.join(search['path'])}, cost "
                           f"{search['cost']}, {search['steps']} steps, {search['nodes_expanded']} nodes expanded.")

    if context["rejected_at"]:
        response["decision"] = "rejected"
        response["message"] = f"{REJECTION_MESSAGES[context['rejected_at']]} {context['stop_reason']}"
        skipped = [module for module in router["selected_pipeline"] if module not in context["modules_run"]]
        explanation.append(f"Rejected / Explain: {context['stop_reason']}")
        if skipped:
            explanation.append(f"Skipped modules: {', '.join(skipped)}.")
    else:
        response["decision"], response["message"] = _success_decision(response)
    return response


def _number(value):
    """
    Function: _number
    Purpose : Show 8.0 as 8 and keep real decimals (feature vector text).
    """
    return int(value) if float(value).is_integer() else value


def _success_decision(response):
    """
    Function: _success_decision
    Purpose : Decision and user message when every module succeeded.
    """
    request_type = response["request_type"]
    if request_type == "Navigation_Only":
        return "completed", "Best route generated successfully."
    if request_type == "Eligibility_Check":
        verdict = "is entailed" if response["eligibility"]["entailed"] else "is NOT entailed"
        return "answered", f"Eligibility query answered successfully: the query {verdict}."
    assignment = response["assignment"]
    message = (f"Your request has been accepted. You are assigned {assignment['room']} "
               f"in slot {assignment['slot']} ({assignment['time']}).")
    if response["priority"]:
        message = message.replace("Your request", f"Your {response['priority']['final_priority']}-priority request")
    if response["route"]:
        message += " Please follow the recommended route."
    elif request_type == "Booking_or_Scheduling":
        message = (f"Booking assigned successfully. You are assigned {assignment['room']} "
                   f"in slot {assignment['slot']} ({assignment['time']}).")
    return "accepted", message


def standard_response(response):
    """
    Function: standard_response
    Purpose : Only the standard response fields (handout 12.7), e.g. to
              print or export as JSON.
    """
    return {key: response[key] for key in STANDARD_KEYS}


def format_response(response):
    """
    Function: format_response
    Purpose : Render the response for the CLI: a summary block, the
              standard response object as JSON and the explanation.
    """
    border = "=" * LINE_WIDTH
    lines = [border, f" SMART CAMPUS RESPONSE  |  {response['request_id']}  |  DECISION: "
             f"{response['decision'].upper()}", border,
             f" Requester   : {response['name']} ({response['role']})",
             f" Request     : {response['request_type']}"
             + (f"  |  category {response['category']}" if response.get("category") else ""),
             f" Pipeline    : {' -> '.join(response['selected_pipeline']) or '(none)'}",
             f" Modules run : {', '.join(response['modules_run']) or '(none)'}"]
    router = response["details"].get("router")
    if router:
        lines.append(f" Router out  : {json.dumps(router)}")
    if response.get("rejected_at"):
        lines.append(f" Rejected at : {response['rejected_at']}")
    lines.append(f" Message     : {response['message']}")
    lines.append(" Final response object:")
    lines.extend("   " + line for line in json.dumps(standard_response(response), indent=2).splitlines())
    lines.append(" Explanation :")
    for number, sentence in enumerate(response["explanation"], start=1):
        lines.append(f"   {number}. {sentence}")
    lines.append(border)
    return "\n".join(lines)


def log_row(response):
    """
    Function: log_row
    Purpose : One flat row of the results log for a response.
    """
    return {
        "timestamp": response["timestamp"],
        "request_id": response["request_id"],
        "name": response["name"],
        "role": response["role"],
        "request_type": response["request_type"],
        "category": response.get("category", ""),
        "selected_pipeline": " > ".join(response["selected_pipeline"]),
        "modules_run": " > ".join(response["modules_run"]),
        "decision": response["decision"],
        "rejected_at": response.get("rejected_at") or "",
        "final_priority": response["priority"].get("final_priority", ""),
        "allowed": response["eligibility"].get("allowed", response["eligibility"].get("entailed", "")),
        "assigned_room": response["assignment"].get("room", ""),
        "assigned_slot": response["assignment"].get("slot", ""),
        "route": " -> ".join(response["route"].get("path", [])),
        "message": response["message"],
    }


def log_response(response, log_path=config.RESULTS_LOG_PATH):
    """
    Function: log_response
    Purpose : Append a one-row summary of the response to the results CSV
              using pandas (header written only when the file is new). A
              log from an older version with different columns is renamed
              to results_log_old.csv first.
    """
    row = pd.DataFrame([log_row(response)], columns=LOG_COLUMNS)
    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        if os.path.exists(log_path):
            with open(log_path, encoding="utf-8") as log_file:
                header = log_file.readline().strip().split(",")
            if header != LOG_COLUMNS:
                os.replace(log_path, log_path.replace(".csv", "_old.csv"))
        row.to_csv(log_path, mode="a", header=not os.path.exists(log_path), index=False)
    except OSError as error:
        print(f"  Warning: could not write results log ({error}).")
