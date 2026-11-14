"""
response_layer.py
-----------------------------------------------------------------------
Module 3.7 - Final Response Layer
Collects the outputs of the modules that actually ran and turns them
into ONE coherent answer: priority summary, eligibility/authorization
result with proof, assignment, route, final status and explanation.
Every response is also appended to a CSV log with pandas.
-----------------------------------------------------------------------
"""

import os
from datetime import datetime

import pandas as pd

import config

LINE_WIDTH = 72


def build_invalid_response(raw_request, error_message):
    """
    Function: build_invalid_response
    Purpose : Response for a request rejected by preprocessing (missing or
              invalid fields). No AI module runs for such a request.
    """
    raw_request = raw_request if isinstance(raw_request, dict) else {}
    return {
        "request_id": "-",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user_id": str(raw_request.get("user_id", "-")),
        "role": str(raw_request.get("role", "-")),
        "request_type": str(raw_request.get("request_type", "-")),
        "pipeline": [],
        "modules_run": ["Preprocessing"],
        "status": "INVALID",
        "priority": None,
        "authorization": None,
        "assignment": None,
        "route": None,
        "explanation": [f"Input validation failed: {error_message}"],
        "details": {},
    }


def build_response(context):
    """
    Function: build_response
    Purpose : Combine module results from the router context into the
              final response dictionary with a step-by-step explanation.
    """
    request, results = context["request"], context["results"]
    explanation = [f"Router selected pipeline: {' -> '.join(context['pipeline'])}"]
    response = {
        "request_id": request["request_id"],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user_id": request["user_id"],
        "role": request["role"],
        "request_type": request["request_type"],
        "pipeline": context["pipeline"],
        "modules_run": context["modules_run"],
        "status": context["status"],
        "priority": None, "authorization": None, "assignment": None, "route": None,
        "explanation": explanation,
        "details": {"request": request},
    }

    if "ANN" in results:
        ann = results["ANN"]
        response["priority"] = ann["priority"]
        response["details"]["ann"] = ann
        explanation.append(f"ANN (MLP) predicted priority {ann['priority']} "
                           f"(confidence {ann['confidence']:.0%}); Perceptron urgent flag = "
                           f"{ann['perceptron_urgent']}. Priority does NOT grant permission.")

    if "Logic" in results:
        logic = results["Logic"]
        response["authorization"] = "ALLOWED" if logic["allowed"] else "DENIED"
        response["details"]["logic"] = logic
        for label, result in logic["checks"]:
            if result["entailed"]:
                explanation.append(f"Logic/KB: {label} OK - {result['query']} is entailed.")
            else:
                explanation.append(f"Logic/KB: {label} FAILED - {result['query']} is not entailed "
                                   f"({'; '.join(result['missing'])}).")

    if "CSP" in results:
        csp = results["CSP"]
        response["details"]["csp"] = csp
        if csp["success"]:
            response["assignment"] = "; ".join(f"{a['slot']} in {a['room']} ({a['building']})"
                                               for a in csp["assignments"])
            explanation.append(f"CSP assigned {response['assignment']} "
                               f"[{csp['statistics']['assignments_tried']} values tried, "
                               f"{csp['statistics']['backtracks']} backtracks].")
            for note in csp["reasons"]:
                explanation.append(f"CSP note: {note}.")
        else:
            explanation.append(f"CSP found no conflict-free assignment: {'; '.join(csp['reasons'])}.")

    if "Search" in results:
        route = results["Search"]
        response["route"] = " -> ".join(route["path"])
        response["details"]["search"] = route
        explanation.append(f"Search: {route['policy']}; cost {route['cost']}, {route['hops']} stops, "
                           f"{route['nodes_expanded']} nodes expanded.")

    skipped = [module for module in context["pipeline"] if module not in context["modules_run"]]
    if context["stop_reason"]:
        explanation.append(f"Pipeline stopped: {context['stop_reason']}")
    if skipped:
        explanation.append(f"Skipped modules: {', '.join(skipped)}")
    return response


def format_response(response):
    """
    Function: format_response
    Purpose : Render the response as a readable text block for the CLI.
    """
    border = "=" * LINE_WIDTH
    lines = [border, f" SMART CAMPUS RESPONSE  |  {response['request_id']}  |  STATUS: {response['status']}",
             border,
             f" User        : {response['user_id']} ({response['role']})",
             f" Request     : {response['request_type']}",
             f" Pipeline    : {' -> '.join(response['pipeline']) or '(none)'}",
             f" Modules run : {', '.join(response['modules_run'])}"]
    details = response["details"]
    if response["priority"]:
        probabilities = ", ".join(f"{k}={v:.2f}" for k, v in details["ann"]["probabilities"].items())
        lines.append(f" Priority    : {response['priority']}  [{probabilities}]")
    if response["authorization"]:
        lines.append(f" Permission  : {response['authorization']}")
        for label, result in details["logic"]["checks"]:
            mark = "+" if result["entailed"] else "x"
            lines.append(f"   [{mark}] {label}: {result['query']}")
            for step in result["proof"]:
                lines.append(f"         proof: {step}")
            for reason in result["missing"]:
                lines.append(f"         missing: {reason}")
    if response["assignment"]:
        lines.append(f" Assignment  : {response['assignment']}")
    if response["route"]:
        search = details["search"]
        lines.append(f" Route       : {response['route']}")
        lines.append(f"   algorithm : {search['algorithm']}  | cost {search['cost']}  | "
                     f"stops {search['hops']}  | expanded {search['nodes_expanded']}")
    lines.append(" Explanation :")
    for number, sentence in enumerate(response["explanation"], start=1):
        lines.append(f"   {number}. {sentence}")
    lines.append(border)
    return "\n".join(lines)


def log_response(response, log_path=config.RESULTS_LOG_PATH):
    """
    Function: log_response
    Purpose : Append a one-row summary of the response to the results CSV
              using pandas (header written only when the file is new).
    """
    row = pd.DataFrame([{
        "timestamp": response["timestamp"],
        "request_id": response["request_id"],
        "user_id": response["user_id"],
        "role": response["role"],
        "request_type": response["request_type"],
        "modules_run": " > ".join(response["modules_run"]),
        "status": response["status"],
        "priority": response["priority"] or "",
        "authorization": response["authorization"] or "",
        "assignment": response["assignment"] or "",
        "route": response["route"] or "",
    }])
    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        row.to_csv(log_path, mode="a", header=not os.path.exists(log_path), index=False)
    except OSError as error:
        print(f"  Warning: could not write results log ({error}).")
