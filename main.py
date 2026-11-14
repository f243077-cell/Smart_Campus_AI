"""
main.py
-----------------------------------------------------------------------
Smart Campus AI Decision Support and Automation System
Command-line entry point.

Run:  python main.py

Menu:
  1. Submit a new request            5. Knowledge base (facts, rules, queries)
  2. Run demo requests               6. Campus navigation (compare 9 search algorithms)
  3. Process batch CSV of requests   7. CSP exam & room schedule
  4. Train / evaluate ANN models     0. Exit
-----------------------------------------------------------------------
"""

import os

import pandas as pd

import config
import visualizer
from csp_scheduler import solve_exam_schedule
from knowledge_base import KnowledgeBaseError, format_atom, parse_atom
from response_layer import format_response, log_response
from router import SmartCampusSystem
from search_navigation import NavigationError, compare_algorithms, find_route

BANNER_TEXT = ("=" * 72 + "\n   SMART CAMPUS AI DECISION SUPPORT AND AUTOMATION SYSTEM\n" + "=" * 72)
MAIN_MENU_TEXT = ("\n  MAIN MENU\n   1. Submit a new request\n   2. Run demo requests\n"
                  "   3. Process batch CSV of requests\n   4. Train / evaluate ANN models\n"
                  "   5. Knowledge base (facts, rules, queries)\n   6. Campus navigation (compare 9 search algorithms)\n"
                  "   7. CSP exam & room schedule\n   0. Exit")

# One example per request type (inputs from the student handout) plus
# rejections at every stage: Logic_KB, CSP and preprocessing.
DEMO_REQUESTS = [
    {"title": "Navigation_Only - Hostel to AI_Lab (A*)",
     "request": {"name": "Ali", "role": "student", "request_type": "Navigation_Only",
                 "current_location": "Hostel", "destination": "AI_Lab"}},
    {"title": "Eligibility_Check - UsesLab(DrKhan, Lab1) is entailed",
     "request": {"name": "DrKhan", "role": "instructor", "request_type": "Eligibility_Check",
                 "query": "UsesLab(DrKhan, Lab1)"}},
    {"title": "Eligibility_Check - Eligible(Sara, AI) is not entailed (missing prerequisite)",
     "request": {"name": "Sara", "role": "student", "request_type": "Eligibility_Check",
                 "query": "Eligible(Sara, AI)"}},
    {"title": "Booking_or_Scheduling - viva for group G4 (preferred slot clashes with its exam)",
     "request": {"name": "Ahmed", "role": "student", "request_type": "Booking_or_Scheduling",
                 "category": "Viva_Scheduling", "preferred_slot": 2, "group_id": "G4",
                 "current_location": "Library"}},
    {"title": "Booking_or_Scheduling - rejected by Logic/KB (Sara may not use the AI lab)",
     "request": {"name": "Sara", "role": "student", "request_type": "booking",
                 "category": "AI_Lab_Support", "preferred_slot": 3}},
    {"title": "Urgent_Service_Request - emergency help from the Hostel",
     "request": {"name": "Fatima", "role": "student", "request_type": "Urgent_Service_Request",
                 "category": "Emergency_Help", "current_location": "Hostel", "severity": 10,
                 "time_sensitivity": 10, "crowd_level": 7}},
    {"title": "Full_Service_Request - urgent AI lab support (ANN -> Logic -> CSP -> Search)",
     "request": {"name": "Ali", "role": "student", "request_type": "Full_Service_Request",
                 "category": "AI_Lab_Support", "current_location": "Hostel", "preferred_slot": 2,
                 "severity": 8, "time_sensitivity": 9, "crowd_level": 5,
                 "description_note": "Need urgent help before practical evaluation."}},
    {"title": "Booking_or_Scheduling - rejected by CSP (no study room fits group G2)",
     "request": {"name": "Sara", "role": "student", "request_type": "Booking_or_Scheduling",
                 "category": "Access_Request", "preferred_slot": 1, "group_id": "G2"}},
    {"title": "Invalid request - unknown location, severity out of range",
     "request": {"name": "Ali", "role": "student", "request_type": "urgent", "category": "AI_Lab_Support",
                 "current_location": "Moon", "severity": 15, "time_sensitivity": 5, "crowd_level": 3}},
]


# -------------------------------------------------------------------
# Safe input helpers
# -------------------------------------------------------------------
def ask_text(prompt, optional=False):
    """
    Function: ask_text
    Purpose : Read a line of text. Re-asks on empty input unless optional.
    """
    while True:
        value = input(f"  {prompt}{' (optional, Enter to skip)' if optional else ''}: ").strip()
        if value or optional:
            return value
        print("  Value cannot be empty, please try again.")


def ask_choice(prompt, options, optional=False):
    """
    Function: ask_choice
    Purpose : Show numbered options; accept the number or the text itself.
              Text is returned as typed so preprocessing can normalize it.
    """
    print(f"  {prompt}:")
    for number, option in enumerate(options, start=1):
        print(f"     {number:>2}. {option}")
    while True:
        value = ask_text("Choose number or type value", optional)
        if not value:
            return ""
        if value.isdigit():
            index = int(value)
            if 1 <= index <= len(options):
                return options[index - 1]
            print(f"  Please enter a number from 1 to {len(options)}.")
            continue
        return value


def ask_yes_no(prompt):
    """
    Function: ask_yes_no
    Purpose : Ask a yes/no question; returns True for y/yes.
    """
    while True:
        answer = input(f"  {prompt} (y/n): ").strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no", ""):
            return False
        print("  Please answer y or n.")


def offer_to_open(file_path):
    """
    Function: offer_to_open
    Purpose : Tell the user where a plot was saved and offer to open it.
    """
    print(f"  Plot saved: {file_path}")
    if ask_yes_no("Open the image now?"):
        visualizer.open_image(file_path)


# -------------------------------------------------------------------
# Menu actions
# -------------------------------------------------------------------
def collect_request_interactively(system):
    """
    Function: collect_request_interactively
    Purpose : Structured CLI request form (handout 1.2): ask for the
              fields one by one - name, role, request type, then only the
              fields that request type uses. Returns a raw request dict.
    """
    locations = system.campus_graph.nodes
    raw_request = {
        "name": ask_text("Enter Name (e.g. Ali, Sara, Ahmed, Fatima, DrKhan, DrAyesha, Bilal)"),
        "role": ask_choice("Enter Role", config.VALID_ROLES),
        "request_type": ask_choice("Enter Request Type", config.REQUEST_TYPES),
    }
    request_type = config.REQUEST_TYPE_ALIASES.get(raw_request["request_type"].strip().lower(),
                                                   raw_request["request_type"])
    fields = config.TYPE_REQUIRED_FIELDS.get(request_type, []) + config.TYPE_OPTIONAL_FIELDS.get(request_type, [])
    optional = set(config.TYPE_OPTIONAL_FIELDS.get(request_type, []))
    level_range = f"({config.LEVEL_MIN}-{config.LEVEL_MAX})"
    prompts = {
        "category": lambda: ask_choice("Enter Category", config.CATEGORIES, "category" in optional),
        "current_location": lambda: ask_choice("Enter Current Location" + (" (for route guidance)"
                                                                           if "current_location" in optional else ""),
                                               locations, "current_location" in optional),
        "destination": lambda: ask_choice("Enter Destination", locations),
        "preferred_slot": lambda: ask_choice("Enter Preferred Slot (1-4)",
                                             [f"{number}  ({time})" for number, time in
                                              zip(config.SLOT_NUMBERS, config.TIME_SLOTS)],
                                             "preferred_slot" in optional).split(" ")[0],
        "severity": lambda: ask_text(f"Enter Severity {level_range}"),
        "time_sensitivity": lambda: ask_text(f"Enter Time Sensitivity {level_range}"),
        "crowd_level": lambda: ask_text(f"Enter Crowd Level {level_range}"),
        "group_id": lambda: ask_choice("Enter Group (if group-based)", config.GROUP_IDS, optional=True),
        "query": lambda: ask_text("Enter Query, e.g. Eligible(Ali, AI) or UsesLab(DrKhan, Lab1)"),
        "description_note": lambda: ask_text("Enter Description Note", optional=True),
    }
    for field in fields:
        raw_request[field] = prompts[field]()
    return raw_request


def handle_new_request(system):
    """
    Function: handle_new_request
    Purpose : Menu 1 - collect a request, process it, show the response,
              and optionally draw the route on the campus map.
    """
    while True:
        print("\n  --- New request ---")
        response = system.process_request(collect_request_interactively(system))
        print(format_response(response))
        log_response(response)
        if response["rejected_at"] != "Preprocessing":
            break
        if not ask_yes_no("Request was invalid. Try again?"):
            return
    if response["route"] and len(response["route"]["path"]) > 1 and ask_yes_no("Draw this route on the campus map?"):
        path = response["route"]["path"]
        offer_to_open(visualizer.plot_campus_graph(system.campus_graph, path,
                                                   f"Route: {' -> '.join(path)}",
                                                   f"route_{response['request_id']}.png"))


def run_demo(system, pause_between=True):
    """
    Function: run_demo
    Purpose : Menu 2 - process the built-in demo requests (one per request
              type plus rejection and invalid-input cases).
    Returns : list of (title, response) tuples
    """
    outputs = []
    for number, demo in enumerate(DEMO_REQUESTS, start=1):
        print(f"\n>>> Demo {number}/{len(DEMO_REQUESTS)}: {demo['title']}")
        response = system.process_request(demo["request"])
        print(format_response(response))
        log_response(response)
        outputs.append((demo["title"], response))
        if pause_between and number < len(DEMO_REQUESTS):
            input("  Press Enter for the next demo...")
    return outputs


def process_batch_csv(system, csv_path=config.BATCH_REQUEST_PATH):
    """
    Function: process_batch_csv
    Purpose : Menu 3 - read requests from a CSV with pandas, process every
              row and print a summary table of decisions.
    """
    try:
        requests_frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        print(f"  Could not read '{csv_path}': {error}")
        return None
    summary_rows = []
    for _, row in requests_frame.iterrows():
        raw_request = {key: value for key, value in row.to_dict().items() if value != ""}
        response = system.process_request(raw_request)
        log_response(response)
        summary_rows.append(batch_summary_row(response))
    summary = pd.DataFrame(summary_rows)
    with pd.option_context("display.max_colwidth", 45, "display.width", 200):
        print(summary.to_string(index=False))
    print("\n  Decision counts:")
    print(summary["decision"].value_counts().to_string())
    print(f"\n  Every response appended to {config.RESULTS_LOG_PATH}")
    return summary


def batch_summary_row(response):
    """
    Function: batch_summary_row
    Purpose : One summary row of a processed request for batch tables.
    """
    assignment = response["assignment"]
    return {"request_id": response["request_id"], "name": response["name"],
            "type": response["request_type"], "decision": response["decision"],
            "rejected_at": response.get("rejected_at") or "-",
            "priority": response["priority"].get("final_priority", "-"),
            "assignment": f"{assignment['room']} slot {assignment['slot']}" if assignment else "-",
            "route": " -> ".join(response["route"].get("path", [])) or "-",
            "message": response["message"]}


def train_models_menu(system, show_prompts=True):
    """
    Function: train_models_menu
    Purpose : Menu 4 - retrain both ANN models, print metrics and save
              training/confusion/distribution plots.
    Returns : (metrics, list_of_plot_paths)
    """
    print("\n  Training ANN models (this takes a few seconds)...")
    metrics = system.priority_predictor.train(verbose=True)
    print(f"\n  Train/test split : {metrics['train_size']} / {metrics['test_size']}")
    print(f"  Perceptron (baseline: urgent vs not_urgent) test accuracy : {metrics['perceptron_accuracy']:.2%}")
    print(f"  MLP 7-4-3-4 (final: urgent/high/normal/low) test accuracy : {metrics['mlp_accuracy']:.2%}")
    print("\n  MLP confusion matrix (rows = actual, cols = predicted):")
    print(pd.DataFrame(metrics["mlp_confusion"], index=config.PRIORITY_CLASSES,
                       columns=config.PRIORITY_CLASSES).to_string())
    plot_paths = [
        visualizer.plot_perceptron_errors(system.priority_predictor.perceptron.errors_per_epoch),
        visualizer.plot_mlp_history(system.priority_predictor.mlp.history),
        visualizer.plot_confusion_matrix(metrics["perceptron_confusion"], config.BINARY_CLASSES,
                                         "Perceptron confusion matrix", "perceptron_confusion.png"),
        visualizer.plot_confusion_matrix(metrics["mlp_confusion"], config.PRIORITY_CLASSES,
                                         "MLP confusion matrix", "mlp_confusion.png"),
        visualizer.plot_class_distribution(metrics["class_distribution"]),
    ]
    print("\n  Plots saved:")
    for path in plot_paths:
        print(f"   - {path}")
    if show_prompts and ask_yes_no("Open the MLP training plot?"):
        visualizer.open_image(plot_paths[1])
    return metrics, plot_paths


def knowledge_base_menu(system):
    """
    Function: knowledge_base_menu
    Purpose : Menu 5 - inspect facts and rules, run queries (backward
              chaining), forward chain, and register new facts.
    """
    knowledge_base = system.knowledge_base
    while True:
        print("\n  --- Knowledge base ---\n   1. Show facts\n   2. Show rules\n"
              "   3. Query (backward chaining), e.g. Eligible(Ali, AI), UsesLab(DrKhan, Lab1) or Eligible(x, AI)\n"
              "   4. Forward chaining - list derived facts\n   5. Add a fact, e.g. Completed(Sara, ProgrammingFundamentals)\n"
              "   0. Back")
        choice = input("  Choice: ").strip()
        try:
            if choice == "1":
                for fact in sorted(knowledge_base.facts):
                    print(f"   {format_atom(fact)}")
            elif choice == "2":
                for rule in knowledge_base.rules:
                    print(f"   {rule}    # {rule.description}")
            elif choice == "3":
                result = knowledge_base.ask(parse_atom(ask_text("Query")))
                print(f"   {result['query']} entailed: {result['entailed']}")
                for answer in result["answers"]:
                    if answer:
                        print(f"     answer: {answer}")
                for step in result["proof"]:
                    print(f"     proof: {step}")
                for reason in result["missing"]:
                    print(f"     missing: {reason}")
            elif choice == "4":
                derived = knowledge_base.forward_chain()
                print(f"   {len(derived)} new facts derived:")
                for fact, (rule_name, premises) in sorted(derived.items()):
                    print(f"   {format_atom(fact):42s} by {rule_name} from "
                          f"{', '.join(format_atom(p) for p in premises)}")
            elif choice == "5":
                atom = parse_atom(ask_text("Fact"))
                knowledge_base.add_fact(atom)
                print(f"   Added {format_atom(atom)}")
            elif choice == "0":
                return
            else:
                print("  Please choose 0-5.")
        except KnowledgeBaseError as error:
            print(f"  KB error: {error}")


def navigation_menu(system):
    """
    Function: navigation_menu
    Purpose : Menu 6 - comparison / demo mode: run all nine search
              algorithms on a chosen route and draw the campus map with
              the operational (policy) route highlighted.
    """
    locations = system.campus_graph.nodes
    source = ask_choice("From", locations)
    destination = ask_choice("To", locations)
    plot_paths = show_route_comparison(system, source, destination)
    if plot_paths:
        offer_to_open(plot_paths[1])


def show_route_comparison(system, source, destination):
    """
    Function: show_route_comparison
    Purpose : Print the comparison table of BFS, DFS, DLS, IDS,
              Bidirectional BFS, UCS, Greedy Best-First, A* and RBFS for
              one query, name the algorithm the policy uses, and save the
              comparison chart and the highlighted map.
    Returns : [comparison_chart_path, map_path] or [] on invalid input
    """
    try:
        results = compare_algorithms(source, destination, system.campus_graph)
        operational = find_route(source, destination, system.graph_type, system.use_heuristic, system.campus_graph)
    except NavigationError as error:
        print(f"  {error}")
        return []
    print(f"\n  Route {source} -> {destination}")
    print(f"  Operational policy: {operational['policy']}  =>  {' -> '.join(operational['path'])}")
    print(f"  A* heuristic = {system.campus_graph.heuristic_scale:.3f} x Euclidean distance (admissible)")
    print(f"\n  {'Algorithm':20s}{'Graph':>11s}{'Cost':>6s}{'Steps':>7s}{'Expanded':>10s}{'ms':>8s}  Path")
    for result in results:
        path_text = " -> ".join(result["path"]) if result["found"] else "(not found within the depth limit)"
        print(f"  {result['algorithm_used']:20s}{result['graph_type']:>11s}{str(result['cost'] if result['found'] else '-'):>6s}"
              f"{str(result['steps'] if result['found'] else '-'):>7s}{result['nodes_expanded']:>10}"
              f"{result['execution_time_ms']:>8.2f}  {path_text}")
    file_suffix = f"{source}_{destination}"
    return [visualizer.plot_algorithm_comparison(results, f"search_comparison_{file_suffix}.png"),
            visualizer.plot_campus_graph(system.campus_graph, operational["path"],
                                         f"{operational['algorithm_used']} route {source} -> {destination}",
                                         f"campus_route_{file_suffix}.png")]


def csp_menu(system):
    """
    Function: csp_menu
    Purpose : Menu 7 - show the exam timetable and room bookings, then
              offer to open the schedule chart.
    """
    offer_to_open(show_schedules(system))


def show_schedules(system):
    """
    Function: show_schedules
    Purpose : Solve the G1-G6 exam timetable from the CSP diagram, verify
              every constraint, print all current room bookings and save
              the schedule chart.
    Returns : path of the schedule chart
    """
    exam_result = solve_exam_schedule()
    print("\n  Exam timetable (variables G1-G6):")
    print(exam_result["schedule"].to_string(index=False))
    stats = exam_result["statistics"]
    print(f"\n  Values tried: {stats['assignments_tried']}, backtracks: {stats['backtracks']}, "
          f"values pruned by AC-3: {stats['ac3_pruned']}")
    print(f"  Constraint violations: {len(exam_result['violations'])} "
          f"{'(all constraints satisfied)' if not exam_result['violations'] else exam_result['violations']}")
    print("\n  All current bookings:")
    booking_table = system.scheduler.booking_table()
    print(booking_table.to_string(index=False))
    return visualizer.plot_schedule(booking_table)


def main():
    """
    Function: main
    Purpose : Start the system and run the menu loop. Ctrl+C or end of
              input exits cleanly instead of crashing.
    """
    print(BANNER_TEXT)
    print("  Loading modules (campus graph, knowledge base, CSP, ANN)...")
    system = SmartCampusSystem(verbose=True)
    menu_actions = {
        "1": handle_new_request, "2": run_demo, "3": process_batch_csv, "4": train_models_menu,
        "5": knowledge_base_menu, "6": navigation_menu, "7": csp_menu,
    }
    while True:
        print(MAIN_MENU_TEXT)
        try:
            choice = input("  Choice: ").strip()
            if choice == "0":
                print("  Goodbye.")
                break
            action = menu_actions.get(choice)
            if action is None:
                print("  Invalid choice, please enter a number from 0 to 7.")
                continue
            action(system)
        except (KeyboardInterrupt, EOFError):
            print("\n  Exiting Smart Campus AI. Goodbye.")
            break
        except Exception as error:          # last-resort guard: report, never crash
            print(f"  Unexpected error: {type(error).__name__}: {error}")


if __name__ == "__main__":
    os.chdir(config.PROJECT_DIRECTORY)
    main()
