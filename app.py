"""
app.py
-----------------------------------------------------------------------
Smart Campus AI Decision Support and Automation System
Streamlit web interface (a second front end next to the CLI).

Run:  python -m streamlit run app.py      (from the SmartCampusAI folder)

The UI only collects input and displays results. Every request still
goes through SmartCampusSystem.process_request() in router.py, exactly
like the CLI in main.py, so both front ends share the same AI modules.
The look ("Campus Line", forest & beige) lives in ui.py: request types
are transit lines and the AI modules are the stations they stop at.
-----------------------------------------------------------------------
"""

import json
import os
import time

import pandas as pd
import streamlit as st

import config
import preprocessing
import ui
import visualizer
from csp_scheduler import EXAM_CONSTRAINT_EDGES, EXAM_GROUPS, EXAM_PRECEDENCE, solve_exam_schedule
from knowledge_base import KnowledgeBaseError, format_atom, parse_atom
from main import DEMO_REQUESTS
from response_layer import LOG_COLUMNS, log_response, standard_response
from router import PIPELINES, SmartCampusSystem
from search_navigation import (OPERATIONAL_ALGORITHMS, NavigationError, compare_algorithms, find_route,
                               trace_search)

SUCCESS_DECISIONS = ("completed", "answered", "accepted")
DETAIL_KEYS = {"ANN": "ann", "Logic_KB": "logic", "CSP": "csp", "Search": "search"}
# Node layout copied from the CSP diagram in the project document (inches).
CSP_NODE_POSITIONS = {"G1": (0, 2.2), "G4": (3.4, 1.5), "G3": (6.8, 2.2),
                      "G6": (9.8, 1.2), "G2": (1.8, 0), "G5": (7.8, 0)}
EXAMPLE_QUERIES = ["Eligible(Ali, AI)", "Eligible(Sara, AI)", "Eligible(x, AI)", "UsesLab(DrKhan, Lab1)",
                   "CanRequest(Sara, AI_Lab_Support)", "RepresentsGroup(x, G4)"]
OTHER_USER = "Other (type a name)"
NO_VALUE = "(none)"
PAGE_INTROS = {
    "Home": ("Smart Campus AI", "One service desk for the campus. Each request rides one of five lines, "
                                "stopping at the AI modules it needs, and arrives as one explained answer."),
    "Submit request": ("Submit a request", "Pick a line, say who is asking and fill in what that line needs. "
                                           "The result shows the request travelling stop by stop."),
    "Demo requests": ("Demo requests", "Nine ready-made requests from the handout: one per request type plus "
                                       "rejections at Logic / KB, CSP and preprocessing. The same requests the CLI demo runs."),
    "Batch CSV": ("Batch processing", "Run a whole file of requests through the network and download the results."),
    "ANN models": ("Priority network", "A Perceptron baseline (urgent vs not urgent) and the final 7-4-3-4 MLP "
                                       "(urgent / high / normal / low), written from scratch in numpy. They predict "
                                       "priority only, never permission."),
    "Knowledge base": ("Knowledge base", "First-order facts and rules. Ask a goal and the system chains backward "
                                         "to known facts, or browse what forward chaining derives."),
    "Navigation": ("Campus navigation", "Comparison mode runs all nine search algorithms on one trip; operational "
                                        "mode (used by requests) runs only the one the policy picks. Step through every "
                                        "decision of BFS, UCS and A* on a separate graph."),
    "CSP schedule": ("Scheduling", "The exam timetable solved as a constraint problem with AC-3, MRV, LCV and "
                                   "forward checking, plus every room booking so far."),
    "History": ("Session history", "Every request made in this browser session. Open any of them again."),
}


def show(html_text):
    """
    Function: show
    Purpose : Render one design block (HTML/SVG built in ui.py).
              st.html drops inline SVG, so blocks that contain a drawing go
              through st.markdown (one HTML block, no markdown inside).
    """
    render_html(st, html_text)


def render_html(target, html_text):
    """
    Function: render_html
    Purpose : Write a design block into a Streamlit container or placeholder.
    """
    block = ui.wrap(html_text).replace("\n", " ") if "<svg" in html_text else ui.wrap(html_text)
    if "<svg" in html_text:
        target.markdown(block, unsafe_allow_html=True)
    else:
        target.html(block)


def saved_chart(cache_key, make_chart):
    """
    Function: saved_chart
    Purpose : Save a matplotlib chart to outputs/plots once per cache_key
              (charts are slow to draw, Streamlit reruns on every click).
    Returns : path of the PNG file
    """
    charts = st.session_state.setdefault("saved_charts", {})
    path = charts.get(cache_key)
    if path is None or not os.path.exists(path):
        path = make_chart()
        charts[cache_key] = path
    return path


def chart_downloads(paths, key):
    """
    Function: chart_downloads
    Purpose : Download buttons for saved chart images, with a note saying
              where they were written.
    """
    paths = [path for path in paths if path and os.path.exists(path)]
    if not paths:
        return
    columns = st.columns(len(paths) + 1, vertical_alignment="center")
    for index, path in enumerate(paths):
        try:
            with open(path, "rb") as chart_file:
                columns[index].download_button(f"⤓  {os.path.basename(path)}", chart_file.read(),
                                               os.path.basename(path), "image/png", key=f"{key}_{index}",
                                               width="stretch")
        except OSError as error:
            columns[index].caption(f"Could not read {os.path.basename(path)}: {error}")
    columns[-1].caption(f"Also saved in {os.path.relpath(config.PLOT_DIRECTORY)}")


@st.cache_resource(show_spinner="Starting the five lines: campus graph, knowledge base, CSP and ANN...")
def get_system():
    """
    Function: get_system
    Purpose : Build one SmartCampusSystem and keep it for the whole session.
              Streamlit reruns the script on every click; caching keeps
              room bookings, KB facts and the trained ANN between reruns.
    """
    return SmartCampusSystem(verbose=False)


# -------------------------------------------------------------------
# Request processing and session history
# -------------------------------------------------------------------
def process_and_store(system, raw_request):
    """
    Function: process_and_store
    Purpose : Run a request through the router, log it to the CSV, and add
              it to this browser session's history.
    """
    response = system.process_request(raw_request)
    log_response(response)
    st.session_state.setdefault("history", []).append(response)
    return response


def history_label(index, response):
    """
    Function: history_label
    Purpose : Unique, readable label for a history entry (invalid requests
              all have request id '-', so the position number is included).
    """
    return f"#{index + 1}  {response['request_id']}  {response['name']}  {response['decision']}"


def open_history_entry(index):
    """
    Function: open_history_entry
    Purpose : Callback - switch to the History page and open the response
              at this position of the session history.
    """
    st.session_state["page"] = "History"
    st.session_state["history_open"] = index


def selected_row(table_key):
    """
    Function: selected_row
    Purpose : Index of the row the user clicked in a selectable table
              (None if no row is selected).
    """
    selection = st.session_state.get(table_key)
    try:
        rows = selection.selection.rows
    except AttributeError:
        return None
    return rows[0] if rows else None


def on_history_row():
    """
    Function: on_history_row
    Purpose : Callback - a row of the History table was clicked.
    """
    index = selected_row("history_table")
    if index is not None:
        st.session_state["history_open"] = index


def find_in_history(request_id):
    """
    Function: find_in_history
    Purpose : Position of the newest session response with this request id
              (None if it was not made in this browser session).
    """
    history = st.session_state.get("history", [])
    for index in range(len(history) - 1, -1, -1):
        if history[index]["request_id"] == request_id and request_id != "-":
            return index
    return None


def response_summary(response):
    """
    Function: response_summary
    Purpose : One table row for a response (history and batch tables).
    """
    assignment, route = response["assignment"], response["route"]
    return {"Request": response["request_id"], "Name": response["name"],
            "Request type": ui.line_of(response["request_type"])["label"],
            "Decision": ui.decision_label(response), "Rejected at": response.get("rejected_at") or "-",
            "Priority": response["priority"].get("final_priority", "-"),
            "Assignment": f"{assignment['room']} · slot {assignment['slot']}" if assignment else "-",
            "Route": " → ".join(route["path"]).replace("_", " ") if route else "-",
            "Message": response["message"]}


def response_table(responses, key, on_click=None):
    """
    Function: response_table
    Purpose : Selectable table of responses: clicking a row opens it.
    """
    frame = pd.DataFrame([response_summary(response) for response in responses])
    frame.index = range(1, len(frame) + 1)
    st.dataframe(frame, key=key, on_select=on_click or "rerun", selection_mode="single-row",
                 width="stretch", height=min(38 + 35 * len(frame), 420))


def known_users(system):
    """
    Function: known_users
    Purpose : Map every registered user in the KB to their role (lower
              case, as in requests), using the query HasRole(?x, ?r).
    """
    users = {}
    for answer in system.knowledge_base.ask(("HasRole", "?x", "?r"))["answers"]:
        users.setdefault(answer["?x"], answer["?r"].lower())
    return dict(sorted(users.items()))


# -------------------------------------------------------------------
# Pipeline states (one entry per station on the request's line)
# -------------------------------------------------------------------
def pipeline_states(response):
    """
    Function: pipeline_states
    Purpose : Decide the state of every pipeline stage for one response:
              'ok' (ran fine), 'error' (stopped the pipeline) or 'skip'.
    Returns : list of (module_name, state, summary_text)
    """
    if response.get("rejected_at") == "Preprocessing":
        return [("Preprocessing", "error", response["explanation"][0].split(" - ", 1)[-1]),
                ("Router", "skip", "not reached")]
    stages = [("Preprocessing", "ok", "validated and normalized"),
              ("Router", "ok", f"{len(response['selected_pipeline'])} module"
                               f"{'s' if len(response['selected_pipeline']) != 1 else ''} selected")]
    for module in response["selected_pipeline"]:
        if module not in response["modules_run"]:
            stages.append((module, "skip", "not reached"))
            continue
        failed = module == response.get("rejected_at")
        stages.append((module, "error" if failed else "ok", module_summary(response, module)))
    return stages


def module_summary(response, module):
    """
    Function: module_summary
    Purpose : One-line result of a module, used under its station.
    """
    details = response["details"].get(DETAIL_KEYS[module])
    if details is None:
        return response["message"]
    if module == "ANN":
        return f"{details['final_priority']} ({details['confidence']:.0%}) · perceptron {details['binary_priority']}"
    if module == "Logic_KB":
        if response["request_type"] == "Eligibility_Check" and details["allowed"]:
            return "query entailed" if details["entailed"] else "query not entailed"
        return "allowed" if details["allowed"] else f"not allowed: {details['query']}"
    if module == "CSP":
        return (f"{details['assigned_room']} · slot {details['assigned_slot']}" if details["decision"] == "accepted"
                else details["notes"])
    return f"{details['algorithm_used']}, cost {details['cost']}"


# -------------------------------------------------------------------
# Response display
# -------------------------------------------------------------------
def render_response(system, response):
    """
    Function: render_response
    Purpose : Show one final response: the request travelling its line,
              the standard response parts next to the route map, details
              at each stop, the explanation, and the router output and
              final response objects in the handout format.
    """
    line = ui.line_of(response["request_type"])
    details = response["details"]
    show(ui.result_line(response, pipeline_states(response)))

    search = details.get("search")
    if response["route"] and search:
        route_meta = (f"{search['algorithm_used']} · cost {search['cost']} · {search['steps']} steps · "
                      f"{search['nodes_expanded']} nodes expanded")
        route_body = (ui.campus_map(system.campus_graph, search["path"], line=line["var"],
                                    label="Route " + " to ".join(search["path"][::max(len(search["path"]) - 1, 1)]))
                      + f'<p class="small" style="margin-top:8px"><b>Path:</b> '
                        f'{" → ".join(ui.nice(node) for node in search["path"])}</p>')
    else:
        route_meta = ("no route guidance needed" if "Search" not in response["selected_pipeline"]
                      else "route not computed")
        route_body = (ui.campus_map(system.campus_graph, line=line["var"])
                      + f'<p class="small muted" style="margin-top:8px">No route on this request: {ui.esc(route_meta)}.</p>')
    show(f'<div class="row" style="--line:{line["var"]}">'
         + ui.panel(ui.key_results(response) + f'<p class="small" style="margin-top:14px">{ui.esc(response["message"])}</p>',
                    "Final response", css_class="grow")
         + ui.panel(route_body, "Route", route_meta, "grow3") + "</div>")
    if response["route"] and search and len(search["path"]) > 1:
        file_name = f"route_{response['request_id']}.png"
        route_png = saved_chart(("route", response["request_id"], tuple(search["path"])),
                                lambda: visualizer.plot_campus_graph(system.campus_graph, search["path"],
                                                                     "Route: " + " -> ".join(search["path"]), file_name))
        chart_downloads([route_png], f"png_{response['request_id']}_{response['timestamp']}")

    show(f'<div style="--line:{line["var"]}"><div class="panel-head" style="margin:6px 0 0"><h2>At each stop</h2>'
         f'<span class="muted small">Only the modules that ran</span></div></div>')
    ran = [module for module in ("ANN", "Logic_KB", "CSP", "Search") if DETAIL_KEYS[module] in details]
    if not ran:
        show(ui.panel('<p class="muted">No AI module ran for this request, so there are no module details. '
                      'The explanation below lists what was wrong with the input.</p>'))
    else:
        tabs = st.tabs([ui.STATION_NAMES[module] for module in ran])
        for tab, module in zip(tabs, ran):
            with tab:
                show(f'<div style="--line:{line["var"]}">{module_detail(response, module)}</div>')
                if module == "Search" and search and len(search["path"]) > 1 and search["algorithm_used"] in ("BFS", "UCS", "A*"):
                    with st.expander("How the search decided, step by step", expanded=False):
                        render_search_steps(system, search["path"][0], search["path"][-1], search["algorithm_used"],
                                            key=f"result_{response['request_id']}_{response['timestamp']}")

    show(ui.explanation(response["explanation"]))
    with st.expander("Standard objects (handout format): request, router output, final response"):
        object_columns = st.columns(3)
        object_columns[0].caption("Request object (after preprocessing)")
        object_columns[0].json(details.get("request") or details.get("raw_request", {}))
        object_columns[1].caption("Router output")
        object_columns[1].json(details.get("router", {}))
        object_columns[2].caption("Final response")
        object_columns[2].json(standard_response(response))


def module_detail(response, module):
    """
    Function: module_detail
    Purpose : HTML for one module's details tab.
    """
    details = response["details"]
    if module == "ANN":
        ann = details["ann"]
        vector = ", ".join(ui.number(value) for value in ann["features"].values())
        return ui.panel(f'<div class="row"><div class="grow">{ui.probability_bars(ann["probabilities"])}'
                        f'<p class="small" style="margin-top:8px">binary_priority (Perceptron): <b>{ui.esc(ann["binary_priority"])}</b>'
                        f' · final_priority (MLP): <b>{ui.esc(ann["final_priority"])}</b></p>'
                        f'<p class="small muted">The ANN predicts priority only. It never grants permission.</p></div>'
                        f'<div class="grow2"><h3 style="margin-bottom:8px">Feature vector x = [{vector}]</h3>'
                        f'{ui.feature_chips(ann["features"], ann["scaled_features"])}</div></div>')
    if module == "Logic_KB":
        logic = details["logic"]
        return ui.panel(ui.logic_checks(logic["checks"])
                        + f'<p class="small muted" style="margin-top:6px">allowed = {logic["allowed"]} · '
                          f'entailed = {logic["entailed"]}</p>')
    if module == "CSP":
        csp = details["csp"]
        body = ""
        if csp["decision"] == "accepted":
            room = config.ROOMS[csp["assigned_room"]]
            rows = [([ui.nice(csp["assigned_room"]), str(csp["assigned_slot"]), ui.esc(csp["slot_time"]),
                      ui.nice(csp["destination"]), ui.esc(room["capacity"]), ui.esc(csp["group_size"])], "var(--line)")]
            body += ui.table(["Assigned room", "Slot", "Time", "Destination", "Capacity", "Group size"], rows)
        body += f'<p class="{"small" if csp["decision"] == "accepted" else "miss"}" style="margin-top:10px">{ui.esc(csp["notes"])}</p>'
        stats = csp["statistics"]
        body += (f'<p class="small muted" style="margin-top:6px">Rooms considered: '
                 f'{", ".join(ui.nice(room) for room in csp["rooms_considered"])} · domain size {csp["domain_size"]} · '
                 f'{stats["assignments_tried"]} values tried · {stats["backtracks"]} backtracks</p>')
        return ui.panel(body)
    search = details["search"]
    body = f'<p><b>Policy:</b> {ui.esc(search["policy"])}</p>'
    if search.get("expanded_order"):
        body += (f'<h3 style="margin:12px 0 8px">Explored in order ({len(search["expanded_order"])})</h3>'
                 + ui.order_chips(search["expanded_order"]))
    return ui.panel(body)


# -------------------------------------------------------------------
# Pages
# -------------------------------------------------------------------
def load_results_log():
    """
    Function: load_results_log
    Purpose : Read outputs/results_log.csv with pandas (None if missing).
              A log in an older column format is moved to
              results_log_old.csv, the same way log_response does.
    """
    try:
        results_log = pd.read_csv(config.RESULTS_LOG_PATH)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return None
    if list(results_log.columns) != LOG_COLUMNS:
        try:
            os.replace(config.RESULTS_LOG_PATH, config.RESULTS_LOG_PATH.replace(".csv", "_old.csv"))
        except OSError:
            pass
        return None
    return results_log


def run_all_demos(system):
    """
    Function: run_all_demos
    Purpose : Button callback - run the nine demo requests once.
    """
    for demo in DEMO_REQUESTS:
        process_and_store(system, demo["request"])


def go_to(page_name):
    """
    Function: go_to
    Purpose : Button callback - switch page.
    """
    st.session_state["page"] = page_name


def page_home(system):
    """
    Function: page_home
    Purpose : Dashboard - totals from the results log, the five lines,
              status and request type breakdowns, and recent requests.
    """
    results_log = load_results_log()
    has_log = results_log is not None and not results_log.empty
    if not has_log:
        show(ui.empty_state("No requests have run yet",
                            "Counts, breakdowns and the latest requests appear here as soon as something travels "
                            "a line. Run the demo set to see all five lines at once."))
        columns = st.columns([1, 1, 3])
        columns[0].button("Run all 9 demos", type="primary", on_click=run_all_demos, args=(system,))
        columns[1].button("Submit a request", on_click=go_to, args=("Submit request",))
    else:
        decision_counts = results_log["decision"].value_counts()
        rejected_at = results_log["rejected_at"].fillna("").value_counts()
        show(ui.tiles([
            (len(results_log), "Requests processed", "brand"),
            (int(decision_counts.reindex(SUCCESS_DECISIONS).fillna(0).sum()), "Completed, answered or accepted", "ok"),
            (int(rejected_at.get("Logic_KB", 0)), "Rejected by Logic / KB", "bad"),
            (int(rejected_at.get("Preprocessing", 0) + rejected_at.get("CSP", 0) + rejected_at.get("Search", 0)),
             "Rejected at input, CSP or Search", "warn"),
        ]))

    facts = [(len(system.campus_graph.nodes), "campus locations"), (len(system.knowledge_base.rules), "KB rules"),
             (len(config.ROOMS), "rooms"), (len(config.TIME_SLOTS), "time slots")]
    users = known_users(system)
    note = (f"Four 90-minute slots: {', '.join(slot[:5] for slot in config.TIME_SLOTS)}. "
            f"Three roles: {', '.join(config.VALID_ROLES)}. {len(users)} known users.")
    show('<div class="row">' + ui.network(PIPELINES) + ui.facts_panel(facts, note) + "</div>")

    if has_log:
        type_counts = results_log["request_type"].value_counts()
        type_items = [(ui.line_of(request_type)["label"], int(type_counts.get(request_type, 0)),
                       ui.line_of(request_type)["var"]) for request_type in ui.LINES]
        show('<div class="row">'
             + ui.panel(ui.status_stack(decision_counts.to_dict()), "By decision", css_class="grow")
             + ui.panel(ui.hbars(type_items), "By request type", css_class="grow") + "</div>")
        show('<div class="panel-head" style="margin:6px 0 0"><h2>Last 10 requests</h2>'
             '<span class="muted small">From outputs/results_log.csv · tick a row to open it</span></div>')
        last = results_log.tail(10).iloc[::-1].reset_index(drop=True)
        shown = last[["request_id", "name", "request_type", "decision", "final_priority", "route"]].fillna("-").copy()
        shown["request_type"] = shown["request_type"].map(lambda request_type: ui.line_of(request_type)["label"])
        shown["route"] = shown["route"].astype(str).str.replace("_", " ")
        shown.columns = ["Request", "Name", "Request type", "Decision", "Priority", "Route"]
        st.session_state["home_last_ids"] = last["request_id"].tolist()
        st.session_state["home_last_rows"] = last.fillna("-").to_dict("records")
        st.dataframe(shown, key="home_table", on_select=on_home_row, selection_mode="single-row",
                     hide_index=True, width="stretch")
        picked = selected_row("home_table")
        logged_row = None
        if picked is not None and picked < len(st.session_state["home_last_rows"]):
            logged_row = st.session_state["home_last_rows"][picked]
            if find_in_history(logged_row["request_id"]) is not None:
                logged_row = None                    # opened on the History page by on_home_row
        if logged_row:
            facts = "".join(f'<div class="kv" style="border-top-color:var(--rule)"><small>{ui.esc(name.replace("_", " ").capitalize())}</small>'
                            f'<b style="font-size:15px">{ui.esc(value)}</b></div>'
                            for name, value in logged_row.items())
            show(ui.panel(f'<p class="small muted" style="margin-bottom:12px">This request was made in an earlier '
                          'session, so only its log entry is kept (the full trace lives in that session).</p>'
                          f'<div class="keys" style="grid-template-columns:repeat(auto-fill,minmax(180px,1fr))">{facts}</div>',
                          f"Logged request {logged_row.get('request_id', '')}"))


def on_home_row():
    """
    Function: on_home_row
    Purpose : Callback - a row of the Home table was clicked. Requests from
              this session open on the History page; older ones show their
              log entry under the table (see page_home).
    """
    index = selected_row("home_table")
    rows = st.session_state.get("home_last_rows", [])
    if index is None or index >= len(rows):
        return
    history_index = find_in_history(rows[index]["request_id"])
    if history_index is not None:
        open_history_entry(history_index)


def page_submit_request(system):
    """
    Function: page_submit_request
    Purpose : Structured request form (same fields as the CLI form): name,
              role and request type first, then only the fields that
              request type uses (handout 2.2). Request type, name and role
              sit outside the form so the fields update immediately; the
              role is filled in from the knowledge base.
    """
    show('<div class="panel-head" style="margin:0"><h2>Choose a request type</h2>'
         '<span class="muted small">The request type decides which AI modules the request stops at</span></div>')
    request_type = st.radio("Request type", config.REQUEST_TYPES, key="req_type", horizontal=True,
                            format_func=lambda request_type_name: f"{ui.line_of(request_type_name)['label']}\n\n"
                                                                  f"{' → '.join(PIPELINES[request_type_name])}")
    line = ui.line_of(request_type)

    show('<div class="panel-head" style="margin:10px 0 0"><h2>Who is asking</h2>'
         '<span class="muted small">Role is filled in from the knowledge base and can be changed to test the role check</span></div>')
    users = known_users(system)
    user_columns = st.columns(2)
    user_choice = user_columns[0].selectbox("Name", list(users) + [OTHER_USER])
    name = user_columns[0].text_input("New name", placeholder="e.g. Zara") if user_choice == OTHER_USER else user_choice
    default_role = users.get(user_choice, "student")
    role = user_columns[1].segmented_control("Role", config.VALID_ROLES, default=default_role,
                                             key=f"role_for_{user_choice}") or default_role

    required = config.TYPE_REQUIRED_FIELDS[request_type]
    optional = config.TYPE_OPTIONAL_FIELDS[request_type]
    fields = required + optional
    locations = system.campus_graph.nodes
    slot_options = config.SLOT_NUMBERS

    def nice_location(location):
        """
        Function: nice_location
        Purpose : Show Main_Gate as 'Main Gate' in the form.
        """
        return location.replace("_", " ")

    show(f'<div class="panel-head" style="margin:10px 0 0"><h2>What a {ui.esc(request_type)} request needs</h2>'
         f'<span class="muted small">Required: {", ".join(required)}'
         + (f' · optional: {", ".join(optional)}' if optional else "") + "</span></div>")
    with st.form("request_form"):
        raw_request = {"name": name, "role": role, "request_type": request_type}
        if "category" in fields:
            raw_request["category"] = st.radio("Category", config.CATEGORIES, horizontal=True,
                                               format_func=lambda category: category.replace("_", " "))
        columns = st.columns(2)
        if "current_location" in fields:
            location_options = locations if "current_location" in required else [NO_VALUE] + locations
            label = "Current location" if "current_location" in required else "Current location (for route guidance)"
            raw_request["current_location"] = columns[0].selectbox(
                label, location_options, index=location_options.index("Hostel"), format_func=nice_location)
        if "destination" in fields:
            raw_request["destination"] = columns[1].selectbox("Destination", locations,
                                                              index=locations.index("AI_Lab"), format_func=nice_location)
        if "preferred_slot" in fields:
            options = slot_options if "preferred_slot" in required else [NO_VALUE] + slot_options
            raw_request["preferred_slot"] = columns[1].segmented_control(
                "Preferred slot" + ("" if "preferred_slot" in required else " (optional)"), options,
                default=2 if "preferred_slot" in required else NO_VALUE,
                format_func=lambda slot: slot if slot == NO_VALUE else f"{slot} · {config.TIME_SLOTS[slot - 1][:5]}")
        if "group_id" in fields:
            raw_request["group_id"] = columns[0].selectbox("Group (if group-based)", [NO_VALUE] + config.GROUP_IDS)
        if "query" in fields:
            raw_request["query"] = st.text_input("Query", value="Eligible(Ali, AI)",
                                                 help="A first-order logic goal, e.g. Eligible(Ali, AI), "
                                                      "UsesLab(DrKhan, Lab1) or CanRequest(Sara, AI_Lab_Support)")
        if "severity" in fields:
            level_columns = st.columns(3)
            raw_request["severity"] = level_columns[0].slider("Severity", config.LEVEL_MIN, config.LEVEL_MAX, 8)
            raw_request["time_sensitivity"] = level_columns[1].slider("Time sensitivity", config.LEVEL_MIN,
                                                                      config.LEVEL_MAX, 9)
            raw_request["crowd_level"] = level_columns[2].slider("Crowd level", config.LEVEL_MIN, config.LEVEL_MAX, 5)
            raw_request["eligibility_claim"] = st.toggle(
                "Eligibility claim", value=False,
                help="Feature x7 of the ANN. When off, x7 says whether the knowledge base knows the requester.")
        if "description_note" in fields:
            raw_request["description_note"] = st.text_area("Description note (optional)", max_chars=300, height=80,
                                                           placeholder="Need urgent help before practical evaluation.")
        raw_request = {key: value for key, value in raw_request.items() if value not in (NO_VALUE, None, "")}
        footer = st.columns([3, 1])
        footer[0].html(ui.wrap('<div style="display:flex;align-items:center;gap:12px;padding-top:10px">'
                               '<span class="muted small">Stops:</span><span style="width:260px">'
                               + ui.mini_line(ui.line_stops(request_type, PIPELINES), line["var"]) + "</span></div>"))
        if footer[1].form_submit_button("▶  Process request", type="primary", width="stretch"):
            st.session_state["last_submit"] = process_and_store(system, raw_request)
    if "last_submit" in st.session_state:
        render_response(system, st.session_state["last_submit"])


def page_demo_requests(system):
    """
    Function: page_demo_requests
    Purpose : Run one of the built-in demo requests (shared with the CLI).
    """
    titles = [demo["title"] for demo in DEMO_REQUESTS]
    columns = st.columns([2, 1], gap="large")
    with columns[0]:
        chosen_index = st.radio("Demo request", range(len(titles)), key="demo_pick",
                                format_func=lambda demo_index: f"{demo_index + 1:02d}  ·  "
                                                               f"{DEMO_REQUESTS[demo_index]['request'].get('name', '?')}\n\n"
                                                               f"{titles[demo_index].replace('_', ' ')}")
    demo = DEMO_REQUESTS[chosen_index]
    with columns[1]:
        show(ui.panel(f'<div class="col"><h2>{ui.esc(demo["title"])}</h2>'
                      '<p class="muted small">Runs through the same pipeline as a submitted request and lands in History.</p>'
                      f'<pre class="code" style="white-space:pre-wrap;padding:14px;margin:0">'
                      f'{ui.esc(json.dumps(demo["request"], indent=2))}</pre></div>'))
        if st.button("▶  Run demo", type="primary", width="stretch"):
            st.session_state["last_demo"] = process_and_store(system, demo["request"])
    if "last_demo" in st.session_state:
        render_response(system, st.session_state["last_demo"])


def page_batch_csv(system):
    """
    Function: page_batch_csv
    Purpose : Process every row of a CSV (default sample file or an
              uploaded one) and show a summary table.
    """
    show('<div class="panel-head" style="margin:0"><h2>Requests file</h2><span class="muted small">CSV with the '
         'standard request fields as columns (name, role, request_type, category, current_location, ...); '
         'the sample file is used if you upload nothing</span></div>')
    uploaded_file = st.file_uploader("Choose a CSV or drop it here", type="csv")
    if st.button("▶  Process CSV", type="primary"):
        try:
            source = uploaded_file if uploaded_file is not None else config.BATCH_REQUEST_PATH
            requests_frame = pd.read_csv(source, dtype=str, keep_default_na=False)
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as error:
            name = uploaded_file.name if uploaded_file is not None else os.path.basename(config.BATCH_REQUEST_PATH)
            show(f'<div class="alert" role="alert"><b>We could not read {ui.esc(name)}.</b>'
                 f'<p>{ui.esc(error)} Check the file and try again, or remove it to use the sample file.</p></div>')
            return
        summary_rows, batch_responses = [], []
        progress_bar = st.progress(0.0, text="Each row rides its own line...")
        for row_number, (_, row) in enumerate(requests_frame.iterrows(), start=1):
            raw_request = {key: value for key, value in row.to_dict().items() if value != ""}
            response = process_and_store(system, raw_request)
            batch_responses.append(response)
            summary_rows.append(response_summary(response))
            progress_bar.progress(row_number / len(requests_frame),
                                  text=f"Processing {row_number} of {len(requests_frame)}")
        progress_bar.empty()
        st.session_state["batch_summary"] = pd.DataFrame(summary_rows)
        st.session_state["batch_responses"] = batch_responses

    summary = st.session_state.get("batch_summary")
    if summary is None:
        show(ui.empty_state("Nothing processed yet", "Press Process CSV to run the sample file, or upload your own first.",
                            ("var(--l-book)", "var(--l-urg)", "var(--l-nav)")))
        return
    show(ui.panel(ui.status_stack(summary["Decision"].value_counts().to_dict()), f"{len(summary)} requests processed"))
    show('<div class="panel-head" style="margin:6px 0 0"><h2>Results</h2>'
         '<span class="muted small">Tick a row (box on the left) to open its full result below</span></div>')
    responses = st.session_state.get("batch_responses", [])
    response_table(responses, "batch_table")
    st.download_button("Download results CSV", summary.to_csv(index=False), "batch_results.csv", "text/csv")
    picked = selected_row("batch_table")
    if picked is not None and picked < len(responses):
        render_response(system, responses[picked])


def page_ann_models(system):
    """
    Function: page_ann_models
    Purpose : What-if priority predictor with the live network, model
              training/evaluation, and the training dataset.
    """
    predictor = system.priority_predictor
    show('<div class="panel-head" style="margin:0"><h2>What if…</h2>'
         '<span class="muted small">Every change runs the trained network again</span></div>')
    input_column, network_column, output_column = st.columns([1.25, 2.3, 1.05], gap="medium")
    with input_column:
        role = st.segmented_control("x1 Role", config.VALID_ROLES, default="student", key="wi_role") or "student"
        category = st.selectbox("x2 RequestType (category)", config.CATEGORIES,
                                format_func=lambda category_name: category_name.replace("_", " "))
        severity = st.slider("x3 Severity", config.LEVEL_MIN, config.LEVEL_MAX, 8)
        time_sensitivity = st.slider("x4 Time sensitivity", config.LEVEL_MIN, config.LEVEL_MAX, 9)
        crowd_level = st.slider("x5 Crowd level", config.LEVEL_MIN, config.LEVEL_MAX, 5)
        distance = st.slider("x6 Distance (route cost)", 0.0, config.MAX_ROUTE_DISTANCE, 4.0, 0.5)
        eligible = st.toggle("x7 Eligibility", value=True)
    feature_vector = preprocessing.build_feature_vector(
        {"role": role, "category": category, "severity": severity,
         "time_sensitivity": time_sensitivity, "crowd_level": crowd_level}, distance, eligible)
    prediction = predictor.predict(feature_vector)
    with network_column:
        show(ui.nn_diagram(predictor.mlp, list(prediction["scaled_features"].values()), config.ANN_FEATURE_NAMES,
                           config.PRIORITY_CLASSES)
             + '<p class="muted small" style="text-align:center">Inputs → 4 hidden → 3 hidden → 4 priorities. '
               "Edge colour is the weight's sign, thickness its strength.</p>")
    with output_column:
        tone = ui.PRIORITY_CLASS.get(prediction["final_priority"], "")
        show(f'<div class="col"><div><span class="muted small">final_priority (MLP)</span>'
             f'<div class="predword {tone}" aria-live="polite">{ui.esc(prediction["final_priority"])}</div>'
             f'<span class="muted small">confidence {prediction["confidence"]:.2f}</span></div>'
             f'<div class="small">binary_priority (Perceptron): <b>{ui.esc(prediction["binary_priority"])}</b></div>'
             f'<div style="--line:var(--brand)">{ui.probability_bars(prediction["probabilities"])}</div>'
             '<p class="small muted">The ANN predicts priority only. It never decides permission, rooms or routes.</p></div>')
    vector = ", ".join(ui.number(value) for value in prediction["features"].values())
    show(f'<p class="small" style="margin-bottom:8px"><b>x = [{vector}]</b> '
         '<span class="muted">(fixed order Role, RequestType, Severity, TimeSensitivity, CrowdLevel, Distance, '
         'Eligibility; the network scales it to [0, 1])</span></p>'
         + ui.feature_chips(prediction["features"], prediction["scaled_features"]))

    st.write("")
    head_columns = st.columns([4, 1])
    head_columns[0].html(ui.wrap('<div class="panel-head" style="margin:0"><h2>Training</h2></div>'
                                 '<p class="muted" style="max-width:72ch">Two models learn from the same generated '
                                 'requests. The Perceptron (baseline / comparison) answers one question: urgent or '
                                 'not_urgent? The 7-4-3-4 MLP (final operational model) picks urgent, high, normal or '
                                 'low; its priority drives scheduling. 80% of rows train, 20% test.</p>'))
    if head_columns[1].button("Retrain models", type="primary", width="stretch"):
        with st.spinner("Training..."):
            st.session_state["ann_metrics"] = predictor.train(verbose=False)
            trained = st.session_state["ann_metrics"]
            st.session_state["ann_pngs"] = [
                visualizer.plot_confusion_matrix(trained["perceptron_confusion"], config.BINARY_CLASSES,
                                                 "Perceptron confusion matrix", "perceptron_confusion.png"),
                visualizer.plot_confusion_matrix(trained["mlp_confusion"], config.PRIORITY_CLASSES,
                                                 "MLP confusion matrix", "mlp_confusion.png"),
                visualizer.plot_class_distribution(trained["class_distribution"]),
                visualizer.plot_perceptron_errors(predictor.perceptron.errors_per_epoch),
                visualizer.plot_mlp_history(predictor.mlp.history)]
    metrics = st.session_state.get("ann_metrics")
    if metrics:
        show('<div class="row">'
             f'<div class="bigstat grow"><span class="muted small">MLP test accuracy</span><b>{metrics["mlp_accuracy"]:.1%}</b>'
             '<span class="muted small">4 classes</span></div>'
             f'<div class="bigstat grow"><span class="muted small">Perceptron test accuracy</span>'
             f'<b>{metrics["perceptron_accuracy"]:.1%}</b><span class="muted small">urgent or not</span></div></div>')
    else:
        show('<p class="muted small">The network in use was loaded from saved weights. Train again to see test accuracy '
             'and confusion matrices for this run.</p>')
    history = predictor.mlp.history
    charts = []
    if history.get("loss"):
        series = [(history["loss"], "var(--l-urg)", "Loss (scaled)")]
        top = max(history["loss"]) or 1
        series = [([loss / top for loss in history["loss"]], "var(--l-urg)", "Loss (scaled)"),
                  (history["train_accuracy"], "var(--brand)", "Train accuracy")]
        if history.get("test_accuracy"):
            series.append((history["test_accuracy"], "var(--accent)", "Test accuracy"))
        charts.append(ui.panel(ui.line_chart(series, "MLP loss falling and accuracy rising", 1.0),
                               "MLP loss and accuracy", css_class="grow"))
    if predictor.perceptron.errors_per_epoch:
        charts.append(ui.panel(ui.line_chart([(predictor.perceptron.errors_per_epoch, "var(--l-elig)", "Mistakes")],
                                             "Perceptron misclassifications per epoch"),
                               "Perceptron errors per epoch", css_class="grow"))
    if charts:
        show('<div class="row">' + "".join(charts) + "</div>")
    if metrics:
        distribution = metrics["class_distribution"]
        colours = {"urgent": "var(--bad)", "high": "var(--accent)", "normal": "var(--l-elig)", "low": "var(--ink-3)"}
        dist_items = [(name, int(distribution.get(name, 0)), colours.get(name, "var(--brand)"))
                      for name in config.PRIORITY_CLASSES]
        show('<div class="row">'
             + ui.panel(ui.confusion(metrics["perceptron_confusion"], ["not urgent", "urgent"]), "Perceptron confusion (2×2)", css_class="grow")
             + ui.panel(ui.confusion(metrics["mlp_confusion"], config.PRIORITY_CLASSES), "MLP confusion (4×4)", css_class="grow2")
             + ui.panel(ui.hbars(dist_items, 70), "Class distribution", css_class="grow") + "</div>")

    if st.session_state.get("ann_pngs"):
        chart_downloads(st.session_state["ann_pngs"], "png_ann")
    if os.path.exists(config.ANN_DATASET_PATH):
        dataset = pd.read_csv(config.ANN_DATASET_PATH)
        with st.expander(f"Dataset · {len(dataset)} rows"):
            st.dataframe(dataset, hide_index=True, height=400)


def set_example_query(query_text):
    """
    Function: set_example_query
    Purpose : Example button callback - put the query in the text box and
              mark it to be answered on this rerun.
    """
    st.session_state["kb_query"] = query_text
    st.session_state["kb_run"] = True


def page_knowledge_base(system):
    """
    Function: page_knowledge_base
    Purpose : Example query chips, backward-chaining queries, adding facts,
              and lists of facts, rules and forward-chained facts.
    """
    knowledge_base = system.knowledge_base
    st.session_state.setdefault("kb_query", "Eligible(Sara, AI)")
    ask_column, browse_column = st.columns([1, 1], gap="large")

    with ask_column:
        show('<div class="panel-head" style="margin:0"><h2>Ask</h2>'
             '<span class="muted small">Backward chaining from the goal to known facts</span></div>')
        with st.container(key="qchips"):
            chip_columns = st.columns(2)
            for index, query_text in enumerate(EXAMPLE_QUERIES):
                chip_columns[index % 2].button(query_text, on_click=set_example_query, args=(query_text,),
                                               key=f"example_{query_text}", width="stretch")
        query_columns = st.columns([4, 1], vertical_alignment="bottom")
        query_columns[0].text_input("Query", key="kb_query",
                                    help="Lower-case single letters like x are variables, e.g. Eligible(x, AI)")
        if query_columns[1].button("Ask", type="primary", width="stretch"):
            st.session_state["kb_run"] = True
        if st.session_state.pop("kb_run", False):
            try:
                result = knowledge_base.ask(parse_atom(st.session_state["kb_query"]))
                ok = result["entailed"]
                answers = [", ".join(f"{variable.lstrip('?')} = {value}" for variable, value in answer.items())
                           for answer in result["answers"] if answer]
                body = (f'<div class="col" aria-live="polite"><div style="display:flex;align-items:baseline;gap:14px;flex-wrap:wrap">'
                        f'<span class="verdict {"ok" if ok else "no"}">{"Yes" if ok else "No"}</span>'
                        f'<span style="font-weight:600">{ui.esc(result["query"])}</span></div>')
                if answers:
                    body += '<div class="chips">' + "".join(
                        f'<span class="ochip" style="animation-delay:{index * 0.05:.2f}s"><i>=</i>{ui.esc(answer_text)}</span>'
                        for index, answer_text in enumerate(answers)) + "</div>"
                if result["proof"]:
                    body += '<h3>Proof</h3>' + ui.proof_steps(result["proof"], [])
                if result["missing"]:
                    body += '<h3>Why not</h3>' + "".join(f'<span class="miss">{ui.esc(reason)}</span>' for reason in result["missing"])
                show(ui.panel(body + "</div>"))
            except KnowledgeBaseError as error:
                show(f'<div class="alert" role="alert"><b>That query could not be read.</b><p>{ui.esc(error)}</p></div>')

        show('<div class="panel-head" style="margin:14px 0 0"><h2>Add a fact</h2></div>')
        with st.form("add_fact_form", clear_on_submit=True):
            fact_columns = st.columns([3, 1], vertical_alignment="bottom")
            fact_text = fact_columns[0].text_input("Fact", placeholder="Completed(Sara, ProgrammingFundamentals)")
            if fact_columns[1].form_submit_button("Add", width="stretch"):
                try:
                    atom = parse_atom(fact_text)
                    knowledge_base.add_fact(atom)
                    st.session_state["kb_msg"] = ("ok", f"Added {format_atom(atom)}")
                except KnowledgeBaseError as error:
                    st.session_state["kb_msg"] = ("no", f"KB error: {error}")
            show('<p class="small muted">Try adding Completed(Sara, ProgrammingFundamentals), '
                 'then ask Eligible(Sara, AI) again.</p>')
        message = st.session_state.pop("kb_msg", None)
        if message:
            show(f'<p class="{"okmsg" if message[0] == "ok" else "alert"}" role="status">{ui.esc(message[1])}</p>')

    with browse_column:
        facts_tab, rules_tab, forward_tab = st.tabs([f"Facts ({len(knowledge_base.facts)})",
                                                     f"Rules ({len(knowledge_base.rules)})", "Forward chaining"])
        with facts_tab:
            show('<div class="scrollbox">' + "".join(
                f'<div class="kbline" style="animation-delay:{min(index, 30) * 0.015:.3f}s"><b>{ui.esc(format_atom(fact))}</b></div>'
                for index, fact in enumerate(sorted(knowledge_base.facts))) + "</div>")
        with rules_tab:
            show('<div class="scrollbox">' + "".join(
                f'<div class="kbline"><span class="rn">{ui.esc(rule.rule_name)}</span>'
                f'<span style="flex:1 1 260px"><b>{ui.esc(str(rule).split(": ", 1)[-1])}</b><br>'
                f'<span class="muted small">{ui.esc(rule.description)}</span></span></div>'
                for rule in knowledge_base.rules) + "</div>")
        with forward_tab:
            derived = knowledge_base.forward_chain()
            show(f'<p class="muted small" style="margin-bottom:8px">{len(derived)} facts derived by firing every rule '
                 'until nothing new appears.</p><div class="scrollbox">' + "".join(
                     f'<div class="kbline" style="animation-delay:{min(index, 30) * 0.015:.3f}s"><span class="rn">{ui.esc(rule_name)}</span>'
                     f'<span style="flex:1 1 260px"><b>{ui.esc(format_atom(fact))}</b><br><span class="muted small">from '
                     f'{ui.esc(", ".join(format_atom(premise) for premise in premises))}</span></span></div>'
                     for index, (fact, (rule_name, premises)) in enumerate(sorted(derived.items()))) + "</div>")


def swap_route():
    """
    Function: swap_route
    Purpose : Button callback - swap From and To on the navigation page.
    """
    st.session_state["nav_from"], st.session_state["nav_to"] = st.session_state["nav_to"], st.session_state["nav_from"]


def group_trace(trace):
    """
    Function: group_trace
    Purpose : Turn the raw search trace into one entry per expansion step.
              A stale priority-queue pop (UCS / A*) is attached to the step
              that follows it.
    Returns : list of (step, previous_step, skipped_stale_entries)
    """
    steps, pending_stale, previous_step = [], [], None
    for entry in trace:
        if entry["kind"] == "stale":
            pending_stale.append(entry)
            continue
        steps.append((entry, previous_step, pending_stale))
        previous_step, pending_stale = entry, []
    return steps


def change_step(step_key, amount, step_count):
    """
    Function: change_step
    Purpose : Previous / Next button callback for the step-by-step view.
    """
    current = st.session_state.get(step_key, 1)
    st.session_state[step_key] = min(max(current + amount, 1), step_count)


def decision_frame(system, steps, step_number, source, destination):
    """
    Function: decision_frame
    Purpose : HTML of one step of the search: decision graph and the
              explanation of the decision side by side.
    """
    step, previous_step, skipped_stale = steps[step_number - 1]
    return ('<div class="col" style="gap:16px">'
            + ui.panel(ui.decision_map(system.campus_graph, step, source, destination), "Decision graph",
                       f"{step['algorithm']} · after step {step_number} of {len(steps)}")
            + ui.panel(ui.decision_panel(step, previous_step, skipped_stale, step_number, len(steps)),
                       "What was decided")
            + "</div>")


def render_search_steps(system, source, destination, algorithm_name, key):
    """
    Function: render_search_steps
    Purpose : Step-by-step view of how one search algorithm decides,
              separate from the final route map: Previous / Next / Play
              controls, the decision graph, and the reasons for each step.
    """
    try:
        _, trace = trace_search(source, destination, algorithm_name, system.campus_graph)
    except NavigationError as error:
        show(f'<div class="alert" role="alert"><b>The search could not be traced.</b><p>{ui.esc(error)}</p></div>')
        return
    steps = group_trace(trace)
    if not steps:
        show('<p class="muted">Start and goal are the same place, so the search makes no decisions.</p>')
        return
    step_key = f"{key}_step"
    step_count = len(steps)
    if not 1 <= st.session_state.get(step_key, 1) <= step_count:
        st.session_state[step_key] = 1
    control_columns = st.columns([1, 5, 1, 1], vertical_alignment="bottom")
    control_columns[0].button("◀  Back", key=f"{key}_back", on_click=change_step, args=(step_key, -1, step_count),
                              width="stretch", disabled=st.session_state.get(step_key, 1) <= 1)
    if step_count > 1:
        control_columns[1].slider("Step", 1, step_count, key=step_key)
    control_columns[2].button("Next  ▶", key=f"{key}_next", on_click=change_step, args=(step_key, 1, step_count),
                              width="stretch", disabled=st.session_state.get(step_key, 1) >= step_count)
    play = control_columns[3].button("▶  Play all", key=f"{key}_play", width="stretch")
    frame = st.empty()
    if play:
        for step_number in range(1, step_count + 1):
            render_html(frame, decision_frame(system, steps, step_number, source, destination))
            time.sleep(1.6)
    else:
        render_html(frame, decision_frame(system, steps, st.session_state.get(step_key, 1), source, destination))


def page_navigation(system):
    """
    Function: page_navigation
    Purpose : Search module in both modes (handout 4.1):
                * operational - the ONE route the policy returns (graph
                  type weighted/unweighted, heuristic on/off),
                * comparison  - all nine algorithms on the same trip,
              then step through the decisions of BFS, UCS or A* on a
              separate decision graph.
    """
    locations = system.campus_graph.nodes
    st.session_state.setdefault("nav_from", "Hostel")
    st.session_state.setdefault("nav_to", "AI_Lab")
    columns = st.columns([2, 2, 1], vertical_alignment="bottom")
    source = columns[0].selectbox("From", locations, key="nav_from", format_func=lambda location: location.replace("_", " "))
    destination = columns[1].selectbox("To", locations, key="nav_to", format_func=lambda location: location.replace("_", " "))
    columns[2].button("⇄  Swap", on_click=swap_route, width="stretch")
    policy_columns = st.columns([2, 2, 1], vertical_alignment="bottom")
    graph_type = policy_columns[0].segmented_control("Graph type (search policy)", config.GRAPH_TYPES,
                                                     default=config.DEFAULT_GRAPH_TYPE, key="nav_graph_type") \
        or config.DEFAULT_GRAPH_TYPE
    use_heuristic = policy_columns[1].toggle("Heuristic available (building coordinates)", value=True,
                                             disabled=graph_type == "unweighted", key="nav_heuristic")
    if source == destination:
        show('<p class="warnbox" role="alert">From and To are the same place. '
             'Pick a different destination to compare routes.</p>')
        return
    try:
        operational = find_route(source, destination, graph_type, use_heuristic, system.campus_graph)
        results = compare_algorithms(source, destination, system.campus_graph)
    except NavigationError as error:
        show(f'<div class="alert" role="alert"><b>No route could be found.</b><p>{ui.esc(error)}</p></div>')
        return

    meta = (f"{operational['algorithm_used']} · cost {operational['cost']} · {operational['steps']} steps · "
            f"{operational['nodes_expanded']} nodes expanded")
    output = {key: operational[key] for key in config.SEARCH_OUTPUT_TEMPLATE}
    side = (f'<div class="col" style="gap:10px"><p><b>Policy:</b> {ui.esc(operational["policy"])}</p>'
            '<h3 class="small muted">Standard search output</h3>'
            f'<pre class="code" style="white-space:pre-wrap;padding:12px;margin:0">{ui.esc(json.dumps(output, indent=2))}</pre>'
            f'<p class="small muted">A* heuristic = {system.campus_graph.heuristic_scale:.3f} × Euclidean distance '
            '(admissible). Unweighted → BFS, weighted with heuristic → A*, weighted without → UCS.</p></div>')
    show('<div class="panel-head" style="margin:6px 0 0"><h2>Operational mode</h2>'
         '<span class="muted small">One algorithm, one route - what requests use</span></div>'
         '<div class="row">'
         + ui.panel(ui.campus_map(system.campus_graph, operational["path"], label="Policy route") + ui.map_legend(),
                    "Route", meta, "grow3")
         + ui.panel(side, "Search policy", css_class="grow") + "</div>")

    show('<div class="panel-head" style="margin:14px 0 0"><h2>Comparison / demo mode</h2>'
         '<span class="muted small">All nine algorithms on the same trip: unweighted BFS, DFS, DLS, IDS, '
         'Bidirectional BFS and weighted UCS, Greedy Best-First, A*, RBFS</span></div>')
    show(ui.algo_cards(results, operational["algorithm_used"]))
    bars = ('<div class="row">'
            + ui.panel(ui.compare_bars(results, "cost"), "Path cost", css_class="grow")
            + ui.panel(ui.compare_bars(results, "steps"), "Steps", css_class="grow")
            + ui.panel(ui.compare_bars(results, "nodes_expanded", True), "Nodes expanded", css_class="grow")
            + "</div>")
    show(bars)
    file_suffix = f"{source}_{destination}"
    route_png = saved_chart(("nav_route", source, destination, operational["algorithm_used"]),
                            lambda: visualizer.plot_campus_graph(system.campus_graph, operational["path"],
                                                                 f"{operational['algorithm_used']} route {source} -> {destination}",
                                                                 f"campus_route_{file_suffix}.png"))
    comparison_png = saved_chart(("nav_compare", source, destination),
                                 lambda: visualizer.plot_algorithm_comparison(results,
                                                                              f"search_comparison_{file_suffix}.png"))
    chart_downloads([route_png, comparison_png], f"png_nav_{file_suffix}")

    show('<div class="panel-head" style="margin:14px 0 0"><h2>How the search decides, step by step</h2>'
         '<span class="muted small">A separate graph from the final route above: each step shows which node is '
         'expanded, why, and what happens to its neighbours (operational algorithms)</span></div>')
    algorithm_name = st.segmented_control("Algorithm", OPERATIONAL_ALGORITHMS,
                                          default=operational["algorithm_used"], key="steps_algo") \
        or operational["algorithm_used"]
    render_search_steps(system, source, destination, algorithm_name,
                        key=f"nav_{algorithm_name}_{source}_{destination}")


def page_csp_schedule(system):
    """
    Function: page_csp_schedule
    Purpose : Solver statistics, the coloured constraint graph, the exam
              timetable and the grid of all room bookings.
    """
    exam_result = solve_exam_schedule()
    statistics = exam_result["statistics"]
    show(ui.stats_row([(statistics["assignments_tried"], "Values tried"), (statistics["backtracks"], "Backtracks"),
                       (statistics["ac3_pruned"], "Pruned by AC-3"),
                       (len(exam_result["violations"]), "Constraint violations")]))
    schedule = exam_result["schedule"]
    slot_index = {slot: index for index, slot in enumerate(config.TIME_SLOTS)}
    exam_rows = [([f'<b>{ui.esc(exam["Group"])}</b>', ui.esc(exam["Students"]),
                   f'<span class="lg slot" style="--sc:var(--s{slot_index[exam["Slot"]]})">{ui.esc(exam["Slot"])}</span>',
                   ui.nice(exam["Room"])], f'var(--s{slot_index[exam["Slot"]]})') for exam in schedule.to_dict("records")]
    show('<div class="row">'
         + ui.panel(ui.constraint_graph(exam_result["solution"], EXAM_GROUPS, EXAM_CONSTRAINT_EDGES, EXAM_PRECEDENCE,
                                        CSP_NODE_POSITIONS), "Constraint graph",
                    "Each exam group is coloured by its time slot", "grow2")
         + ui.panel(ui.table(["Group", "Students", "Slot", "Room"], exam_rows), "Exam timetable", css_class="grow")
         + "</div>")

    booking_table = system.scheduler.booking_table()
    head_columns = st.columns([3, 1], vertical_alignment="bottom")
    head_columns[0].html(ui.wrap('<div class="panel-head" style="margin:0"><h2>Room bookings</h2>'
                                 f'<span class="muted small">{len(booking_table)} of '
                                 f'{len(config.ROOMS) * len(config.TIME_SLOTS)} room slots taken</span></div>'))
    view = head_columns[1].segmented_control("View", ["Grid", "List"], default="Grid", key="booking_view",
                                             label_visibility="collapsed") or "Grid"
    if view == "Grid":
        show(ui.panel(ui.booking_grid(booking_table)))
    else:
        rows = [([ui.esc(booking["Slot"]), ui.nice(booking["Room"]), ui.esc(booking["Booked_By"])], "var(--brand)")
                for booking in booking_table.to_dict("records")]
        show(ui.panel(ui.table(["Slot", "Room", "Booked by"], rows, scroll=True)))
    schedule_png = saved_chart(("schedule", booking_table.to_csv(index=False)),
                               lambda: visualizer.plot_schedule(booking_table))
    chart_downloads([schedule_png], "png_schedule")


def page_history(system):
    """
    Function: page_history
    Purpose : List every response from this browser session and re-open
              any of them.
    """
    history = st.session_state.get("history", [])
    if not history:
        show(ui.empty_state("Nothing in this session yet",
                            "Requests you submit, demos you run and batch rows all land here.", ("var(--rule)",)))
        st.button("Submit a request", type="primary", on_click=go_to, args=("Submit request",))
        return
    show('<div class="panel-head" style="margin:0"><h2>All requests this session</h2>'
         '<span class="muted small">Tick a row (box on the left) to open its result</span></div>')
    response_table(history, "history_table", on_history_row)
    open_index = st.session_state.get("history_open", len(history) - 1)
    if not 0 <= open_index < len(history):
        open_index = len(history) - 1
    render_response(system, history[open_index])


PAGES = {
    "Home": page_home,
    "Submit request": page_submit_request,
    "Demo requests": page_demo_requests,
    "Batch CSV": page_batch_csv,
    "ANN models": page_ann_models,
    "Knowledge base": page_knowledge_base,
    "Navigation": page_navigation,
    "CSP schedule": page_csp_schedule,
    "History": page_history,
}


def render_sidebar_navigation():
    """
    Function: render_sidebar_navigation
    Purpose : Brand and the page selector, drawn as a line of stations.
    """
    st.sidebar.html(ui.brand())
    page_name = st.sidebar.radio("Screens", list(PAGES), key="page")
    return page_name


def render_sidebar_history():
    """
    Function: render_sidebar_history
    Purpose : Last five requests of the session as sidebar buttons that open
              them on the History page. Drawn after the page so a request
              made on this run already appears.
    """
    history = st.session_state.get("history", [])
    if history:
        st.sidebar.html(ui.wrap('<p class="side-h">Recent this session</p>'))
        for index in range(len(history) - 1, max(len(history) - 6, -1), -1):
            label = history_label(index, history[index])
            st.sidebar.button(label, key=f"sidebar_{index}", on_click=open_history_entry, args=(index,),
                              width="stretch")


def main():
    """
    Function: main
    Purpose : Page setup, sidebar navigation and dispatch to the chosen
              page. Unexpected errors are shown instead of crashing.
    """
    st.set_page_config(page_title="Smart Campus AI", page_icon="🚉", layout="wide")
    st.html(ui.page_css())
    visualizer.set_dark_mode(False)
    config.ensure_directories()
    page_name = render_sidebar_navigation()
    title, intro = PAGE_INTROS[page_name]
    show(ui.page_header(title, intro))
    try:
        PAGES[page_name](get_system())
    except Exception as error:              # last-resort guard: report, never crash
        show(f'<div class="alert" role="alert"><b>The AI modules stopped responding.</b>'
             f'<p>{ui.esc(type(error).__name__)}: {ui.esc(error)}. Your inputs are kept on this page. '
             'Try again; if it keeps failing, restart the app server.</p></div>')
    render_sidebar_history()


main()
