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
from csp_scheduler import EXAM_CONSTRAINT_EDGES, EXAM_GROUPS, EXAM_PRECEDENCE, solve_exam_schedule
from knowledge_base import KnowledgeBaseError, format_atom, parse_atom
from main import DEMO_REQUESTS
from response_layer import log_response
from router import PIPELINES, SmartCampusSystem
from search_navigation import NavigationError, compare_algorithms

POSITIVE_STATUSES = ("COMPLETED", "APPROVED", "DISPATCHED")
STOPPING_STATUSES = ("REJECTED", "UNSCHEDULABLE", "ERROR")
DETAIL_KEYS = {"ANN": "ann", "Logic": "logic", "CSP": "csp", "Search": "search"}
# Node layout copied from the CSP diagram in the project document (inches).
CSP_NODE_POSITIONS = {"G1": (0, 2.2), "G4": (3.4, 1.5), "G3": (6.8, 2.2),
                      "G6": (9.8, 1.2), "G2": (1.8, 0), "G5": (7.8, 0)}
EXAMPLE_QUERIES = ["Eligible(Ali, AI)", "Eligible(Sara, AI)", "Eligible(x, AI)",
                   "UseLab(DrKhan, Lab1)", "CanBook(Fatima, SeminarRoom)", "HasRole(x, Instructor)"]
OTHER_USER = "Other (type a name)"
ROUTE_PREFERENCE_LABELS = {"distance": "Shortest distance (A*)", "stops": "Fewest stops (BFS)",
                           "distance_no_heuristic": "Shortest, no heuristic (UCS)"}
PAGE_INTROS = {
    "Home": ("Smart Campus AI", "One service desk for the campus. Each request rides one of five lines, "
                                "stopping at the AI modules it needs, and arrives as one explained answer."),
    "Submit request": ("Submit a request", "Pick a line, say who is asking and fill in what that line needs. "
                                           "The result shows the request travelling stop by stop."),
    "Demo requests": ("Demo requests", "Nine ready-made requests, one per line plus a rejection and an invalid input. "
                                       "The same requests the CLI demo runs."),
    "Batch CSV": ("Batch processing", "Run a whole file of requests through the network and download the results."),
    "ANN models": ("Priority network", "A perceptron and a 7-4-3-4 MLP, written from scratch in numpy. "
                                       "They predict priority only, never permission."),
    "Knowledge base": ("Knowledge base", "First-order facts and rules. Ask a goal and the system chains backward "
                                         "to known facts, or browse what forward chaining derives."),
    "Navigation": ("Campus navigation", "The same trip searched three ways. Compare BFS, UCS and A*, then step through "
                                        "the order each one explores the campus."),
    "CSP schedule": ("Scheduling", "The exam timetable solved as a constraint problem with AC-3, MRV, LCV and "
                                   "forward checking, plus every room booking so far."),
    "History": ("Session history", "Every request made in this browser session. Open any of them again."),
}


def theme_name():
    """
    Function: theme_name
    Purpose : 'dark' or 'light', from the theme the user picked in
              Streamlit's settings menu (light if unknown).
    """
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except AttributeError:
        return "light"


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
    return f"#{index + 1}  {response['request_id']}  {response['user_id']}  {response['status']}"


def open_history_entry(label):
    """
    Function: open_history_entry
    Purpose : Sidebar button callback - switch to the History page and
              select the clicked response.
    """
    st.session_state["page"] = "History"
    st.session_state["history_select"] = label


def known_users(system):
    """
    Function: known_users
    Purpose : Map every registered user in the KB to their role, using the
              query HasRole(?x, ?r).
    """
    users = {}
    for answer in system.knowledge_base.ask(("HasRole", "?x", "?r"))["answers"]:
        users.setdefault(answer["?x"], answer["?r"])
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
    if response["status"] == "INVALID":
        return [("Preprocessing", "error", response["explanation"][0]), ("Router", "skip", "not reached")]
    stages = [("Preprocessing", "ok", "request validated"),
              ("Router", "ok", " → ".join(response["pipeline"]))]
    last_module = response["modules_run"][-1] if response["modules_run"] else None
    for module in response["pipeline"]:
        if module not in response["modules_run"]:
            stages.append((module, "skip", "not served"))
            continue
        failed = module == last_module and response["status"] in STOPPING_STATUSES
        stages.append((module, "error" if failed else "ok", module_summary(response, module)))
    return stages


def module_summary(response, module):
    """
    Function: module_summary
    Purpose : One-line result of a module, used under its station.
    """
    details = response["details"].get(DETAIL_KEYS[module])
    if details is None:
        return response["explanation"][-1]
    if module == "ANN":
        return f"{details['priority']} ({details['confidence']:.0%})"
    if module == "Logic":
        return "permission allowed" if details["allowed"] else "permission denied"
    if module == "CSP":
        return (response["assignment"] if details["success"]
                else f"no room: {'; '.join(details['reasons'])}")
    return f"{details['algorithm']}, cost {details['cost']}"


# -------------------------------------------------------------------
# Response display
# -------------------------------------------------------------------
def render_response(system, response):
    """
    Function: render_response
    Purpose : Show one final response: the request travelling its line,
              key results next to the route map, details at each stop, and
              the numbered explanation.
    """
    line = ui.line_of(response["request_type"])
    details = response["details"]
    show(ui.result_line(response, pipeline_states(response)))

    search = details.get("search")
    if response["route"] and search:
        route_meta = (f"{search['algorithm']} · cost {search['cost']} · {search['hops']} stops · "
                      f"{search['nodes_expanded']} nodes expanded")
        route_body = (ui.campus_map(system.campus_graph, search["path"], line=line["var"],
                                    label=f"Route {response['route']}")
                      + f'<p class="small" style="margin-top:8px"><b>Path:</b> '
                        f'{" → ".join(ui.nice(n) for n in search["path"])}</p>')
    else:
        route_meta = "this line does not stop at Search" if "Search" not in response["pipeline"] else "no route"
        route_body = (ui.campus_map(system.campus_graph, line=line["var"])
                      + f'<p class="small muted" style="margin-top:8px">No route on this request: {ui.esc(route_meta)}.</p>')
    show(f'<div class="row" style="--line:{line["var"]}">'
         + ui.panel(ui.key_results(response), "Key results", cls="grow")
         + ui.panel(route_body, "Route", route_meta, "grow3") + "</div>")

    show(f'<div style="--line:{line["var"]}"><div class="panel-head" style="margin:6px 0 0"><h2>At each stop</h2>'
         f'<span class="muted small">Only the modules that ran</span></div></div>')
    ran = [m for m in ("ANN", "Logic", "CSP", "Search") if DETAIL_KEYS[m] in details]
    if not ran:
        show(ui.panel('<p class="muted">No AI module ran for this request, so there are no module details. '
                      'The explanation below lists what was wrong with the input.</p>'))
    else:
        tabs = st.tabs([ui.STATION_NAMES[m] for m in ran])
        for tab, module in zip(tabs, ran):
            with tab:
                show(f'<div style="--line:{line["var"]}">{module_detail(response, module)}</div>')

    show(ui.explanation(response["explanation"]))
    if details.get("request"):
        with st.expander("Raw request data (after preprocessing)"):
            st.json(details["request"])


def module_detail(response, module):
    """
    Function: module_detail
    Purpose : HTML for one module's details tab.
    """
    details = response["details"]
    if module == "ANN":
        ann = details["ann"]
        return ui.panel(f'<div class="row"><div class="grow">{ui.probability_bars(ann["probabilities"])}'
                        f'<p class="small" style="margin-top:8px">Perceptron says urgent: '
                        f'<b>{"Yes" if ann["perceptron_urgent"] else "No"}</b></p>'
                        f'<p class="small muted">The ANN predicts priority only. It never grants permission.</p></div>'
                        f'<div class="grow2"><h3 style="margin-bottom:8px">Scaled inputs</h3>{ui.feature_chips(ann["features"])}</div></div>')
    if module == "Logic":
        return ui.panel(ui.logic_checks(details["logic"]["checks"]))
    if module == "CSP":
        csp = details["csp"]
        body = ""
        if csp["assignments"]:
            rows = [([ui.esc(a["session"]), ui.nice(a["room"]), ui.esc(a["capacity"]), ui.esc(a["slot"]),
                      ui.nice(a["building"])], "var(--line)") for a in csp["assignments"]]
            body += ui.table(["Session", "Room", "Capacity", "Slot", "Building"], rows)
        body += "".join(f'<p class="miss" style="margin-top:8px">{ui.esc(r)}</p>' for r in csp["reasons"])
        stats = csp["statistics"]
        body += (f'<p class="small muted" style="margin-top:10px">Domain size {csp["domain_size"]} · '
                 f'{stats["assignments_tried"]} values tried · {stats["backtracks"]} backtracks · '
                 f'{stats["ac3_pruned"]} pruned by AC-3</p>')
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
    """
    try:
        return pd.read_csv(config.RESULTS_LOG_PATH)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return None


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
        status_counts = results_log["status"].value_counts()
        show(ui.tiles([
            (len(results_log), "Requests processed", "brand"),
            (int(status_counts.reindex(POSITIVE_STATUSES).fillna(0).sum()), "Completed or approved", "ok"),
            (int(status_counts.get("REJECTED", 0)), "Rejected by the knowledge base", "bad"),
            (int(status_counts.get("INVALID", 0) + status_counts.get("UNSCHEDULABLE", 0)),
             "Invalid or unschedulable", "warn"),
        ]))

    facts = [(len(system.campus_graph.nodes), "campus locations"), (len(system.knowledge_base.rules), "KB rules"),
             (len(config.ROOMS), "rooms"), (len(config.TIME_SLOTS), "time slots")]
    users = known_users(system)
    note = (f"Four 90-minute slots: {', '.join(s[:5] for s in config.TIME_SLOTS)}. "
            f"Three roles: {', '.join(config.VALID_ROLES)}. {len(users)} known users.")
    show('<div class="row">' + ui.network(PIPELINES) + ui.facts_panel(facts, note) + "</div>")

    if has_log:
        type_counts = results_log["request_type"].value_counts()
        type_items = [(ui.line_of(t)["label"], int(type_counts.get(t, 0)), ui.line_of(t)["var"]) for t in ui.LINES]
        show('<div class="row">'
             + ui.panel(ui.status_stack(status_counts.to_dict()), "By status", cls="grow")
             + ui.panel(ui.hbars(type_items), "By request type", cls="grow") + "</div>")
        last = results_log.tail(10).iloc[::-1]
        rows = []
        for record in last.to_dict("records"):
            colour = ui.line_of(record.get("request_type"))["var"]
            rows.append(([ui.id_cell(record.get("request_id"), colour), ui.esc(record.get("user_id")),
                          ui.esc(ui.line_of(record.get("request_type"))["label"]), ui.status_pill(record.get("status")),
                          ui.priority_pill(record.get("priority") if isinstance(record.get("priority"), str) else None)],
                         colour))
        show(ui.panel(ui.table(["Request", "User", "Line", "Status", "Priority"], rows), "Last 10 requests",
                      "From outputs/results_log.csv"))


def page_submit_request(system):
    """
    Function: page_submit_request
    Purpose : Form for a new request. Request type, user and role sit
              outside the form so the fields update immediately; the role
              is auto-filled from the knowledge base.
    """
    show('<div class="panel-head" style="margin:0"><h2>Choose a line</h2>'
         '<span class="muted small">The line decides which AI modules the request stops at</span></div>')
    request_type = st.radio("Request type", config.REQUEST_TYPES, key="req_type", horizontal=True,
                            format_func=lambda t: f"{ui.line_of(t)['label']}\n\n{' → '.join(PIPELINES[t])}")
    line = ui.line_of(request_type)

    show('<div class="panel-head" style="margin:10px 0 0"><h2>Who is asking</h2>'
         '<span class="muted small">Role is filled in from the knowledge base and can be changed to test the role check</span></div>')
    users = known_users(system)
    user_columns = st.columns(2)
    user_choice = user_columns[0].selectbox("User", list(users) + [OTHER_USER])
    user_id = user_columns[0].text_input("New user name", placeholder="e.g. Zara") if user_choice == OTHER_USER else user_choice
    default_role = users.get(user_choice, "Student")
    role = user_columns[1].segmented_control("Role", config.VALID_ROLES, default=default_role,
                                             key=f"role_for_{user_choice}") or default_role

    locations = system.campus_graph.nodes
    show(f'<div class="panel-head" style="margin:10px 0 0"><h2>What the {ui.esc(line["label"])} line needs</h2></div>')
    with st.form("request_form"):
        raw_request = {"request_type": request_type, "user_id": user_id, "role": role}
        columns = st.columns(2)
        if request_type in ("Navigation_Only", "Urgent_Service_Request", "Full_Service_Request"):
            raw_request["source"] = columns[0].selectbox("Current location", locations, index=locations.index("Main_Gate"),
                                                         format_func=lambda n: n.replace("_", " "))
        if request_type == "Navigation_Only":
            raw_request["destination"] = columns[1].selectbox("Destination", locations, index=locations.index("AI_Lab"),
                                                              format_func=lambda n: n.replace("_", " "))
        if request_type == "Eligibility_Check":
            raw_request["target"] = columns[0].selectbox("Course or resource", system.eligibility_targets())
        if request_type in ("Booking_or_Scheduling", "Full_Service_Request"):
            booking_columns = st.columns(4)
            raw_request["resource_type"] = booking_columns[0].selectbox("Resource type", config.RESOURCE_TYPES)
            raw_request["group_size"] = booking_columns[1].number_input("Group size", 1, 500, 10)
            raw_request["preferred_slot"] = booking_columns[2].selectbox(
                "Preferred time slot", [""] + config.TIME_SLOTS, format_func=lambda s: s or "Any slot")
            raw_request["sessions"] = booking_columns[3].number_input("Sessions", 1, len(config.TIME_SLOTS), 1)
        if request_type in ("Navigation_Only", "Full_Service_Request"):
            raw_request["route_preference"] = st.radio(
                "Route preference", config.ROUTE_PREFERENCES, horizontal=True,
                format_func=lambda p: ROUTE_PREFERENCE_LABELS.get(p, p))
        if request_type == "Urgent_Service_Request":
            raw_request["service_type"] = st.radio("Service needed", list(config.SERVICE_DESTINATIONS), horizontal=True,
                                                   format_func=lambda s: s.replace("_", " "))
        if request_type in ("Urgent_Service_Request", "Full_Service_Request"):
            level_columns = st.columns(3)
            raw_request["severity"] = level_columns[0].slider("Severity", 1, 5, 3)
            raw_request["time_sensitivity"] = level_columns[1].slider("Time sensitivity", 1, 5, 3)
            raw_request["crowd_level"] = level_columns[2].slider("Crowd level", 1, 5, 2)
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
                                format_func=lambda i: f"{i + 1:02d}  ·  {DEMO_REQUESTS[i]['request'].get('user_id', '?')}\n\n"
                                                      f"{titles[i].replace('_', ' ')}")
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
    show('<div class="panel-head" style="margin:0"><h2>Requests file</h2><span class="muted small">CSV with a user_id '
         'and request_type column; the sample file is used if you upload nothing</span></div>')
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
        summary_rows = []
        progress_bar = st.progress(0.0, text="Each row rides its own line...")
        for row_number, (_, row) in enumerate(requests_frame.iterrows(), start=1):
            raw_request = {key: value for key, value in row.to_dict().items() if value != ""}
            response = process_and_store(system, raw_request)
            summary_rows.append({"request_id": response["request_id"], "user": response["user_id"],
                                 "type": response["request_type"], "status": response["status"],
                                 "priority": response["priority"] or "-",
                                 "assignment": response["assignment"] or "-",
                                 "route": response["route"] or "-",
                                 "explanation": response["explanation"][-1]})
            progress_bar.progress(row_number / len(requests_frame),
                                  text=f"Processing {row_number} of {len(requests_frame)}")
        progress_bar.empty()
        st.session_state["batch_summary"] = pd.DataFrame(summary_rows)

    summary = st.session_state.get("batch_summary")
    if summary is None:
        show(ui.empty_state("Nothing processed yet", "Press Process CSV to run the sample file, or upload your own first.",
                            ("var(--l-book)", "var(--l-urg)", "var(--l-nav)")))
        return
    show(ui.panel(ui.status_stack(summary["status"].value_counts().to_dict()), f"{len(summary)} requests processed"))
    rows = []
    for record in summary.to_dict("records"):
        colour = ui.line_of(record["type"])["var"]
        rows.append(([ui.id_cell(record["request_id"], colour), ui.esc(record["user"]),
                      ui.esc(ui.line_of(record["type"])["label"]), ui.status_pill(record["status"]),
                      ui.priority_pill(record["priority"]), ui.esc(record["assignment"]), ui.esc(record["route"]),
                      f'<span class="small">{ui.esc(record["explanation"])}</span>'], colour))
    show(ui.panel(ui.table(["Request", "User", "Line", "Status", "Priority", "Assignment", "Route", "Explanation"],
                           rows, scroll=True), "Results"))
    st.download_button("Download results CSV", summary.to_csv(index=False), "batch_results.csv", "text/csv")


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
        role = st.segmented_control("Role", config.VALID_ROLES, default="Student", key="wi_role") or "Student"
        request_type = st.selectbox("Request type", config.REQUEST_TYPES, index=3,
                                    format_func=lambda t: ui.line_of(t)["label"])
        severity = st.slider("Severity", 1, 5, 4)
        time_sensitivity = st.slider("Time sensitivity", 1, 5, 4)
        crowd_level = st.slider("Crowd level", 1, 5, 2)
        distance = st.slider("Route distance", 0.0, config.MAX_ROUTE_DISTANCE, 5.0, 0.5)
        eligible = st.toggle("Registered / eligible user", value=True)
    feature_vector = preprocessing.build_feature_vector(
        {"role": role, "request_type": request_type, "severity": severity,
         "time_sensitivity": time_sensitivity, "crowd_level": crowd_level}, distance, eligible)
    prediction = predictor.predict(feature_vector)
    with network_column:
        show(ui.nn_diagram(predictor.mlp, list(prediction["features"].values()), config.ANN_FEATURE_NAMES,
                           config.PRIORITY_CLASSES)
             + '<p class="muted small" style="text-align:center">Inputs → 4 hidden → 3 hidden → 4 priorities. '
               "Edge colour is the weight's sign, thickness its strength.</p>")
    with output_column:
        tone = ui.PRIORITY_CLASS.get(prediction["priority"], "")
        show(f'<div class="col"><div><span class="muted small">Predicted priority</span>'
             f'<div class="predword {tone}" aria-live="polite">{ui.esc(prediction["priority"])}</div>'
             f'<span class="muted small">{prediction["confidence"]:.0%} confidence</span></div>'
             f'<div class="small">Perceptron says urgent: <b>{"Yes" if prediction["perceptron_urgent"] else "No"}</b></div>'
             f'<div style="--line:var(--brand)">{ui.probability_bars(prediction["probabilities"])}</div>'
             '<p class="small muted">The ANN predicts priority only. It never decides permission, rooms or routes.</p></div>')
    show(ui.feature_chips(prediction["features"]))

    st.write("")
    head_columns = st.columns([4, 1])
    head_columns[0].html(ui.wrap('<div class="panel-head" style="margin:0"><h2>Training</h2></div>'
                                 '<p class="muted" style="max-width:72ch">Two models learn from the same generated '
                                 'requests. A perceptron answers one question: is this urgent? A 7-4-3-4 multilayer '
                                 'perceptron picks one of four priorities. 80% of rows train, 20% test.</p>'))
    if head_columns[1].button("Retrain models", type="primary", width="stretch"):
        with st.spinner("Training..."):
            st.session_state["ann_metrics"] = predictor.train(verbose=False)
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
        series = [([v / top for v in history["loss"]], "var(--l-urg)", "Loss (scaled)"),
                  (history["train_accuracy"], "var(--brand)", "Train accuracy")]
        if history.get("test_accuracy"):
            series.append((history["test_accuracy"], "var(--accent)", "Test accuracy"))
        charts.append(ui.panel(ui.line_chart(series, "MLP loss falling and accuracy rising", 1.0),
                               "MLP loss and accuracy", cls="grow"))
    if predictor.perceptron.errors_per_epoch:
        charts.append(ui.panel(ui.line_chart([(predictor.perceptron.errors_per_epoch, "var(--l-elig)", "Mistakes")],
                                             "Perceptron misclassifications per epoch"),
                               "Perceptron errors per epoch", cls="grow"))
    if charts:
        show('<div class="row">' + "".join(charts) + "</div>")
    if metrics:
        distribution = metrics["class_distribution"]
        colours = {"Urgent": "var(--bad)", "High": "var(--accent)", "Normal": "var(--l-elig)", "Low": "var(--ink-3)"}
        dist_items = [(name, int(distribution.get(name, 0)), colours.get(name, "var(--brand)"))
                      for name in config.PRIORITY_CLASSES]
        show('<div class="row">'
             + ui.panel(ui.confusion(metrics["perceptron_confusion"], ["No", "Yes"]), "Perceptron confusion (2×2)", cls="grow")
             + ui.panel(ui.confusion(metrics["mlp_confusion"], config.PRIORITY_CLASSES), "MLP confusion (4×4)", cls="grow2")
             + ui.panel(ui.hbars(dist_items, 70), "Class distribution", cls="grow") + "</div>")

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
                answers = [", ".join(f"{k.lstrip('?')} = {v}" for k, v in a.items()) for a in result["answers"] if a]
                body = (f'<div class="col" aria-live="polite"><div style="display:flex;align-items:baseline;gap:14px;flex-wrap:wrap">'
                        f'<span class="verdict {"ok" if ok else "no"}">{"Yes" if ok else "No"}</span>'
                        f'<span style="font-weight:600">{ui.esc(result["query"])}</span></div>')
                if answers:
                    body += '<div class="chips">' + "".join(
                        f'<span class="ochip" style="animation-delay:{i * 0.05:.2f}s"><i>=</i>{ui.esc(a)}</span>'
                        for i, a in enumerate(answers)) + "</div>"
                if result["proof"]:
                    body += '<h3>Proof</h3>' + ui.proof_steps(result["proof"], [])
                if result["missing"]:
                    body += '<h3>Why not</h3>' + "".join(f'<span class="miss">{ui.esc(m)}</span>' for m in result["missing"])
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
                f'<div class="kbline" style="animation-delay:{min(i, 30) * 0.015:.3f}s"><b>{ui.esc(format_atom(f))}</b></div>'
                for i, f in enumerate(sorted(knowledge_base.facts))) + "</div>")
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
                     f'<div class="kbline" style="animation-delay:{min(i, 30) * 0.015:.3f}s"><span class="rn">{ui.esc(rule_name)}</span>'
                     f'<span style="flex:1 1 260px"><b>{ui.esc(format_atom(fact))}</b><br><span class="muted small">from '
                     f'{ui.esc(", ".join(format_atom(p) for p in premises))}</span></span></div>'
                     for i, (fact, (rule_name, premises)) in enumerate(sorted(derived.items()))) + "</div>")


def swap_route():
    """
    Function: swap_route
    Purpose : Button callback - swap From and To on the navigation page.
    """
    st.session_state["nav_from"], st.session_state["nav_to"] = st.session_state["nav_to"], st.session_state["nav_from"]


def walk_caption(result, step):
    """
    Function: walk_caption
    Purpose : One sentence describing the current search step.
    """
    order = result["expanded_order"]
    if step >= len(order):
        return (f"{result['algorithm']} finished after {len(order)} expansions: "
                f"{' → '.join(n.replace('_', ' ') for n in result['path'])} (cost {result['cost']}).")
    return f"Step {step} of {len(order)}: {result['algorithm']} expands {order[step - 1].replace('_', ' ')}."


def walk_frame(system, result, step):
    """
    Function: walk_frame
    Purpose : HTML of one search walkthrough frame.
    """
    order = result["expanded_order"]
    finished = step >= len(order)
    current = None if finished else order[step - 1]
    return (f'<p class="caption" aria-live="polite" style="margin:4px 0 10px">{ui.esc(walk_caption(result, step))}</p>'
            + ui.campus_map(system.campus_graph, result["path"] if finished else None, order[:step], current,
                            line="var(--accent)", label="Search walkthrough")
            + ui.map_legend(walk=True, line="var(--accent)")
            + f'<div style="margin-top:14px">{ui.order_chips(order[:step], current)}</div>')


def page_navigation(system):
    """
    Function: page_navigation
    Purpose : Compare BFS, UCS and A* on a route, and step through the
              order in which each algorithm expands nodes.
    """
    locations = system.campus_graph.nodes
    st.session_state.setdefault("nav_from", "Main_Gate")
    st.session_state.setdefault("nav_to", "AI_Lab")
    columns = st.columns([2, 2, 1], vertical_alignment="bottom")
    source = columns[0].selectbox("From", locations, key="nav_from", format_func=lambda n: n.replace("_", " "))
    destination = columns[1].selectbox("To", locations, key="nav_to", format_func=lambda n: n.replace("_", " "))
    columns[2].button("⇄  Swap", on_click=swap_route, width="stretch")
    if source == destination:
        show('<p class="warnbox" role="alert">From and To are the same place. '
             'Pick a different destination to compare routes.</p>')
        return
    try:
        results = compare_algorithms(source, destination, system.campus_graph)
    except NavigationError as error:
        show(f'<div class="alert" role="alert"><b>No route could be found.</b><p>{ui.esc(error)}</p></div>')
        return
    best = min(results, key=lambda r: (r["cost"], r["nodes_expanded"]))
    show(ui.algo_cards(results, best["algorithm"]))
    meta = f"{best['algorithm']} · cost {best['cost']} · {best['hops']} stops"
    side = ('<div class="col" style="gap:10px"><h3 class="small muted">Path cost</h3>' + ui.compare_bars(results, "cost")
            + '<h3 class="small muted" style="margin-top:6px">Stops</h3>' + ui.compare_bars(results, "hops")
            + '<h3 class="small muted" style="margin-top:6px">Nodes expanded</h3>' + ui.compare_bars(results, "nodes_expanded", True)
            + f'<p class="small muted" style="margin-top:6px">A* heuristic = {system.campus_graph.heuristic_scale:.3f} × '
              'Euclidean distance (admissible).</p></div>')
    show('<div class="row">'
         + ui.panel(ui.campus_map(system.campus_graph, best["path"], label="Best route") + ui.map_legend(),
                    "Best route", meta, "grow3")
         + ui.panel(side, "Side by side", cls="grow") + "</div>")

    show('<div class="panel-head" style="margin:10px 0 0"><h2>Watch the search</h2>'
         '<span class="muted small">Step through the order each algorithm explores the campus</span></div>')
    control_columns = st.columns([2, 3, 1], vertical_alignment="bottom")
    algorithm_name = control_columns[0].segmented_control("Algorithm", [r["algorithm"] for r in results],
                                                          default=results[2]["algorithm"], key="walk_algo") \
        or results[2]["algorithm"]
    result = next(r for r in results if r["algorithm"] == algorithm_name)
    order = result["expanded_order"]
    step = control_columns[1].slider("Step", 1, len(order), len(order)) if len(order) > 1 else 1
    play = control_columns[2].button("▶  Play", width="stretch")
    frame = st.empty()
    if play:
        for frame_step in range(1, len(order) + 1):
            render_html(frame, ui.panel(walk_frame(system, result, frame_step)))
            time.sleep(0.7)
    else:
        render_html(frame, ui.panel(walk_frame(system, result, step)))


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
    slot_index = {s: i for i, s in enumerate(config.TIME_SLOTS)}
    exam_rows = [([f'<b>{ui.esc(r["Group"])}</b>', ui.esc(r["Students"]),
                   f'<span class="lg slot" style="--sc:var(--s{slot_index[r["Slot"]]})">{ui.esc(r["Slot"])}</span>',
                   ui.nice(r["Room"])], f'var(--s{slot_index[r["Slot"]]})') for r in schedule.to_dict("records")]
    show('<div class="row">'
         + ui.panel(ui.constraint_graph(exam_result["solution"], EXAM_GROUPS, EXAM_CONSTRAINT_EDGES, EXAM_PRECEDENCE,
                                        CSP_NODE_POSITIONS), "Constraint graph",
                    "Each exam group is coloured by its time slot", "grow2")
         + ui.panel(ui.table(["Group", "Students", "Slot", "Room"], exam_rows), "Exam timetable", cls="grow")
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
        rows = [([ui.esc(r["Slot"]), ui.nice(r["Room"]), ui.esc(r["Booked_By"])], "var(--brand)")
                for r in booking_table.to_dict("records")]
        show(ui.panel(ui.table(["Slot", "Room", "Booked by"], rows, scroll=True)))


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
    rows = []
    for index, r in enumerate(history):
        colour = ui.line_of(r["request_type"])["var"]
        rows.append(([str(index + 1), ui.id_cell(r["request_id"], colour), ui.esc(r["user_id"]),
                      ui.esc(ui.line_of(r["request_type"])["label"]), ui.status_pill(r["status"]),
                      ui.priority_pill(r["priority"])], colour))
    show(ui.panel(ui.table(["#", "Request", "User", "Line", "Status", "Priority"], rows, scroll=True)))
    labels = [history_label(index, response) for index, response in enumerate(history)]
    if st.session_state.get("history_select") not in labels:
        st.session_state["history_select"] = labels[-1]
    chosen_label = st.selectbox("Open a response", labels, key="history_select")
    render_response(system, history[labels.index(chosen_label)])


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
            st.sidebar.button(label, key=f"sidebar_{index}", on_click=open_history_entry, args=(label,),
                              width="stretch")
    st.sidebar.html(ui.wrap('<p class="side-note" style="margin-top:14px">Light / dark: ⋮ menu (top right) → Settings</p>'))


def main():
    """
    Function: main
    Purpose : Page setup, sidebar navigation and dispatch to the chosen
              page. Unexpected errors are shown instead of crashing.
    """
    st.set_page_config(page_title="Smart Campus AI", page_icon="🚉", layout="wide")
    st.html(ui.page_css(theme_name()))
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
