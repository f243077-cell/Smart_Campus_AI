"""
app.py
-----------------------------------------------------------------------
Smart Campus AI Decision Support and Automation System
Streamlit web interface (a second front end next to the CLI).

Run:  python -m streamlit run app.py      (from the SmartCampusAI folder)

The UI only collects input and displays results. Every request still
goes through SmartCampusSystem.process_request() in router.py, exactly
like the CLI in main.py, so both front ends share the same AI modules.
-----------------------------------------------------------------------
"""

import os
import time

import pandas as pd
import streamlit as st

import config
import preprocessing
import visualizer
from csp_scheduler import EXAM_CONSTRAINT_EDGES, EXAM_GROUPS, EXAM_PRECEDENCE, solve_exam_schedule
from knowledge_base import KnowledgeBaseError, format_atom, parse_atom
from main import DEMO_REQUESTS
from response_layer import log_response
from router import PIPELINES, SmartCampusSystem
from search_navigation import NavigationError, compare_algorithms

STATUS_STYLE = {
    "COMPLETED": st.success, "APPROVED": st.success, "DISPATCHED": st.success,
    "REJECTED": st.error, "INVALID": st.error, "ERROR": st.error,
    "UNSCHEDULABLE": st.warning,
}
POSITIVE_STATUSES = ("COMPLETED", "APPROVED", "DISPATCHED")
STOPPING_STATUSES = ("REJECTED", "UNSCHEDULABLE", "ERROR")
DETAIL_KEYS = {"ANN": "ann", "Logic": "logic", "CSP": "csp", "Search": "search"}
MODULE_LABELS = {"ANN": "ANN\\nPriority", "Logic": "Logic / KB", "CSP": "CSP\\nScheduler",
                 "Search": "Search &\\nNavigation"}
STEP_ICONS = {"ok": "✅", "error": "❌", "skip": "⏭️"}
SLOT_COLOURS = {"light": ["#9ecae1", "#a1d99b", "#fdae6b", "#bcbddc"],
                "dark": ["#1F4E79", "#1F5F3A", "#7A4A12", "#4B3F7A"]}
# Diagram colours per theme: pipeline stage fill/border/font and CSP edge colours.
DIAGRAM_COLOURS = {
    "light": {"ok": ("#c7e9c0", "#2ca02c", "#1A1A1A"), "error": ("#fcbba1", "#d62728", "#1A1A1A"),
              "skip": ("#eeeeee", "#aaaaaa", "#888888"), "edge": "#555555", "font": "#1A1A1A",
              "not_equal": "#d62728", "examiner": "#7b3294", "supervisor": "#e66101", "precedence": "#1f4e9c"},
    "dark": {"ok": ("#1F4D2C", "#3FB950", "#E6EDF3"), "error": ("#5C1F24", "#F85149", "#E6EDF3"),
             "skip": ("#1A2538", "#4A5A72", "#7D8CA3"), "edge": "#8B9BB4", "font": "#E6EDF3",
             "not_equal": "#FF7B72", "examiner": "#D2A8FF", "supervisor": "#FFA657", "precedence": "#79C0FF"},
}
SLOT_LEGEND_ICONS = ["🟦", "🟩", "🟧", "🟪"]
# Node layout copied from the CSP diagram in the project document (inches).
CSP_NODE_POSITIONS = {"G1": (0, 2.2), "G4": (3.4, 1.5), "G3": (6.8, 2.2),
                      "G6": (9.8, 1.2), "G2": (1.8, 0), "G5": (7.8, 0)}
EXAMPLE_QUERIES = ["Eligible(Ali, AI)", "Eligible(Sara, AI)", "Eligible(x, AI)",
                   "UseLab(DrKhan, Lab1)", "CanBook(Fatima, SeminarRoom)", "HasRole(x, Instructor)"]
OTHER_USER = "Other (type a name)"


def theme_name():
    """
    Function: theme_name
    Purpose : 'dark' or 'light', from the theme the user picked in
              Streamlit's settings menu (light if unknown).
    """
    return "dark" if st.context.theme.type == "dark" else "light"


def custom_css(theme):
    """
    Function: custom_css
    Purpose : Card styling for metric tiles in the active theme.
    """
    background, accent = ("#1A2538", "#2F81F7") if theme == "dark" else ("#EEF3F8", "#1F4E79")
    return (f"<style>[data-testid='stMetric'] {{background: {background}; border-radius: 10px; "
            f"padding: 10px 14px; border-left: 4px solid {accent};}}</style>")


@st.cache_resource(show_spinner="Loading campus graph, knowledge base, CSP and ANN...")
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
# Pipeline visualisation
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
    stages = [("Preprocessing", "ok", "request validated and normalized"),
              ("Router", "ok", f"pipeline {' -> '.join(response['pipeline'])}")]
    last_module = response["modules_run"][-1] if response["modules_run"] else None
    for module in response["pipeline"]:
        if module not in response["modules_run"]:
            stages.append((module, "skip", "skipped"))
            continue
        failed = module == last_module and response["status"] in STOPPING_STATUSES
        stages.append((module, "error" if failed else "ok", module_summary(response, module)))
    return stages


def module_summary(response, module):
    """
    Function: module_summary
    Purpose : One-line result of a module, used in the pipeline trace.
    """
    details = response["details"].get(DETAIL_KEYS[module])
    if details is None:
        return response["explanation"][-1]
    if module == "ANN":
        return f"priority {details['priority']} ({details['confidence']:.0%} confidence)"
    if module == "Logic":
        return f"permission {'ALLOWED' if details['allowed'] else 'DENIED'}"
    if module == "CSP":
        return (f"assigned {response['assignment']}" if details["success"]
                else f"no assignment: {'; '.join(details['reasons'])}")
    return f"{details['algorithm']} route {response['route']} (cost {details['cost']})"


def pipeline_dot(response):
    """
    Function: pipeline_dot
    Purpose : Graphviz DOT text of the request pipeline. Green = module ran,
              red = module stopped the request, grey = skipped.
    """
    colours = DIAGRAM_COLOURS[theme_name()]
    labels = {"Preprocessing": "Input &\\nPreprocessing", "Router": "Request\\nRouter", **MODULE_LABELS}
    lines = ['digraph pipeline {', 'rankdir=LR; bgcolor="transparent";',
             'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11];',
             f'edge [color="{colours["edge"]}"];']
    stages = pipeline_states(response) + [("Response", "ok", "")]
    labels["Response"] = "Final\\nResponse"
    for module, state, _ in stages:
        fill, border, font = colours[state]
        lines.append(f'"{module}" [label="{labels[module]}", fillcolor="{fill}", color="{border}", '
                     f'fontcolor="{font}", penwidth=2];')
    for (source, _, _), (target, state, _) in zip(stages[:-1], stages[1:]):
        lines.append(f'"{source}" -> "{target}" [style={"dashed" if state == "skip" else "solid"}];')
    lines.append("}")
    return "\n".join(lines)


def render_pipeline_trace(response):
    """
    Function: render_pipeline_trace
    Purpose : Step-by-step trace of the request lifecycle in a status box.
    """
    stages = pipeline_states(response)
    failed = any(state == "error" for _, state, _ in stages)
    with st.status(f"Pipeline trace - {response['status']}", state="error" if failed else "complete",
                   expanded=True):
        for module, state, summary in stages:
            st.markdown(f"{STEP_ICONS[state]} **{module}**: {summary}")
        st.markdown("✅ **Final Response**: answer combined and logged")


def render_probabilities(probabilities):
    """
    Function: render_probabilities
    Purpose : Show class probabilities as ordered progress bars.
    """
    for class_name in config.PRIORITY_CLASSES:
        value = float(probabilities[class_name])
        st.progress(min(max(value, 0.0), 1.0), text=f"{class_name}: {value:.0%}")


# -------------------------------------------------------------------
# Response display
# -------------------------------------------------------------------
def render_response(system, response):
    """
    Function: render_response
    Purpose : Show one final response in three tabs: Summary (pipeline
              diagram, trace, key results, route map), AI module details,
              and the numbered explanation.
    """
    STATUS_STYLE.get(response["status"], st.info)(
        f"**{response['status']}** | {response['request_id']} | {response['request_type']} | "
        f"{response['user_id']} ({response['role']})")
    details = response["details"]
    summary_tab, details_tab, explanation_tab = st.tabs(["Summary", "AI module details", "Explanation"])

    with summary_tab:
        st.graphviz_chart(pipeline_dot(response))
        render_pipeline_trace(response)
        metric_columns = st.columns(4)
        metric_columns[0].metric("Priority (ANN)", response["priority"] or "-")
        metric_columns[1].metric("Permission (KB)", response["authorization"] or "-")
        metric_columns[2].metric("Assignment (CSP)", "Yes" if response["assignment"] else "-")
        metric_columns[3].metric("Route cost", details.get("search", {}).get("cost", "-"))
        if response["assignment"]:
            st.markdown(f"**Assignment:** {response['assignment']}")
        if response["route"]:
            search = details["search"]
            st.markdown(f"**Route:** {response['route']}  \nAlgorithm **{search['algorithm']}**, "
                        f"cost {search['cost']}, {search['hops']} stops, {search['nodes_expanded']} nodes expanded")
            if len(search["path"]) > 1:
                st.image(visualizer.plot_campus_graph(system.campus_graph, search["path"],
                                                      f"Route: {response['route']}",
                                                      f"route_{response['request_id']}.png"))

    with details_tab:
        if not any(key in details for key in DETAIL_KEYS.values()):
            st.info("No AI module ran for this request.")
        if "ann" in details:
            st.subheader("ANN priority prediction")
            render_probabilities(details["ann"]["probabilities"])
            st.write(f"Perceptron urgent flag: **{details['ann']['perceptron_urgent']}**")
            st.write("Feature vector:", details["ann"]["features"])
            st.caption("The ANN predicts priority only. It never grants permission.")
        if "logic" in details:
            st.subheader("Logic / Knowledge Base checks")
            for label, result in details["logic"]["checks"]:
                st.markdown(f"{'✅' if result['entailed'] else '❌'} **{label}**: `{result['query']}`")
                for step in result["proof"]:
                    st.markdown(f"- proof: {step}")
                for reason in result["missing"]:
                    st.markdown(f"- missing: {reason}")
        if "csp" in details:
            st.subheader("CSP assignment")
            if details["csp"]["assignments"]:
                st.dataframe(pd.DataFrame(details["csp"]["assignments"]), hide_index=True)
            for reason in details["csp"]["reasons"]:
                st.write(f"Note: {reason}")
            st.caption(f"Domain size {details['csp']['domain_size']}, solver statistics {details['csp']['statistics']}")
        if "search" in details:
            st.subheader("Search")
            st.write(details["search"]["policy"])
            if details["search"].get("expanded_order"):
                st.write("Expansion order:", " -> ".join(details["search"]["expanded_order"]))

    with explanation_tab:
        for number, sentence in enumerate(response["explanation"], start=1):
            st.markdown(f"{number}. {sentence}")
        if details.get("request"):
            with st.expander("Standard request object (after preprocessing)"):
                st.json(details["request"])


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


def page_home(system):
    """
    Function: page_home
    Purpose : Dashboard - totals from the results log, status and request
              type charts, recent requests and the pipeline table.
    """
    st.title("🎓 Smart Campus AI")
    st.markdown("One smart service desk for the campus: **Input & Preprocessing -> Request Router -> "
                "ANN / Logic / CSP / Search -> Final Response**.")
    results_log = load_results_log()
    if results_log is None or results_log.empty:
        st.info("No requests processed yet. Use *Submit request* or *Demo requests* to get started.")
    else:
        status_counts = results_log["status"].value_counts()
        columns = st.columns(4)
        columns[0].metric("Requests processed", len(results_log))
        columns[1].metric("Completed / approved", int(status_counts.reindex(POSITIVE_STATUSES).fillna(0).sum()))
        columns[2].metric("Rejected by KB", int(status_counts.get("REJECTED", 0)))
        columns[3].metric("Invalid / unschedulable",
                          int(status_counts.get("INVALID", 0) + status_counts.get("UNSCHEDULABLE", 0)))
        chart_columns = st.columns(2)
        chart_columns[0].markdown("**Requests by status**")
        chart_columns[0].bar_chart(status_counts)
        chart_columns[1].markdown("**Requests by type**")
        chart_columns[1].bar_chart(results_log["request_type"].value_counts())
        st.subheader("Recent requests")
        st.dataframe(results_log.tail(10).iloc[::-1], hide_index=True)
    st.subheader("Request pipelines")
    st.dataframe(pd.DataFrame([{"Request type": name, "Pipeline": " -> ".join(modules)}
                               for name, modules in PIPELINES.items()]), hide_index=True)
    st.caption(f"{len(system.campus_graph.nodes)} campus locations | {len(system.knowledge_base.rules)} KB rules | "
               f"{len(config.ROOMS)} rooms | {len(config.TIME_SLOTS)} time slots")


def page_submit_request(system):
    """
    Function: page_submit_request
    Purpose : Form for a new request. Request type, user and role sit
              outside the form so the fields update immediately; the role
              is auto-filled from the knowledge base.
    """
    st.header("Submit a new request")
    request_type = st.selectbox("Request type", config.REQUEST_TYPES)
    st.caption(f"Pipeline for this type: {' -> '.join(PIPELINES[request_type])}")
    users = known_users(system)
    user_columns = st.columns(2)
    user_choice = user_columns[0].selectbox("User", list(users) + [OTHER_USER])
    user_id = user_columns[0].text_input("New user name") if user_choice == OTHER_USER else user_choice
    default_role = users.get(user_choice, "Student")
    role = user_columns[1].selectbox("Role", config.VALID_ROLES, index=config.VALID_ROLES.index(default_role),
                                     help="Auto-filled from the knowledge base. Change it to test the role check.")
    locations = system.campus_graph.nodes
    with st.form("request_form"):
        raw_request = {"request_type": request_type, "user_id": user_id, "role": role}
        if request_type in ("Navigation_Only", "Urgent_Service_Request", "Full_Service_Request"):
            raw_request["source"] = st.selectbox("Current location", locations, index=locations.index("Main_Gate"))
        if request_type == "Navigation_Only":
            raw_request["destination"] = st.selectbox("Destination", locations, index=locations.index("AI_Lab"))
        if request_type in ("Navigation_Only", "Full_Service_Request"):
            raw_request["route_preference"] = st.selectbox(
                "Route preference", config.ROUTE_PREFERENCES,
                help="distance = A*, stops = BFS, distance_no_heuristic = UCS")
        if request_type == "Eligibility_Check":
            raw_request["target"] = st.selectbox("Course or resource", system.eligibility_targets())
        if request_type in ("Booking_or_Scheduling", "Full_Service_Request"):
            booking_columns = st.columns(2)
            raw_request["resource_type"] = booking_columns[0].selectbox("Resource type", config.RESOURCE_TYPES)
            raw_request["group_size"] = booking_columns[1].number_input("Group size", 1, 500, 10)
            raw_request["preferred_slot"] = booking_columns[0].selectbox("Preferred slot", [""] + config.TIME_SLOTS)
            raw_request["sessions"] = booking_columns[1].number_input("Sessions", 1, len(config.TIME_SLOTS), 1)
        if request_type == "Urgent_Service_Request":
            raw_request["service_type"] = st.selectbox("Service needed", list(config.SERVICE_DESTINATIONS))
        if request_type in ("Urgent_Service_Request", "Full_Service_Request"):
            level_columns = st.columns(3)
            raw_request["severity"] = level_columns[0].slider("Severity", 1, 5, 3)
            raw_request["time_sensitivity"] = level_columns[1].slider("Time sensitivity", 1, 5, 3)
            raw_request["crowd_level"] = level_columns[2].slider("Crowd level", 1, 5, 2)
        if st.form_submit_button("Process request", type="primary"):
            st.session_state["last_submit"] = process_and_store(system, raw_request)
    if "last_submit" in st.session_state:
        st.divider()
        render_response(system, st.session_state["last_submit"])


def page_demo_requests(system):
    """
    Function: page_demo_requests
    Purpose : Run one of the built-in demo requests (shared with the CLI).
    """
    st.header("Demo requests")
    titles = [demo["title"] for demo in DEMO_REQUESTS]
    chosen_title = st.selectbox("Choose a demo", titles)
    demo = DEMO_REQUESTS[titles.index(chosen_title)]
    with st.expander("Raw request", expanded=True):
        st.json(demo["request"])
    if st.button("Run demo", type="primary"):
        st.session_state["last_demo"] = process_and_store(system, demo["request"])
    if "last_demo" in st.session_state:
        st.divider()
        render_response(system, st.session_state["last_demo"])


def page_batch_csv(system):
    """
    Function: page_batch_csv
    Purpose : Process every row of a CSV (default sample file or an
              uploaded one) and show a summary table with pandas.
    """
    st.header("Batch processing from CSV")
    uploaded_file = st.file_uploader("Upload a requests CSV (or use the sample file)", type="csv")
    if not st.button("Process CSV", type="primary"):
        return
    try:
        source = uploaded_file if uploaded_file is not None else config.BATCH_REQUEST_PATH
        requests_frame = pd.read_csv(source, dtype=str, keep_default_na=False)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as error:
        st.error(f"Could not read the CSV: {error}")
        return
    summary_rows = []
    progress_bar = st.progress(0.0, text="Processing requests...")
    for row_number, (_, row) in enumerate(requests_frame.iterrows(), start=1):
        raw_request = {key: value for key, value in row.to_dict().items() if value != ""}
        response = process_and_store(system, raw_request)
        summary_rows.append({"request_id": response["request_id"], "user": response["user_id"],
                             "type": response["request_type"], "status": response["status"],
                             "priority": response["priority"] or "-",
                             "assignment": response["assignment"] or "-",
                             "route": response["route"] or "-",
                             "explanation": response["explanation"][-1]})
        progress_bar.progress(row_number / len(requests_frame), text=f"Processed {row_number}/{len(requests_frame)}")
    summary = pd.DataFrame(summary_rows)
    st.dataframe(summary, hide_index=True)
    st.bar_chart(summary["status"].value_counts())
    st.download_button("Download results CSV", summary.to_csv(index=False), "batch_results.csv", "text/csv")


def page_ann_models(system):
    """
    Function: page_ann_models
    Purpose : What-if priority predictor, model training/evaluation, and
              the training dataset.
    """
    st.header("ANN priority models")
    predictor = system.priority_predictor
    what_if_tab, training_tab, dataset_tab = st.tabs(["What-if predictor", "Training & evaluation", "Dataset"])

    with what_if_tab:
        st.write("Move the inputs and watch the MLP prediction change. The ANN predicts priority only.")
        columns = st.columns(2)
        role = columns[0].selectbox("x1 Role", config.VALID_ROLES)
        request_type = columns[1].selectbox("x2 Request type", config.REQUEST_TYPES, index=3)
        level_columns = st.columns(3)
        severity = level_columns[0].slider("x3 Severity", 1, 5, 4)
        time_sensitivity = level_columns[1].slider("x4 Time sensitivity", 1, 5, 4)
        crowd_level = level_columns[2].slider("x5 Crowd level", 1, 5, 2)
        distance_column, eligibility_column = st.columns(2)
        distance = distance_column.slider("x6 Route distance", 0.0, config.MAX_ROUTE_DISTANCE, 5.0, 0.5)
        eligible = eligibility_column.checkbox("x7 Registered / eligible user", value=True)
        feature_vector = preprocessing.build_feature_vector(
            {"role": role, "request_type": request_type, "severity": severity,
             "time_sensitivity": time_sensitivity, "crowd_level": crowd_level}, distance, eligible)
        prediction = predictor.predict(feature_vector)
        result_columns = st.columns(3)
        result_columns[0].metric("MLP priority", prediction["priority"])
        result_columns[1].metric("Confidence", f"{prediction['confidence']:.0%}")
        result_columns[2].metric("Perceptron urgent?", "Yes" if prediction["perceptron_urgent"] else "No")
        render_probabilities(prediction["probabilities"])
        st.caption(f"Scaled feature vector: {prediction['features']}")

    with training_tab:
        st.write("Perceptron = binary baseline (Urgent/High vs Normal/Low). "
                 "MLP 7-4-3-4 = final multiclass model (Urgent, High, Normal, Low).")
        if st.button("Train / retrain models", type="primary"):
            with st.spinner("Training..."):
                st.session_state["ann_metrics"] = predictor.train(verbose=False)
        metrics = st.session_state.get("ann_metrics")
        if metrics:
            columns = st.columns(2)
            columns[0].metric("Perceptron test accuracy", f"{metrics['perceptron_accuracy']:.1%}")
            columns[1].metric("MLP test accuracy", f"{metrics['mlp_accuracy']:.1%}")
            columns = st.columns(2)
            columns[0].image(visualizer.plot_confusion_matrix(metrics["perceptron_confusion"],
                                                              ["Not urgent", "Urgent"],
                                                              "Perceptron confusion matrix",
                                                              "perceptron_confusion.png"))
            columns[1].image(visualizer.plot_confusion_matrix(metrics["mlp_confusion"], config.PRIORITY_CLASSES,
                                                              "MLP confusion matrix", "mlp_confusion.png"))
            st.image(visualizer.plot_class_distribution(metrics["class_distribution"]))
        else:
            st.info("Models are loaded from saved weights. Press the button to retrain and see test metrics.")
        if predictor.perceptron.errors_per_epoch:
            st.image(visualizer.plot_perceptron_errors(predictor.perceptron.errors_per_epoch))
        if predictor.mlp.history.get("loss"):
            st.image(visualizer.plot_mlp_history(predictor.mlp.history))

    with dataset_tab:
        if os.path.exists(config.ANN_DATASET_PATH):
            dataset = pd.read_csv(config.ANN_DATASET_PATH)
            st.write(f"{len(dataset)} synthetic requests (first 100 shown).")
            st.dataframe(dataset.head(100), hide_index=True)
        else:
            st.info("Dataset is created the first time the models are trained.")


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
    Purpose : Example query buttons, backward-chaining queries, adding
              facts, and tables of facts, rules and forward-chained facts.
    """
    st.header("Logic / Knowledge Base")
    knowledge_base = system.knowledge_base
    st.session_state.setdefault("kb_query", "Eligible(Sara, AI)")
    st.write("Try an example:")
    example_columns = st.columns(len(EXAMPLE_QUERIES))
    for column, query_text in zip(example_columns, EXAMPLE_QUERIES):
        column.button(query_text, on_click=set_example_query, args=(query_text,), key=f"example_{query_text}")
    st.text_input("Query (backward chaining)", key="kb_query",
                  help="Lower-case single letters like x are variables, e.g. Eligible(x, AI)")
    if st.button("Ask", type="primary"):
        st.session_state["kb_run"] = True
    if st.session_state.pop("kb_run", False):
        try:
            result = knowledge_base.ask(parse_atom(st.session_state["kb_query"]))
            (st.success if result["entailed"] else st.error)(f"{result['query']} entailed: {result['entailed']}")
            for answer in result["answers"]:
                if answer:
                    st.write("answer:", answer)
            for step in result["proof"]:
                st.markdown(f"- proof: {step}")
            for reason in result["missing"]:
                st.markdown(f"- missing: {reason}")
        except KnowledgeBaseError as error:
            st.error(f"KB error: {error}")

    with st.form("add_fact_form", clear_on_submit=True):
        fact_text = st.text_input("Add a fact", placeholder="Completed(Sara, ProgrammingFundamentals)")
        if st.form_submit_button("Add fact"):
            try:
                atom = parse_atom(fact_text)
                knowledge_base.add_fact(atom)
                st.success(f"Added {format_atom(atom)}")
            except KnowledgeBaseError as error:
                st.error(f"KB error: {error}")

    facts_tab, rules_tab, forward_tab = st.tabs(["Facts", "Rules", "Forward chaining"])
    facts_tab.dataframe(pd.DataFrame({"Fact": [format_atom(f) for f in sorted(knowledge_base.facts)]}),
                        hide_index=True)
    rules_tab.dataframe(pd.DataFrame([{"Rule": str(rule), "Meaning": rule.description}
                                      for rule in knowledge_base.rules]), hide_index=True)
    derived = knowledge_base.forward_chain()
    forward_tab.dataframe(pd.DataFrame([{"Derived fact": format_atom(fact), "Rule": rule_name,
                                         "From": ", ".join(format_atom(p) for p in premises)}
                                        for fact, (rule_name, premises) in sorted(derived.items())]),
                          hide_index=True)


def search_frame(system, result, step):
    """
    Function: search_frame
    Purpose : Draw one animation frame: nodes expanded so far (blue), the
              node being expanded (yellow), and the final path on the last step.
    """
    order = result["expanded_order"]
    finished = step >= len(order)
    current = order[step - 1]
    title = (f"{result['algorithm']}: finished after {len(order)} expansions, cost {result['cost']}" if finished
             else f"{result['algorithm']}: step {step}/{len(order)} - expanding {current}")
    return visualizer.plot_campus_graph(system.campus_graph, result["path"] if finished else None, title,
                                        "search_animation.png", expanded_nodes=order[:step], current_node=current)


def page_navigation(system):
    """
    Function: page_navigation
    Purpose : Compare BFS, UCS and A* on a route, and animate the order in
              which each algorithm expands nodes.
    """
    st.header("Campus navigation")
    locations = system.campus_graph.nodes
    columns = st.columns(2)
    source = columns[0].selectbox("From", locations, index=locations.index("Main_Gate"))
    destination = columns[1].selectbox("To", locations, index=locations.index("AI_Lab"))
    if source == destination:
        st.warning("Choose two different locations.")
        return
    try:
        results = compare_algorithms(source, destination, system.campus_graph)
    except NavigationError as error:
        st.error(str(error))
        return
    compare_tab, animation_tab = st.tabs(["Compare algorithms", "Step-by-step search animation"])

    with compare_tab:
        st.caption(f"A* heuristic = {system.campus_graph.heuristic_scale:.3f} x Euclidean distance (admissible)")
        st.dataframe(pd.DataFrame([{"Algorithm": r["algorithm"], "Cost": r["cost"], "Stops": r["hops"],
                                    "Nodes expanded": r["nodes_expanded"], "Path": " -> ".join(r["path"])}
                                   for r in results]), hide_index=True)
        file_suffix = f"{source}_{destination}"
        st.image(visualizer.plot_campus_graph(system.campus_graph, results[2]["path"],
                                              f"A* route {source} -> {destination}",
                                              f"campus_route_{file_suffix}.png"))
        st.image(visualizer.plot_algorithm_comparison(results, f"search_comparison_{file_suffix}.png"))

    with animation_tab:
        algorithm_name = st.radio("Algorithm", [r["algorithm"] for r in results], horizontal=True)
        result = next(r for r in results if r["algorithm"] == algorithm_name)
        order = result["expanded_order"]
        st.caption("🟨 expanding now   🟦 already expanded   🟥 final path")
        if len(order) > 1:
            step = st.slider("Step", 1, len(order), len(order))
        else:
            step = 1
        frame_placeholder = st.empty()
        if st.button("▶ Play animation"):
            for frame_step in range(1, len(order) + 1):
                frame_placeholder.image(search_frame(system, result, frame_step))
                time.sleep(0.7)
        else:
            frame_placeholder.image(search_frame(system, result, step))
        st.write("Expansion order:", " -> ".join(order[:step]))


def constraint_graph_dot(solution):
    """
    Function: constraint_graph_dot
    Purpose : Graphviz DOT of the G1-G6 constraint graph from the CSP
              diagram. Nodes are coloured by assigned time slot; edge colours
              follow the diagram (red !=, purple examiner, orange supervisor,
              blue precedence arrow).
    """
    theme = theme_name()
    colours = DIAGRAM_COLOURS[theme]
    lines = ['graph csp {', 'bgcolor="transparent"; layout=neato; splines=true; overlap=false;',
             f'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, '
             f'fontcolor="{colours["font"]}", color="{colours["edge"]}"];',
             'edge [fontname="Helvetica", fontsize=9];']
    for group, (slot, room) in sorted(solution.items()):
        colour = SLOT_COLOURS[theme][config.TIME_SLOTS.index(slot)]
        x_position, y_position = CSP_NODE_POSITIONS[group]
        lines.append(f'"{group}" [label="{group}\\n{EXAM_GROUPS[group]} students\\n{slot}\\n{room}", '
                     f'fillcolor="{colour}", pos="{x_position},{y_position}!"];')
    for group_a, group_b, label in EXAM_CONSTRAINT_EDGES:
        kind = "examiner" if "examiner" in label else "supervisor" if "supervisor" in label else "not_equal"
        colour = colours[kind]
        short_label = label.replace(" (shared students)", "")
        lines.append(f'"{group_a}" -- "{group_b}" [label="{short_label}", color="{colour}", fontcolor="{colour}"];')
    for group_a, group_b, label in EXAM_PRECEDENCE:
        colour = colours["precedence"]
        lines.append(f'"{group_a}" -- "{group_b}" [label="{label}", color="{colour}", fontcolor="{colour}", '
                     f'dir=forward, penwidth=2];')
    lines.append("}")
    return "\n".join(lines)


def page_csp_schedule(system):
    """
    Function: page_csp_schedule
    Purpose : Exam timetable with solver statistics, the coloured
              constraint graph, and the grid of all room bookings.
    """
    st.header("CSP exam timetable and room bookings")
    exam_result = solve_exam_schedule()
    statistics = exam_result["statistics"]
    columns = st.columns(4)
    columns[0].metric("Values tried", statistics["assignments_tried"])
    columns[1].metric("Backtracks", statistics["backtracks"])
    columns[2].metric("Pruned by AC-3", statistics["ac3_pruned"])
    columns[3].metric("Constraint violations", len(exam_result["violations"]))
    timetable_tab, graph_tab, bookings_tab = st.tabs(["Exam timetable", "Constraint graph", "Room bookings"])
    with timetable_tab:
        st.dataframe(exam_result["schedule"], hide_index=True)
    with graph_tab:
        st.graphviz_chart(constraint_graph_dot(exam_result["solution"]))
        st.caption("   ".join(f"{icon} {slot}" for icon, slot in zip(SLOT_LEGEND_ICONS, config.TIME_SLOTS)))
    with bookings_tab:
        booking_table = system.scheduler.booking_table()
        st.image(visualizer.plot_schedule(booking_table))
        st.dataframe(booking_table, hide_index=True)


def page_history(system):
    """
    Function: page_history
    Purpose : List every response from this browser session and re-open
              any of them.
    """
    st.header("Session history")
    history = st.session_state.get("history", [])
    if not history:
        st.info("No requests in this session yet.")
        return
    st.dataframe(pd.DataFrame([{"#": index + 1, "request_id": r["request_id"], "user": r["user_id"],
                                "type": r["request_type"], "status": r["status"],
                                "priority": r["priority"] or "-"} for index, r in enumerate(history)]),
                 hide_index=True)
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
    Purpose : Title and page selector at the top of the sidebar.
    """
    st.sidebar.title("🎓 Smart Campus AI")
    st.sidebar.caption("Decision Support and Automation System")
    page_name = st.sidebar.radio("Go to", list(PAGES), key="page")
    st.sidebar.caption("🌗 Light / dark mode: ⋮ menu (top right) → Settings → Theme")
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
        st.sidebar.divider()
        st.sidebar.markdown("**Recent requests**")
        for index in range(len(history) - 1, max(len(history) - 6, -1), -1):
            label = history_label(index, history[index])
            st.sidebar.button(label, key=f"sidebar_{index}", on_click=open_history_entry, args=(label,),
                              width="stretch")


def main():
    """
    Function: main
    Purpose : Page setup, sidebar navigation and dispatch to the chosen
              page. Unexpected errors are shown instead of crashing.
    """
    st.set_page_config(page_title="Smart Campus AI", page_icon="🎓", layout="wide")
    theme = theme_name()
    visualizer.set_dark_mode(theme == "dark")
    st.markdown(custom_css(theme), unsafe_allow_html=True)
    config.ensure_directories()
    page_name = render_sidebar_navigation()
    try:
        PAGES[page_name](get_system())
    except Exception as error:              # last-resort guard: report, never crash
        st.error(f"Unexpected error: {type(error).__name__}: {error}")
    render_sidebar_history()


main()
