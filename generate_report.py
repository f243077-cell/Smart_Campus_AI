"""
generate_report.py
-----------------------------------------------------------------------
Builds the project report (Microsoft Word .docx, plus PDF when Word is
installed) with python-docx:
  * project overview, objectives, architecture and run sequence
  * description of every AI concept used and the modules completed
  * console screenshots of every task (rendered with matplotlib)
  * all generated plots
  * the complete source code of every Python file

Run:  python generate_report.py
-----------------------------------------------------------------------
"""

import contextlib
import io
import os
import sys

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

import config
import main
import visualizer
from knowledge_base import format_atom, parse_atom
from response_layer import format_response, log_response
from router import PIPELINES, SmartCampusSystem
from search_navigation import CampusGraph, find_route

REPORT_DIRECTORY = os.path.dirname(config.PROJECT_DIRECTORY)
REPORT_DOCX_PATH = os.path.join(REPORT_DIRECTORY, "Smart_Campus_AI_Report.docx")
REPORT_PDF_PATH = os.path.join(REPORT_DIRECTORY, "Smart_Campus_AI_Report.pdf")
SOURCE_FILES = ["main.py", "app.py", "ui.py", "config.py", "preprocessing.py", "router.py", "ann_module.py",
                "knowledge_base.py", "csp_scheduler.py", "search_navigation.py",
                "response_layer.py", "visualizer.py", "generate_report.py"]
MAX_LINES_PER_SCREENSHOT = 58


def capture_output(function, *arguments, **keyword_arguments):
    """
    Function: capture_output
    Purpose : Run a function while capturing everything it prints.
    Returns : (function_result, captured_text)
    """
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result = function(*arguments, **keyword_arguments)
    return result, buffer.getvalue()


def text_to_screenshots(console_text, base_name, title):
    """
    Function: text_to_screenshots
    Purpose : Split long console output into chunks and render each chunk
              as a terminal-style PNG. Returns the list of image paths.
    """
    lines = console_text.strip("\n").splitlines()
    paths = []
    for part, start in enumerate(range(0, max(len(lines), 1), MAX_LINES_PER_SCREENSHOT), start=1):
        chunk = "\n".join(lines[start:start + MAX_LINES_PER_SCREENSHOT])
        suffix = f"_{part}" if len(lines) > MAX_LINES_PER_SCREENSHOT else ""
        paths.append(visualizer.render_console_screenshot(chunk, f"{base_name}{suffix}.png", title))
    return paths


def collect_evidence():
    """
    Function: collect_evidence
    Purpose : Run every task of the project non-interactively, capture
              the console output, render screenshots and plots.
    Returns : list of sections: dict(title, description, images)
    """
    config.ensure_directories()
    if os.path.exists(config.RESULTS_LOG_PATH):
        os.remove(config.RESULTS_LOG_PATH)
    sections = []
    system, startup_text = capture_output(SmartCampusSystem, verbose=True)
    sections.append({"title": "Program start and main menu",
                     "description": "Running python main.py loads every module and shows the main menu.",
                     "images": text_to_screenshots(main.BANNER_TEXT + "\n  Loading modules (campus graph, "
                                                   "knowledge base, CSP, ANN)...\n" + startup_text
                                                   + main.MAIN_MENU_TEXT + "\n  Choice: 2",
                                                   "00_main_menu", "python main.py")})

    # ANN training and evaluation
    (metrics, ann_plots), ann_text = capture_output(main.train_models_menu, system, show_prompts=False)
    sections.append({"title": "Task: ANN Priority Module (menu 4)",
                     "description": f"Perceptron test accuracy {metrics['perceptron_accuracy']:.2%}, "
                                    f"MLP 7-4-3-4 test accuracy {metrics['mlp_accuracy']:.2%}.",
                     "images": text_to_screenshots(ann_text, "01_ann_training", "Menu 4 - Train / evaluate ANN")
                     + ann_plots})

    # Demo requests: one screenshot per request type
    for number, demo in enumerate(main.DEMO_REQUESTS, start=1):
        response = system.process_request(demo["request"])
        log_response(response)
        request_lines = "\n".join(f"    {key}: {value}" for key, value in demo["request"].items())
        text = f">>> Demo {number}: {demo['title']}\n  Raw request:\n{request_lines}\n\n" + format_response(response)
        images = text_to_screenshots(text, f"02_demo_{number:02d}", f"Menu 2 - Demo {number}")
        if response["route"]:
            images.append(visualizer.plot_campus_graph(system.campus_graph, response["details"]["search"]["path"],
                                                       f"{demo['title']}: {response['route']}",
                                                       f"demo_{number:02d}_route.png"))
        sections.append({"title": f"Task: {demo['title']}",
                         "description": f"Status {response['status']}; modules run: "
                                        f"{', '.join(response['modules_run'])}.",
                         "images": images})

    # Batch CSV
    _, batch_text = capture_output(main.process_batch_csv, system)
    sections.append({"title": "Task: Batch processing from CSV (menu 3)",
                     "description": "All rows of data/campus_requests.csv processed with pandas, including "
                                    "invalid rows (non-numeric group size, missing user id).",
                     "images": text_to_screenshots(batch_text, "03_batch", "Menu 3 - Batch CSV")})

    # Knowledge base
    knowledge_base = system.knowledge_base
    kb_lines = ["  --- Rules ---"] + [f"   {rule}" for rule in knowledge_base.rules]
    derived = knowledge_base.forward_chain()
    kb_lines += ["", f"  --- Forward chaining: {len(derived)} derived facts (first 20) ---"]
    for fact, (rule_name, _) in sorted(derived.items())[:20]:
        kb_lines.append(f"   {format_atom(fact):40s} by {rule_name}")
    for query_text in ("Eligible(x, AI)", "UseLab(DrKhan, Lab1)", "Eligible(Sara, AI)", "CanBook(Fatima, SeminarRoom)"):
        result = knowledge_base.ask(parse_atom(query_text))
        kb_lines += ["", f"  Query: {query_text}  ->  entailed: {result['entailed']}"]
        kb_lines += [f"     answer: {answer}" for answer in result["answers"] if answer]
        kb_lines += [f"     proof: {step}" for step in result["proof"]]
        kb_lines += [f"     missing: {reason}" for reason in result["missing"]]
    sections.append({"title": "Task: Logic / Knowledge Base (menu 5)",
                     "description": "Rules, forward chaining and backward-chaining queries with proofs "
                                    "and explanations of failures.",
                     "images": text_to_screenshots("\n".join(kb_lines), "04_knowledge_base", "Menu 5 - Knowledge base")})

    # Navigation
    navigation_images, navigation_text = [], ""
    for source, destination in (("Main_Gate", "AI_Lab"), ("Bus_Stop", "Seminar_Room"), ("Hostel", "Exam_Hall")):
        paths, text = capture_output(main.show_route_comparison, system, source, destination)
        navigation_images += paths
        navigation_text += text
    no_heuristic_graph = CampusGraph(coordinates={})
    fallback_text = (f"\n  Graph without coordinates -> heuristic available: "
                     f"{no_heuristic_graph.heuristic_available()}\n"
                     f"  find_route policy result: {system_fallback_route(no_heuristic_graph)}")
    sections.append({"title": "Task: Search & Navigation (menu 6)",
                     "description": "BFS (unweighted), UCS (weighted fallback) and A* (weighted with admissible "
                                    "heuristic) compared on the same queries.",
                     "images": text_to_screenshots(navigation_text + fallback_text, "05_navigation",
                                                   "Menu 6 - Campus navigation") + navigation_images})

    # CSP
    schedule_plot, csp_text = capture_output(main.show_schedules, system)
    sections.append({"title": "Task: CSP Scheduler (menu 7)",
                     "description": "Exam timetable for G1-G6 from the CSP diagram and every room booking "
                                    "made by the requests above.",
                     "images": text_to_screenshots(csp_text, "06_csp", "Menu 7 - CSP schedule") + [schedule_plot]})

    # Invalid menu input / error handling, captured from a real run of main()
    scripted_input = "9\nabc\n5\n3\nEligible Sara AI\n0\n1\n1\nAli\n1\nParking\nParking\n\nn\n"
    _, error_text = capture_output(run_main_with_input, scripted_input)
    sections.append({"title": "Task: Exception handling",
                     "description": "Wrong menu options, a malformed KB query and an invalid request "
                                    "(same source and destination) are reported without crashing; end of "
                                    "input exits cleanly.",
                     "images": text_to_screenshots(error_text, "07_errors", "Error handling")})
    return sections, metrics


class EchoInput(io.StringIO):
    """
    Class: EchoInput
    Purpose: Stand-in for sys.stdin that prints every line it hands to
             input(), so scripted answers appear in the captured console.
    """

    def readline(self, *arguments):
        """
        Function: readline
        Purpose : Return the next scripted line and echo it to stdout.
        """
        line = super().readline(*arguments)
        print(line.rstrip("\n"))
        return line


def run_main_with_input(scripted_input):
    """
    Function: run_main_with_input
    Purpose : Run the real CLI main() with scripted keyboard input. When
              the script runs out, input() raises EOFError and main()
              exits through its normal handler.
    """
    original_stdin = sys.stdin
    sys.stdin = EchoInput(scripted_input)
    try:
        main.main()
    finally:
        sys.stdin = original_stdin


def system_fallback_route(graph_without_heuristic):
    """
    Function: system_fallback_route
    Purpose : Show that find_route falls back to UCS when no heuristic exists.
    """
    result = find_route("Main_Gate", "AI_Lab", "distance", graph_without_heuristic)
    return f"{result['algorithm']} ({result['policy']}), path {' -> '.join(result['path'])}, cost {result['cost']}"


# -------------------------------------------------------------------
# Word document helpers
# -------------------------------------------------------------------
def add_bullets(document, items):
    """
    Function: add_bullets
    Purpose : Add a bulleted list to the document.
    """
    for item in items:
        document.add_paragraph(item, style="List Bullet")


def add_table(document, header, rows):
    """
    Function: add_table
    Purpose : Add a simple grid table with a bold header row.
    """
    table = document.add_table(rows=1, cols=len(header))
    table.style = "Light Grid Accent 1"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for cell, text in zip(table.rows[0].cells, header):
        cell.text = text
        cell.paragraphs[0].runs[0].bold = True
    for row in rows:
        cells = table.add_row().cells
        for cell, text in zip(cells, row):
            cell.text = str(text)
    document.add_paragraph()


def add_code_block(document, code_text):
    """
    Function: add_code_block
    Purpose : Add source code in a small monospace font.
    """
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(0)
    run = paragraph.add_run(code_text)
    run.font.name = "Consolas"
    run.font.size = Pt(7.5)


def build_document(sections, metrics):
    """
    Function: build_document
    Purpose : Write the full Word report and return the saved path.
    """
    document = Document()
    document.styles["Normal"].font.name = "Calibri"
    document.styles["Normal"].font.size = Pt(11)

    # Title page
    for text, size, bold in (("National University of Computing & Emerging Sciences", 14, True),
                             ("AL2002 - Artificial Intelligence Lab (Fall 2026)", 12, False),
                             ("Semester Capstone Project", 12, False), ("", 12, False),
                             ("Smart Campus AI Decision Support and Automation System", 22, True),
                             ("", 12, False), ("Project Report", 14, False),
                             ("Section 5A1  |  Department of Software Engineering", 11, False)):
        paragraph = document.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = paragraph.add_run(text)
        run.font.size, run.bold = Pt(size), bold
        if size == 22:
            run.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)
    document.add_page_break()

    document.add_heading("1. Project Overview", level=1)
    document.add_paragraph(
        "The Smart Campus AI Decision Support and Automation System is one connected service desk for a "
        "university campus. A user (Student, Instructor or Staff) submits a structured request. The Input & "
        "Preprocessing module cleans and validates it, the Request Router chooses the pipeline for the request "
        "type, the selected AI modules (ANN, Logic / Knowledge Base, CSP, Search) do their part, and the Final "
        "Response Layer combines all outputs into a single, explained answer. The project is written in Python; "
        "numpy is used for the neural networks and geometry, pandas for datasets, batch input, schedules and "
        "logs, and matplotlib for every chart and screenshot.")
    document.add_heading("2. Objectives", level=1)
    add_bullets(document, [
        "Design a unified AI-based smart campus platform.",
        "Integrate classical AI (Search, Logic, CSP) with modern AI (ANN).",
        "Model realistic campus decision-support scenarios.",
        "Route structured requests through different pipelines instead of isolated algorithm demos.",
        "Produce explainable and modular AI behaviour."])

    document.add_heading("3. Architecture and Request Lifecycle", level=1)
    add_bullets(document, [
        "1. User submits a structured request (CLI form, demo or CSV row).",
        "2. Input & Preprocessing validates and standardizes it, and prepares the ANN feature vector.",
        "3. Request Router selects the pipeline from request_type.",
        "4. Relevant AI modules execute in order; Logic / KB can stop the pipeline.",
        "5. Final Response is generated, shown to the user and logged to outputs/results_log.csv."])
    add_table(document, ["Request type", "Pipeline (in order)"],
              [(name, " -> ".join(modules)) for name, modules in PIPELINES.items()])
    add_table(document, ["Responsibility", "Module", "File"], [
        ("User intent", "User input", "main.py"), ("Validation / features", "Input & Preprocessing", "preprocessing.py"),
        ("Control flow", "Request Router", "router.py"), ("Priority estimation", "ANN", "ann_module.py"),
        ("Permission & rules", "Logic / KB", "knowledge_base.py"), ("Allocation", "CSP", "csp_scheduler.py"),
        ("Navigation", "Search", "search_navigation.py"), ("Communication", "Final Response", "response_layer.py")])

    document.add_heading("4. Modules Completed", level=1)
    add_table(document, ["Component", "Marks", "Status", "What was implemented"], [
        ("Input & Preprocessing", 10, "Completed", "Required-field checks, alias & case normalization, range checks, "
                                                    "custom exception, 7-feature ANN vector"),
        ("ANN Priority Prediction", 20, "Completed", f"Perceptron ({metrics['perceptron_accuracy']:.1%}) and MLP "
                                                      f"7-4-3-4 ({metrics['mlp_accuracy']:.1%}) from scratch in numpy"),
        ("Logic / Knowledge Base", 20, "Completed", "FOL facts and rules R1-R16, unification, forward and backward "
                                                     "chaining, proofs and failure explanations"),
        ("CSP Scheduler", 15, "Completed", "AC-3, backtracking, MRV, LCV, forward checking; G1-G6 exam timetable "
                                            "and room booking"),
        ("Search Navigation", 15, "Completed", "Campus graph from the diagram; BFS, UCS and A* with admissible heuristic"),
        ("Final Response & Integration", 10, "Completed", "Router pipelines, combined explained response, CSV log"),
        ("Documentation & Code Quality", 10, "Completed", "Comment block on every function, clear names, "
                                                           "exception handling, this report")])

    document.add_heading("5. AI Concepts Used", level=1)
    concepts = [
        ("5.1 Input & Preprocessing",
         "Every request must contain user_id, role and request_type plus the fields of its type. Text is "
         "normalized so 'nav', 'Navigation' and 'navigation_only' all become Navigation_Only and 'ai lab' becomes "
         "AI_Lab. Numbers are range checked (severity 1-5, group size 1-500). Errors raise RequestValidationError "
         "so the system never crashes on bad data. For ANN requests the module builds the feature vector "
         "x1 Role, x2 RequestType, x3 Severity, x4 TimeSensitivity, x5 CrowdLevel, x6 Distance, x7 Eligibility, "
         "min-max scaled to [0, 1]."),
        ("5.2 Request Router",
         "A table maps each request type to an ordered list of modules. The router runs them one by one, "
         "collects outputs, and stops when Logic / KB rejects a request or CSP finds no assignment. Skipped "
         "modules are reported in the explanation."),
        ("5.3 ANN Priority Module",
         "A synthetic dataset of 1500 requests is generated with numpy and stored with pandas. A hidden urgency "
         "score (weighted mostly by severity and time sensitivity, plus noise) is cut into four classes. The "
         "Perceptron uses a step activation and the perceptron learning rule as a binary baseline "
         "(Urgent/High vs Normal/Low). The MLP has the exact shape of the project diagram: 7 inputs, hidden "
         "layers of 4 and 3 tanh neurons and 4 softmax outputs (Urgent, High, Normal, Low). It is trained with "
         "cross-entropy loss, hand-written backpropagation and mini-batch gradient descent with momentum. "
         "The ANN predicts priority only: it never approves or rejects a request (see the unregistered user "
         "'Zain' in the batch run, who receives Urgent priority but is still rejected by the KB)."),
        ("5.4 Logic / Knowledge Base",
         "Knowledge is stored as First-Order Logic definite clauses. Facts and rules R1-R4 come from the FOL "
         "diagram (Teaches => Instructor, Enrolled in AI => UseLab, Student AND Completed(ProgrammingFundamentals) "
         "=> Eligible(AI), Instructor of AI => UseLab). Extra rules define roles, prerequisites and booking "
         "permissions. Unification with variable renaming (standardizing apart) supports forward chaining "
         "(derive all entailed facts) and backward chaining (goal-directed proof with a proof trace). When a "
         "query fails, the engine lists the missing premises, e.g. 'R3 needs Completed(Sara, "
         "ProgrammingFundamentals)'. The KB is the gatekeeper before CSP: the claimed role is verified first, then "
         "the specific permission."),
        ("5.5 CSP Scheduler",
         "Variables receive (time slot, room) values. Unary constraints remove rooms of the wrong type or "
         "capacity. Binary constraints from the CSP diagram: G1!=G2, G1!=G4, G2!=G4, G3!=G5, G3!=G6, G5!=G6, "
         "G1!=G3 (examiner clash), G2!=G5 (supervisor clash) and G4 < G3 (precedence), plus no room "
         "double-booking. AC-3 prunes values first, then backtracking search uses MRV with a degree tie-break, LCV "
         "value ordering and forward checking. Booking requests are solved as a CSP against all existing bookings; "
         "the ANN priority only changes value ordering (urgent requests prefer the earliest slot)."),
        ("5.6 Search & Navigation",
         "The campus map (13 buildings, 19 weighted edges, coordinates) is copied from the project diagrams. "
         "BFS solves unweighted problems (fewest stops). A* solves weighted problems with f(n) = g(n) + h(n). "
         "Plain Euclidean distance would overestimate some edges (Admin_Block-Student_Services costs 1 but is 3 "
         f"units apart), so h(n) = {CampusGraph().heuristic_scale:.3f} x Euclidean distance, where the factor is "
         "the minimum cost/length ratio over all edges. This keeps the heuristic admissible and consistent, so A* "
         "stays optimal. UCS is the fallback for weighted graphs without a heuristic."),
        ("5.7 Final Response Layer",
         "Combines priority, authorization with proofs, assignment, route and a numbered explanation into one "
         "answer with a final status (COMPLETED, APPROVED, DISPATCHED, REJECTED, UNSCHEDULABLE or INVALID)."),
    ]
    for heading, body in concepts:
        document.add_heading(heading, level=2)
        document.add_paragraph(body)

    document.add_heading("6. How to Run (Sequence)", level=1)
    add_bullets(document, [
        "1. Install Python 3.10+ and the libraries: pip install -r requirements.txt",
        "2. Open a terminal in the SmartCampusAI folder.",
        "3. Run: python main.py (the ANN is trained automatically on the first run).",
        "4. Menu 2 runs one demo per request type; menu 1 lets you enter your own request.",
        "5. Menus 3-7 show the batch CSV run, ANN training, knowledge base, navigation and CSP schedule.",
        "6. Web UI (optional): python -m streamlit run app.py opens the same 7 functions in the browser "
        "(sidebar pages). The CLI and the web UI share all AI modules.",
        "7. Plots are saved in outputs/plots and every response is logged in outputs/results_log.csv.",
        "8. Optional: python generate_report.py rebuilds this report."])
    add_table(document, ["Sample user", "Role in KB", "Useful for"], [
        ("Ali", "Student, completed ProgrammingFundamentals, enrolled in AI", "Eligible for AI, can book ComputerLab"),
        ("Sara", "Student, no completed courses", "Rejection cases"),
        ("Ahmed / Fatima", "Students with completed courses", "Eligibility for DataStructures, MachineLearning"),
        ("DrKhan / DrAyesha", "Instructors (AI / DataStructures)", "Seminar room and exam hall bookings"),
        ("Bilal", "Staff", "Meeting room bookings"),
        ("Any other name", "Not registered", "Role check fails")])

    document.add_heading("7. Results and Screenshots", level=1)
    for section in sections:
        document.add_heading(section["title"], level=2)
        document.add_paragraph(section["description"])
        for image_path in section["images"]:
            document.add_picture(image_path, width=Inches(6.3))
            document.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

    document.add_heading("8. Assumptions and Design Constraints", level=1)
    add_bullets(document, [
        "Users provide structured input (CLI form fields or CSV columns).",
        "The campus map is predefined (taken from the project diagrams).",
        "Knowledge base rules are consistent definite clauses (no negation).",
        "The ANN only predicts priority; it never approves or rejects requests.",
        "The ANN dataset is synthetic because no real campus data was provided.",
        "Bookings are kept in memory for the session and logged to CSV."])

    document.add_heading("9. Source Code", level=1)
    for file_name in SOURCE_FILES:
        document.add_heading(file_name, level=2)
        with open(os.path.join(config.PROJECT_DIRECTORY, file_name), encoding="utf-8") as source_file:
            add_code_block(document, source_file.read())

    document.save(REPORT_DOCX_PATH)
    return REPORT_DOCX_PATH


def convert_to_pdf(docx_path, pdf_path):
    """
    Function: convert_to_pdf
    Purpose : Export the report to PDF with Microsoft Word (Windows only).
              Returns True on success; failure is reported, not raised.
    """
    try:
        import win32com.client
    except ImportError:
        print("  PDF export skipped (pywin32 not installed). Open the .docx in Word and Save as PDF.")
        return False
    word_application = None
    try:
        word_application = win32com.client.DispatchEx("Word.Application")
        word_application.Visible = False
        word_document = word_application.Documents.Open(os.path.abspath(docx_path))
        word_document.SaveAs(os.path.abspath(pdf_path), FileFormat=17)   # 17 = wdFormatPDF
        word_document.Close(False)
        return True
    except Exception as error:                  # COM errors have many types
        print(f"  PDF export failed ({error}). Open the .docx in Word and Save as PDF.")
        return False
    finally:
        if word_application is not None:
            word_application.Quit()


if __name__ == "__main__":
    os.chdir(config.PROJECT_DIRECTORY)
    print("Collecting outputs, screenshots and plots...")
    report_sections, ann_metrics = collect_evidence()
    print(f"Writing Word report: {build_document(report_sections, ann_metrics)}")
    if convert_to_pdf(REPORT_DOCX_PATH, REPORT_PDF_PATH):
        print(f"PDF report: {REPORT_PDF_PATH}")
