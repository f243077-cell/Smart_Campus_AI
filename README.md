# Smart Campus AI Decision Support and Automation System

AL2002 Artificial Intelligence Lab (Fall 2026), semester capstone project.

A user fills a structured request form. **Input & Preprocessing** validates and normalizes it into the
standard request object, the **Request Router** returns the control object
`{request_id, selected_pipeline, needs_ann, needs_logic, needs_csp, needs_search}`, the selected AI modules
run, and the **Final Response Layer** returns
`{request_id, decision, priority, eligibility, assignment, route, message}` with an explanation.

| Request type           | Pipeline (brackets = optional)            | Request fields (besides name, role)                                         |
|------------------------|-------------------------------------------|-----------------------------------------------------------------------------|
| Navigation_Only        | Search                                    | current_location, destination                                               |
| Eligibility_Check      | Logic_KB                                  | query, e.g. `UsesLab(DrKhan, Lab1)`                                          |
| Booking_or_Scheduling  | Logic_KB -> CSP -> (Search)               | category, preferred_slot; optional group_id, current_location, note         |
| Urgent_Service_Request | ANN -> Logic_KB -> CSP -> (Search)        | category, current_location, severity, time_sensitivity, crowd_level; optional preferred_slot |
| Full_Service_Request   | ANN -> Logic_KB -> CSP -> Search          | category, current_location, preferred_slot, severity, time_sensitivity, crowd_level; optional note |

- Categories: AI_Lab_Support, Viva_Scheduling, Access_Request, Maintenance, Emergency_Help
- Roles: student, instructor, staff. Preferred slot 1-4 (09:00, 11:00, 14:00, 16:00). Levels 1-10.
- Search is optional when no `current_location` is given (no route guidance needed).
- Logic_KB is the gatekeeper ("Eligible / Allowed?"): if it says no, CSP and Search never run.
- `decision` is `completed`, `answered`, `accepted` or `rejected` (with `rejected_at` and the reason).

## How to run

```
pip install -r requirements.txt
python main.py
```

The ANN trains automatically the first time (about 5 seconds). Then use the menu:

1. Submit a new request (structured form: name, role, request type, then that type's fields)
2. Run demo requests (one per request type, plus rejections at Logic_KB, CSP and preprocessing)
3. Process batch CSV (`data/campus_requests.csv`)
4. Train / evaluate ANN models (accuracy, confusion matrix, plots)
5. Knowledge base: facts, rules, forward chaining, backward-chaining queries
6. Campus navigation: compare BFS, DFS, DLS, IDS, Bidirectional BFS, UCS, Greedy Best-First, A* and RBFS
7. CSP exam timetable and room bookings

### Web UI (Streamlit)

```
python -m streamlit run app.py
```

This opens the browser at http://localhost:8501. The sidebar has the same 7 functions as the CLI menu. The CLI
and the web UI share all AI modules, so you can use either one.

The web UI uses the **Campus Line (Forest & Beige)** design: the five request types are five transit lines
and the AI modules are the stations they stop at. A result shows the request travelling its line station by
station (skipped stops are dashed, a failure ends at a red buffer stop), and the campus map, priority network,
constraint graph and booking grid are drawn live from the real modules. All styling and drawing code is in `ui.py`.

- **Step-by-step search decisions** (Navigation page, and the Search tab of any routed request): a decision
  graph separate from the final route shows, for every step of BFS, UCS or A*, which node is expanded and why
  (FIFO order, lowest g, or lowest f = g + h), what happens to each neighbour (added, updated or ignored, with
  its g, h and f), and the queue left for the next step. Use Back / Next, the step slider or Play all.
- **Clickable tables**: tick a row in History, Batch results or Home's last 10 requests to open that result.
- **Chart images**: route maps, search comparisons, the room schedule and ANN training charts are saved to
  `outputs/plots` and can be downloaded from the page.

`python generate_report.py` rebuilds the Word/PDF report with screenshots and the full source code.

## Files

| File                   | Module                                                         |
|------------------------|----------------------------------------------------------------|
| `main.py`              | CLI menu (entry point)                                         |
| `app.py`               | Streamlit web UI (second entry point)                          |
| `ui.py`                | Web UI design: theme tokens, CSS, HTML/SVG drawings            |
| `config.py`            | Standard templates, roles, categories, rooms, slots, encodings |
| `preprocessing.py`     | 3.1 Input & Preprocessing                                      |
| `router.py`            | 3.2 Request Router                                             |
| `ann_module.py`        | 3.3 ANN: Perceptron + MLP 7-4-3-4 (numpy, from scratch)        |
| `knowledge_base.py`    | 3.4 Logic / KB: FOL, unification, forward and backward chaining |
| `csp_scheduler.py`     | 3.5 CSP: AC-3, backtracking, MRV, LCV, forward checking        |
| `search_navigation.py` | 3.6 Search: policy BFS / A* / UCS + 6 comparison algorithms    |
| `response_layer.py`    | 3.7 Final Response Layer + CSV log                             |
| `visualizer.py`        | matplotlib charts and console screenshots                      |
| `generate_report.py`   | Builds the project report                                      |

## Sample users in the knowledge base

- **Ali**: student, enrolled in AI (may use Lab1 / AI lab support), group G1.
- **Sara**: student with no completed courses, group G2. Use her to see rejections.
- **Ahmed** (group G4), **Fatima** (group G3): students with completed courses.
- **DrKhan**, **DrAyesha**: instructors (may request for any group).
- **Bilal**: staff.

Any other name is unregistered, so the role check rejects the request.
