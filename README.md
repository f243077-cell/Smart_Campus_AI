# Smart Campus AI Decision Support and Automation System

AL2002 Artificial Intelligence Lab (Fall 2026), semester capstone project.

A user submits a structured request. **Input & Preprocessing** validates it, the **Request Router**
picks the pipeline, the selected AI modules run, and the **Final Response Layer** combines their
outputs into one explained answer.

| Request type           | Pipeline                          |
|------------------------|-----------------------------------|
| Navigation_Only        | Search                            |
| Eligibility_Check      | Logic                             |
| Booking_or_Scheduling  | Logic -> CSP                      |
| Urgent_Service_Request | ANN -> Logic -> Search            |
| Full_Service_Request   | ANN -> Logic -> CSP -> Search     |

## How to run

```
pip install -r requirements.txt
python main.py
```

The ANN trains automatically the first time (about 5 seconds). Then use the menu:

1. Submit a new request (guided form)
2. Run demo requests (one per request type, plus a rejection and an invalid input)
3. Process batch CSV (`data/campus_requests.csv`)
4. Train / evaluate ANN models (accuracy, confusion matrix, plots)
5. Knowledge base: facts, rules, forward chaining, backward-chaining queries
6. Campus navigation: BFS vs UCS vs A*
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
| `config.py`            | Roles, request types, rooms, slots, encodings, paths           |
| `preprocessing.py`     | 3.1 Input & Preprocessing                                      |
| `router.py`            | 3.2 Request Router                                             |
| `ann_module.py`        | 3.3 ANN: Perceptron + MLP 7-4-3-4 (numpy, from scratch)        |
| `knowledge_base.py`    | 3.4 Logic / KB: FOL, unification, forward and backward chaining |
| `csp_scheduler.py`     | 3.5 CSP: AC-3, backtracking, MRV, LCV, forward checking        |
| `search_navigation.py` | 3.6 Search: BFS, UCS, A* on the campus graph                   |
| `response_layer.py`    | 3.7 Final Response Layer + CSV log                             |
| `visualizer.py`        | matplotlib charts and console screenshots                      |
| `generate_report.py`   | Builds the project report                                      |

## Sample users in the knowledge base

- **Ali**: Student. Completed ProgrammingFundamentals and enrolled in AI.
- **Sara**: Student with no completed courses. Use her to see rejections.
- **Ahmed**, **Fatima**: Students with completed courses.
- **DrKhan**, **DrAyesha**: Instructors.
- **Bilal**: Staff.

Any other user id is unregistered, so the role check rejects the request.
