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

`python generate_report.py` rebuilds the Word/PDF report with screenshots and the full source code.

## Files

| File                   | Module                                                         |
|------------------------|----------------------------------------------------------------|
| `main.py`              | CLI menu (entry point)                                         |
| `app.py`               | Streamlit web UI (second entry point)                          |
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
