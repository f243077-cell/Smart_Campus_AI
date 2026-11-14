"""
visualizer.py
-----------------------------------------------------------------------
All matplotlib figures of the project. Every function saves a PNG in
outputs/plots (or outputs/screenshots) and returns the file path, so
the figures can be opened from the CLI and embedded in the report.
-----------------------------------------------------------------------
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")                      # render to files; works without a display
import matplotlib.pyplot as plt            # noqa: E402
import numpy as np                         # noqa: E402

import config                              # noqa: E402

# Colour palettes. Light is the default (CLI and report); the Streamlit UI
# switches to dark when the user picks the dark theme.
LIGHT_PALETTE = {
    "background": "#FFFFFF", "panel": "#FFFFFF", "text": "#1A1A1A", "axis": "#555555", "grid": "#CCCCCC",
    "edge": "#9A9A9A", "edge_label_background": "#FFFFFF",
    "node": "#FDF6D8", "node_border": "#555555",
    "route": "#D62728", "route_fill": "#FFD6D6",
    "current_fill": "#FFE08A", "current_border": "#B8860B",
    "expanded_fill": "#CFE2FF", "expanded_border": "#1F77B4",
    "exam_cell": "#C6DBEF", "booked_cell": "#FDD0A2", "free_cell": "#F2F2F2", "cell_border": "#777777",
}
DARK_PALETTE = {
    "background": "#0E1726", "panel": "#1A2538", "text": "#E6EDF3", "axis": "#8B9BB4", "grid": "#2A3A52",
    "edge": "#5B6B84", "edge_label_background": "#0E1726",
    "node": "#1A2538", "node_border": "#4A5A72",
    "route": "#F85149", "route_fill": "#5C1F24",
    "current_fill": "#5C4A12", "current_border": "#E3B341",
    "expanded_fill": "#173A5E", "expanded_border": "#58A6FF",
    "exam_cell": "#1D3B5C", "booked_cell": "#5A3A1A", "free_cell": "#16213A", "cell_border": "#2A3A52",
}
palette = dict(LIGHT_PALETTE)
_dark_mode = False


def set_dark_mode(enabled):
    """
    Function: set_dark_mode
    Purpose : Switch every later figure between the light and dark palette
              (matplotlib rcParams plus the colours used by the map and
              schedule plots). Dark figures are saved with a '_dark' suffix
              so light and dark images never overwrite each other.
    """
    global _dark_mode
    _dark_mode = bool(enabled)
    palette.clear()
    palette.update(DARK_PALETTE if _dark_mode else LIGHT_PALETTE)
    plt.rcParams.update({
        "figure.facecolor": palette["background"], "axes.facecolor": palette["background"],
        "savefig.facecolor": palette["background"], "text.color": palette["text"],
        "axes.labelcolor": palette["text"], "axes.titlecolor": palette["text"],
        "axes.edgecolor": palette["axis"], "xtick.color": palette["text"], "ytick.color": palette["text"],
        "grid.color": palette["grid"], "legend.facecolor": palette["panel"],
        "legend.edgecolor": palette["axis"], "legend.labelcolor": palette["text"],
    })


def _save_figure(figure, file_name, directory=config.PLOT_DIRECTORY):
    """
    Function: _save_figure
    Purpose : Save a figure as PNG, close it, and return the saved path.
    """
    os.makedirs(directory, exist_ok=True)
    if _dark_mode:
        stem, extension = os.path.splitext(file_name)
        file_name = f"{stem}_dark{extension}"
    file_path = os.path.join(directory, file_name)
    figure.savefig(file_path, dpi=130, bbox_inches="tight")
    plt.close(figure)
    return file_path


def _edge_curve(start, end, other_points, bulge=0.7):
    """
    Function: _edge_curve
    Purpose : Points for drawing one edge. A straight line is used unless
              another building lies on it (e.g. Admin_Block-Exam_Hall passes
              Student_Services); then a quadratic curve bending upward is
              drawn so the edge and its label stay visible.
    Returns : (x_values, y_values, label_point)
    """
    direction = end - start
    length = np.linalg.norm(direction)
    blocked = False
    for point in other_points:
        along = np.dot(point - start, direction) / (length ** 2)
        distance_to_line = abs(direction[0] * (point - start)[1] - direction[1] * (point - start)[0]) / length
        if 0.05 < along < 0.95 and distance_to_line < 0.3:
            blocked = True
            break
    if not blocked:
        return [start[0], end[0]], [start[1], end[1]], (start + end) / 2
    normal = np.array([-direction[1], direction[0]]) / length
    if normal[1] < 0:
        normal = -normal
    control = (start + end) / 2 + normal * bulge * 2
    t_values = np.linspace(0, 1, 40)[:, None]
    curve = (1 - t_values) ** 2 * start + 2 * (1 - t_values) * t_values * control + t_values ** 2 * end
    return curve[:, 0], curve[:, 1], curve[20]


def plot_campus_graph(campus_graph, route_path=None, title="Campus Map", file_name="campus_map.png",
                      expanded_nodes=None, current_node=None):
    """
    Function: plot_campus_graph
    Purpose : Draw the weighted campus graph using building coordinates and
              highlight a route (list of nodes) in red if given. For search
              animations, expanded nodes are shaded blue and the node being
              expanded right now is shaded yellow.
    """
    expanded_nodes = set(expanded_nodes or [])
    figure, axis = plt.subplots(figsize=(15, 7))
    x_stretch = 1.7                        # display only: spreads close buildings apart
    position = {node: np.array([x * x_stretch, y], dtype=float)
                for node, (x, y) in campus_graph.coordinates.items()}
    route_edges = set()
    if route_path and len(route_path) > 1:
        route_edges = {frozenset(pair) for pair in zip(route_path[:-1], route_path[1:])}
    for node_a, node_b, cost in campus_graph.edges:
        start, end = position[node_a], position[node_b]
        curve_x, curve_y, label_point = _edge_curve(start, end, [p for n, p in position.items()
                                                                 if n not in (node_a, node_b)])
        on_route = frozenset((node_a, node_b)) in route_edges
        axis.plot(curve_x, curve_y, color=palette["route"] if on_route else palette["edge"],
                  linewidth=3.5 if on_route else 1.2, zorder=1)
        axis.text(label_point[0], label_point[1], str(cost), fontsize=9, ha="center", va="center",
                  bbox=dict(boxstyle="round,pad=0.15", facecolor=palette["edge_label_background"],
                            edgecolor="none"), zorder=2)
    for node_name, (x_value, y_value) in campus_graph.coordinates.items():
        if route_path and node_name in route_path:
            face_colour, border_colour, border_width = palette["route_fill"], palette["route"], 1.6
        elif node_name == current_node:
            face_colour, border_colour, border_width = palette["current_fill"], palette["current_border"], 2.0
        elif node_name in expanded_nodes:
            face_colour, border_colour, border_width = palette["expanded_fill"], palette["expanded_border"], 1.4
        else:
            face_colour, border_colour, border_width = palette["node"], palette["node_border"], 0.8
        axis.text(x_value * x_stretch, y_value, f"{node_name}\n({x_value},{y_value})", fontsize=8.5,
                  ha="center", va="center", zorder=3,
                  bbox=dict(boxstyle="round,pad=0.35", linewidth=border_width,
                            facecolor=face_colour, edgecolor=border_colour))
    axis.set_title(title, fontsize=13, fontweight="bold")
    axis.set_xlim(-1.2, 10 * x_stretch + 1.2)
    axis.set_ylim(-0.8, 6.6)
    axis.axis("off")
    return _save_figure(figure, file_name)


def plot_algorithm_comparison(results, file_name="search_comparison.png"):
    """
    Function: plot_algorithm_comparison
    Purpose : Grouped bar chart of path cost, stops and nodes expanded for
              BFS, UCS and A* on the same query.
    """
    algorithms = [result["algorithm"] for result in results]
    metrics = {"Path cost": [r["cost"] for r in results],
               "Stops": [r["hops"] for r in results],
               "Nodes expanded": [r["nodes_expanded"] for r in results]}
    positions = np.arange(len(algorithms))
    bar_width = 0.25
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for offset, (metric_name, values) in enumerate(metrics.items()):
        bars = axis.bar(positions + (offset - 1) * bar_width, values, bar_width, label=metric_name)
        axis.bar_label(bars, fontsize=9)
    axis.set_xticks(positions, algorithms)
    path_text = f"{results[0]['path'][0]} -> {results[0]['path'][-1]}"
    axis.set_title(f"BFS vs UCS vs A*  ({path_text})", fontweight="bold")
    axis.legend()
    axis.spines[["top", "right"]].set_visible(False)
    return _save_figure(figure, file_name)


def plot_perceptron_errors(errors_per_epoch, file_name="perceptron_training.png"):
    """
    Function: plot_perceptron_errors
    Purpose : Line chart of perceptron misclassifications per epoch.
    """
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(range(1, len(errors_per_epoch) + 1), errors_per_epoch, color="#1f77b4", linewidth=1.8)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Weight updates (mistakes)")
    axis.set_title("Perceptron (binary baseline) - training errors", fontweight="bold")
    axis.grid(alpha=0.3)
    return _save_figure(figure, file_name)


def plot_mlp_history(history, file_name="mlp_training.png"):
    """
    Function: plot_mlp_history
    Purpose : Two panels - cross-entropy loss, and train/test accuracy per epoch.
    """
    figure, (loss_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["loss"]) + 1)
    loss_axis.plot(epochs, history["loss"], color="#d62728")
    loss_axis.set_title("MLP 7-4-3-4 - cross-entropy loss", fontweight="bold")
    loss_axis.set_xlabel("Epoch")
    loss_axis.grid(alpha=0.3)
    accuracy_axis.plot(epochs, history["train_accuracy"], label="train")
    if history.get("test_accuracy"):
        accuracy_axis.plot(epochs, history["test_accuracy"], label="test")
    accuracy_axis.set_title("MLP accuracy", fontweight="bold")
    accuracy_axis.set_xlabel("Epoch")
    accuracy_axis.set_ylim(0, 1)
    accuracy_axis.legend()
    accuracy_axis.grid(alpha=0.3)
    return _save_figure(figure, file_name)


def plot_confusion_matrix(matrix, class_names, title, file_name):
    """
    Function: plot_confusion_matrix
    Purpose : Heatmap of a confusion matrix with the count in every cell.
    """
    matrix = np.asarray(matrix)
    figure, axis = plt.subplots(figsize=(5.5, 4.8))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046)
    axis.set_xticks(range(len(class_names)), class_names)
    axis.set_yticks(range(len(class_names)), class_names)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, matrix[row, column], ha="center", va="center",
                      color="white" if matrix[row, column] > threshold else "black")
    axis.set_title(title, fontweight="bold")
    return _save_figure(figure, file_name)


def plot_class_distribution(class_counts, file_name="priority_distribution.png"):
    """
    Function: plot_class_distribution
    Purpose : Bar chart of how many training samples fall in each priority.
    """
    figure, axis = plt.subplots(figsize=(6, 4))
    colours = ["#d62728", "#ff7f0e", "#1f77b4", "#2ca02c"]
    bars = axis.bar(class_counts.index.astype(str), class_counts.values, color=colours[:len(class_counts)])
    axis.bar_label(bars)
    axis.set_title("ANN training data - priority classes", fontweight="bold")
    axis.set_ylabel("Samples")
    axis.spines[["top", "right"]].set_visible(False)
    return _save_figure(figure, file_name)


def plot_schedule(booking_table, file_name="room_schedule.png", title="Campus room schedule (CSP)"):
    """
    Function: plot_schedule
    Purpose : Grid of time slots x rooms showing who occupies each cell.
    """
    rooms = list(config.ROOMS)
    slots = config.TIME_SLOTS
    figure, axis = plt.subplots(figsize=(13, 4.8))
    for column, room in enumerate(rooms):
        for row, slot in enumerate(slots):
            match = booking_table[(booking_table["Room"] == room) & (booking_table["Slot"] == slot)]
            occupied = not match.empty
            label = match.iloc[0]["Booked_By"] if occupied else "free"
            if occupied and label.startswith("Exam"):
                colour = palette["exam_cell"]
            elif occupied:
                colour = palette["booked_cell"]
            else:
                colour = palette["free_cell"]
            axis.add_patch(plt.Rectangle((column, row), 0.96, 0.92, facecolor=colour,
                                         edgecolor=palette["cell_border"]))
            axis.text(column + 0.48, row + 0.46, label.replace(" (", "\n("), ha="center", va="center", fontsize=7.5)
    axis.set_xlim(0, len(rooms))
    axis.set_ylim(len(slots), 0)
    axis.set_xticks(np.arange(len(rooms)) + 0.48, [r.replace("_", "\n") for r in rooms], fontsize=8)
    axis.set_yticks(np.arange(len(slots)) + 0.46, slots, fontsize=9)
    axis.tick_params(length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.set_title(title, fontweight="bold")
    return _save_figure(figure, file_name)


def render_console_screenshot(console_text, file_name, title="Smart Campus AI - console"):
    """
    Function: render_console_screenshot
    Purpose : Render captured console output as a terminal-style PNG so it
              can be used as a screenshot in the report.
    """
    lines = console_text.rstrip("\n").splitlines() or [""]
    longest = max(len(line) for line in lines)
    width = max(8.0, min(0.075 * longest + 0.6, 16.0))
    height = 0.19 * len(lines) + 0.7
    figure = plt.figure(figsize=(width, height))
    figure.patch.set_facecolor("#1e1e1e")
    figure.text(0.012, 1 - 0.18 / height, title, color="#9cdcfe", fontsize=9,
                family="monospace", va="top", fontweight="bold")
    figure.text(0.012, 1 - 0.5 / height, "\n".join(lines), color="#e6e6e6", fontsize=8.5,
                family="monospace", va="top", linespacing=1.25)
    return _save_figure(figure, file_name, config.SCREENSHOT_DIRECTORY)


def open_image(file_path):
    """
    Function: open_image
    Purpose : Open a saved PNG with the operating system's image viewer.
              Failures are reported but never crash the program.
    """
    try:
        if sys.platform.startswith("win"):
            os.startfile(file_path)              # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            os.system(f'open "{file_path}"')
        else:
            os.system(f'xdg-open "{file_path}" >/dev/null 2>&1 &')
    except OSError as error:
        print(f"  Could not open image automatically ({error}). File: {file_path}")
