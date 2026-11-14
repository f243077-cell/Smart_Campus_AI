"""
config.py
-----------------------------------------------------------------------
Central configuration for the Smart Campus AI Decision Support and
Automation System. Every module imports its constants from here so that
roles, request types, encodings, rooms, slots and file paths are defined
in exactly one place.
-----------------------------------------------------------------------
"""

import os

# -------------------------------------------------------------------
# File system paths
# -------------------------------------------------------------------
PROJECT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
DATA_DIRECTORY = os.path.join(PROJECT_DIRECTORY, "data")
OUTPUT_DIRECTORY = os.path.join(PROJECT_DIRECTORY, "outputs")
MODEL_DIRECTORY = os.path.join(OUTPUT_DIRECTORY, "models")
PLOT_DIRECTORY = os.path.join(OUTPUT_DIRECTORY, "plots")
SCREENSHOT_DIRECTORY = os.path.join(OUTPUT_DIRECTORY, "screenshots")

ANN_DATASET_PATH = os.path.join(DATA_DIRECTORY, "ann_training_data.csv")
BATCH_REQUEST_PATH = os.path.join(DATA_DIRECTORY, "campus_requests.csv")
RESULTS_LOG_PATH = os.path.join(OUTPUT_DIRECTORY, "results_log.csv")
# "_v2": models trained on the handout schema (categories, levels 1-10, raw x)
PERCEPTRON_WEIGHTS_PATH = os.path.join(MODEL_DIRECTORY, "perceptron_weights_v2.npz")
MLP_WEIGHTS_PATH = os.path.join(MODEL_DIRECTORY, "mlp_weights_v2.npz")

# -------------------------------------------------------------------
# Users, roles and request types (handout: Standard Python Data Structures)
# -------------------------------------------------------------------
VALID_ROLES = ["student", "instructor", "staff"]

REQUEST_TYPES = [
    "Navigation_Only",
    "Eligibility_Check",
    "Booking_or_Scheduling",
    "Urgent_Service_Request",
    "Full_Service_Request",
]

# Friendly aliases the user may type; all are mapped to canonical names.
REQUEST_TYPE_ALIASES = {
    "navigation": "Navigation_Only", "nav": "Navigation_Only", "route": "Navigation_Only",
    "eligibility": "Eligibility_Check", "eligible": "Eligibility_Check",
    "booking": "Booking_or_Scheduling", "scheduling": "Booking_or_Scheduling",
    "schedule": "Booking_or_Scheduling",
    "urgent": "Urgent_Service_Request", "urgent_service": "Urgent_Service_Request",
    "full": "Full_Service_Request", "full_service": "Full_Service_Request",
}

# Service categories (the ANN's "RequestType" feature uses this encoding).
CATEGORIES = ["AI_Lab_Support", "Viva_Scheduling", "Access_Request", "Maintenance", "Emergency_Help"]
CATEGORY_ALIASES = {
    "ai lab": "AI_Lab_Support", "lab": "AI_Lab_Support", "lab support": "AI_Lab_Support",
    "viva": "Viva_Scheduling", "access": "Access_Request",
    "repair": "Maintenance", "emergency": "Emergency_Help", "medical": "Emergency_Help",
}

# Universal field pool (handout 2.3). Not every request uses every field.
REQUEST_TEMPLATE = {
    "request_id": "",
    "name": "",
    "role": "",
    "request_type": "",
    "category": "",
    "current_location": "",
    "destination": "",
    "preferred_slot": None,
    "severity": 0,
    "time_sensitivity": 0,
    "crowd_level": 0,
    "group_id": "",
    "query": "",
    "eligibility_claim": False,
    "description_note": "",
}

# Fields every request must contain, and the extra fields per request type
# (handout 2.1 / 2.2). Optional fields are listed separately.
COMMON_REQUIRED_FIELDS = ["name", "role", "request_type"]
TYPE_REQUIRED_FIELDS = {
    "Navigation_Only": ["current_location", "destination"],
    "Eligibility_Check": ["query"],
    "Booking_or_Scheduling": ["category", "preferred_slot"],
    "Urgent_Service_Request": ["category", "current_location", "severity", "time_sensitivity", "crowd_level"],
    "Full_Service_Request": ["category", "current_location", "preferred_slot", "severity",
                             "time_sensitivity", "crowd_level"],
}
TYPE_OPTIONAL_FIELDS = {
    "Navigation_Only": [],
    "Eligibility_Check": [],
    "Booking_or_Scheduling": ["group_id", "current_location", "description_note"],
    "Urgent_Service_Request": ["preferred_slot", "group_id", "description_note"],
    "Full_Service_Request": ["group_id", "description_note"],
}

# -------------------------------------------------------------------
# Router: pipelines per request type (handout 7.1). An optional module
# runs only when the request needs it (Search: route guidance wanted).
# -------------------------------------------------------------------
PIPELINE_TEMPLATES = {
    "Navigation_Only": [("Search", False)],
    "Eligibility_Check": [("Logic_KB", False)],
    "Booking_or_Scheduling": [("Logic_KB", False), ("CSP", False), ("Search", True)],
    "Urgent_Service_Request": [("ANN", False), ("Logic_KB", False), ("CSP", False), ("Search", True)],
    "Full_Service_Request": [("ANN", False), ("Logic_KB", False), ("CSP", False), ("Search", False)],
}

# Standard output templates (handout 12.2 - 12.7)
ROUTER_OUTPUT_TEMPLATE = {"request_id": "", "selected_pipeline": [], "needs_ann": False,
                          "needs_logic": False, "needs_csp": False, "needs_search": False}
PRIORITY_OUTPUT_TEMPLATE = {"binary_priority": "", "final_priority": "", "confidence": 0.0}
LOGIC_OUTPUT_TEMPLATE = {"allowed": False, "entailed": False, "explanation": ""}
CSP_OUTPUT_TEMPLATE = {"decision": "", "assigned_room": "", "assigned_slot": None, "destination": "", "notes": ""}
SEARCH_OUTPUT_TEMPLATE = {"algorithm_used": "", "path": [], "cost": 0, "steps": 0}
FINAL_RESPONSE_TEMPLATE = {"request_id": "", "decision": "", "priority": {}, "eligibility": {},
                           "assignment": {}, "route": {}, "message": ""}

# -------------------------------------------------------------------
# Search policy (handout 7.2): unweighted -> BFS, weighted + heuristic ->
# A*, weighted without heuristic -> UCS.
# -------------------------------------------------------------------
GRAPH_TYPES = ["weighted", "unweighted"]
DEFAULT_GRAPH_TYPE = "weighted"
USE_HEURISTIC = True
DEPTH_LIMIT = 3                         # default limit for Depth-Limited Search (comparison mode)

# -------------------------------------------------------------------
# Rooms / resources used by the CSP scheduler
# -------------------------------------------------------------------
ROOMS = {
    "Study_Room_L1":  {"type": "StudyRoom",   "building": "Library",          "capacity": 6},
    "Study_Room_L2":  {"type": "StudyRoom",   "building": "Library",          "capacity": 10},
    "Seminar_Hall":   {"type": "SeminarRoom", "building": "Seminar_Room",     "capacity": 60},
    "Lab1":           {"type": "ComputerLab", "building": "AI_Lab",           "capacity": 30},
    "Science_Lab":    {"type": "ComputerLab", "building": "Science_Block",    "capacity": 40},
    "Meeting_Room_A": {"type": "MeetingRoom", "building": "Admin_Block",      "capacity": 12},
    "Meeting_Room_B": {"type": "MeetingRoom", "building": "Student_Services", "capacity": 8},
    "Help_Desk":      {"type": "ServiceDesk", "building": "Student_Services", "capacity": 4},
    "Medical_Room":   {"type": "MedicalRoom", "building": "Medical_Center",   "capacity": 4},
    "Exam_Hall_Main": {"type": "ExamHall",    "building": "Exam_Hall",        "capacity": 120},
    "Exam_Room_S1":   {"type": "ExamHall",    "building": "Science_Block",    "capacity": 50},
}

# Which rooms can serve each category (CSP domain before constraints).
CATEGORY_ROOMS = {
    "AI_Lab_Support": ["Lab1", "Science_Lab"],
    "Viva_Scheduling": ["Seminar_Hall", "Exam_Room_S1", "Meeting_Room_A", "Meeting_Room_B"],
    "Access_Request": ["Study_Room_L1", "Study_Room_L2"],
    "Maintenance": ["Help_Desk", "Meeting_Room_B"],
    "Emergency_Help": ["Medical_Room", "Help_Desk"],
}

# Exam / viva groups (same as the CSP diagram); used by group_id requests.
GROUP_IDS = ["G1", "G2", "G3", "G4", "G5", "G6"]

# Time slots are numbered 1-4 in requests (handout: preferred_slot in {1,2,3,4}).
TIME_SLOTS = ["09:00-10:30", "11:00-12:30", "14:00-15:30", "16:00-17:30"]
SLOT_NUMBERS = list(range(1, len(TIME_SLOTS) + 1))

# Slots (1-4) in which a user already has a fixed commitment (cannot book then).
USER_BUSY_SLOTS = {
    "Ali": [1],
    "DrKhan": [2],
}

# -------------------------------------------------------------------
# ANN feature encodings (handout 12.8 / 12.9, fixed order x1..x7)
# -------------------------------------------------------------------
FEATURE_ORDER = ["Role", "RequestType", "Severity", "TimeSensitivity", "CrowdLevel", "Distance", "Eligibility"]
ANN_FEATURE_NAMES = FEATURE_ORDER
ROLE_ENCODING = {"student": 0, "instructor": 1, "staff": 2}
REQUEST_TYPE_ENCODING = {category: index for index, category in enumerate(CATEGORIES)}
PRIORITY_CLASSES = ["urgent", "high", "normal", "low"]
BINARY_CLASSES = ["not_urgent", "urgent"]
LEVEL_MIN, LEVEL_MAX = 1, 10            # severity / time sensitivity / crowd level scale
MAX_ROUTE_DISTANCE = 15.0               # largest route cost, used to scale the distance feature
MLP_HIDDEN_LAYERS = (4, 3)              # matches the MLP diagram: 7 -> 4 -> 3 -> 4
RANDOM_SEED = 42


def ensure_directories():
    """
    Function: ensure_directories
    Purpose : Create the data/output/model/plot/screenshot folders if they
              do not exist yet, so later file writes never fail.
    Returns : None
    """
    for directory_path in (DATA_DIRECTORY, OUTPUT_DIRECTORY, MODEL_DIRECTORY,
                           PLOT_DIRECTORY, SCREENSHOT_DIRECTORY):
        os.makedirs(directory_path, exist_ok=True)
