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
PERCEPTRON_WEIGHTS_PATH = os.path.join(MODEL_DIRECTORY, "perceptron_weights.npz")
MLP_WEIGHTS_PATH = os.path.join(MODEL_DIRECTORY, "mlp_weights.npz")

# -------------------------------------------------------------------
# Users, roles and request types
# -------------------------------------------------------------------
VALID_ROLES = ["Student", "Instructor", "Staff"]

REQUEST_TYPES = [
    "Navigation_Only",
    "Eligibility_Check",
    "Booking_or_Scheduling",
    "Urgent_Service_Request",
    "Full_Service_Request",
]

# Friendly aliases the user may type; all are mapped to canonical names.
REQUEST_TYPE_ALIASES = {
    "navigation_only": "Navigation_Only",
    "navigation": "Navigation_Only",
    "nav": "Navigation_Only",
    "route": "Navigation_Only",
    "eligibility_check": "Eligibility_Check",
    "eligibility": "Eligibility_Check",
    "eligible": "Eligibility_Check",
    "booking_or_scheduling": "Booking_or_Scheduling",
    "booking": "Booking_or_Scheduling",
    "scheduling": "Booking_or_Scheduling",
    "schedule": "Booking_or_Scheduling",
    "urgent_service_request": "Urgent_Service_Request",
    "urgent": "Urgent_Service_Request",
    "emergency": "Urgent_Service_Request",
    "full_service_request": "Full_Service_Request",
    "full_service": "Full_Service_Request",
    "full": "Full_Service_Request",
}

# Fields every request must contain, and the extra fields per request type.
COMMON_REQUIRED_FIELDS = ["user_id", "role", "request_type"]
TYPE_REQUIRED_FIELDS = {
    "Navigation_Only": ["source", "destination"],
    "Eligibility_Check": ["target"],
    "Booking_or_Scheduling": ["resource_type", "group_size"],
    "Urgent_Service_Request": ["source", "service_type", "severity",
                               "time_sensitivity", "crowd_level"],
    "Full_Service_Request": ["source", "resource_type", "group_size", "severity",
                             "time_sensitivity", "crowd_level"],
}

# -------------------------------------------------------------------
# Route preferences (decide BFS / A* / UCS according to project policy)
# -------------------------------------------------------------------
ROUTE_PREFERENCES = ["distance", "stops", "distance_no_heuristic"]
DEFAULT_ROUTE_PREFERENCE = "distance"

# -------------------------------------------------------------------
# Urgent services and the campus building that handles each one
# -------------------------------------------------------------------
SERVICE_DESTINATIONS = {
    "Medical": "Medical_Center",
    "Security": "Admin_Block",
    "IT_Support": "Science_Block",
    "Maintenance": "Student_Services",
}

# -------------------------------------------------------------------
# Rooms / resources used by the CSP scheduler
# -------------------------------------------------------------------
RESOURCE_TYPES = ["StudyRoom", "SeminarRoom", "ComputerLab", "MeetingRoom", "ExamHall"]

ROOMS = {
    "Study_Room_L1":  {"type": "StudyRoom",   "building": "Library",          "capacity": 6},
    "Study_Room_L2":  {"type": "StudyRoom",   "building": "Library",          "capacity": 10},
    "Seminar_Hall":   {"type": "SeminarRoom", "building": "Seminar_Room",     "capacity": 60},
    "Lab1":           {"type": "ComputerLab", "building": "AI_Lab",           "capacity": 30},
    "Science_Lab":    {"type": "ComputerLab", "building": "Science_Block",    "capacity": 40},
    "Meeting_Room_A": {"type": "MeetingRoom", "building": "Admin_Block",      "capacity": 12},
    "Meeting_Room_B": {"type": "MeetingRoom", "building": "Student_Services", "capacity": 8},
    "Exam_Hall_Main": {"type": "ExamHall",    "building": "Exam_Hall",        "capacity": 120},
    "Exam_Room_S1":   {"type": "ExamHall",    "building": "Science_Block",    "capacity": 50},
}

TIME_SLOTS = ["09:00-10:30", "11:00-12:30", "14:00-15:30", "16:00-17:30"]

# Slots in which a user already has a fixed commitment (cannot book then).
USER_BUSY_SLOTS = {
    "Ali": ["09:00-10:30"],
    "DrKhan": ["11:00-12:30"],
}

# -------------------------------------------------------------------
# ANN feature encodings (fixed input order x1..x7 from the spec diagram)
# -------------------------------------------------------------------
ANN_FEATURE_NAMES = ["Role", "RequestType", "Severity", "TimeSensitivity",
                     "CrowdLevel", "Distance", "Eligibility"]
ROLE_ENCODING = {"Student": 0, "Instructor": 1, "Staff": 2}
REQUEST_TYPE_ENCODING = {name: index for index, name in enumerate(REQUEST_TYPES)}
PRIORITY_CLASSES = ["Urgent", "High", "Normal", "Low"]
LEVEL_MIN, LEVEL_MAX = 1, 5            # severity / time sensitivity / crowd level scale
MAX_ROUTE_DISTANCE = 15.0              # used to scale the distance feature to [0, 1]
MLP_HIDDEN_LAYERS = (4, 3)             # matches the MLP diagram: 7 -> 4 -> 3 -> 4
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
