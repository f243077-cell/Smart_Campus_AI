"""
preprocessing.py
-----------------------------------------------------------------------
Module 3.1 - Input & Preprocessing
First stage of the platform. It:
  1. checks that all required fields are present,
  2. normalizes values (case, aliases, building names, numbers),
  3. validates ranges and allowed options,
  4. builds the standard request object, and
  5. prepares the ANN feature vector when the request needs the ANN.
-----------------------------------------------------------------------
"""

import itertools

import numpy as np

import config


class RequestValidationError(Exception):
    """Raised when a request is missing fields or contains invalid values."""


_request_counter = itertools.count(1)


def _clean_text(value):
    """
    Function: _clean_text
    Purpose : Convert a raw value to a stripped string ('' for None/NaN).
    """
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value).strip()


def _lookup_key(text):
    """
    Function: _lookup_key
    Purpose : Build a comparison key that ignores case, spaces, dashes and
              underscores, so 'ai lab', 'AI-Lab' and 'AI_Lab' all match.
    """
    return "".join(character for character in text.lower() if character.isalnum())


def normalize_choice(raw_value, allowed_values, field_name, aliases=None):
    """
    Function: normalize_choice
    Purpose : Match free text against a list of allowed values (and an
              optional alias dictionary). Raises a clear error otherwise.
    """
    text = _clean_text(raw_value)
    if not text:
        raise RequestValidationError(f"Field '{field_name}' is empty.")
    key = _lookup_key(text)
    if aliases:
        for alias, canonical in aliases.items():
            if _lookup_key(alias) == key:
                return canonical
    for allowed in allowed_values:
        if _lookup_key(allowed) == key:
            return allowed
    raise RequestValidationError(f"Invalid {field_name} '{text}'. "
                                 f"Allowed values: {', '.join(allowed_values)}")


def normalize_user_id(raw_value):
    """
    Function: normalize_user_id
    Purpose : User ids must be a single word of letters/digits; returned
              with the first letter capitalized (ali -> Ali, drkhan -> DrKhan
              keeps internal capitals when given).
    """
    text = _clean_text(raw_value)
    if not text:
        raise RequestValidationError("Field 'user_id' is empty.")
    if not text.replace("_", "").isalnum():
        raise RequestValidationError(f"user_id '{text}' may contain only letters, digits or '_'.")
    return text[0].upper() + text[1:]


def parse_integer(raw_value, field_name, minimum, maximum):
    """
    Function: parse_integer
    Purpose : Convert text to int and check minimum <= value <= maximum.
    """
    text = _clean_text(raw_value)
    try:
        number = float(text)
    except ValueError:
        raise RequestValidationError(f"Field '{field_name}' must be a whole number, got '{text}'.") from None
    if not number.is_integer():
        raise RequestValidationError(f"Field '{field_name}' must be a whole number, got '{text}'.")
    number = int(number)
    if not minimum <= number <= maximum:
        raise RequestValidationError(f"Field '{field_name}' must be between {minimum} and {maximum}, got {number}.")
    return number


def validate_request(raw_request, campus_locations, known_targets=None):
    """
    Function: validate_request
    Purpose : Validate and normalize a raw request dictionary and return
              the standard request object used by every other module.
    Inputs  : raw_request      - dict from CLI input or a CSV row
              campus_locations - list of valid building names (graph nodes)
              known_targets    - valid Eligibility_Check targets (optional)
    Returns : dict (standard request object)
    Raises  : RequestValidationError with a readable message
    """
    if not isinstance(raw_request, dict):
        raise RequestValidationError("Request must be a dictionary of field -> value.")
    request = {key.strip().lower(): value for key, value in raw_request.items() if isinstance(key, str)}

    missing_common = [field for field in config.COMMON_REQUIRED_FIELDS if not _clean_text(request.get(field))]
    if missing_common:
        raise RequestValidationError(f"Missing required field(s): {', '.join(missing_common)}")

    request_type = normalize_choice(request["request_type"], config.REQUEST_TYPES,
                                    "request_type", config.REQUEST_TYPE_ALIASES)
    missing_specific = [field for field in config.TYPE_REQUIRED_FIELDS[request_type]
                        if not _clean_text(request.get(field))]
    if missing_specific:
        raise RequestValidationError(f"{request_type} requires field(s): {', '.join(missing_specific)}")

    standard_request = {
        "request_id": f"REQ-{next(_request_counter):04d}",
        "user_id": normalize_user_id(request["user_id"]),
        "role": normalize_choice(request["role"], config.VALID_ROLES, "role"),
        "request_type": request_type,
    }

    if _clean_text(request.get("source")):
        standard_request["source"] = normalize_choice(request["source"], campus_locations, "source location")
    if _clean_text(request.get("destination")):
        standard_request["destination"] = normalize_choice(request["destination"], campus_locations,
                                                           "destination location")

    route_text = _clean_text(request.get("route_preference"))
    standard_request["route_preference"] = (
        normalize_choice(route_text, config.ROUTE_PREFERENCES, "route_preference")
        if route_text else config.DEFAULT_ROUTE_PREFERENCE)

    if request_type == "Navigation_Only" and standard_request["source"] == standard_request["destination"]:
        raise RequestValidationError("Source and destination are the same location.")

    if request_type == "Eligibility_Check":
        target_text = _clean_text(request["target"])
        allowed_targets = list(known_targets) if known_targets else [target_text]
        standard_request["target"] = normalize_choice(target_text, allowed_targets, "eligibility target")

    if request_type in ("Booking_or_Scheduling", "Full_Service_Request"):
        standard_request["resource_type"] = normalize_choice(request["resource_type"],
                                                             config.RESOURCE_TYPES, "resource_type")
        standard_request["group_size"] = parse_integer(request["group_size"], "group_size", 1, 500)
        preferred_text = _clean_text(request.get("preferred_slot"))
        standard_request["preferred_slot"] = (
            normalize_choice(preferred_text, config.TIME_SLOTS, "preferred_slot") if preferred_text else None)
        sessions_text = _clean_text(request.get("sessions"))
        standard_request["sessions"] = (parse_integer(sessions_text, "sessions", 1, len(config.TIME_SLOTS))
                                        if sessions_text else 1)

    if request_type == "Urgent_Service_Request":
        standard_request["service_type"] = normalize_choice(request["service_type"],
                                                            list(config.SERVICE_DESTINATIONS), "service_type")
        if "destination" not in standard_request:
            standard_request["destination"] = config.SERVICE_DESTINATIONS[standard_request["service_type"]]

    if request_type in ("Urgent_Service_Request", "Full_Service_Request"):
        for level_field in ("severity", "time_sensitivity", "crowd_level"):
            standard_request[level_field] = parse_integer(request[level_field], level_field,
                                                          config.LEVEL_MIN, config.LEVEL_MAX)
    return standard_request


def build_feature_vector(standard_request, route_distance=0.0, eligibility_flag=0):
    """
    Function: build_feature_vector
    Purpose : Prepare the ANN input vector in the fixed order of the
              diagrams: x1 Role, x2 RequestType, x3 Severity,
              x4 TimeSensitivity, x5 CrowdLevel, x6 Distance, x7 Eligibility.
              Every value is min-max scaled to [0, 1].
    Inputs  : standard_request - validated request object
              route_distance   - weighted route cost to the destination
              eligibility_flag - 1 if the user is a registered campus user
    Returns : numpy array of shape (7,)
    """
    def scaled_level(field_name):
        """
        Function: scaled_level
        Purpose : Scale a 1-5 level field to [0, 1]; missing fields count
                  as the minimum level (requests without ANN inputs).
        """
        value = standard_request.get(field_name, config.LEVEL_MIN)
        return (value - config.LEVEL_MIN) / (config.LEVEL_MAX - config.LEVEL_MIN)

    return np.array([
        config.ROLE_ENCODING[standard_request["role"]] / (len(config.VALID_ROLES) - 1),
        config.REQUEST_TYPE_ENCODING[standard_request["request_type"]] / (len(config.REQUEST_TYPES) - 1),
        scaled_level("severity"),
        scaled_level("time_sensitivity"),
        scaled_level("crowd_level"),
        min(max(float(route_distance), 0.0) / config.MAX_ROUTE_DISTANCE, 1.0),
        float(1 if eligibility_flag else 0),
    ], dtype=float)
