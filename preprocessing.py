"""
preprocessing.py
-----------------------------------------------------------------------
Module 3.1 - Input & Preprocessing
First stage of the platform (handout 4 and 8). It:
  1. validates fields (required fields present, legal values, slot in
     1-4, levels in 1-10, known locations, valid request type),
  2. normalizes values ("ai lab" -> "AI_Lab", "urgent" ->
     "Urgent_Service_Request", "viva" -> "Viva_Scheduling"),
  3. takes the request type chosen explicitly by the user,
  4. builds ONE standard request object (universal template),
  5. prepares module inputs: logic query, search source/destination,
     ANN feature fields, CSP details and the needs_* flags,
  6. which the router then uses to decide the pipeline.
-----------------------------------------------------------------------
"""

import itertools

import numpy as np

import config
from knowledge_base import KnowledgeBaseError, format_atom, parse_atom


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


def normalize_name(raw_value):
    """
    Function: normalize_name
    Purpose : The requester's name must be a single word of letters/digits
              (it is used as a constant in the knowledge base); returned
              with the first letter capitalized (ali -> Ali, DrKhan stays).
    """
    text = _clean_text(raw_value)
    if not text:
        raise RequestValidationError("Field 'name' is empty.")
    if not text.replace("_", "").isalnum():
        raise RequestValidationError(f"name '{text}' may contain only letters, digits or '_' (one word).")
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


def parse_boolean(raw_value, field_name):
    """
    Function: parse_boolean
    Purpose : Read yes/no style input (true/false, yes/no, y/n, 1/0).
    """
    text = _clean_text(raw_value).lower()
    if text in ("true", "yes", "y", "1"):
        return True
    if text in ("false", "no", "n", "0"):
        return False
    raise RequestValidationError(f"Field '{field_name}' must be true or false, got '{raw_value}'.")


def validate_request(raw_request, campus_locations):
    """
    Function: validate_request
    Purpose : Steps 1-4 of preprocessing (handout 8.1): read the fields,
              validate them, normalize them to the project naming and build
              ONE standard request object with every field of the
              universal template (unused fields keep their empty value).
    Inputs  : raw_request      - dict from the CLI form, the web form or a CSV row
              campus_locations - list of valid building names (graph nodes)
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

    standard_request = dict(config.REQUEST_TEMPLATE)
    standard_request.update({
        "request_id": f"REQ{100 + next(_request_counter)}",
        "name": normalize_name(request["name"]),
        "role": normalize_choice(request["role"], config.VALID_ROLES, "role"),
        "request_type": request_type,
    })
    allowed_fields = set(config.TYPE_REQUIRED_FIELDS[request_type] + config.TYPE_OPTIONAL_FIELDS[request_type])

    def given(field_name):
        """
        Function: given
        Purpose : True when this request type uses the field and it has a value.
        """
        return field_name in allowed_fields and bool(_clean_text(request.get(field_name)))

    if given("category"):
        standard_request["category"] = normalize_choice(request["category"], config.CATEGORIES, "category",
                                                        config.CATEGORY_ALIASES)
    for location_field in ("current_location", "destination"):
        if given(location_field):
            standard_request[location_field] = normalize_choice(request[location_field], campus_locations,
                                                                location_field.replace("_", " "))
    if request_type == "Navigation_Only" and standard_request["current_location"] == standard_request["destination"]:
        raise RequestValidationError("current_location and destination are the same place.")
    if given("preferred_slot"):
        standard_request["preferred_slot"] = parse_integer(request["preferred_slot"], "preferred_slot",
                                                           config.SLOT_NUMBERS[0], config.SLOT_NUMBERS[-1])
    for level_field in ("severity", "time_sensitivity", "crowd_level"):
        if given(level_field):
            standard_request[level_field] = parse_integer(request[level_field], level_field,
                                                          config.LEVEL_MIN, config.LEVEL_MAX)
    if given("group_id"):
        standard_request["group_id"] = normalize_choice(request["group_id"], config.GROUP_IDS, "group_id")
    if given("query"):
        try:
            standard_request["query"] = format_atom(parse_atom(_clean_text(request["query"])))
        except KnowledgeBaseError as error:
            raise RequestValidationError(f"Invalid query: {error}") from None
    if _clean_text(request.get("eligibility_claim")):
        standard_request["eligibility_claim"] = parse_boolean(request["eligibility_claim"], "eligibility_claim")
    if given("description_note"):
        note = _clean_text(request["description_note"])
        if len(note) > 300:
            raise RequestValidationError("description_note may be at most 300 characters.")
        standard_request["description_note"] = note
    return standard_request


def prepare_module_inputs(standard_request):
    """
    Function: prepare_module_inputs
    Purpose : Step 5 of preprocessing (handout 4.5 / 8.1): mark which
              modules the request needs (needs_* flags) and prepare their
              inputs - the logic query, the search source/destination,
              the ANN fields and the CSP details.
    Returns : dict with needs_ann, needs_logic, needs_csp, needs_search
              and module_inputs
    """
    request_type = standard_request["request_type"]
    template = config.PIPELINE_TEMPLATES[request_type]
    modules = {module for module, _ in template}
    route_guidance = bool(standard_request["current_location"])
    needs_search = "Search" in modules and (request_type in ("Navigation_Only", "Full_Service_Request")
                                            or route_guidance)
    module_inputs = {}
    if "Logic_KB" in modules:
        module_inputs["logic_query"] = (standard_request["query"] if request_type == "Eligibility_Check"
                                        else f"CanRequest({standard_request['name']}, {standard_request['category']})")
    if needs_search:
        module_inputs["search"] = {"source": standard_request["current_location"],
                                   "destination": standard_request["destination"] or "assigned by CSP"}
    if "ANN" in modules:
        module_inputs["ann_fields"] = {field: standard_request[field] for field in
                                       ("role", "category", "severity", "time_sensitivity", "crowd_level")}
    if "CSP" in modules:
        module_inputs["csp"] = {"category": standard_request["category"],
                                "preferred_slot": standard_request["preferred_slot"],
                                "group_id": standard_request["group_id"]}
    return {"needs_ann": "ANN" in modules, "needs_logic": "Logic_KB" in modules,
            "needs_csp": "CSP" in modules, "needs_search": needs_search, "module_inputs": module_inputs}


def build_feature_vector(standard_request, route_distance=0.0, eligibility_flag=0):
    """
    Function: build_feature_vector
    Purpose : Prepare the ANN input in the fixed handout order
              x = [Role, RequestType, Severity, TimeSensitivity,
                   CrowdLevel, Distance, Eligibility]
              with the handout encodings, e.g. [0, 0, 8, 9, 5, 4, 1].
              RequestType is the service category (AI_Lab_Support = 0 ...).
    Inputs  : standard_request - validated request object
              route_distance   - weighted route cost to the service location
              eligibility_flag - 1 if the requester is eligible / registered
    Returns : list of 7 numbers
    """
    def level(field_name):
        """
        Function: level
        Purpose : A 1-10 level field (missing counts as the minimum).
        """
        return float(standard_request.get(field_name) or config.LEVEL_MIN)

    return [
        float(config.ROLE_ENCODING[standard_request["role"]]),
        float(config.REQUEST_TYPE_ENCODING.get(standard_request["category"], 0)),
        level("severity"),
        level("time_sensitivity"),
        level("crowd_level"),
        round(max(float(route_distance), 0.0), 2),
        float(1 if eligibility_flag else 0),
    ]
