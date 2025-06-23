import re
import json
import logging
from typing import Optional, Tuple, Dict, Any

logger = logging.getLogger(__name__)


def robust_json_parse(response: str, llm_client, expected_format: str = "auto") -> Any:
    """
    Parse JSON with targeted escape error fixing and LLM fallback.

    Strategy:
    1. Extract JSON string
    2. Try direct parsing
    3. If escape error, apply targeted fix
    4. For all other errors, use LLM fallback

    Args:
        response: LLM response containing JSON
        llm_client: LLM client for fallback repairs
        expected_format: "object", "array", or "auto"

    Returns:
        Parsed JSON data
    """
    try:
        # Step 1: Extract JSON string based on expected format
        if expected_format == "array":
            json_str = extract_json_array(response)
        elif expected_format == "object":
            json_str = extract_json(response)
        else:
            json_str = extract_json_from_response(response)

        # Step 2: Try direct parsing first
        return json.loads(json_str)

    except json.JSONDecodeError as e:
        error_msg = str(e)

        # Step 3: Handle specific error types with targeted fixes
        if "Invalid \\escape" in error_msg:
            try:
                fixed_json = fix_escape_errors(json_str)
                return json.loads(fixed_json)
            except json.JSONDecodeError:
                logger.info("Escape fix failed, falling back to LLM")

        # Step 4: For all other errors (including escape fix failures), use LLM
        logger.info(f"JSON error ({error_msg}), using LLM to fix")
        return llm_repair_json(
            broken_json=json_str,
            original_response=response,
            error_type=error_msg,
            expected_format=expected_format,
            llm_client=llm_client,
        )

    except ValueError as e:
        if "No valid JSON" in str(e):
            # No JSON structure found, ask LLM to generate
            logger.info("No JSON found in response, asking LLM to generate")
            return llm_repair_json(
                broken_json=None,
                original_response=response,
                error_type="No JSON structure found",
                expected_format=expected_format,
                llm_client=llm_client,
            )
        raise


def fix_escape_errors(json_str: str) -> str:
    """
    Fix escape sequence errors with safe, targeted approach.

    Only handles actual control characters and obvious invalid escapes.
    Does not modify valid escape sequences.
    """
    # Step 1: Fix actual control characters first (safest operation)
    control_char_map = {
        ord("\n"): "\\n",  # newline
        ord("\t"): "\\t",  # tab
        ord("\r"): "\\r",  # carriage return
        ord("\b"): "\\b",  # backspace
        ord("\f"): "\\f",  # form feed
    }
    fixed = json_str.translate(control_char_map)

    # Step 2: Fix invalid backslash escape sequences
    # Pattern matches backslashes NOT followed by valid JSON escape characters
    # Valid escapes: \" \\ \/ \b \f \n \r \t \uXXXX
    fixed = re.sub(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r"\\\\", fixed)

    return fixed


def llm_repair_json(
    broken_json: Optional[str],
    original_response: str,
    error_type: str,
    expected_format: str,
    llm_client,
) -> Any:
    """
    Unified LLM function to repair or generate JSON based on the specific error type.

    Args:
        broken_json: The broken JSON string (None if no JSON was found)
        original_response: The original LLM response
        error_type: Specific error description
        expected_format: Expected JSON format
        llm_client: LLM client

    Returns:
        Parsed JSON data
    """
    if broken_json is None:
        # Case 1: No JSON structure found - generate from original response
        repair_prompt = f"""The following response does not contain valid JSON. Please extract the information and provide it as a valid JSON {expected_format}:

Original response:
{original_response}

Error: {error_type}

Please return ONLY valid JSON {expected_format} without any explanation or additional text."""

    else:
        # Case 2: JSON found but has errors - fix the broken JSON
        repair_prompt = f"""The following JSON has errors and needs to be fixed:

Broken JSON:
{broken_json}

Error: {error_type}

Please return ONLY the corrected JSON without any explanation. Fix the errors while preserving all the original data and structure.

Return only valid JSON:"""

    try:
        response = llm_client.generate(
            repair_prompt, max_tokens=len(original_response) + 1000
        )

        # Extract JSON from the repair response
        fixed_json = extract_json_from_response(response)

        # Validate it parses correctly
        result = json.loads(fixed_json)
        logger.info(f"Successfully repaired JSON using LLM (error: {error_type})")
        return result

    except Exception as e:
        logger.error(f"LLM JSON repair failed for '{error_type}': {e}")
        raise ValueError(f"Could not repair JSON using LLM: {e}")


def extract_json_from_response(response: str) -> str:
    """Extract JSON from LLM response, trying multiple strategies."""
    # Try existing extraction methods
    try:
        return extract_json(response)
    except ValueError:
        pass

    try:
        return extract_json_array(response)
    except ValueError:
        pass

    # Find any JSON-like structure
    json_obj = find_first_json_object(response)
    if json_obj:
        return json_obj

    json_arr = find_first_json_array(response)
    if json_arr:
        return json_arr

    raise ValueError("No valid JSON found in response")


# TODO: Add specific handlers for other error types as needed
# Example structure for future extensions:
#
# def fix_delimiter_errors(json_str: str) -> str:
#     """Fix missing comma or delimiter issues."""
#     # Implement specific delimiter fixing logic
#     pass
#
# def fix_property_name_errors(json_str: str) -> str:
#     """Fix unquoted property names."""
#     # Implement property name quoting logic
#     pass
#
# Then add to robust_json_parse():
# elif "Expecting ',' delimiter" in error_msg:
#     try:
#         fixed_json = fix_delimiter_errors(json_str)
#         return json.loads(fixed_json)
#     except json.JSONDecodeError:
#         logger.info("Delimiter fix failed, falling back to LLM")


def extract_json(response: str) -> str:
    """Extract JSON from the plan response."""
    json_code_block_pattern = re.compile(
        r"```json\s*(\[\s*{.*?}\s*\])\s*```", re.DOTALL
    )
    match = json_code_block_pattern.search(response)
    if match:
        json_str = match.group(1)
        return "".join(char for char in json_str if ord(char) >= 32 or char in "\r\t")

    json_code_block_pattern = re.compile(r"```json\s*([\s\S]*?)\s*```", re.DOTALL)
    match = json_code_block_pattern.search(response)
    if match:
        json_str = match.group(1)
        return "".join(char for char in json_str if ord(char) >= 32 or char in "\r\t")

    json_str = find_first_json_object(response)
    if not json_str:
        raise ValueError(f"No valid JSON array found in the response, {response}")

    return "".join(char for char in json_str if ord(char) >= 32 or char in "\r\t")


def extract_json_array(response: str) -> str:
    """Extract JSON array from the plan response."""
    json_code_block_pattern = re.compile(
        r"```json\s*(\[\s*{.*?}\s*\])\s*```", re.DOTALL
    )
    match = json_code_block_pattern.search(response)
    if match:
        json_str = match.group(1)
        return "".join(char for char in json_str if ord(char) >= 32 or char in "\r\t")

    json_code_block_pattern = re.compile(r"```json\s*([\s\S]*?)\s*```", re.DOTALL)
    match = json_code_block_pattern.search(response)
    if match:
        json_str = match.group(1)
        return "".join(char for char in json_str if ord(char) >= 32 or char in "\r\t")

    json_str = find_first_json_array(response)
    if not json_str:
        raise ValueError(f"No valid JSON array found in the response, {response}")

    return "".join(char for char in json_str if ord(char) >= 32 or char in "\r\t")


def find_first_json_object(text: str) -> Optional[str]:
    """Find the first JSON object in the given text."""
    stack = []
    start = -1
    for i, char in enumerate(text):
        if char == "{":
            if not stack:
                start = i
            stack.append(i)
        elif char == "}":
            if stack:
                stack.pop()
                if not stack:
                    return text[start : i + 1]
    return None


def find_first_json_array(text: str) -> Optional[str]:
    """Find the first array in the given text."""
    stack = []
    start = -1
    for i, char in enumerate(text):
        if char == "[":
            if not stack:
                start = i
            stack.append(i)
        elif char == "]":
            if stack:
                stack.pop()
                if not stack:
                    return text[start : i + 1]
    return None
