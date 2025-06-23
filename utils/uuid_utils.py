import uuid
from typing import List, Optional


def is_valid_uuid(id_string: str) -> bool:
    """
    Validate if a string is a valid UUID format.
    
    Args:
        id_string: String to validate
        
    Returns:
        bool: True if string is a valid UUID, False otherwise
        
    Example:
        >>> is_valid_uuid("2d74d3d9-8f17-421c-a56b-0072472ad8a6")
        True
        >>> is_valid_uuid("2")
        False
        >>> is_valid_uuid("invalid-uuid")
        False
    """
    if not isinstance(id_string, str):
        return False
        
    try:
        uuid.UUID(id_string)
        return True
    except (ValueError, TypeError):
        return False


def validate_uuid_list(ids: List[str], strict: bool = True) -> List[str]:
    """
    Filter and return only valid UUIDs from a list of ID strings.
    
    Args:
        ids: List of ID strings to validate
        strict: If True, log warnings for invalid UUIDs
        
    Returns:
        List[str]: List containing only valid UUID strings
        
    Example:
        >>> validate_uuid_list(["2d74d3d9-8f17-421c-a56b-0072472ad8a6", "2", "invalid"])
        ["2d74d3d9-8f17-421c-a56b-0072472ad8a6"]
    """
    if not ids:
        return []
    
    valid_ids = []
    for id_str in ids:
        if is_valid_uuid(id_str):
            valid_ids.append(id_str)
        elif strict:
            print(f"Warning: Skipping invalid UUID format: '{id_str}'")
    
    return valid_ids


def validate_single_uuid(id_string: str, raise_error: bool = False) -> Optional[str]:
    """
    Validate a single UUID string.
    
    Args:
        id_string: UUID string to validate
        raise_error: If True, raise ValueError for invalid UUIDs
        
    Returns:
        str: The validated UUID string, or None if invalid
        
    Raises:
        ValueError: If raise_error is True and UUID is invalid
        
    Example:
        >>> validate_single_uuid("2d74d3d9-8f17-421c-a56b-0072472ad8a6")
        "2d74d3d9-8f17-421c-a56b-0072472ad8a6"
        >>> validate_single_uuid("2")
        None
    """
    if is_valid_uuid(id_string):
        return id_string
    
    if raise_error:
        raise ValueError(f"Invalid UUID format: '{id_string}'")
    
    return None


def generate_uuid() -> str:
    """
    Generate a new UUID string.
    
    Returns:
        str: A new UUID string in standard format
        
    Example:
        >>> uuid_str = generate_uuid()
        >>> is_valid_uuid(uuid_str)
        True
    """
    return str(uuid.uuid4())


def normalize_uuid(id_string: str) -> Optional[str]:
    """
    Normalize a UUID string to standard format (lowercase with dashes).
    
    Args:
        id_string: UUID string to normalize
        
    Returns:
        str: Normalized UUID string, or None if invalid
        
    Example:
        >>> normalize_uuid("2D74D3D9-8F17-421C-A56B-0072472AD8A6")
        "2d74d3d9-8f17-421c-a56b-0072472ad8a6"
        >>> normalize_uuid("2")
        None
    """
    try:
        return str(uuid.UUID(id_string)).lower()
    except (ValueError, TypeError):
        return None 