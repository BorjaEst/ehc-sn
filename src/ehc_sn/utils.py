from collections import defaultdict
from typing import Any, Dict, List

from torch import nn


def map_by_key(items: List[Any], key: str) -> Dict[Any, List[Any]]:
    """
    Maps items into a dictionary based on a shared attribute specified by 'key'.

    Args:
        items (List[Any]): A list of objects expected to have the attribute 'key'.
        key (str): The name of the attribute used for grouping.

    Returns:
        Dict[Any, List[Any]]: A dictionary where each key is an attribute value and the corresponding
        value is a list of items that share that attribute.

    Raises:
        ValueError: If an item does not have the specified attribute.
    """
    mapping: defaultdict[Any, List[Any]] = defaultdict(list)
    for item in items:
        try:
            attr = getattr(item, key)
        except AttributeError as e:
            raise ValueError(f"One or more items do not have the attribute '{key}'") from e
        mapping[attr].append(item)
    return dict(mapping)


def torch_function(name: str) -> Any:
    """
    Retrieve a torch function by name from torch.nn. This function normalizes the name
    to lowercase, checks for a special case, and then attempts to locate the corresponding
    function as an attribute of torch.nn.

    Args:
        name (str): The name of the torch function to retrieve.

    Returns:
        Any: The corresponding torch function from torch.nn.

    Raises:
        ValueError: If the torch function cannot be found.
    """
    normalized_name = name.lower()

    # Check special cases first; for example, "linear" returns an identity lambda.
    special_functions = {
        "linear": lambda x: x,  # Placeholder: replace with actual implementation if needed
    }
    if normalized_name in special_functions:
        return special_functions[normalized_name]

    torch_func = getattr(nn, normalized_name, None)
    if torch_func is None:
        raise ValueError(f"Unknown torch function: {name}")

    return torch_func
