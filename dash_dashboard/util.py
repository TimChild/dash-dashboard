import dash
from typing import List, Dict, Tuple, Union


def get_trig_id() -> str:
    """Pass in dash.callback_context to get the id of the trigger returned"""
    ctx = dash.callback_context
    return ctx.triggered[0]['prop_id'].split('.')[0]


def triggered_by(id_name: str) -> bool:
    trig_id = get_trig_id()
    if id_name in trig_id:  # Check if 'in' because might have some additional ID prepended to it by multipage etc
        return True
    return False


def list_to_options(options_list: List[str]) -> List[Dict[str, str]]:
    """
    Turns a list into the List[Dict]] format that dropdown.options take in dash
    Args:
        options_list (): List of options to show in dropdown (will use same name as value)

    Returns:

    """
    return [{'label': k, 'value': k} for k in options_list]


def new_options_keep_old_selection(new_options_list, current_selection) -> Tuple[List[Dict[str, str]], str]:
    """
    Converts a simple list of new options and the current selection values into a list of dash options and any still
    valid selections
    Args:
        new_options_list (): Simple list of new options to show in dropdown
        current_selection (): Currently selected values in dropdown for which only values which are still relevant
            will be returned

    Returns:
        dash style options, dash style values (which are still valid)
    """
    val = _valid_values(new_options_list, current_selection)
    opts = list_to_options(new_options_list)
    return opts, val


def _valid_values(options_list: List[str], current: Union[str, List[str]], default_to_first=True) -> Union[str, List[str]]:
    """
    Returns the list of current values which exist in the options_list. Useful for only keeping current values which
    are valid options.

    Args:
        options_list (): Simple list of options
        current (): List of values which may or may not exist in options
        default_to_first (): If True, and no 'current' values are in options, then the first option will be returned

    Returns:
        Only values which exist in options_list
    """
    if isinstance(current, str):
        current = [current]

    values = []
    if options_list is not None and current is not None:
        for x in current:
            if x in options_list:
                values.append(x)
    if len(values) == 1:
        values = values[0]  # return str for only one value selected to keep in line with how dash does things
    elif len(values) == 0:
        if len(options_list) > 0 and default_to_first:
            values = options_list[0]
        else:
            values = ''
    return values


