import dash


def get_trig_id() -> str:
    """Pass in dash.callback_context to get the id of the trigger returned"""
    ctx = dash.callback_context
    return ctx.triggered[0]['prop_id'].split('.')[0]


def triggered_by(id_name: str) -> bool:
    trig_id = get_trig_id()
    if id_name in trig_id:  # Check if 'in' because might have some additional ID prepended to it by multipage etc
        return True
    return False

