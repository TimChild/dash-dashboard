"""
The small changes that need to be made to dash_labs and dash_extensions in order for them to play nicely together.

Want dash_extensions mostly for the implementation of multipage_app (but also has some other nice features)
Want dash_labs for the component plugins and templates. Also makes working with Callbacks nicer.
"""
from dash_labs.plugins import dx_callback
import dash.dependencies as dd
from functools import partial
from types import MethodType
from dash import Dash
from dash_extensions.enrich import DashProxy


class MyFlexibleCallbacks:
    """
    Dash app plugin to enable advanced callback behaviors including:
      - Property grouping
      - Keyword argument support
      - Support for providing full components in place of ids when using the
        dl.Output/dl.Input/sl.State dependency objects.

    Usage:
    """

    def __init__(self):
        pass

    def plug(self, app: Dash):
        # Instead of wrapping Dash.callback, wrap type(app) (e.g. in case it is a DashProxy from dash_extensions
        _wrapped_callback = type(app).callback  # type(app) so that I don't wrap a method which has 'self' as first arg
        app._wrapped_callback = _wrapped_callback
        app.callback = MethodType(
            partial(dx_callback, _wrapped_callback=_wrapped_callback), app
        )


def register_dash_proxy_callbacks_to_app(proxy_app: DashProxy, new_app: Dash):
    """Register all callbacks in a DashProxy to a Dash app"""
    callbacks = list(proxy_app._resolve_callbacks())
    new_app = super() if new_app is None else new_app
    for callback in callbacks:
        # Replacing this line from dash_extensions. Otherwise scalar returns end up wrapped in a list
        # outputs = callback[Output][0] if len(callback[Output]) == 1 else callback[Output]
        outputs = callback[dd.Output]
        new_app.callback(outputs, callback[dd.Input], callback[dd.State], **callback["kwargs"])(callback["f"])
