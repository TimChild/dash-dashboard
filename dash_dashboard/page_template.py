from __future__ import annotations
from typing import Optional, Union, List, Tuple, Dict

import dash
import dash_labs as dl
from dash_labs import Input, Output, State
from dash_extensions.enrich import DashProxy
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc

from dash_dashboard.dash_labs_extensions_overrides import MyFlexibleCallbacks
from dash_dashboard.run_app import PageInfo
import dash_dashboard.component_defaults as cds
import dash_dashboard as dd  # Can get to all dash_dashboard stuff from here e.g. dd.util.get_trig_id()


def make_app(app: Optional[DashProxy] = None):
    if app is None:
        app = dash.Dash(plugins=[MyFlexibleCallbacks()], external_stylesheets=[dbc.themes.FLATLY])
        # app = DashProxy(transforms=[], plugins=[MyFlexibleCallbacks()])  # Use this to use additional dash_extensions
        # Note: Do NOT use a PrefixID transform here (it doesn't work with the dash_labs callback/template system)
        # PrefixIDs are handled by multipage

    """
    Make the app (i.e. all layout and app.callback() normal stuff)
    
    Examples:
        tpl = dl.templates.DbcCard(app, title='Card Title')

        @app.callback(
            tpl.textbox_input(label='Input text here'),
            template=tpl
        )
        def return_text(text):
            return f'Text: {text}'

        app.layout = dbc.Container(children=tpl.children)
        
        return app
    """

    return app


# Multipage app will look for this in order to add to multipage
page_info = PageInfo(
    page_name='Template',  # The name which will show up in the NavBar
    app_function=make_app,  # This function should take a DashProxy instance
)


if __name__ == '__main__':
    from dash_dashboard.run_app import run_app
    page_app = make_app()
    run_app(page_app, debug=True, debug_port=8050, threaded=True)

