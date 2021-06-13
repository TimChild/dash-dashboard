from __future__ import annotations
from typing import Optional

from dash_extensions.enrich import DashProxy
from dash_dashboard.dash_labs_extensions_overrides import MyFlexibleCallbacks

from dash_dashboard.run_app import PageInfo


def make_app(app: Optional[DashProxy] = None):
    if app is None:
        app = DashProxy(transforms=[], plugins=[MyFlexibleCallbacks()])

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
    page_app = make_app()
    page_app.run_server(debug=True)

