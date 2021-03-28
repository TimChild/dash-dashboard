import logging
# logging.basicConfig(level=logging.DEBUG)
from typing import Callable

from dash_extensions.enrich import DashProxy, ServersideOutputTransform
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc
from dash_extensions.multipage import PageCollection, Page, module_to_page


# Everything below here shouldn't really need to be touched
CONTENT_ID = 'content'
URL_ID = 'url'


def default_layout(*args):
    return html.Div([html.Div(id=CONTENT_ID), dcc.Location(id=URL_ID)] + list(args))


def get_app(pages) -> DashProxy:
    # Make PageCollection (which handles a lot of the navigation between pages)
    pc = PageCollection(pages=[
        module_to_page(page, id=page.URL_ID, label=page.NAME) for page in pages
        # Page(id=page.url_id, label=page.name, layout=page.layout, callbacks=page.run_all_callbacks) for page in pages
    ])

    # Tell each page what the page collection is so that it can use the labels/ids in the top bar layout for each
    for page in pages:
        page.page_collection = pc

    # Create app.
    app = DashProxy(name=__name__, suppress_callback_exceptions=False, external_stylesheets=[dbc.themes.BOOTSTRAP],
                    transforms=[ServersideOutputTransform()])
    app.layout = html.Div([default_layout()])

    # Register callbacks.
    pc.navigation(app, content_id=CONTENT_ID, url_id=URL_ID)
    pc.callbacks(app)

    # Make validation layout to check if callbacks all make sense
    layouts = [page.layout() for page in pc.pages]
    app.validation_layout = html.Div([page.layout() for page in pc.pages] + [default_layout()])
    return app


# Run Server
if __name__ == '__main__':
    # app.run_server(port=8060, debug=True)
    pass


def test_page(layout: Callable, callbacks: Callable, single_threaded=False, port=8050):
    """
    Makes a Dash app and runs loads layout and callbacks from layout_class in a similar way to how
    the PageCollection will when added in main app
    """
    app = DashProxy(
        transforms=[
            ServersideOutputTransform()
        ],
        name=__name__, external_stylesheets=[dbc.themes.BOOTSTRAP]
    )

    app.layout = layout
    callbacks(app)
    app.run_server(port=port, debug=True, threaded=not single_threaded)

