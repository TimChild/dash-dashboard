from dataclasses import dataclass
from typing import List, Callable, Optional

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash import Dash
from dash.dependencies import Output, Input
from dash_extensions.enrich import DashProxy, PrefixIdTransform
from slugify import slugify

from dash_dashboard.dash_labs_extensions_overrides import MyFlexibleCallbacks, register_dash_proxy_callbacks_to_app

CONTENT_ID = 'content'
URL_ID = 'url'


@dataclass
class PageInfo:
    """
    Mostly new pages are made as if they are a single page, then this class just ties things together so that a
    multipage app can more predictably interact with each page. I.e. it can expect to find a page name for example
    """
    page_name: str  # The name of the page which will show up in a Nav bar
    app_function: Callable  # Function which returns the complete app (and can be passed in an app to fill)
    page_id: Optional[str] = None  # Prefix to add to all interactive layout/callback components

    def __post_init__(self):
        self.page_id = self.page_id if self.page_id else self.page_name


def run_app(app: dash.Dash, debug: bool = True, debug_port: int = 8051, real_port: int = None, threaded=True):
    """Handles running dash app with reasonable settings for debug or real"""
    if isinstance(app, DashProxy):
        real_app = dash.Dash(__name__, plugins=[MyFlexibleCallbacks()])
        register_dash_proxy_callbacks_to_app(app, real_app)  # Otherwise single outputs get turned into lists when they shouldn't
        real_app.layout = app._layout_value()
        app = real_app

    # Resizes things better based on actual device width rather than just pixels (good for mobile)
    meta_tags = [
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ]
    app.config.meta_tags = meta_tags
    if debug is False:
        assert isinstance(real_port, int)
        app.run_server(debug=False, port=real_port, host='0.0.0.0', threaded=threaded)
    else:
        app.run_server(debug=True, port=debug_port, threaded=threaded)


def make_multipage_app(module_pages: list,
                       app_name: str) -> Dash:
    """
    Make a multipage app from a list of module_pages (which each need to implement a PageInfo, see template page)

    Args:
        module_pages (): list of modules which are app pages
        app_name (): Name of multipage app to display in Navbar

    Returns:

    """
    multipage_app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
    pages = _modules_to_pages(module_pages, multipage_app)
    navbar = _make_navbar(pages, app_name)
    _page_select_callbacks(multipage_app, pages)
    multipage_app.layout = _multipage_layout(navbar)
    multipage_app.validation_layout = html.Div([multipage_app.layout, *[p.layout for p in pages]])
    return multipage_app


@dataclass
class _ProcessedPage:
    """Used when generating a multipage app"""
    name: str
    layout: list


def _modules_to_pages(modules: list, app: Dash) -> List[_ProcessedPage]:
    pages = []
    for page in modules:
        page_info = getattr(page, 'page_info', None)
        if page_info is None or not isinstance(page_info, PageInfo):
            raise RuntimeError(f'Need to implement `page_info` for {page}')
        # TODO: Need to add more transforms if used in other pages, or maybe add that to PageInfo so as not to
        # TODO: load more transforms than necessary for each page
        page_app = page_info.app_function(DashProxy(__name__,
                                                    transforms=[PrefixIdTransform(page_info.page_id)],
                                                    plugins=[MyFlexibleCallbacks()])
                                          )
        pages.append(
            _ProcessedPage(
                name=page_info.page_name,
                layout=page_app._layout_value(),
            )
        )
        # Callbacks must be registered AFTER page_app._layout_value() to work with dl.Plugins.FlexibleCallbacks() where
        # component is made inside of app.callback(...), otherwise prefix is added twice!
        register_dash_proxy_callbacks_to_app(page_app, app)
    return pages


def _page_select_callbacks(app: Dash, pages: List[_ProcessedPage]):
    page_dict = {slugify(p.name): p for p in pages}

    @app.callback(Output(CONTENT_ID, 'children'), Input(URL_ID, 'pathname'))
    def page_select(pathname):
        if pathname:
            pathname = pathname[1:]  # get rid of the first leading "/"
        if pathname in page_dict:
            return page_dict[pathname].layout
        else:
            return pages[0].layout


def _multipage_layout(navbar: dbc.NavbarSimple) -> dbc.Container:
    return dbc.Container(
        [
            dcc.Location(id=URL_ID),
            navbar,
            html.Div(id=CONTENT_ID),
        ], fluid=True
    )


def _make_navbar(pages: List[_ProcessedPage], app_name: str) -> dbc.NavbarSimple:
    navbar = dbc.NavbarSimple(
        [
            *[dbc.NavItem(dbc.NavLink(p.name, href=slugify(p.name))) for p in pages]
        ], brand=app_name,
    )
    return navbar