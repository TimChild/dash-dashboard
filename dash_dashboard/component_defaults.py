from __future__ import annotations
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import pandas as pd

from typing import Optional, List, Tuple, Dict, Any, Union, Callable

from dash.dependencies import Output, Input, State
from dash.exceptions import PreventUpdate
from dash_extensions import Download
from dash_extensions.snippets import send_file
from plotly import graph_objects as go

from dash_dashboard.base_classes import CallbackInfo, PendingCallbacks


def input_box(id_name: Optional[str] = None, val_type='number', debounce=True,
              placeholder: str = '', persistence=False,
              **kwargs) -> dbc.Input:
    """

    Args:
        id_name ():
        val_type (): "text", 'number', 'password', 'email', 'range', 'search', 'tel', 'url', 'hidden', None
        debounce ():
        placeholder ():
        persistence ():
        **kwargs ():

    Returns:

    """
    inp = dbc.Input(id=id_name, type=val_type, placeholder=placeholder, debounce=debounce, **kwargs,
                    persistence=persistence, persistence_type='local')
    return inp


def dropdown(id_name: str, multi=False, placeholder='Select',
             persistence=False) -> Union[dbc.Select, dcc.Dropdown]:
    if multi is False:
        dd = dbc.Select(id=id_name, placeholder=placeholder, persistence=persistence,
                        persistence_type='local')
    else:
        dd = dcc.Dropdown(id=id_name, placeholder=placeholder, style={'width': '80%'}, multi=True,
                          persistence=persistence, persistence_type='local')
    return dd


def toggle(id_name: str, persistence=False) -> dbc.Checklist:
    tog = dbc.Checklist(id=id_name, options=[{'label': '', 'value': True}], switch=True, persistence=persistence,
                        persistence_type='local')
    return tog


def slider(id_name: str, updatemode='mouseup', persistence=False) -> dcc.Slider:
    return dcc.Slider(id=id_name, updatemode=updatemode, persistence=persistence,
                        persistence_type='local')


def checklist(id_name: str,
              options: Optional[List[dict]] = None, persistence=False) -> dbc.Checklist:
    if options is None:
        options = []
    checklist = dbc.Checklist(id=id_name, options=options, switch=False, persistence=persistence,
                              persistence_type='local')
    return checklist


def table(id_name: str, dataframe: Optional[pd.Dataframe] = None,
          **kwargs) -> dbc.Table:
    """https://dash.plotly.com/datatable"""
    # table = dbc.Table(dataframe, id=self.id(id_name), striped=True, bordered=True, hover=True)
    if dataframe is not None:
        cols = [{'name': n, 'id': n} for n in dataframe.columns()]
        data = dataframe.to_dict('records')
    else:
        cols, data = None, None
    table_component = dash_table.DataTable(id=id_name, columns=cols, data=data, **kwargs)
    return table_component


def button(id_name: str, text: str, color='secondary') -> dbc.Button:
    """Note: text is just children of Button"""
    return dbc.Button(text, id=id_name, color=color)


def div(id_name: str, **kwargs) -> html.Div:
    div = html.Div(id=id_name, **kwargs)
    return div


def date_picker_single(id_name: str, **kwargs) -> dcc.DatePickerSingle:
    dps = dcc.DatePickerSingle(id=id_name, **kwargs, )
    return dps


def date_picker_range(id_name: str, **kwargs) -> dcc.DatePickerRange:
    dpr = dcc.DatePickerRange(id=id_name, **kwargs)
    return dpr


def graph_area(id_name: str, graph_header: str, pending_callbacks: Optional[PendingCallbacks] = None) -> dbc.Card:
    """

    Args:
        id_name (): id for component (all extra components in here derive from this as well)
        graph_header (): Name to appear above figure in graph header
        pending_callbacks (): Instance of PendingCallbacks to add download button callbacks etc to.
    Returns:
        (dbc.Card): Card containing a header and figure. Note: id of graph is saved in self.graph_id
            (instead of the usual self.id which in this case would refer to the Card object)

    """
    def _graph_save_options(graph_id):
        layout = dbc.Row([
            dbc.Col(_download_button(graph_id, 'html'), width='auto'),
            dbc.Col(_download_button(graph_id, 'jpg'), width='auto'),
            dbc.Col(_download_button(graph_id, 'svg'), width='auto'),
            dbc.Col(_download_name(graph_id), width='auto'),
        ], no_gutters=True)
        return layout

    def _get_graph_callbacks(graph_id: str) -> List[CallbackInfo]:
        return [_download_callback(graph_id, 'html'),
                _download_callback(graph_id, 'jpg'),
                _download_callback(graph_id, 'svg')]

    def _download_callback(graph_id: str, file_type: str) -> CallbackInfo:
        """https://pypi.org/project/dash-extensions/"""

        def make_file(n_clicks, fig: dict, filename: str):
            if n_clicks:
                fig = go.Figure(fig)
                if not filename:
                    filename = fig.layout.title.text
                    if not filename:
                        filename = 'DashFigure'

                fname = filename + f'.{file_type}'
                bytes_ = False
                if file_type == 'html':
                    data = fig.to_html()
                    mtype = 'text/html'
                elif file_type == 'jpg':
                    fig.write_image('temp/dash_temp.jpg', format='jpg')
                    return send_file('temp/dash_temp.jpg', filename=fname, mime_type='image/jpg')
                elif file_type == 'svg':
                    fig.write_image('temp/dash_temp.svg', format='svg')
                    return send_file('temp/dash_temp.svg', fname, 'image/svg+xml')
                else:
                    raise ValueError(f'{file_type} not supported')

                return dict(content=data, filename=fname, mimetype=mtype, byte=bytes_)
            else:
                raise PreventUpdate

        if file_type not in ['html', 'jpg', 'svg']:
            raise ValueError(f'{file_type} not supported')

        dl_id = f'{graph_id}_download-{file_type}'  # Download extension
        but_id = f'{graph_id}_but-{file_type}-download'
        name_id = f'{graph_id}_inp-download-name'

        callback_info = CallbackInfo(func=make_file, outputs=Output(dl_id, 'data'), inputs=Input(but_id, 'n_clicks'),
                                     states=[State(graph_id, 'figure'), State(name_id, 'value')])
        return callback_info

    def _download_button(graph_id: str, file_type: str):
        if file_type not in ['html', 'jpg', 'svg']:
            raise ValueError(f'{file_type} not supported')
        button = [dbc.Button(f'Download {file_type.upper()}', id=f'{graph_id}_but-{file_type}-download'),
                  Download(id=f'{graph_id}_download-{file_type}')]
        return button

    def _download_name(graph_id: str):
        name = dbc.Input(id=f'{graph_id}_inp-download-name', type='text', placeholder='Download Name')
        return name

    header_layout = dbc.CardHeader(
        dbc.Row([
            dbc.Col(html.H3(graph_header), width='auto'),
            _graph_save_options(id_name),
        ], justify='between')
    )

    graph_body = dcc.Graph(id=id_name, figure=None)
    graph = dbc.Card([
        header_layout, graph_body
    ])
    if pending_callbacks is not None:
        callback_infos = _get_graph_callbacks(id_name)  # ALl the callbacks for downloading the graph
        pending_callbacks.extend(callback_infos)
    graph.graph_id = id_name  # So that it is easy to get to the graphs id through the returned card
    return graph


