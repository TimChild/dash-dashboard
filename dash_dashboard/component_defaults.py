from __future__ import annotations
import abc
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import pandas as pd
from dataclasses import dataclass
import json
import numpy as np

from typing import Optional, List, Tuple, Dict, Any, Union, Callable

from dash.dependencies import Output, Input, State
from dash.exceptions import PreventUpdate
from dash_extensions import Download
from dash_extensions.snippets import send_file
from plotly import graph_objects as go
from dash_dashboard.base_classes import CallbackInfo, PendingCallbacks, CALLBACK_TYPE


class SetupCallbackTemplate(abc.ABC):
    @abc.abstractmethod
    def __init__(self, *args):
        pass

    @staticmethod
    @abc.abstractmethod
    def get_outputs(id_name: str) -> Union[List[CALLBACK_TYPE], CALLBACK_TYPE]:
        pass

    @classmethod
    @abc.abstractmethod
    def get_inputs(cls, *args) -> Union[List[CALLBACK_TYPE], CALLBACK_TYPE]:
        """Override to return the Inputs that will trigger the callback (these go first in __init__ args"""
        pass

    @classmethod
    def get_states(cls, *args) -> Union[List[CALLBACK_TYPE], CALLBACK_TYPE]:
        """Override to return the States for the callback (these go after Inputs in __init__ args"""
        return []

    @abc.abstractmethod
    def callback_return(self):
        """Override this to return the list of values which match self.get_outputs"""
        pass

    @classmethod
    def get_callback_func(cls):
        """Generates a callback function which takes all arguments that __init__ takes"""
        def callback_func(*args):
            inst = cls(*args)
            return inst.callback_return()
        return callback_func


class RangeSliderSetupCallback(SetupCallbackTemplate, abc.ABC):
    # noinspection PyMissingConstructor
    def __init__(self, min: float, max: float, step: float,
                 marks: Dict[float, str],
                 value: Tuple[float]):
        self.min = min
        self.max = max
        self.step = step
        self.marks = marks
        self.value = value

    @staticmethod
    def get_outputs(id_name: str) -> Union[List[CALLBACK_TYPE], CALLBACK_TYPE]:
        return [
            (id_name, 'min'),
            (id_name, 'max'),
            (id_name, 'step'),
            (id_name, 'marks'),
            (id_name, 'value'),
        ]

    def callback_return(self):
        return self.min, self.max, self.step, self.marks, self.value


def space(height: Optional[str] = None, width: Optional[str] = None) -> html.Div:
    """Just for adding a blank div which takes up some space"""
    return html.Div(style={f'height': height, 'width': width})


def store(id_name: str, storage_type: str = 'memory') -> dcc.Store:
    """For storing data only on clientside (or serverside if used with ServersideOutput from dash-extensions)

    Usual Callback Format:
        'data': Json serializable for clientside, Any for serverside
        'clear_data': set True to clear data

    Args:
        id_name:
        storage_type: 'memory' = resets on page refresh, 'session' = resets on browser reload, 'local' = local memory

    """
    return dcc.Store(id=id_name, storage_type=storage_type)


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
    """
    Either a single select or multi selectable dropdown.

    Usual callbacks format:
        'options': [{'label': <name>, 'value', <val>}]

    Args:
        id_name ():
        multi ():  Whether multiple selections can be made
        placeholder ():
        persistence ():

    Returns:

    """
    if multi is False:
        dd = dbc.Select(id=id_name, placeholder=placeholder, persistence=persistence,
                        persistence_type='local')
    else:
        dd = dcc.Dropdown(id=id_name, placeholder=placeholder, style={'width': '80%'}, multi=True,
                          persistence=persistence, persistence_type='local')
    return dd


def toggle(id_name: str, label: str = '', persistence=False) -> dbc.Checklist:
    tog = dbc.Checklist(id=id_name, options=[{'label': label, 'value': True}], switch=True, persistence=persistence,
                        persistence_type='local')
    return tog


def slider(id_name: str, updatemode='mouseup', persistence=False) -> dcc.Slider:
    """

    Args:
        id_name ():
        updatemode (): 'mouseup' or 'drag'
        persistence ():

    Returns:

    """
    return dcc.Slider(id=id_name, updatemode=updatemode, persistence=persistence,
                      persistence_type='local')


def range_slider(id_name: str, updatemode='mouseup', persistence=False) -> dcc.RangeSlider:
    return dcc.RangeSlider(id=id_name, updatemode=updatemode,
                           persistence=persistence, persistence_type='local')


def checklist(id_name: str,
              options: Optional[List[dict]] = None, persistence=False) -> dbc.Checklist:
    if options is None:
        options = []
    checklist = dbc.Checklist(id=id_name, options=options, switch=False, persistence=persistence,
                              persistence_type='local')
    return checklist


def table(id_name: str, dataframe: Optional[pd.Dataframe] = None,
          **kwargs) -> dbc.Table:
    """
    https://dash.plotly.com/datatable

    Usual callbacks format:
        'columns': List[{"name": name, "id": name} for name in df.columns]
        'data': df.to_dict('records')

    Args:
        id_name ():
        dataframe ():
        **kwargs ():

    Returns:

    """
    # table = dbc.Table(dataframe, id=self.id(id_name), striped=True, bordered=True, hover=True)
    if dataframe is not None:
        cols = [{'name': n, 'id': n} for n in dataframe.columns()]
        data = dataframe.to_dict('records')
    else:
        cols, data = None, None
    table_component = dash_table.DataTable(id=id_name, columns=cols, data=data, **kwargs)
    return table_component


def button(id_name: str, text: str, color='secondary', spinner: Optional[dbc.Spinner] = None) -> dbc.Button:
    """
    Makes a button which shows <text> and if <spinner> is passed will also show a loading spinner

    Note: If using a spinner, remember to put the item you want to wait for in the children of spinner
    (i.e. an empty div that waits for an update)

    Args:
        id_name ():
        text ():
        color ():
        spinner ():

    Returns:

    """
    if spinner:
        children = [spinner, text]
    else:
        children = text
    return dbc.Button(children, id=id_name, color=color)


def div(id_name: str, **kwargs) -> html.Div:
    d = html.Div(id=id_name, **kwargs)
    return d


def collapse(id_name: str) -> dbc.Collapse:
    """
    Usual callbacks format:
        'is_open': bool

    Args:
        id_name ():

    Returns:

    """
    c = dbc.Collapse(id=id_name)
    return c


def date_picker_single(id_name: str, **kwargs) -> dcc.DatePickerSingle:
    dps = dcc.DatePickerSingle(id=id_name, **kwargs, )
    return dps


def date_picker_range(id_name: str, **kwargs) -> dcc.DatePickerRange:
    dpr = dcc.DatePickerRange(id=id_name, **kwargs)
    return dpr


def graph_area(id_name: str, graph_header: str, pending_callbacks: Optional[PendingCallbacks] = None) -> dbc.Card:
    """
    A graph in a Card component with download buttons.

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
            dbc.Col(_download_button(graph_id, 'fig_json'), width='auto'),
            dbc.Col(_download_button(graph_id, 'data_json'), width='auto'),
            dbc.Col(_download_button(graph_id, 'igor'), width='auto'),
            dbc.Col(_download_name(graph_id), width='auto'),
        ], no_gutters=True)
        return layout

    def _get_graph_callbacks(graph_id: str) -> List[CallbackInfo]:
        return [_download_callback(graph_id, 'html'),
                _download_callback(graph_id, 'jpg'),
                _download_callback(graph_id, 'svg'),
                _download_callback(graph_id, 'fig_json'),
                _download_callback(graph_id, 'data_json'),
                _download_callback(graph_id, 'igor'),
                ]

    def _download_callback(graph_id: str, file_type: str) -> CallbackInfo:
        """https://pypi.org/project/dash-extensions/"""

        def make_file(n_clicks, fig: dict, filename: str):
            def data_from_fig(f: go.Figure) -> Dict[str, np.ndarray]:
                all_data = {}
                for i, d in enumerate(f.data):
                    name = getattr(d, 'name', None)
                    if name is None:
                        name = f'data{i}'
                    elif name in all_data.keys():
                        name = name + f'_{i}'
                    if 'z' in d:  # Then it is 2D
                        all_data[name] = getattr(d, 'z')
                        all_data[name + '_y'] = getattr(d, 'y')
                    else:
                        all_data[name] = getattr(d, 'y')
                    all_data[name + '_x'] = getattr(d, 'x')
                return all_data

            def itx_from_fig(f: go.Figure) -> str:
                from igorwriter import IgorWave
                import io
                d = data_from_fig(f)
                waves = []
                for k in d:
                    if not k.endswith('_x') and not k.endswith('_y'):
                        wave = IgorWave(d[k], name=k)
                        wave.set_datascale(f.layout.yaxis.title.text)
                        for dim in ['x', 'y']:
                            if f'{k}_{dim}' in d:
                                dim_arr = d[f'{k}_{dim}']
                                wave.set_dimscale('x', dim_arr[0], np.mean(np.diff(dim_arr)),
                                                                           units=f.layout.xaxis.title.text)
                        waves.append(wave)
                buffer = io.StringIO()
                for wave in waves:
                    wave.save_itx(buffer, image=True)  # Image = True hopefully makes np and igor match in x/y
                buffer.seek(0)
                return buffer.read()

            import base64
            if n_clicks:
                fig = go.Figure(fig)
                if not filename:
                    filename = fig.layout.title.text
                    if not filename:
                        filename = 'DashFigure'

                download_info = DownloadInfo(data=None, mtype=None, base64=False, file_extension=file_type)

                if file_type == 'html':
                    download_info.mtype = 'text/html'
                    download_info.data = fig.to_html()
                elif file_type == 'jpg':
                    download_info.data = base64.b64encode(fig.to_image(format='jpg')).decode()
                    download_info.mtype = 'image/jpg'
                    download_info.base64 = True
                elif file_type == 'svg':
                    download_info.data = base64.b64encode(fig.to_image(format='svg')).decode()
                    download_info.mtype = 'image/svg+xml'
                    download_info.base64 = True
                elif file_type == 'fig_json':
                    filename = 'Figure_'+filename
                    download_info.data = fig.to_json()
                    download_info.mtype = 'application/json'
                    download_info.file_extension = 'json'
                elif file_type == 'data_json':
                    filename = 'Data_'+filename
                    data = data_from_fig(fig)
                    data_json = json.dumps(data, default=lambda arr: arr.tolist())
                    download_info.data = data_json
                    download_info.mtype = 'application/json'
                    download_info.file_extension = 'json'
                elif file_type == 'igor':
                    download_info.data = itx_from_fig(fig)
                    download_info.mtype = 'application/octet-stream'
                    download_info.file_extension = 'itx'
                else:
                    raise ValueError(f'{file_type} not supported')

                fname = filename + f'.{download_info.file_extension}'
                return dict(content=download_info.data, filename=fname, mimetype=download_info.mtype,
                            base64=download_info.base64)
            else:
                raise PreventUpdate

        if file_type not in ['html', 'jpg', 'svg', 'fig_json', 'data_json', 'igor']:
            raise ValueError(f'{file_type} not supported')

        dl_id = f'{graph_id}_download-{file_type}'  # Download extension
        but_id = f'{graph_id}_but-{file_type}-download'
        name_id = f'{graph_id}_inp-download-name'

        callback_info = CallbackInfo(func=make_file, outputs=Output(dl_id, 'data'), inputs=Input(but_id, 'n_clicks'),
                                     states=[State(graph_id, 'figure'), State(name_id, 'value')])
        return callback_info

    def _download_button(graph_id: str, file_type: str):
        if file_type not in ['html', 'jpg', 'svg', 'fig_json', 'data_json', 'igor']:
            raise ValueError(f'{file_type} not supported')
        button = [dbc.Button(f'Download {file_type.upper()}', id=f'{graph_id}_but-{file_type}-download'),
                  Download(id=f'{graph_id}_download-{file_type}')]
        return button

    def _download_name(graph_id: str):
        name = dbc.Input(id=f'{graph_id}_inp-download-name', type='text', placeholder='Download Name')
        return name

    header_id = f'h3-{id_name}'
    header_layout = dbc.CardHeader(
        dbc.Row([
            dbc.Col(html.H3(id=header_id, children=graph_header), width='auto'),
            _graph_save_options(id_name),
        ], justify='between')
    )

    graph_body = dcc.Graph(id=id_name, config=dict(editable=True))  # editable allows changing text and labels
    graph = dbc.Card([
        header_layout, graph_body
    ])
    if pending_callbacks is not None:
        callback_infos = _get_graph_callbacks(id_name)  # ALl the callbacks for downloading the graph
        pending_callbacks.extend(callback_infos)
    graph.graph_id = id_name  # So that it is easy to get to the graphs id through the returned card
    graph.header_id = header_id  # So easy to get to header of graph through returned card
    return graph


@dataclass
class DownloadInfo:
    data: Optional[Any]  # The data to be sent for download
    mtype: Optional[str]  # Mimetype of data (google to find them)
    base64: bool  # Whether this is base64 info, necessary for dash-extensions to know
    file_extension: str  # Extension to filename
