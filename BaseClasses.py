"""
This provides some helpful classes for making layouts of pages easier. Everything in here should be fully
general to ANY Dash app, not just Dat analysis. For Dat analysis specific, implement in DatSpecificDash.
"""
from __future__ import annotations
from deprecation import deprecated
from dataclasses import dataclass
from dash_extensions.enrich import Input, Output, State
import threading
from typing import Optional, List, Union, Callable, Tuple, Type
import abc
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc
from dash_extensions.enrich import Trigger
from dash_extensions.multipage import PageCollection
import dash
import logging


logger = logging.getLogger(__name__)

CALLBACK_TYPE = Tuple[str, str]  # All Inputs/Outputs/States/Triggers require (<id>, <target>)
LAYOUT_TYPE = Union[dbc.Container, html.Div, dbc.Row, dbc.Col, dbc.Card]

# e.g. ('button-test', 'n_clicks')


def get_trig_id(ctx=None) -> str:
    """Pass in dash.callback_context to get the id of the trigger returned"""
    if ctx is None:
        ctx = dash.callback_context
    return ctx.triggered[0]['prop_id'].split('.')[0]


@deprecated(details='Think I should always avoid using this in order to achieve a stateless backend')
class EnforceSingleton:
    """Simplified from https://www.reddit.com/r/Python/comments/2qkwgh/class_to_enforce_singleton_pattern_on_subclasses/

    Enforces that subclasses are singletons (or are only instantiated once).
    """
    singletonExists = False
    lock = threading.Lock()

    def __init__(self):
        with EnforceSingleton.lock:
            if self.__class__.singletonExists:
                raise Exception(f'Instance already exists for {self.__class__.__name__}. There should only be 1 '
                                f'instance. Consider adding @singleton decorator to the subclass')
            self.__class__.singletonExists = True


class PendingCallbacks(list):
    """
    Stores callbacks functions which should be executed at app runtime
    """

    def output_exists(self, output: Output) -> bool:
        """Checks if (output_id, property) is in any of the current CallbackInfos"""
        for item in self:
            if output in item.outputs:
                return True
        return False

    def get_callback_for_output(self, output: Output) -> Optional[CallbackInfo]:
        for item in self:
            if output in item.outputs:
                return item
        return None

    def append(self, item: CallbackInfo):
        for output in item.outputs:
            if self.output_exists(output):
                self.get_callback_for_output(output)
                raise RuntimeError(f'{output} already exists: {item}')
        super().append(item)

    def __init__(self, items: List[CallbackInfo] = None):
        if items is None:
            items = []
        super().__init__(items)


@dataclass
class CallbackInfo:
    func: Callable
    outputs: Union[List[Output], Output]
    inputs: Optional[Union[List[Input], Input]]
    states: Optional[Union[List[State], State]]
    triggers: Optional[Union[List[Trigger], Trigger]]


class BaseDashRequirements(abc.ABC):
    """
    Things that are useful or necessary for all of my Dash page classes
    """

    def __init__(self, page_components: PageInteractiveComponents):
        self.components = page_components  # All the interactive components of a page
        self.pending_callbacks: PendingCallbacks = PendingCallbacks()  # List of CallbackInfo

    @abc.abstractmethod
    def layout(self):
        """Should return the full layout of whatever the relevant part is
        Examples:
            layout = html.Div([
                        child1,
                        child2,
                        etc,
                        ])
            return layout
        """
        raise NotImplementedError

    def make_callback(self, inputs: Union[List[CALLBACK_TYPE], CALLBACK_TYPE] = None,
                      outputs: Union[List[CALLBACK_TYPE], CALLBACK_TYPE] = None,
                      func: Callable = None,
                      states: Union[List[CALLBACK_TYPE], CALLBACK_TYPE] = None,
                      triggers: Union[List[CALLBACK_TYPE], CALLBACK_TYPE] = None):

        """
        Helper function for attaching callbacks more easily

        Args:
            inputs (List[CALLBACK_TYPE]): The tuples that would go into dash.dependencies.Input() (i.e. (<id>, <property>)
            outputs (List[CALLBACK_TYPE]): Similar, (<id>, <property>)
            states (List[CALLBACK_TYPE]): Similar, (<id>, <property>)
            func (Callable): The function to wrap with the callback (make sure it takes the right number of inputs in order and returns the right number of outputs in order)
            triggers (): Triggers callback but is not passed to function

        Returns:

        """

        def ensure_list(val) -> List[CALLBACK_TYPE]:
            if isinstance(val, tuple):
                return [val]
            elif val is None:
                return []
            elif isinstance(val, list):
                return val
            else:
                raise TypeError(f'{val} is not valid')

        if inputs is None and triggers is None:
            raise ValueError(f"Can't have both inputs and triggers set as None... "
                             f"\n{inputs, triggers, outputs, states}")

        inputs, outputs, states, triggers = [ensure_list(v) for v in [inputs, outputs, states, triggers]]

        Inputs = [Input(*inp) for inp in inputs]
        Outputs = [Output(*out) for out in outputs]
        States = [State(*s) for s in states]
        Triggers = [Trigger(*t) for t in triggers]

        callback_info = CallbackInfo(func=func, outputs=Outputs, inputs=Inputs, states=States, triggers=Triggers)
        self.pending_callbacks.append(callback_info)


class BasePageLayout(BaseDashRequirements):
    """
    The overall page layout which should be used per page of the app.
    Switching between whole pages will reset things when going back to previous pages.

    For switching between similar sections where it is beneficial to move back and forth, the contents area should be
    hidden/unhidden to "switch" back and forth. This will not reset progress, and any callbacks which apply to several
    unhidden/hidden parts will be applied to all of them.
    """

    def __init__(self, page_components: PageInteractiveComponents):
        super().__init__(page_components)
        self.mains = self.get_mains()
        self.sidebar = self.get_sidebar()
        self.page_collection: PageCollection = None

    @abc.abstractmethod
    def get_mains(self) -> List[BaseMain]:
        """
        Override to return list of BaseMain areas to use in Page

        Note: Pass in self.components to initialization so that the same components are shared
        Examples:
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_sidebar(self) -> BaseSideBar:
        """
        Override to return the SideBar which will be used for the Page

        Note: pass in self.components for initialization so that the same components are shared
        Examples:
            return TestSideBar()  # where TestSideBar is a subclass of SideBar
        """
        raise NotImplementedError

    def layout(self, *args, **kwargs) -> LAYOUT_TYPE:
        """
        Overall layout of main Pages, generally this should not need to be changed, but more complex pages can be
        created by subclassing this. Just don't forget to still include a top_bar_layout which has the links to all
        pages.
        """
        logger.debug(f'Layout called on {self}: args = {args}, kwargs = {kwargs}')
        self.pending_callbacks = PendingCallbacks()  # Reset pending callbacks for each load of layout
        self.sidebar.pending_callbacks = PendingCallbacks()
        for main in self.mains:
            main.pending_callbacks = PendingCallbacks()

        layout = dbc.Container(fluid=True, className='p-0', children=[
            dbc.Row(
                className='header-bar',
                children=dbc.Col(self.top_bar_layout())
            ),
            dbc.Container(
                fluid=True,
                className='below-header',
                children=[
                    dbc.Container(fluid=True, className='sidebar',
                                  children=self.side_bar_layout()
                                  ),
                    dbc.Container(fluid=True, className='content-area',
                                  children=self.main_area_layout()
                                  )
                ])
        ])
        return layout

    def top_bar_layout(self) -> dbc.NavbarSimple:
        """
        This generally should not be changed since this is what is used to switch between whole Pages, but in case
        something else should be added to the top bar for a certain page, this can be overridden (just remember to
        include a call to super().top_bar_layout() and incorporate that into the new top_bar_layout.
        """
        if self.page_collection is not None:
            layout = dbc.NavbarSimple(
                [dbc.NavItem(dbc.NavLink(page.label, href=page.id)) for page in self.page_collection.pages],
                brand=self.top_bar_title(),
                )
        else:
            layout = dbc.NavbarSimple('No pagecollection passed', brand=self.top_bar_title())
        return layout

    @abc.abstractmethod
    def top_bar_title(self) -> str:
        """override to return a title to show in the top navbar"""
        raise NotImplementedError

    def main_area_layout(self) -> LAYOUT_TYPE:
        """
        Makes the main area layout based on self.get_mains()
        """
        # A basic layout which enables showing only one "main" part at a time
        layout = html.Div([html.Div(main.layout(), id=main.name) for main in self.mains])
        return layout

    def side_bar_layout(self):
        """
        Override this to return a layout from BaseSideBar
        Examples:
            return BaseSideBar().layout()
        """
        return self.sidebar.layout()

    def set_callbacks(self):
        def _main_dd_callback_func() -> Callable:
            opts = [d['value'] for d in self.components.dd_main.options]

            def func(inp):
                outs = {k: True for k in opts}
                if inp is not None and inp in outs:
                    outs[inp] = False  # Set selected page to False (not hidden)
                else:
                    outs[next(iter(outs))] = False  # Otherwise set first page to False (not hidden)
                ret = list(outs.values())
                if len(ret) == 1:  # If only outputting to 1 main, then return value not list
                    ret = False
                return ret
            return func

        # Main dd selection
        # First add some more initialization info
        labels = [main.name for main in self.mains]
        self.components.dd_main.options = [{'label': label, 'value': label} for label in labels]
        self.components.dd_main.value = labels[0]
        # Then make callback
        self.make_callback(inputs=(self.components.dd_main.id, 'value'),
                           outputs=[(id_, 'hidden') for id_ in labels],
                           func=_main_dd_callback_func())

    def run_all_callbacks(self, app: dash.Dash):
        """This is the only place where callbacks should be run, and will only be run ONCE on initialization of app"""
        self.set_callbacks()
        self.sidebar.set_callbacks()
        all_callbacks = self.pending_callbacks + self.sidebar.pending_callbacks
        for main in self.mains:
            main.set_callbacks()
            all_callbacks.extend(main.pending_callbacks)

        for callback in all_callbacks:
            app.callback(*callback.inputs, *callback.outputs, *callback.states, *callback.triggers)(callback.func)


class BaseMain(BaseDashRequirements):
    """
    This is the area that should be hidden/unhidden for sections of app which are closely related, i.e. looking at
    different aspects of the same dat, or different ways to look at multiple dats. Everything shown in this main area
    should rely on the same sidebar

    There may be several different instances/subclasses of this for a single full page, but all of which share the same
    sidebar and hide/unhide in the same main area
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Override to provide the name which will show up in the main dropdown
        Note: can just set a class attribute instead of a whole property"""
        pass

    @abc.abstractmethod
    def set_callbacks(self):
        """
        Override this to run all callbacks for main area (generally callbacks depending on things in self.sidebar)

        This method is called by the BaseLayout
        """
        raise NotImplementedError

    def layout(self) -> LAYOUT_TYPE:
        return html.Div()


class PageInteractiveComponents(abc.ABC):
    """A class for listing all components involved in callbacks on page so that it is easier to work with them in Sidebar, Main etc

    Intention is to subclass this for each page.
    """
    def __init__(self):
        self.dd_main = dcc.Dropdown(id='dd-main')


class CommonInputCallbacks(abc.ABC):
    """Helper class for generating callbacks which rely on a lot of the same input information. Makes it much easier to
    make new functions which common information without making WET code.
    Examples:
        Several figures on a page which show different things, but all based on the save 5 inputs.

    """
    @abc.abstractmethod
    def __init__(self, *args):
        """Override to accept all common inputs and store them however you like. Then you can rely on any of this info
        in any callback function
        Examples:
            def __init__(self, a, b, c):
                self.a = a
                self.b = b
                self.c = c
                self.d = a*b*c  # i.e. can also run some common initialization here
        """
        pass

    @abc.abstractmethod
    def callback_names_funcs(self):
        """Override to return a dict of {<name>: <callback_func>}
        Examples:
            return {
                        'func_a': self.do_func_a(),
                        'func_b': self.do_func_b(),
                    }

            where self.do_func_a() is:
            def do_func_a():
                return self.a*self.b*self.c, self.d  # i.e. two outputs in this example
        """
        pass

    @classmethod
    def get_callback(cls, callback_name: str):
        """Generates a callback function which takes all arguments that __init__ takes and will do whatever
        the corresponding function is for "callback_name" in self.callback_names_funcs"""
        def callback_func(*args):
            inst = cls(*args)
            callback_options = inst.callback_names_funcs()
            if callback_name in callback_options:
                return callback_options[callback_name]
            else:
                raise KeyError(f'{callback_name} not found in {callback_options.keys()}')
        return callback_func


# class ComponentStore(dict):
#     """For keeping a dictionary of all components in side bar in {name: component} format"""
#     # def __getitem__(self, item):
#     #     """Because dash-extensions modifies item.id to add a prefix, return copy only so that if called again, it
#     #     doesn't add more and more id_prefixes"""
#     #     return copy.copy(super().__getitem__(item))
#     pass


class BaseSideBar(BaseDashRequirements):
    """
    This should be subclassed for each full page to give relevant sidebar options for each main section of the app
    (i.e. working with single dats will require different options in general than comparing multiple dats)
    """

    @property
    @abc.abstractmethod
    def id_prefix(self):
        """Something which returns an ID prefix for any ID in the sidebar"""
        return "BaseSidebar"

    def layout(self):
        """Return the full layout of sidebar to be used"""
        layout = html.Div([
            self.input_wrapper(text='This adds a label', children=[
                self.input_box(id_name='inp-number', placeholder='Choose Number', val_type='number', autoFocus=True)
            ]),
        ])
        return layout

    @abc.abstractmethod
    def set_callbacks(self):
        """Override this to set any callbacks which update items in the sidebar"""
        pass

    @staticmethod
    def input_wrapper(text: str, children, mode: str = 'addon') -> Union[dbc.InputGroup, dbc.FormGroup]:
        """
        "Wraps 'children' in a group which adds an addon prefix or label"
        Args:
            text (): Text to show in addon or label
            children (): Things to wrap in group (i.e. the input)
            mode (): 'addon' or 'label' for how children should be wrapped

        Returns:
            InputGroup or FormGroup depending on mode
        """
        if not isinstance(children, list):
            children = [children]
        if mode == 'addon':
            addon = dbc.InputGroupAddon(text, addon_type='prepend')
            ret = dbc.InputGroup([addon, *children])
        elif mode == 'label':
            label = dbc.Label(text)
            ret = dbc.FormGroup([label, *children])
        else:
            raise ValueError(f'{mode} not recognized')
        return ret


def test_page(layout: Callable, callbacks: Callable):
    """
    Makes a Dash app and runs loads layout and callbacks from layout_class in a similar way to how
    the PageCollection will when added in main app
    """
    from dash_extensions.enrich import DashProxy, TriggerTransform
    app = DashProxy(
        transforms=[
            TriggerTransform(),
        ],
        name=__name__, external_stylesheets=[dbc.themes.BOOTSTRAP]
    )
    app.layout = layout
    callbacks(app)
    app.run_server(port=8056, debug=True)
