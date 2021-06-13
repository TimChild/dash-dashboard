"""
Intended to work with Dash Extensions and Dash Labs
"""
from __future__ import annotations
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html

from typing import List

import dash_labs as dl
from plotly import graph_objects as go


class MultiGraph(dl.component_plugins.ComponentPlugin):
    def __init__(self):
        self.output_id = dl.util.build_id('multi-graph')
        output = dl.Output(html.Div(id=self.output_id), 'children', role='output')
        args = tuple()
        template = None
        super().__init__(args, output, template)

    def get_output_values(self, args_value, figs: List[go.Figure]) -> html.Div:
        """Generate a grid of figures inside an html.Div"""
        items = [self.generate_single_fig(fig) for fig in figs]
        grid = self.items_to_grid(items)
        return grid

    def generate_single_fig(self, fig: go.Figure):
        """Turns a go.Figure into an item which displays nicely in a grid of figures"""
        title = fig.layout.title.text
        fig.layout.title.text = ''
        fig.update_layout(margin=dict(l=40, r=10, b=50, t=10, pad=4),
                          height=300)
        card = dbc.Card(
            [dbc.CardHeader(title),
             dbc.CardBody(dcc.Graph(figure=fig))]
        )
        return card

    def items_to_grid(self, items) -> html.Div:
        col_items = [dbc.Col(item, width=6, md=4, lg=3) for item in items]
        return html.Div(dbc.Row(col_items))


