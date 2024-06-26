from io import BytesIO
from typing import List, Tuple

from matplotlib.axes import Axes

from .pipeline import Pipeline

import networkx as nx  # type: ignore
from matplotlib import pyplot as plt


__all__ = ["draw_pipeline", "show_pipeline", "bytes_draw_pipeline"]


def _draw_pipeline_network(
    pipeline: Pipeline,
    graph_axes: Axes,
    execution_axes: Axes,
    g: nx.DiGraph,
    colors: List[str],
    exec_edges: List[Tuple[str, str]],
):
    edge_labels = {}
    for u, vs in pipeline.graph.items():
        for v in vs:
            if v in pipeline.nodes:
                typ = pipeline.nodes[v].out_type
            else:
                typ = pipeline.in_type
            edge_labels[(v, u)] = typ.__name__
    edge_labels[(pipeline.output_node.name, "$output")] = pipeline.output_node.out_type.__name__
    g1 = g.copy()
    g1.add_edges_from(exec_edges)

    layout = nx.kamada_kawai_layout(g1, pos=nx.circular_layout(g))

    # Draw dataflow
    nx.draw_networkx(g, layout, ax=graph_axes, node_size=1500, node_color=colors, with_labels=True)
    nx.draw_networkx_edges(g, layout, ax=graph_axes, node_size=1500, edge_color="lightgrey")
    nx.draw_networkx_edge_labels(g, layout, ax=graph_axes, edge_labels=edge_labels, label_pos=0.30)

    # Draw execution flow
    nx.draw_networkx_nodes(g, layout, ax=execution_axes, node_size=1500, node_color=colors)
    nx.draw_networkx_labels(g, layout, ax=execution_axes)
    nx.draw_networkx_edges(
        g,
        layout,
        ax=execution_axes,
        node_size=1500,
        edgelist=exec_edges,
        edge_color="tab:red",
    )


def _draw_execution_line(
    pipeline: Pipeline,
    ax: Axes,
    g: nx.DiGraph,
    colors: List[str],
    exec_edges: List[Tuple[str, str]],
):
    exec_order = pipeline.default_execution_order
    node_count = len(exec_order)
    distance = 2 / (node_count + 1)
    layout = {node: (-1 + (i + 1) * distance, 0) for i, node in enumerate(exec_order)} | {
        "$input": (-1, 0),
        "$output": (+1, 0),
    }

    nx.draw(
        g,
        pos=layout,
        ax=ax,
        edgelist=exec_edges,
        node_size=500,
        node_color=colors,
        edge_color="tab:red",
    )


def _draw_pipeline(pipeline: Pipeline, ax: Tuple[Axes, Axes, Axes]):
    g = _pipeline_into_networkx(pipeline)
    colors = []
    for node, auxiliary in g.nodes(data="auxiliary", default=False):
        if node == "$input":
            colors.append("tab:green")
        elif node == "$output":
            colors.append("tab:red")
        elif auxiliary:
            colors.append("tab:grey")
        else:
            colors.append("skyblue")

    exec_edges = [("$input", pipeline.input_node.name)]
    exec_order = pipeline.default_execution_order
    for i in range(len(exec_order) - 1):
        exec_edges.append((exec_order[i], exec_order[i + 1]))
    exec_edges.append((pipeline.output_node.name, "$output"))

    _draw_pipeline_network(pipeline, ax[0], ax[1], g, colors, exec_edges)
    _draw_execution_line(pipeline, ax[2], g, colors, exec_edges)
    ax[0].set_title("Data flow")
    ax[1].set_title("Execution order")
    ax[2].set_title("Execution order")


def draw_pipeline(pipeline: Pipeline) -> plt.Figure:
    fig, ((ax11, ax12), (ax21, ax22)) = plt.subplots(2, 2, figsize=(30, 18), height_ratios=[11.6, 0.4])

    gs = ax21.get_gridspec()
    ax21.remove()
    ax22.remove()
    ax3 = fig.add_subplot(gs[1, :])
    fig.tight_layout()

    _draw_pipeline(pipeline, (ax11, ax12, ax3))

    return fig


def bytes_draw_pipeline(pipeline: Pipeline) -> BytesIO:
    fig = draw_pipeline(pipeline)
    img = BytesIO()
    fig.savefig(img, format="png")
    img.seek(0)
    return img


def show_pipeline(pipeline: Pipeline):
    draw_pipeline(pipeline)
    plt.show()


def _pipeline_into_networkx(pipeline: Pipeline) -> nx.DiGraph:
    g = nx.DiGraph(pipeline.graph).reverse()
    for node in pipeline.auxiliary_nodes:
        g.nodes[node]["auxiliary"] = True

    g.add_edge(pipeline.output_node.name, "$output")
    return g
