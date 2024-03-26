from typing import List, Tuple

from matplotlib.axes import Axes

from .pipeline import Pipeline

import networkx as nx  # type: ignore
from matplotlib import pyplot as plt


def draw_network(pipeline: Pipeline,
                 graph_axes: Axes, execution_axes: Axes,
                 g: nx.DiGraph, colors: List[str], exec_edges: List[Tuple[str, str]]):
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

    layout = nx.kamada_kawai_layout(g1, pos=nx.circular_layout(g) | {'$input': (-1, 1)})

    # Draw dataflow
    nx.draw_networkx(g, layout, ax=graph_axes, node_size=1500, node_color=colors, with_labels=True)
    nx.draw_networkx_edges(g, layout, ax=graph_axes, node_size=1500, edge_color="grey")
    nx.draw_networkx_edge_labels(g, layout, ax=graph_axes, edge_labels=edge_labels, label_pos=0.30)

    # Draw execution flow
    nx.draw_networkx_nodes(g, layout, ax=execution_axes, node_size=1500, node_color=colors)
    nx.draw_networkx_labels(g, layout, ax=execution_axes)
    nx.draw_networkx_edges(g, layout, ax=execution_axes, node_size=1500, edgelist=exec_edges, edge_color="tab:red",
                           connectionstyle="arc3,rad=0.35")


def draw_execution(pipeline: Pipeline, ax: Axes, g: nx.DiGraph, colors: List[str], exec_edges: List[Tuple[str, str]]):
    node_count = len(pipeline.execution_plan)
    distance = 2 / (node_count + 1)
    layout = {
                 node.name: (-1 + (i + 1) * distance, 0)
                 for i, node in enumerate(pipeline.execution_plan)
             } | {"$input": (-1, 0), "$output": (+1, 0)}

    nx.draw(g,
            pos=layout,
            ax=ax,
            edgelist=exec_edges,
            node_size=500,
            node_color=colors,
            edge_color="tab:red")


def draw(pipeline: Pipeline, ax: Tuple[Axes, Axes, Axes]):
    g = __into_networkx(pipeline)
    colors = []
    for node in g.nodes():
        if node == "$input":
            colors.append("tab:green")
        elif node == "$output":
            colors.append("tab:red")
        else:
            colors.append("skyblue")

    exec_edges = [("$input", pipeline.input_node.name)]
    for i in range(len(pipeline.execution_plan) - 1):
        exec_edges.append((pipeline.execution_plan[i].name, pipeline.execution_plan[i + 1].name))
    exec_edges.append((pipeline.output_node.name, "$output"))

    draw_network(pipeline, ax[0], ax[1], g, colors, exec_edges)
    draw_execution(pipeline, ax[2], g, colors, exec_edges)
    ax[0].set_title("Data flow")
    ax[1].set_title("Execution order")
    ax[2].set_title("Execution order")


def show(pipeline: Pipeline):
    fig, ((ax11, ax12), (ax21, ax22)) = plt.subplots(2, 2, figsize=(20, 12), height_ratios=[11.6, 0.4])

    gs = ax21.get_gridspec()
    ax21.remove()
    ax22.remove()
    ax3 = fig.add_subplot(gs[1, :])
    fig.tight_layout()

    draw(pipeline, (ax11, ax12, ax3))
    plt.show()


def __into_networkx(self) -> nx.DiGraph:
    g = nx.DiGraph(self.graph).reverse()
    g.add_edge(self.output_node.name, "$output")
    return g

