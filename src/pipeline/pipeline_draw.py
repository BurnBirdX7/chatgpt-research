from typing import List, Tuple

from matplotlib.axes import Axes

from .pipeline import Pipeline

import networkx as nx  # type: ignore
from matplotlib import pyplot as plt


def draw_network(pipeline: Pipeline, ax: Axes, g: nx.DiGraph, colors: List[str], exec_edges: List[Tuple[str, str]]):
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
    nx.draw_networkx(g, layout, ax=ax, node_size=1500, node_color=colors, with_labels=True)
    nx.draw_networkx_edges(g, layout, ax=ax, node_size=1500, edgelist=exec_edges, edge_color="tab:red",
                           connectionstyle="arc3,rad=0.35")
    nx.draw_networkx_edges(g, layout, ax=ax, node_size=1500, edge_color="grey")
    nx.draw_networkx_edge_labels(g, layout, ax=ax, edge_labels=edge_labels, label_pos=0.30)


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


def draw(pipeline: Pipeline, ax1: Axes, ax2: Axes):
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

    draw_network(pipeline, ax1, g, colors, exec_edges)
    draw_execution(pipeline, ax2, g, colors, exec_edges)
    ax1.set_title("Data flow")
    ax2.set_title("Execution order")


def show(pipeline: Pipeline):
    fix, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), height_ratios=[11.6, 0.4])
    draw(pipeline, ax1, ax2)
    plt.show()


def __into_networkx(self) -> nx.DiGraph:
    g = nx.DiGraph(self.graph).reverse()
    g.add_edge(self.output_node.name, "$output")
    return g

