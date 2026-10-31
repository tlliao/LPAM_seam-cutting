from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GraphCutResult:
    cut_value: float
    labels: np.ndarray
    backend: str


def _finite_capacities(terminal_weights: np.ndarray, edge_weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tw = np.asarray(terminal_weights, dtype=np.float64).copy()
    ew = np.asarray(edge_weights, dtype=np.float64).copy()
    if tw.ndim != 2 or tw.shape[1] != 2:
        raise ValueError("terminal_weights must have shape (num_nodes, 2)")
    if ew.ndim != 2 or ew.shape[1] != 4:
        raise ValueError("edge_weights must have shape (num_edges, 4)")

    finite_values = []
    if np.isfinite(tw).any():
        finite_values.append(np.nanmax(tw[np.isfinite(tw)]))
    if np.isfinite(ew[:, 2:]).any():
        finite_values.append(np.nanmax(ew[:, 2:][np.isfinite(ew[:, 2:])]))
    max_finite = max(finite_values) if finite_values else 1.0
    hard_cap = max(1e9, float(max_finite) * (tw.shape[0] + ew.shape[0] + 1) * 10.0)
    tw[~np.isfinite(tw)] = hard_cap
    ew[:, 2:][~np.isfinite(ew[:, 2:])] = hard_cap
    return tw, ew


def graph_cut(terminal_weights: np.ndarray, edge_weights: np.ndarray) -> GraphCutResult:
    """Run binary s-t min-cut using MATLAB graphCutMex-compatible inputs.

    `terminal_weights[:, 0]` is the source-to-node capacity and
    `terminal_weights[:, 1]` is the node-to-sink capacity. Labels follow
    graphCutMex: 0 for source side, 1 for sink side.
    """

    tw, ew = _finite_capacities(terminal_weights, edge_weights)
    try:
        import maxflow  # type: ignore

        return _graph_cut_pymaxflow(tw, ew, maxflow)
    except ImportError:
        return _graph_cut_networkx(tw, ew)


def _graph_cut_pymaxflow(tw: np.ndarray, ew: np.ndarray, maxflow_module) -> GraphCutResult:
    graph = maxflow_module.Graph[float](tw.shape[0], ew.shape[0])
    graph.add_nodes(tw.shape[0])

    for node_id, (source_cap, sink_cap) in enumerate(tw):
        if source_cap != 0.0 or sink_cap != 0.0:
            graph.add_tedge(node_id, float(source_cap), float(sink_cap))

    for src, dst, cap_forward, cap_backward in ew:
        graph.add_edge(int(src), int(dst), float(cap_forward), float(cap_backward))

    cut_value = float(graph.maxflow())
    labels = np.fromiter((graph.get_segment(i) for i in range(tw.shape[0])), dtype=np.uint8)
    return GraphCutResult(cut_value=cut_value, labels=labels, backend="pymaxflow")


def _graph_cut_networkx(tw: np.ndarray, ew: np.ndarray) -> GraphCutResult:
    import networkx as nx

    source = "__source__"
    sink = "__sink__"
    graph = nx.DiGraph()
    graph.add_node(source)
    graph.add_node(sink)
    graph.add_nodes_from(range(tw.shape[0]))

    for node_id, (source_cap, sink_cap) in enumerate(tw):
        if source_cap > 0:
            graph.add_edge(source, node_id, capacity=float(source_cap))
        if sink_cap > 0:
            graph.add_edge(node_id, sink, capacity=float(sink_cap))

    for src, dst, cap_forward, cap_backward in ew:
        src_i = int(src)
        dst_i = int(dst)
        if cap_forward > 0:
            graph.add_edge(src_i, dst_i, capacity=float(cap_forward))
        if cap_backward > 0:
            graph.add_edge(dst_i, src_i, capacity=float(cap_backward))

    cut_value, partition = nx.minimum_cut(graph, source, sink, capacity="capacity")
    source_side, _ = partition
    labels = np.ones(tw.shape[0], dtype=np.uint8)
    node_ids = [node for node in source_side if isinstance(node, int)]
    labels[node_ids] = 0
    return GraphCutResult(cut_value=float(cut_value), labels=labels, backend="networkx")


def cut_energy(labels: np.ndarray, terminal_weights: np.ndarray, edge_weights: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.uint8)
    tw, ew = _finite_capacities(terminal_weights, edge_weights)
    energy = float(np.sum(tw[labels == 0, 1]) + np.sum(tw[labels == 1, 0]))

    src = ew[:, 0].astype(np.int64)
    dst = ew[:, 1].astype(np.int64)
    src_labels = labels[src]
    dst_labels = labels[dst]
    energy += float(np.sum(ew[(src_labels == 0) & (dst_labels == 1), 2]))
    energy += float(np.sum(ew[(src_labels == 1) & (dst_labels == 0), 3]))
    return energy
