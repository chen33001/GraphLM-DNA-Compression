"""de Bruijn graph construction utilities."""

from collections import Counter, defaultdict
import re

DeBruijnGraph = dict[str, Counter[str]]


def build_debruijn_graph(sequence: str, k: int = 15) -> DeBruijnGraph:
    """Build a directed, edge-counted de Bruijn graph from a sequence."""
    if not sequence or k <= 0 or k >= len(sequence):
        raise ValueError("k must be positive and smaller than the sequence length")
    graph: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for index in range(len(sequence) - k):
        source = sequence[index : index + k]
        target = sequence[index + 1 : index + k + 1]
        graph[source][target] += 1
    return dict(graph)


def graph_path_frequencies(graph: DeBruijnGraph) -> dict[tuple[str, str], int]:
    """Flatten edge frequencies for inspection and benchmarking."""
    return {
        (source, target): count
        for source, targets in graph.items()
        for target, count in targets.items()
    }


def graph_to_mermaid(
    graph: DeBruijnGraph,
    *,
    max_edges: int | None = None,
    min_count: int = 1,
) -> str:
    """Render a de Bruijn graph as Mermaid flowchart text."""
    if min_count <= 0:
        raise ValueError("min_count must be positive")

    edge_items = [
        (source, target, count)
        for source, targets in graph.items()
        for target, count in targets.items()
        if count >= min_count
    ]
    edge_items.sort(key=lambda item: (-item[2], item[0], item[1]))
    if max_edges is not None:
        if max_edges <= 0:
            raise ValueError("max_edges must be positive when provided")
        edge_items = edge_items[:max_edges]

    lines = ["flowchart LR"]
    if not edge_items:
        lines.append('    empty["No edges to display"]')
        return "\n".join(lines)

    seen_nodes: set[str] = set()
    for source, target, count in edge_items:
        for node in (source, target):
            if node not in seen_nodes:
                lines.append(f'    {_mermaid_node_id(node)}["{node}"]')
                seen_nodes.add(node)
        lines.append(
            f"    {_mermaid_node_id(source)} -->|{count}| {_mermaid_node_id(target)}"
        )
    return "\n".join(lines)


def _mermaid_node_id(label: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_]", "_", label)
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"node_{cleaned}"
    return cleaned
