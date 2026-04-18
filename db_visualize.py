"""Visualize the supply-chain DiGraph built from the SQLite DB."""

from pyvis.network import Network

from app import build_graph

NODE_STYLE = {
    "Company":  {"color": "#E74C3C", "shape": "box"},
    "Product":  {"color": "#3498DB", "shape": "dot"},
    "Supplier": {"color": "#2ECC71", "shape": "triangle"},
}

EDGE_COLOR = {
    "OWNS":     "#E74C3C",
    "CONTAINS": "#F39C12",
    "OFFERS":   "#2ECC71",
}


def _node_label(data: dict) -> str:
    node_type = data.get("node_type", "")
    if node_type == "Company":
        return f"{data.get('Name', '')}"
    if node_type == "Supplier":
        return f"{data.get('Name', '')}"
    if node_type == "Product":
        return f"{data.get('SKU', '')} ({data.get('Type', '')})"
    return str(data.get("Id", ""))


def _node_tooltip(node_id: str, data: dict) -> str:
    lines = [f"<b>{node_id}</b>"]
    for k, v in data.items():
        lines.append(f"{k}: {v}")
    return "<br>".join(lines)


def _styled_pyvis(G, directed: bool = True) -> Network:
    net = Network(height="800px", width="100%", directed=directed, notebook=False)
    net.force_atlas_2based()

    for node_id, data in G.nodes(data=True):
        style = NODE_STYLE.get(data.get("node_type"), {"color": "#BDC3C7", "shape": "dot"})
        net.add_node(
            node_id,
            label=_node_label(data),
            title=_node_tooltip(node_id, data),
            color=style["color"],
            shape=style["shape"],
        )

    for src, dst, data in G.edges(data=True):
        edge_type = data.get("edge_type", "")
        net.add_edge(
            src,
            dst,
            label=edge_type,
            title=edge_type,
            color=EDGE_COLOR.get(edge_type, "#95A5A6"),
        )

    return net


def visualize_full_graph(G=None, output: str = "supply_chain_graph.html") -> str:
    """Render the entire graph to an interactive HTML file."""
    if G is None:
        G = build_graph()
    net = _styled_pyvis(G)
    net.write_html(output, notebook=False)
    print(f"Wrote full graph ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges) -> {output}")
    return output


def visualize_subgraph(G, product_id, output: str = "subgraph.html") -> str:
    """Render the ego subgraph around a single product."""
    center = f"PROD_{product_id}"
    if center not in G:
        raise ValueError(f"{center} not in graph")

    neighbors = set(G.predecessors(center)) | set(G.successors(center))
    nodes = neighbors | {center}
    subgraph = G.subgraph(nodes)

    net = _styled_pyvis(subgraph)
    net.write_html(output, notebook=False)
    print(f"Wrote subgraph for {center} ({subgraph.number_of_nodes()} nodes) -> {output}")
    return output


if __name__ == "__main__":
    visualize_full_graph()
