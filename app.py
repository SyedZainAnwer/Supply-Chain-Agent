"""Migrate the supply-chain SQLite schema into a networkx DiGraph."""

import sqlite3
from collections import Counter

import networkx as nx
import pandas as pd

DB_PATH = "db/db.sqlite"

COMPANY_PREFIX = "COMP"
PRODUCT_PREFIX = "PROD"
SUPPLIER_PREFIX = "SUPP"


def _node_id(prefix: str, raw_id) -> str:
    return f"{prefix}_{raw_id}"


def _add_nodes_from_df(
    G: nx.DiGraph, df: pd.DataFrame, prefix: str, node_type: str, id_col: str = "Id"
) -> None:
    for row in df.to_dict(orient="records"):
        node_id = _node_id(prefix, row[id_col])
        G.add_node(node_id, node_type=node_type, **row)


def build_graph(db_path: str = DB_PATH) -> nx.DiGraph:
    """Read the relational schema from SQLite and return a populated DiGraph."""
    G = nx.DiGraph()

    with sqlite3.connect(db_path) as conn:
        companies = pd.read_sql_query("SELECT * FROM Company", conn)
        products = pd.read_sql_query("SELECT * FROM Product", conn)
        suppliers = pd.read_sql_query("SELECT * FROM Supplier", conn)
        supplier_product = pd.read_sql_query("SELECT * FROM Supplier_Product", conn)
        bom_edges = pd.read_sql_query(
            """
            SELECT b.ProducedProductId AS produced_id,
                   bc.ConsumedProductId AS consumed_id
            FROM BOM AS b
            JOIN BOM_Component AS bc ON bc.BOMId = b.Id
            """,
            conn,
        )

    _add_nodes_from_df(G, companies, COMPANY_PREFIX, "Company")
    _add_nodes_from_df(G, products, PRODUCT_PREFIX, "Product")
    _add_nodes_from_df(G, suppliers, SUPPLIER_PREFIX, "Supplier")

    for row in products.itertuples(index=False):
        G.add_edge(
            _node_id(COMPANY_PREFIX, row.CompanyId),
            _node_id(PRODUCT_PREFIX, row.Id),
            edge_type="OWNS",
        )

    for row in bom_edges.itertuples(index=False):
        G.add_edge(
            _node_id(PRODUCT_PREFIX, row.produced_id),
            _node_id(PRODUCT_PREFIX, row.consumed_id),
            edge_type="CONTAINS",
        )

    for row in supplier_product.itertuples(index=False):
        G.add_edge(
            _node_id(SUPPLIER_PREFIX, row.SupplierId),
            _node_id(PRODUCT_PREFIX, row.ProductId),
            edge_type="OFFERS",
        )

    return G


def _print_diagnostics(G: nx.DiGraph) -> None:
    node_counts = Counter(data.get("node_type", "Unknown") for _, data in G.nodes(data=True))
    edge_counts = Counter(data.get("edge_type", "Unknown") for _, _, data in G.edges(data=True))

    print(f"Total nodes: {G.number_of_nodes()}")
    for node_type, count in sorted(node_counts.items()):
        print(f"  {node_type}: {count}")

    print(f"Total edges: {G.number_of_edges()}")
    for edge_type, count in sorted(edge_counts.items()):
        print(f"  {edge_type}: {count}")


if __name__ == "__main__":
    graph = build_graph()
    _print_diagnostics(graph)
