from langchain_neo4j import Neo4jGraph
from tqdm import tqdm


def create_nodes(graph: Neo4jGraph, data: dict, node_label: str, node_name: str) -> None:
    main_node_query = f"""
    MERGE (main:{node_label} {{name: $name}})
    """
    graph.query(main_node_query, params={"name": f"{node_name}_info"})

    for section, content in data.items():
        query = f"""
        MERGE (s:Section {{type: $type, parent_name: $name}})
        """
        params = {
            "type": section,
            "name": f"{node_name}_info"
        }
        graph.query(query, params=params)
