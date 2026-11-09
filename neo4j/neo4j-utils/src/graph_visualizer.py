"""Display Neo4j query results in a Jupyter notebook."""

from typing import Any, LiteralString

from IPython.display import display
from neo4j import AsyncSession
from neo4j_viz.neo4j import from_neo4j


async def visualize(
    session: AsyncSession,
    query: LiteralString,
    parameters: dict[str, Any] | None = None,
    *,
    height: str = "600px",
) -> None:
    """Query and display a graph using a caller-owned session.

    The caller opens and closes the session. Query parameters are passed to
    the driver unchanged, so this helper does not require a memory client
    or a particular graph schema. Full properties remain available even
    when captions are shortened.
    """
    result = await session.run(query, parameters=parameters)
    graph = await result.graph()

    if not graph.nodes:
        print("시각화할 그래프가 없다.")
        return

    visualization = from_neo4j(graph)
    for node in visualization.nodes:
        props = node.properties
        if "Fact" in props.get("labels", []):
            caption = (
                f"{props.get('subject', '')} — {props.get('predicate', '')}"
                f" → {props.get('object', '')}"
            )
        else:
            caption = str(
                props.get("name") or props.get("content") or props.get("session_id") or node.caption
            )
        node.caption = caption if len(caption) <= 80 else caption[:79] + "…"
    for relationship in visualization.relationships:
        relationship.caption = relationship.properties.get("__type") or relationship.caption
    display(visualization.render_widget(height=height))
