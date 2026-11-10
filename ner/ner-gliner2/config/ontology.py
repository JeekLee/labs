"""POLE+O ontology shared by the GLiNER and LLM extractors.

Entity and relationship definitions mirror the built-in POLE+O template of
``neo4j-agent-memory`` 0.8.0, so results stay comparable with the agent-memory
experiments without depending on that SDK.
"""

from dataclasses import dataclass

ENTITY_TYPES = {
    "PERSON": "A named human individual — a real person referred to by name, alias or persona. "
              "Not a job title, not a fictional character, and not an organization named after a person.",
    "ORGANIZATION": "A named company, institution, government agency, non-profit or other formal group "
                    "of people. Not an industry or market segment, and not a product the organization sells.",
    "LOCATION": "A place: an address, city, region, country, or named landmark. Not an organization "
                "that happens to occupy a building, and not a nationality or demonym.",
    "EVENT": "Something that happened at a point or interval in time — an incident, meeting, transaction "
             "or milestone. Not a recurring process or a standing activity, and not a bare date.",
    "OBJECT": "A physical or digital item: a vehicle, device, document, product, phone number or account "
              "identifier. Last resort — check every other type first. Not a person, place, organization or event.",
}

ALL = tuple(ENTITY_TYPES)


@dataclass(frozen=True)
class Relation:
    name: str
    heads: tuple[str, ...]
    tails: tuple[str, ...]
    description: str
    threshold: float | None = None
    unique_head: bool = False
    acyclic: bool = False


RELATIONS = (
    Relation("KNOWS", ("PERSON",), ("PERSON",),
             "One person has a personal or professional acquaintance with another."),
    Relation("ALIAS_OF", ("PERSON",), ("PERSON",),
             "One person record is an alternative identity of another; the same human.", acyclic=True),
    Relation("MEMBER_OF", ("PERSON",), ("ORGANIZATION",),
             "A person belongs to an organization without necessarily being paid by it."),
    Relation("EMPLOYED_BY", ("PERSON",), ("ORGANIZATION",),
             "A person is paid to work for an organization. Their current employer only.", unique_head=True),
    Relation("OWNS", ("PERSON", "ORGANIZATION"), ("OBJECT",),
             "A person or organization holds legal ownership of an item."),
    Relation("USES", ("PERSON",), ("OBJECT",),
             "A person operates or makes use of an item without necessarily owning it."),
    Relation("LOCATED_AT", ("PERSON", "OBJECT", "ORGANIZATION", "EVENT"), ("LOCATION",),
             "An entity is physically at a place. Use the more specific RESIDES_AT / HEADQUARTERS_AT / "
             "OCCURRED_AT where they apply."),
    Relation("RESIDES_AT", ("PERSON",), ("LOCATION",),
             "A person lives at a place. Their current home only.", unique_head=True),
    Relation("HEADQUARTERS_AT", ("ORGANIZATION",), ("LOCATION",),
             "An organization has its principal place of business at a location.", unique_head=True),
    Relation("PARTICIPATED_IN", ("PERSON", "ORGANIZATION"), ("EVENT",),
             "A person or organization took an active part in an event."),
    Relation("OCCURRED_AT", ("EVENT",), ("LOCATION",), "An event took place at a location."),
    Relation("INVOLVED", ("EVENT",), ("OBJECT",),
             "An item featured in an event as instrument, subject or evidence."),
    Relation("SUBSIDIARY_OF", ("ORGANIZATION",), ("ORGANIZATION",),
             "One organization is owned or controlled by another.", acyclic=True),
    Relation("PARTNER_WITH", ("ORGANIZATION",), ("ORGANIZATION",),
             "Two organizations have a declared partnership or alliance."),
    Relation("RELATED_TO", ("PERSON", "ORGANIZATION", "LOCATION", "EVENT"), ALL,
             "A stated connection that no more specific relationship covers. "
             "Last resort — check every other relationship first.", threshold=0.6),
)


def allowed(relation_type: str, head_type: str, tail_type: str) -> bool:
    """Whether the ontology permits this typed edge."""
    return any(r.name == relation_type and head_type in r.heads and tail_type in r.tails
               for r in RELATIONS)
