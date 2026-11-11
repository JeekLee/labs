"""GLiNER2 JointIE and LLM extractors returning one common result shape.

Both extractors return ``{"entities": [...], "relations": [...]}`` where an
entity is ``{"name", "type", "confidence"}`` and a relation is
``{"source", "type", "target", "confidence"}``. Entity types are the
upper-case POLE+O names in :mod:`config.ontology`.
"""

from functools import cache
import json
import warnings

from config.ontology import ENTITY_TYPES, RELATIONS, allowed


@cache
def kiwi():
    from kiwipiepy import Kiwi

    return Kiwi()


def kiwi_splitter(text, lower=True):
    """Korean morpheme splitter for ``GlinerExtractor.set_word_splitter()``.

    A GLiNER2 splitter yields ``(token, start, end)`` with exclusive-end offsets
    into the original text.
    """
    last_end = 0
    for token in kiwi().tokenize(text):
        start, end = token.start, token.start + token.len
        if start < last_end or end <= start:  # contracted or empty morphemes
            continue
        surface = text[start:end]
        last_end = end
        yield (surface.lower() if lower else surface), start, end


class GlinerExtractor:
    """GLiNER2.5 JointIE over the POLE+O schema, compiled the way the SDK does."""

    def __init__(self, model: str, *, threshold: float = 0.2, device: str = "cpu"):
        from gliner2 import AutoExtractor
        from gliner2.joint_ie import JointIEConfig, JointIEEngine

        self.model_id = model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = AutoExtractor.from_pretrained(model, map_location=device)
        self.joint = JointIEEngine(self.model, device=device)
        self.config = JointIEConfig(entity_threshold=threshold)
        self.schema = self._compile_schema()

    def _compile_schema(self):
        schema = self.joint.create_schema()
        for label, description in ENTITY_TYPES.items():
            schema.entity(label, description)
        for relation in RELATIONS:
            schema.relation(
                relation.name, list(relation.heads), list(relation.tails), relation.description,
                threshold=relation.threshold, unique_head=relation.unique_head,
                acyclic=relation.acyclic,
            )
            schema.no_self_loops(relation.name)
        return schema

    def set_word_splitter(self, splitter) -> None:
        """Replace the pre-tokenization word splitter (name or callable)."""
        self.model.set_word_splitter(splitter)

    def extract(self, text: str) -> dict:
        result = self.joint.extract(text, self.schema, config=self.config)
        by_id = {}
        entities = []
        for raw in result.entities:
            entity = {"name": raw.text.strip(), "type": raw.type.upper(),
                      "confidence": float(raw.confidence)}
            by_id[raw.id] = entity
            entities.append(entity)
        relations = [
            {"source": by_id[raw.head]["name"], "type": raw.type.upper(),
             "target": by_id[raw.tail]["name"], "confidence": float(raw.confidence)}
            for raw in result.relations if raw.head in by_id and raw.tail in by_id
        ]
        return {"entities": entities, "relations": relations}


PROMPT = """Extract named entities and the relationships between them from the user's text.

## Entity types
{entity_types}

## Relationship types (HEAD_TYPE -> TAIL_TYPE)
{relation_types}

Rules:
- Use the exact surface form from the text as the entity name, without attached particles or punctuation.
- Only extract relationships that the text states; do not infer unstated facts.
- Every relationship endpoint must be one of the extracted entity names.
- Give each item a confidence between 0 and 1.
"""

RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["entities", "relations"],
    "properties": {
        "entities": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["name", "type", "confidence"],
            "properties": {"name": {"type": "string"},
                           "type": {"type": "string", "enum": list(ENTITY_TYPES)},
                           "confidence": {"type": "number"}},
        }},
        "relations": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["source", "type", "target", "confidence"],
            "properties": {"source": {"type": "string"},
                           "type": {"type": "string", "enum": sorted({r.name for r in RELATIONS})},
                           "target": {"type": "string"},
                           "confidence": {"type": "number"}},
        }},
    },
}


class LLMExtractor:
    """OpenAI structured-output extractor over the same POLE+O schema."""

    def __init__(self, model: str):
        from openai import AsyncOpenAI

        self.model_id = model
        self.client = AsyncOpenAI()
        self.instructions = PROMPT.format(
            entity_types="\n".join(f"- {name}: {desc}" for name, desc in ENTITY_TYPES.items()),
            relation_types="\n".join(
                f"- {r.name} ({'|'.join(r.heads)} -> {'|'.join(r.tails)}): {r.description}"
                for r in RELATIONS),
        )

    async def extract(self, text: str) -> dict:
        response = await self.client.responses.create(
            model=self.model_id,
            instructions=self.instructions,
            input=text,
            text={"format": {"type": "json_schema", "name": "extraction",
                             "schema": RESPONSE_SCHEMA, "strict": True}},
        )
        data = json.loads(response.output_text)
        entities = {e["name"]: e for e in data["entities"]}
        # Same rule the GLiNER decoder enforces: endpoints exist and the typed edge is legal.
        relations = [
            r for r in data["relations"]
            if r["source"] in entities and r["target"] in entities and r["source"] != r["target"]
            and allowed(r["type"], entities[r["source"]]["type"], entities[r["target"]]["type"])
        ]
        return {"entities": data["entities"], "relations": relations}
