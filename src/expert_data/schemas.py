"""Structured records used by the mock pair-generation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass
class ObjectInfo:
    """Describe an object mentioned by a fact record."""

    object_id: str
    name: str
    category: str
    color: str | None = None
    aliases: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ObjectInfo":
        """Build an object record from a JSON-compatible mapping."""

        return cls(
            object_id=str(payload["object_id"]),
            name=str(payload["name"]),
            category=str(payload["category"]),
            color=payload.get("color"),
            aliases=[str(alias) for alias in payload.get("aliases", [])],
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the object record into a JSON-compatible dictionary."""

        return {
            "object_id": self.object_id,
            "name": self.name,
            "category": self.category,
            "color": self.color,
            "aliases": list(self.aliases),
        }


@dataclass
class RelationInfo:
    """Describe a binary relation between two objects."""

    subject_id: str
    predicate: str
    object_id: str
    subject_category: str | None = None
    object_category: str | None = None
    dx: float | None = None
    dy: float | None = None
    iou: float | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RelationInfo":
        """Build a relation record from a JSON-compatible mapping."""

        return cls(
            subject_id=str(payload["subject_id"]),
            predicate=str(payload["predicate"]),
            object_id=str(payload["object_id"]),
            subject_category=(
                str(payload["subject_category"])
                if payload.get("subject_category") is not None
                else None
            ),
            object_category=(
                str(payload["object_category"])
                if payload.get("object_category") is not None
                else None
            ),
            dx=float(payload["dx"]) if payload.get("dx") is not None else None,
            dy=float(payload["dy"]) if payload.get("dy") is not None else None,
            iou=float(payload["iou"]) if payload.get("iou") is not None else None,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the relation record into a JSON-compatible dictionary."""

        return {
            "subject_id": self.subject_id,
            "predicate": self.predicate,
            "object_id": self.object_id,
            "subject_category": self.subject_category,
            "object_category": self.object_category,
            "dx": self.dx,
            "dy": self.dy,
            "iou": self.iou,
        }


@dataclass
class FactRecord:
    """Represent one source fact that can be rendered into a pair record."""

    fact_id: str
    image_id: str
    subtype: str
    subject: ObjectInfo
    object: ObjectInfo | None = None
    relation: RelationInfo | None = None
    positive_value: Any = None
    negative_candidates: list[Any] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FactRecord":
        """Build a fact record from a JSON-compatible mapping."""

        object_payload = payload.get("object")
        relation_payload = payload.get("relation")
        return cls(
            fact_id=str(payload["fact_id"]),
            image_id=str(payload["image_id"]),
            subtype=str(payload["subtype"]),
            subject=ObjectInfo.from_dict(payload["subject"]),
            object=ObjectInfo.from_dict(object_payload) if object_payload else None,
            relation=RelationInfo.from_dict(relation_payload) if relation_payload else None,
            positive_value=payload.get("positive_value"),
            negative_candidates=list(payload.get("negative_candidates", [])),
            meta=dict(payload.get("meta", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the fact record into a JSON-compatible dictionary."""

        return {
            "fact_id": self.fact_id,
            "image_id": self.image_id,
            "subtype": self.subtype,
            "subject": self.subject.to_dict(),
            "object": self.object.to_dict() if self.object else None,
            "relation": self.relation.to_dict() if self.relation else None,
            "positive_value": self.positive_value,
            "negative_candidates": list(self.negative_candidates),
            "meta": dict(self.meta),
        }


@dataclass
class PairRecord:
    """Represent a rendered fact-counterfact pair for training or evaluation."""

    pair_id: str
    fact_id: str
    image_id: str
    subtype: str
    question: str
    response_pos: str
    response_neg: str
    pos_label: Any = None
    neg_label: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the pair record into a JSON-compatible dictionary."""

        return {
            "pair_id": self.pair_id,
            "fact_id": self.fact_id,
            "image_id": self.image_id,
            "subtype": self.subtype,
            "question": self.question,
            "response_pos": self.response_pos,
            "response_neg": self.response_neg,
            "pos_label": self.pos_label,
            "neg_label": self.neg_label,
            "metadata": dict(self.metadata),
        }

    @property
    def prompt(self) -> str:
        """Expose the question text under the legacy prompt name."""

        return self.question

    @property
    def positive_text(self) -> str:
        """Expose the positive response under the legacy field name."""

        return self.response_pos

    @property
    def negative_text(self) -> str:
        """Expose the negative response under the legacy field name."""

        return self.response_neg
