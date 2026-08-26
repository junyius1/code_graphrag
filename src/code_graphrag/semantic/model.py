"""Code Semantic Model: typed entities, relations, and the knowledge graph container.

Entity ids are stable hashes of (type, repo, file, qualified_name) so that the same
symbol always maps to the same id across runs, and same-named symbols in different
files stay distinct.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


def stable_entity_id(etype: str, repo: str, file: str, qualified_name: str) -> str:
    key = f"{etype}|{repo}|{file}|{qualified_name}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:20]


@dataclass
class CodeEntity:
    id: str
    name: str
    qualified_name: str
    type: str  # controlled vocabulary, see GraphRAGAdapterConfig.entity_types
    file: str  # relative path ("__repo__" for repository-level entities)
    start_line: int = 0
    end_line: int = 0
    description: str = ""
    language: str | None = None
    attributes: dict[str, str] = field(default_factory=dict)

    @property
    def location(self) -> str:
        if self.start_line and self.end_line and self.end_line > self.start_line:
            return f"{self.file}:{self.start_line}-{self.end_line}"
        if self.start_line:
            return f"{self.file}:{self.start_line}"
        return self.file


@dataclass
class CodeRelation:
    source_id: str
    target_id: str
    type: str
    description: str = ""
    weight: float = 1.0
    file: str | None = None

    def key(self) -> tuple[str, str, str]:
        return (self.source_id, self.target_id, self.type)


@dataclass
class CodeKnowledgeGraph:
    repo: str
    entities: dict[str, CodeEntity] = field(default_factory=dict)
    relations: dict[tuple[str, str, str], CodeRelation] = field(default_factory=dict)

    def add_entity(self, entity: CodeEntity) -> CodeEntity:
        """Add an entity, deduplicating by id (merges attributes)."""
        existing = self.entities.get(entity.id)
        if existing is not None:
            existing.attributes.update(entity.attributes)
            if not existing.description and entity.description:
                existing.description = entity.description
            return existing
        self.entities[entity.id] = entity
        return entity

    def add_relation(self, relation: CodeRelation) -> CodeRelation:
        """Add a relation, deduplicating by (source, target, type) (sums weight)."""
        key = relation.key()
        existing = self.relations.get(key)
        if existing is not None:
            existing.weight += relation.weight
            return existing
        self.relations[key] = relation
        return relation

    # -- convenience builders ------------------------------------------------

    def entity(
        self,
        etype: str,
        name: str,
        qualified_name: str,
        file: str,
        start_line: int = 0,
        end_line: int = 0,
        description: str = "",
        language: str | None = None,
        **attributes: str,
    ) -> CodeEntity:
        eid = stable_entity_id(etype, self.repo, file, qualified_name)
        return self.add_entity(
            CodeEntity(
                id=eid,
                name=name,
                qualified_name=qualified_name,
                type=etype,
                file=file,
                start_line=start_line,
                end_line=end_line,
                description=description,
                language=language,
                attributes=dict(attributes),
            )
        )

    def relate(
        self,
        source_id: str,
        target_id: str,
        rtype: str,
        description: str = "",
        weight: float = 1.0,
        file: str | None = None,
    ) -> CodeRelation:
        return self.add_relation(
            CodeRelation(
                source_id=source_id,
                target_id=target_id,
                type=rtype,
                description=description,
                weight=weight,
                file=file,
            )
        )

    def by_type(self, etype: str) -> list[CodeEntity]:
        return [e for e in self.entities.values() if e.type == etype]

    def entity_by_file(self, file: str, etype: str) -> CodeEntity | None:
        for e in self.entities.values():
            if e.type == etype and e.file == file:
                return e
        return None

    def stats(self) -> dict[str, int]:
        from collections import Counter

        return {
            "entities": len(self.entities),
            "relations": len(self.relations),
            "entity_types": sum(Counter(e.type for e in self.entities.values()).values())
            and len(set(e.type for e in self.entities.values())),
            "by_entity_type": dict(Counter(e.type for e in self.entities.values())),
            "by_relation_type": dict(Counter(r.type for r in self.relations.values())),
        }
