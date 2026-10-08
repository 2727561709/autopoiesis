"""M08 语义记忆模块的输入输出类型定义（接口契约）。
M08 semantic memory — input/output type definitions (interface
contract).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping

import numpy as np


class SemanticError(ValueError):
    """概念/关系不存在、自环、非法参数时抛出。
    Raised on missing concept/relation, self-loops, or invalid
    arguments."""


@dataclass
class Concept:
    """一个概念节点。One concept node.

    Attributes:
        cid: 概念标识（唯一键）。Concept id (unique key).
        attrs: 任意属性（如 kind、valence、energy…）。
            Arbitrary attributes (kind, valence, energy, ...).
    """
    cid: str
    attrs: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """读属性（带默认值）。Read an attribute with a default."""
        return self.attrs.get(key, default)


@dataclass
class Link:
    """一条带类型的加权有向边。One typed, weighted, directed edge.

    Attributes:
        src / dst: 起止概念 id。Source / target concept ids.
        relation: 关系类型（如 "is-a"、"near"、"causes"）。
            Relation type ("is-a", "near", "causes", ...).
        weight: 连接强度 (0, 1]。Connection strength in (0, 1].
    """
    src: str
    dst: str
    relation: str
    weight: float = 1.0


#: 激活值表：概念 id -> 激活强度（扩散激活的输出）。
#: Activation table: concept id -> activation strength (the output
#: of spreading activation).
Activation = Dict[str, float]
