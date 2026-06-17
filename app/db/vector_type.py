import json
from typing import Any

from sqlalchemy.ext.compiler import compiles
from sqlalchemy.types import UserDefinedType


class VectorType(UserDefinedType):
    """Small pgvector type with SQLite-friendly storage for tests."""

    cache_ok = True

    def __init__(self, dimension: int = 1536) -> None:
        self.dimension = dimension

    def get_col_spec(self, **_kw: Any) -> str:
        return f"VECTOR({self.dimension})"

    def bind_processor(self, dialect):
        def process(value: Any) -> Any:
            if value is None:
                return None
            if isinstance(value, str):
                return value
            if dialect.name == "sqlite":
                return json.dumps(list(value))
            return "[" + ",".join(f"{float(item):.8f}" for item in value) + "]"

        return process

    def result_processor(self, dialect, coltype):
        def process(value: Any) -> Any:
            if value is None:
                return None
            if isinstance(value, list):
                return value
            if isinstance(value, str):
                stripped = value.strip()
                if not stripped:
                    return []
                try:
                    return json.loads(stripped)
                except json.JSONDecodeError:
                    if stripped.startswith("[") and stripped.endswith("]"):
                        return [float(item) for item in stripped[1:-1].split(",") if item.strip()]
            return value

        return process


@compiles(VectorType, "postgresql")
def _compile_pgvector(type_: VectorType, compiler, **kw: Any) -> str:
    return f"VECTOR({type_.dimension})"


@compiles(VectorType, "sqlite")
def _compile_sqlite_vector(type_: VectorType, compiler, **kw: Any) -> str:
    return "TEXT"
