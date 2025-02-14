from pydantic import BaseModel
from typing import Type

class SchemaManager:
    _schema: Type[BaseModel] = None

    @classmethod
    def set_schema(cls, schema: Type[BaseModel]):
        """Sets the user-defined Pydantic schema."""
        cls._schema = schema

    @classmethod
    def get_schema(cls) -> Type[BaseModel]:
        """Returns the currently set schema."""
        if cls._schema is None:
            raise ValueError("No schema has been set. Use `set_schema()` first.")
        return cls._schema
