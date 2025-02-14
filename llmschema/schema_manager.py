from pydantic import BaseModel
from typing import Type, Union, Dict, Any

class SchemaManager:
    _schema: Union[Type[BaseModel], Dict[str, Any]] = None

    @classmethod
    def set_schema(cls, schema: Union[Type[BaseModel], Dict[str, Any]]):
        """Sets the user-defined schema, either as a Pydantic model or a JSON schema dict."""
        if not isinstance(schema, dict) and not (isinstance(schema, type) and issubclass(schema, BaseModel)):
            raise ValueError("Schema must be either a Pydantic model or a JSON schema dictionary.")
        cls._schema = schema

    @classmethod
    def get_schema(cls) -> Union[Type[BaseModel], Dict[str, Any]]:
        """Returns the currently set schema."""
        if cls._schema is None:
            raise ValueError("No schema has been set. Use `set_schema()` first.")
        return cls._schema
