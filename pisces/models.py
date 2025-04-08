class ModelRegistry:
    """A registry for managing models, their schemas, and invocations."""

    _models = {}

    @classmethod
    def register_model(cls, name: str, schema: dict, invoke_function: callable):
        """Register a model with its schema and invocation function."""
        cls._models[name] = {
            "schema": schema,
            "invoke": invoke_function
        }

    @classmethod
    def get_schema(cls, name: str) -> dict:
        """Retrieve the schema for a given model."""
        model = cls._models.get(name)
        if not model:
            raise ValueError(f"Model {name} is not registered.")
        return model["schema"]

    @classmethod
    def invoke(cls, name: str, input_data: dict) -> dict:
        """Invoke a model with the provided input data."""
        model = cls._models.get(name)
        if not model:
            raise ValueError(f"Model {name} is not registered.")
        return model["invoke"](input_data)

# Example usage:
# ModelRegistry.register_model(
#     "example_model",
#     schema={"input": {"type": "string"}, "output": {"type": "string"}},
#     invoke_function=lambda data: {"output": data["input"].upper()}
# )