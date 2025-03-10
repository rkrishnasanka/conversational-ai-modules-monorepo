from dataclasses import dataclass

@dataclass
class Entity:
    name: str
    type: str
    metadata: dict = None

ENTITY_SCHEMA = """
                Entity Schema:
                - name (str): The name of the entity
                - type (str): The type of entity
                - metadata (dict, optional): Additional information about the entity
                """


