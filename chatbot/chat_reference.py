from dataclasses import dataclass


@dataclass
class ChatReference:
    """Chat Reference Dataclass"""

    title: str
    description: str
    ref_url: str
    context: str
