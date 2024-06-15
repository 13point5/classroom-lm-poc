from pydantic import BaseModel
from typing import List, Type, Dict, Any, get_origin, get_args
from collections import defaultdict

from instructor import Instructor
from openai import OpenAI

from utils import aggregate_dict, deduplicate_dict


def get_structured_knowledge(
    input: str,
    knowledge_class: BaseModel,
    instructor_client: Instructor,
):
    return instructor_client.chat.completions.create(
        model="gpt-4-turbo",
        response_model=knowledge_class,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert at extracting structured knowledge "
                    "from an AI agent's response."
                ),
            },
            {"role": "user", "content": input},
        ],
        max_retries=3,
    )


def aggregate_knowledge(
    knowledge_list: List[BaseModel],
    knowledge_class: Type[BaseModel],
    openai_client: OpenAI,
):
    def to_dict(obj: Any) -> Any:
        if isinstance(obj, BaseModel):
            return {k: to_dict(v) for k, v in obj.model_dump().items()}
        elif isinstance(obj, list):
            return [to_dict(v) for v in obj]
        else:
            return obj

    def from_dict(cls: Type[BaseModel], data: Dict[str, Any]) -> BaseModel:
        field_values = {}
        for name, field in cls.model_fields.items():
            if name in data:
                value = data[name]
                origin = get_origin(field.annotation)
                if origin is not None and issubclass(origin, list):
                    field_type = get_args(field.annotation)[0]
                    if issubclass(field_type, BaseModel):
                        field_values[name] = [
                            from_dict(field_type, v) for v in value
                        ]
                    else:
                        field_values[name] = value
                elif isinstance(field.annotation, type) and issubclass(
                    field.annotation, BaseModel
                ):
                    field_values[name] = from_dict(field.annotation, value)
                else:
                    field_values[name] = value
        return cls(**field_values)

    # Initialize the aggregated structure dynamically
    aggregated = defaultdict(lambda: defaultdict(list))

    # Aggregate all knowledge
    for knowledge in knowledge_list:
        knowledge_dict = to_dict(knowledge)
        aggregate_dict(aggregated, knowledge_dict)

    # Deduplicate lists
    deduplicate_dict(data=aggregated, openai_client=openai_client)

    # Construct a new knowledge object dynamically
    return from_dict(knowledge_class, aggregated)
