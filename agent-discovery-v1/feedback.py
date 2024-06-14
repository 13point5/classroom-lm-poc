from pydantic import BaseModel
from instructor import Instructor


def get_structured_feedback(
    instructor_client: Instructor, input: str, feedback_cls: BaseModel
):
    return instructor_client.chat.completions.create(
        model="gpt-3.5-turbo",
        response_model=feedback_cls,
        messages=[
            {"role": "user", "content": input},
        ],
    )
