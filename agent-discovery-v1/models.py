from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class MessageRole(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"
    thought = "thought"


class Message(BaseModel):
    role: MessageRole
    content: str


class TeacherAgentThought(BaseModel):
    knowledge_needed: Optional[str]


class TopicKnowledge(BaseModel):
    topic: str
    strengths: List[str]
    misconceptions: List[str]


class MathAgentKnowledge(BaseModel):
    knowledge: List[TopicKnowledge]


class CriteraFeedback(BaseModel):
    strengths: List[str] = Field(
        ..., description="Concise list of strengths for this criteria"
    )
    weaknesses: List[str] = Field(
        ..., description="Concise list of weaknesses for this criteria"
    )
    suggestions: List[str] = Field(
        ...,
        description=(
            "Concise list of suggestions provided that are not weaknesses"
        ),
    )


class EssayAgentKnowledge(BaseModel):
    introduction: CriteraFeedback = Field(
        ...,
        description="Analysis of the introduction of the essay with these criteria: 1. Clarity of thesis statement, 2. Engagement and relevance of opening statements",
    )
    structure: CriteraFeedback = Field(
        ...,
        description="Analysis of the structure of the essay's body with these criteria: 1. Organization and clarity of paragraphs, 2. Logical flow of ideas",
    )
    argumentation: CriteraFeedback = Field(
        ...,
        description="Analysis of the argumentation of the essay with these criteria: 1. Strength and clarity of arguments, 2. Use of critical reasoning",
    )
    evidence: CriteraFeedback = Field(
        ...,
        description="Analysis of the evidence used in the essay with these criteria: 1. Relevance and quality of evidence, 2. Use of citations and references",
    )
    conclusion: CriteraFeedback = Field(
        ...,
        description="Analysis of the conclusion of the essay with these criteria: 1. Restatement of thesis, 2. Summary of main points, 3. Closing statements",
    )
