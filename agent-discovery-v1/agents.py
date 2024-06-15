import os
from typing import List, Optional
from pydantic import BaseModel

from openai import OpenAI
from instructor import Instructor
from chromadb import Client as ChromaClient
import chromadb.utils.embedding_functions as embedding_functions

import knowledge
from models import Message, MessageRole, TeacherAgentThought

KNOWLEDGE_DESC_GENERATOR_SYSTEM_PROMPT = """
You are an expert in creating detailed, human-readable descriptions (NOT JSON)
for intelligent agents based on their specific system prompts and knowledge schemas.
Your task is to generate a first-person description of the agent, explaining its purpose
and capabilities. Additionally, you will describe the structured knowledge the agent stores
according to the provided JSON schema. Ensure that the description is clear, comprehensive,
and accessible to a general audience
"""


class StudentAgent:
    def __init__(
        self,
        id: str,
        name: str,
        system_prompt: str,
        knowledge_class: BaseModel,
        openai_client: OpenAI,
        instructor_client: Instructor,
    ) -> None:
        self.id = id
        self.name = name
        self.system_prompt = system_prompt

        self.openai_client = openai_client
        self.instructor_client = instructor_client

        self.knowledge = None
        self.knowledge_class = knowledge_class
        self.knowledge_description = self._generate_knowledge_description()

    def respond(self, input: str, chat_history: List[Message]) -> Message:
        response = self._get_response(input=input, chat_history=chat_history)

        self._add_knowledge(input=response)

        return Message(role=MessageRole.assistant, content=response)

    def _get_response(self, input: str, chat_history: List[Message]) -> str:
        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=chat_history
            + [Message(role=MessageRole.user, content=input)],
        )

        return response.choices[0].message.content

    def _add_knowledge(self, input: str):
        new_knowledge = knowledge.get_structured_knowledge(
            input=input,
            knowledge_class=self.knowledge_class,
            instructor_client=self.instructor_client,
        )

        if self.knowledge is None:
            self.knowledge = new_knowledge
        else:
            self.knowledge = knowledge.aggregate_knowledge(
                knowledge_list=[self.knowledge, new_knowledge],
                knowledge_class=self.knowledge_class,
                openai_client=self.openai_client,
            )

    def _generate_knowledge_description(self) -> str:
        knowledge_schema = self.knowledge_class.model_json_schema()

        response = self.openai_client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                Message(
                    role=MessageRole.system,
                    content=KNOWLEDGE_DESC_GENERATOR_SYSTEM_PROMPT,
                ),
                Message(
                    role=MessageRole.user,
                    content=(
                        f"Agent name: {self.name}\n\n"
                        f"Agent System prompt: {self.system_prompt}\n\n"
                        f"JSON Schema for Knowledge stored: {knowledge_schema}"
                    ),
                ),
            ],
        )

        return response.choices[0].message.content


class StudentAgentStore:
    def __init__(
        self, chroma_client: ChromaClient, openai_client: OpenAI
    ) -> None:
        self.agents = []

        self.EMBEDDING_MODEL_NAME = "text-embedding-3-small"
        self.AGENT_KNOWLEDGE_COLLECTION_NAME = "agent_knowledge_collection"

        self.chroma_client = chroma_client
        self.openai_client = openai_client

        self.agent_knowledge_collection = (
            self.chroma_client.get_or_create_collection(
                name=self.AGENT_KNOWLEDGE_COLLECTION_NAME,
                embedding_function=embedding_functions.OpenAIEmbeddingFunction(
                    api_key=os.environ.get("OPENAI_API_KEY"),
                    model_name=self.EMBEDDING_MODEL_NAME,
                ),
            )
        )

    def get_agent(self, id: str) -> StudentAgent | None:
        for agent in self.agents:
            if agent.id == id:
                return agent

        return None

    def add_agents(self, agents: List[StudentAgent]):
        self.agents += agents
        self.agent_knowledge_collection.add(
            documents=[agent.knowledge_description for agent in agents],
            ids=[agent.id for agent in agents],
            embeddings=[
                self._embed_text(agent.knowledge_description)
                for agent in agents
            ],
        )

    def get_relevant_agent(self, query: str) -> StudentAgent | None:
        res = self.agent_knowledge_collection.query(
            query_texts=[query], n_results=1
        )

        res_agent = None

        if len(res["ids"][0]) == 0:
            return res_agent

        agent_id = res["ids"][0][0]

        for agent in self.agents:
            if agent.id == agent_id:
                res_agent = agent
                break

        return res_agent

    def _embed_text(self, text: str):
        return (
            self.openai_client.embeddings.create(
                model=self.EMBEDDING_MODEL_NAME, input=text
            )
            .data[0]
            .embedding
        )


class TeacherAgent:
    def __init__(
        self,
        id: str,
        name: str,
        system_prompt: str,
        student_agent_store: StudentAgentStore,
        openai_client: OpenAI,
        instructor_client: Instructor,
    ) -> None:
        self.id = id
        self.name = name
        self.system_prompt = system_prompt
        self.student_agent_store = student_agent_store

        self.openai_client = openai_client
        self.instructor_client = instructor_client

    def respond(self, input: str, chat_history: List[Message]) -> Message:
        knowledge_needed = self._think(input=input)

        print("Knowledge needed:", knowledge_needed)

        if not knowledge_needed:
            messages = chat_history + Message(
                role=MessageRole.user, content=input
            )
        else:
            agent = self.student_agent_store.get_relevant_agent(
                query=knowledge_needed
            )

            if not agent:
                print("Agent not found")
                messages = chat_history + Message(
                    role=MessageRole.user, content=input
                )
            else:
                print()
                print("Agent:", agent.name)
                print()
                print(
                    "Agent Knowledge Description:", agent.knowledge_description
                )
                print()
                print("Agent Knowledge:", agent.knowledge)
                print()

                messages = chat_history + [
                    Message(
                        role=MessageRole.system,
                        content=(
                            "Use this knowledge in combination with the teacher's query"
                            "to personalise your response and best help the teacher.\n\n"
                            f"Knowledge: {agent.knowledge}"
                        ),
                    ),
                    Message(role=MessageRole.user, content=input),
                ]

        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo", messages=messages
        )

        return Message(
            role=MessageRole.assistant,
            content=response.choices[0].message.content,
        )

    def _think(self, input: str) -> Optional[str]:
        response = self.instructor_client.chat.completions.create(
            model="gpt-3.5-turbo",
            response_model=TeacherAgentThought,
            messages=[
                Message(
                    role=MessageRole.system,
                    content=(
                        "You are an expert at generating a thought to decide"
                        "if you need knowledge from student facing AI agents"
                        "about student performance"
                        "to best respond to the teacher's query"
                        "If you do not need knowledge about students then you will return None"
                        "Otherwise you will describe the knowledge you need"
                    ),
                ),
                Message(role=MessageRole.user, content=input),
            ],
        )

        return response.knowledge_needed
