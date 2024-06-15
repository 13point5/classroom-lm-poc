from typing import Dict, Any, List
from collections import defaultdict
import numpy as np
from openai import OpenAI


def get_embeddings(openai_client, texts: List[str]):
    result = openai_client.embeddings.create(
        model="text-embedding-3-small", input=texts
    )
    embeddings = [e.embedding for e in result.data]

    return embeddings


def cosine_similarity(vector1, vector2):
    return np.dot(vector1, vector2) / (
        np.linalg.norm(vector1) * np.linalg.norm(vector2)
    )


def cosine_similarity_matrix(embeddings):
    # Normalize the embeddings to unit length
    embeddings_norm = embeddings / np.linalg.norm(
        embeddings, axis=1, keepdims=True
    )
    # Compute the cosine similarity matrix
    similarity_matrix = np.dot(embeddings_norm, embeddings_norm.T)
    return similarity_matrix


def dynamic_threshold(similarity_matrix):
    # Exclude the diagonal elements and flatten the array
    upper_triangle_indices = np.triu_indices_from(similarity_matrix, k=1)
    similarities = similarity_matrix[upper_triangle_indices]

    # Calculate a dynamic threshold: e.g., one standard deviation above the mean
    mean_similarity = np.mean(similarities)
    std_dev = np.std(similarities)
    threshold = mean_similarity + std_dev

    return threshold


def remove_semantic_duplicates(
    openai_client: OpenAI, strings: List[str]
) -> List[str]:
    strings = [s.strip() for s in strings if s.strip() != "None"]

    n = len(strings)
    keep = [True] * n

    if n == 0:
        return []

    embeddings = get_embeddings(openai_client, strings)
    similarity_matrix = cosine_similarity_matrix(embeddings)
    threshold = dynamic_threshold(similarity_matrix)
    threshold = round(threshold, 2)

    # Compute pairwise cosine similarity and filter duplicates
    for i in range(n):
        for j in range(i + 1, n):
            similarity = round(
                cosine_similarity(embeddings[i], embeddings[j]), 2
            )
            if keep[j] and (similarity > threshold or similarity >= 0.8):
                keep[j] = False

    # Filter out duplicates
    unique_strings = [strings[i] for i in range(n) if keep[i]]
    return unique_strings


def remove_duplicates(items: List[Any], openai_client: OpenAI):
    if len(items) == 0:
        return items

    if isinstance(items[0], str):
        return remove_semantic_duplicates(
            openai_client=openai_client, strings=items
        )

    seen = set()
    unique_items = []
    for item in items:
        if isinstance(item, (int, float, tuple)):
            if item not in seen:
                seen.add(item)
                unique_items.append(item)
        else:
            unique_items.append(item)
    return unique_items


def aggregate_dict(aggregated: Dict[str, Any], data: Dict[str, Any]):
    for key, value in data.items():
        if isinstance(value, list):
            if key not in aggregated:
                aggregated[key] = []
            aggregated[key].extend(value)
        elif isinstance(value, dict):
            if key not in aggregated:
                aggregated[key] = defaultdict(list)
            aggregate_dict(aggregated[key], value)


def deduplicate_dict(data: Dict[str, Any], openai_client: OpenAI):
    for key, value in data.items():
        if isinstance(value, list):
            data[key] = remove_duplicates(
                items=value, openai_client=openai_client
            )
        elif isinstance(value, dict):
            deduplicate_dict(data=value, openai_client=openai_client)
