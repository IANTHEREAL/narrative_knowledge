import openai
import json
import os

from setting.base import EMBEDDING_MODEL, EMBEDDING_BASE_URL, EMBEDDING_MODEL_API_KEY


def get_text_embedding(text: str, model):
    embedding_model = openai.OpenAI(
        base_url=EMBEDDING_BASE_URL,
        api_key=EMBEDDING_MODEL_API_KEY,
    )
    text = text.replace("\n", " ")
    return (
        embedding_model.embeddings.create(input=[text], model=EMBEDDING_MODEL).data[0].embedding
    )


def get_entity_description_embedding(name: str, description: str):
    combined_text = f"{name}: {description}"
    return get_text_embedding(combined_text)


def get_entity_metadata_embedding(metadata: dict):
    combined_text = json.dumps(metadata)
    return get_text_embedding(combined_text)
