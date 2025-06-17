import openai
import json
import os


def get_text_embedding(text: str, model="hf.co/Qwen/Qwen3-Embedding-8B-GGUF:Q8_0"):
    embedding_model = openai.OpenAI(
        base_url=os.getenv("EMBEDDING_MODEL_BASE_URL"),
        api_key=os.getenv("EMBEDDING_MODEL_API_KEY"),
    )
    text = text.replace("\n", " ")
    return (
        embedding_model.embeddings.create(input=[text], model=model).data[0].embedding
    )


def get_entity_description_embedding(name: str, description: str):
    combined_text = f"{name}: {description}"
    return get_text_embedding(combined_text)


def get_entity_metadata_embedding(metadata: dict):
    combined_text = json.dumps(metadata)
    return get_text_embedding(combined_text)
