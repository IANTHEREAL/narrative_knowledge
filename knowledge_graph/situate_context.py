import boto3
import os
import logging

from llm.providers.bedrock import BedrockProvider
from llm.factory import LLMInterface
from utils.token import calculate_tokens

logger = logging.getLogger(__name__)

DOCUMENT_CONTEXT_PROMPT = """
<document>
{doc_content}
</document>
"""

CHUNK_CONTEXT_PROMPT = """
Here is the chunk we want to situate within the whole document
<chunk>
{chunk_content}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
Answer only with the succinct context and nothing else.
"""


def gen_situate_context(llm_client: LLMInterface, doc: str, chunk: str) -> str:
    prompt = (
        DOCUMENT_CONTEXT_PROMPT.format(doc_content=doc)
        + "\n\n"
        + CHUNK_CONTEXT_PROMPT.format(chunk_content=chunk)
    )
    token_count = calculate_tokens(prompt)
    if token_count > 40960:
        logger.warning(f"Chunk is too long to situate: {token_count} tokens")
        return None
    max_tokens = 8192
    if token_count + 500 > max_tokens:
        max_tokens = token_count + 500

    response_stream = llm_client.generate_stream(prompt, max_tokens=max_tokens)
    response = ""
    for chunk in response_stream:
        response += chunk

    # Remove think sections if present
    if "</think>" in response:
        # Extract content after </think>
        think_end = response.find("</think>")
        if think_end != -1:
            response = response[think_end + 8 :].strip()  # 8 = len("</think>")

    return response
