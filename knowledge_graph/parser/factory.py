from utils.file import extract_file_info
from llm.factory import LLMInterface
from .markdown import MarkdownParser


def get_parser(path: str, llm_client: LLMInterface):
    name, extension = extract_file_info(path)
    if extension == ".md":
        return MarkdownParser(llm_client)

    raise NotImplementedError(f"Not suitable parser for {name}{extension}")


def get_parser_by_content_type(content_type: str, llm_client: LLMInterface):
    """
    Get parser by content type/MIME type.
    
    Args:
        content_type: MIME type (e.g., "text/markdown", "text/plain")
        llm_client: LLM interface for processing
        
    Returns:
        Parser instance
    """
    content_type = content_type.lower()
    
    # Handle MIME types and map to appropriate parser
    if content_type in ["text/markdown", "text/x-markdown"]:
        return MarkdownParser(llm_client)
    elif content_type in ["text/plain", "application/octet-stream"]:
        # Default to markdown parser for plain text files
        return MarkdownParser(llm_client)

    raise NotImplementedError(f"PDF parser not yet implemented")
