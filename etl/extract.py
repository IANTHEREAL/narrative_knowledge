import json
import logging
from pathlib import Path
import pymupdf
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
    from magic_pdf.data.dataset import PymuDocDataset
    from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze

from utils.file import read_file_content

logger = logging.getLogger(__name__)



try:
    # Magic PDF imports
    from magic_pdf.data.data_reader_writer import (
        FileBasedDataWriter,
        FileBasedDataReader,
    )
    from magic_pdf.data.dataset import PymuDocDataset
    from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze

    MAGIC_PDF_AVAILABLE = True
except ImportError:
    MAGIC_PDF_AVAILABLE = False
    logger.warning(
        "Warning: magic_pdf not available. Install it using `pip install magic-pdf` to use magic_pdf extraction."
    )


def convert_pdf_to_markdown_using_pymupdf(pdf_path):
    """
    Convert a single PDF file to markdown and JSON, creating a directory structure
    similar to magic_pdf output

    Parameters:
    - pdf_path: Path to the PDF file
    - output_dir: Path where the output directory will be created

    Returns:
    - True if conversion successful, False otherwise
    """
    try:
        pdf_path_obj = Path(pdf_path)
        if not pdf_path_obj.is_file():
            logger.error(f"Path {pdf_path} is not a file")
            raise ValueError(f"Path {pdf_path} is not a file")

        if pdf_path_obj.suffix != ".pdf":
            logger.error(f"Path {pdf_path} is not a PDF file")
            raise ValueError(f"Path {pdf_path} is not a PDF file")

        output_dir_name = get_output_dir_using_pymupdf(pdf_path_obj)
        output_dir = pdf_path_obj.parent / output_dir_name
        output_dir_path = Path(output_dir)

        # Create output directory
        output_dir_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Processing with pymupdf: {pdf_path}, output directory: {output_dir_path}"
        )

        # Convert PDF to text
        doc = pymupdf.open(pdf_path)
        markdown_content = []
        pages_data = []
        total_pages = doc.page_count

        for page_num in range(total_pages):
            page = doc.load_page(page_num)
            text_page = page.get_textpage()  # get TextPage object
            text = text_page.extractText()  # extract plain text as string
            if text.strip():  # Only add non-empty pages
                # Add to markdown content
                markdown_content.append(f"<!-- Page {page_num+1} -->\n\n")
                markdown_content.append(text.strip() + "\n\n")

                # Add to JSON structure
                page_data = {
                    "type": "text",
                    "text": text.strip(),
                    "page_idx": page_num - 1,  # 0-based index to match magic_pdf
                    "page_number": page_num,  # 1-based for readability
                }
                pages_data.append(page_data)

        doc.close()

        # Generate output filenames
        name_without_suffix = pdf_path_obj.stem
        markdown_file = output_dir_path / f"{name_without_suffix}.md"
        json_file = output_dir_path / f"{name_without_suffix}_content_list.json"

        # Write markdown file
        with open(markdown_file, "w", encoding="utf-8") as f:
            f.write("".join(markdown_content))

        # Write JSON file
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(pages_data, f, ensure_ascii=False, indent=4)

        logger.info(
            f"✅ Successfully processed with pymupdf: {pdf_path}, output directory: {output_dir_path}, generated files: {name_without_suffix}.md (markdown content), {name_without_suffix}_content.json (page structure)"
        )
        return markdown_file

    except Exception as e:
        logger.error(f"Error converting {pdf_path} using pymupdf: {e}", exc_info=True)
        raise RuntimeError(f"Error converting {pdf_path} using pymupdf: {e}")


def get_output_dir_using_pymupdf(pdf_path):
    """
    Generate the output directory name for pymupdf extraction
    Format: {pdf_file_name}_pymupdf/
    """
    pdf_name = pdf_path.stem  # Get filename without extension
    return f"{pdf_name}_pymupdf"


def get_output_dir_using_magic_pdf(pdf_path):
    """
    Generate the output directory name for magic_pdf extraction
    Format: {pdf_file_name}_magic_pdf/
    """
    pdf_name = pdf_path.stem  # Get filename without extension
    return f"{pdf_name}_magic_pdf"


def convert_pdf_using_magic_pdf(pdf_path):
    """
    Convert a single PDF file using magic_pdf
    Creates a directory with markdown, images, and metadata files

    Parameters:
    - pdf_path: Path to the PDF file

    Returns:
    - True if conversion successful, False otherwise
    """

    if not MAGIC_PDF_AVAILABLE:
        logger.error("❌ magic_pdf not available. Please install it first.")
        raise RuntimeError("magic_pdf not available. Please install it first.")

    try:
        pdf_path_obj = Path(pdf_path)
        if not pdf_path_obj.is_file():
            logger.error(f"Path {pdf_path} is not a file")
            raise ValueError(f"Path {pdf_path} is not a file")

        if pdf_path_obj.suffix != ".pdf":
            logger.error(f"Path {pdf_path} is not a PDF file")
            raise ValueError(f"Path {pdf_path} is not a PDF file")

        output_dir_name = get_output_dir_using_magic_pdf(pdf_path_obj)
        output_dir = pdf_path_obj.parent / output_dir_name
        output_dir_path = Path(output_dir)

        # Create output directory structure
        output_dir_path.mkdir(parents=True, exist_ok=True)
        local_image_dir = output_dir_path / "images"
        local_image_dir.mkdir(parents=True, exist_ok=True)

        name_without_suff = pdf_path_obj.stem
        image_dir = "images"  # relative path for markdown references

        # Prepare data writers
        image_writer = FileBasedDataWriter(str(local_image_dir))
        md_writer = FileBasedDataWriter(str(output_dir_path))

        # Read PDF bytes
        reader = FileBasedDataReader("")
        pdf_bytes = reader.read(str(pdf_path))

        logger.info(
            f"Processing with magic_pdf: {pdf_path}, output directory: {output_dir_path}"
        )

        # Create Dataset Instance
        ds = PymuDocDataset(pdf_bytes)

        # Apply document analysis with OCR
        infer_result = ds.apply(doc_analyze, ocr=True)

        # Process with OCR mode
        pipe_result = infer_result.pipe_ocr_mode(image_writer)

        # 1. Main markdown content
        pipe_result.dump_md(md_writer, f"{name_without_suff}.md", image_dir)

        # 2. Content list JSON
        pipe_result.dump_content_list(
            md_writer, f"{name_without_suff}_content_list.json", image_dir
        )

        # 3. Middle JSON (metadata)
        pipe_result.dump_middle_json(md_writer, f"{name_without_suff}_middle.json")

        logger.info(
            f"✅ Successfully processed with magic_pdf: {pdf_path}, output directory: {output_dir_path}, generated files: {name_without_suff}.md (main markdown), {name_without_suff}_content_list.json (content structure), {name_without_suff}_middle.json (metadata)"
        )

        return f"{output_dir_path}/{name_without_suff}.md"

    except Exception as e:
        logger.error(
            f"❌ Error processing {pdf_path} with magic_pdf: {e}", exc_info=True
        )
        raise RuntimeError(f"Error processing {pdf_path} with magic_pdf: {e}")


def extract_data_from_text_file(path):
    """
    Extract data from text-based files (markdown, txt, sql)
    """
    try:
        logger.info(f"Reading text file: {path}")
        full_content = read_file_content(path)

        # Determine file type from extension
        path_obj = Path(path)
        file_extension = path_obj.suffix.lower()

        if file_extension == ".md":
            file_type = "markdown"
        elif file_extension == ".txt":
            file_type = "document"
        elif file_extension == ".sql":
            file_type = "sql"
        else:
            raise RuntimeError(f"Unsupported file type: {file_extension} for {path}")

        return {
            "status": "success",
            "content": full_content,
            "file_type": file_type,
        }
    except Exception as e:
        logger.error(f"Error reading text file {path}: {e}", exc_info=True)
        raise RuntimeError(f"Error reading text file {path}: {e}")


def extract_data_from_pdf(path):
    """
    Extract data from files in the docs directory
    """

    try:
        if MAGIC_PDF_AVAILABLE:
            logger.info("Using magic_pdf to extract data from PDFs")
            markdown_file = convert_pdf_using_magic_pdf(path)
        else:
            logger.info("Using pymupdf to extract data from PDFs")
            markdown_file = convert_pdf_to_markdown_using_pymupdf(path)

        full_content = read_file_content(str(markdown_file))
        return {
            "status": "success",
            "content": full_content,
            "file_type": "pdf",
        }
    except Exception as e:
        logger.error(f"Error extracting data from {path}: {e}", exc_info=True)
        raise RuntimeError(f"Error extracting data from {path}: {e}")


def extract_source_data(path: str) -> dict[str, str]:
    """
    Extract source data from a file
    """
    pdf_path_obj = Path(path)
    if not pdf_path_obj.is_file():
        logger.error(f"Path {path} is not a file")
        raise ValueError(f"Path {path} is not a file")

    file_extension = pdf_path_obj.suffix.lower()

    if file_extension == ".pdf":
        return extract_data_from_pdf(path)
    elif file_extension in [".md", ".txt", ".sql"]:
        return extract_data_from_text_file(path)

    raise RuntimeError(f"Unsupported file type: {file_extension} for {path}")
