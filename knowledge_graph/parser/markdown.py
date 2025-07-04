import json
from typing import List
import logging
import re

from .base import SourceData, Block
from utils.file import read_file_content, extract_file_info
from llm.factory import LLMInterface
from utils.token import calculate_tokens
from utils.json_utils import robust_json_parse

logger = logging.getLogger(__name__)


class MarkdownParser:
    """
    A builder class for constructing knowledge graphs from documents.
    """

    def __init__(self, llm_client: LLMInterface):
        self.llm_client = llm_client

    def parse(self, path: str, max_tokens=4096, split_threshold=2048) -> SourceData:
        # Extract basic info
        name, extension = extract_file_info(path)
        markdown_content = read_file_content(path)

        return self._parse_content_internal(markdown_content, name, max_tokens, split_threshold)

    def parse_content(self, content: str, name: str, max_tokens=4096, split_threshold=2048) -> SourceData:
        """
        Parse markdown content directly.
        
        Args:
            content: The markdown content to parse
            name: Name for the document
            max_tokens: Maximum tokens per merged block
            split_threshold: Token threshold for splitting blocks
            
        Returns:
            SourceData object with parsed blocks
        """
        return self._parse_content_internal(content, name, max_tokens, split_threshold)

    def _parse_content_internal(self, markdown_content: str, name: str, max_tokens=4096, split_threshold=2048) -> SourceData:
        # Phase 1: Hierarchical Top-Down Splitting
        logger.info("[Parser] Starting Phase 1: Hierarchical Splitting")

        # Start splitting by level 1 headings first.
        initial_chunks = self._split_content_by_heading(markdown_content, 1)
        if not initial_chunks:
            # If no L1 headings, treat the whole doc as one chunk.
            initial_chunks = [{"title": name, "content": markdown_content}]

        atomic_blocks = []
        position_counter = 0
        for chunk in initial_chunks:
            # Now, recursively split each L1 chunk starting from L2 headings.
            split_blocks = self._hierarchical_split(
                chunk["title"],
                chunk["content"],
                position_counter,
                start_level=2,  # Start recursion from level 2
                split_threshold=split_threshold,
            )
            atomic_blocks.extend(split_blocks)
            position_counter += len(split_blocks)

        logger.info(f"[Parser] Phase 1 produced {len(atomic_blocks)} atomic blocks.")

        # Phase 2: Thematic Bottom-Up Merging
        if len(atomic_blocks) > 1:
            logger.info("[Parser] Starting Phase 2: Thematic Merging")
            try:
                final_blocks = self._thematic_merge_with_llm(atomic_blocks, max_tokens)
                logger.info(
                    f"[Parser] Phase 2 merged into {len(final_blocks)} final blocks."
                )
            except Exception as e:
                logger.error(
                    f"[Parser] Thematic merging failed: {e}. Using atomic blocks."
                )
                final_blocks = atomic_blocks
        else:
            final_blocks = atomic_blocks
            logger.info("[Parser] Skipping Phase 2: Only one block produced.")

        # Update final positions
        for i, block in enumerate(final_blocks):
            block.position = i

        return SourceData(name=name, content=markdown_content, blocks=final_blocks)

    def _hierarchical_split(
        self,
        base_title: str,
        content: str,
        position_offset: int,
        start_level=2,
        split_threshold=2048,
    ) -> List[Block]:
        """
        Recursively splits content based on headings, down to a certain size.
        """
        # Stop recursion if content is small enough or we've reached max heading depth
        if self._estimate_tokens(content) <= split_threshold or start_level > 6:
            return [Block(name=base_title, content=content, position=position_offset)]

        # Check for the next level of headings to split by
        has_next_level_headings = self._has_lower_level_headings(
            content, start_level - 1
        )

        if has_next_level_headings:
            # Split by the next heading level
            split_chunks = self._split_content_by_heading(content, start_level)
            if split_chunks:
                final_blocks = []
                for i, chunk in enumerate(split_chunks):
                    # Recurse on the new, smaller chunk
                    recursive_blocks = self._hierarchical_split(
                        chunk["title"],
                        chunk["content"],
                        position_offset + i,  # Position will be updated later
                        start_level + 1,
                        split_threshold,
                    )
                    final_blocks.extend(recursive_blocks)
                return final_blocks

        # Fallback: No more headings to split by, but chunk is still too large.
        # Use simple splitting as a last resort.
        logger.info(
            f"[Hierarchical Split] Block '{base_title}' is > {split_threshold} tokens but has no more sub-headings. Applying simple split."
        )
        return self._simple_split(content, base_title, position_offset, split_threshold)

    def _thematic_merge_with_llm(
        self, blocks: List[Block], max_tokens: int
    ) -> List[Block]:
        """
        Uses a single LLM call to merge blocks based on their semantic topic
        and generate a new title for each merged group.
        """
        # Prepare the input for the LLM, sending full content
        numbered_chunks_str = ""
        for i, block in enumerate(blocks):
            chunk_info = json.dumps(
                {"original_title": block.name, "content": block.content}
            )
            numbered_chunks_str += f"{i+1}: {chunk_info}\n"

        prompt = (
            prompt
        ) = f"""You are an expert technical writer and information architect. Your mission is to transform a fragmented series of document chunks into a well-structured, coherent document by grouping them into meaningful topics.

### CONTEXT
I have a document that has been pre-processed into numbered chunks. These chunks are out of their original, larger context, making the document hard to read. Your task is to reverse this fragmentation by creating logical topic groups.

### STEP-BY-STEP THINKING PROCESS TO FOLLOW
To ensure the highest quality result, please follow this exact thinking process:

1.  **Holistic Comprehension**: First, read through ALL the numbered chunks below. Before grouping, form a mental summary of the entire document's purpose and subject matter. What is this document about as a whole?

2.  **Theme Identification**: Based on your overall understanding, identify the main themes, concepts, or stages discussed across the chunks. A theme could be a project phase (e.g., "Initial Setup & Configuration"), a specific feature (e.g., "User Authentication Logic"), or a core concept (e.g., "Data Privacy Principles").

3.  **Grouping and Mapping**: For each theme you identified, create a group and map the relevant chunk indices to it.
    * **Justify your grouping**: The chunks in a group must be strongly related and form a continuous, logical narrative or explanation.
    * **Encourage overlaps**: It is highly encouraged to include a chunk in multiple groups if it serves as a natural bridge between two different topics (e.g., a chunk that concludes one feature and introduces the next).

4.  **Title Synthesis**: For each group, create a new, highly descriptive title. The title should accurately synthesize the core content of the chunks within that group. Avoid generic titles.

5.  **Constraint Verification**: Finally, review your generated topic groups against the critical constraints below.
    * If a group is semantically perfect but exceeds the token limit, try to split it into two or more smaller, still-coherent sub-topics. **Semantic integrity is more important than creating the largest possible groups.**

### CRITICAL CONSTRAINTS
1.  **Complete Coverage**: EVERY chunk from 1 to {len(blocks)} MUST be included in AT LEAST ONE topic group. No chunks left behind.
2.  **Overlaps Allowed**: A chunk can be included in multiple topics.
3.  **Title Quality**: Titles must be new, descriptive, and reflect the specific content of the group.
4.  **Size Constraint**: The total token count of the content within any single merged group must NOT exceed {max_tokens} tokens.
5.  **Output Format**: Your final output MUST BE ONLY a JSON object, enclosed in ```json and ```. Do not include any explanations or text outside the JSON block.

### NUMBERED CHUNKS TO MERGE
<chunks>
{numbered_chunks_str}
</chunks>

### CHUNK INDEX RANGE SPECIFICATION
**IMPORTANT**: Use `chunk_index_range` to specify continuous chunk ranges:

1. **Single chunk**: `[5, 5]` (only chunk 5)
2. **Continuous range**: `[3, 11]` (chunks 3 through 11)

### EXAMPLE JSON RESPONSE (surrounding by ```json and ```)
```json
{{
  "topics": [
    {{
      "new_title": "Full Introduction and Project Goals",
      "chunk_index_range": [1, 2]
    }},
    {{
      "new_title": "Core Feature Details and Implementation",
      "chunk_index_range": [3, 11]
    }},
    {{
      "new_title": "Final Summary",
      "chunk_index_range": [12, 12]
    }}
  ]
}}
```
Note: [3, 11] means chunks 3, 4, 5, 6, 7, 8, 9, 10, 11. For single chunks use [N, N].

Now, apply the thinking process to the provided chunks and generate the final JSON output.
"""
        logger.info("[Thematic Merge] Calling LLM for unified merge and title plan...")
        token_count = calculate_tokens(prompt)
        response_stream = self.llm_client.generate_stream(
            prompt, max_tokens=token_count + 150
        )
        response = ""
        for chunk in response_stream:
            response += chunk

        logger.info("[Thematic Merge] LLM response received. Parsing merge plan.")
        logger.info("prompt %s\n\nresponse %s", prompt, response)
        llm_response_obj = robust_json_parse(response, "object", self.llm_client)
        if not llm_response_obj or "topics" not in llm_response_obj:
            raise ValueError(
                "LLM did not return a valid merge plan with a 'topics' key."
            )

        merge_plan = llm_response_obj["topics"]

        # Validate chunk_index_range format
        for topic in merge_plan:
            if "chunk_index_range" not in topic:
                raise ValueError(
                    f"Topic '{topic.get('new_title', 'Unknown')}' missing 'chunk_index_range' field."
                )
            range_val = topic["chunk_index_range"]
            if not isinstance(range_val, list) or len(range_val) != 2:
                raise ValueError(
                    f"chunk_index_range must be [start, end] format, got: {range_val}"
                )

        final_blocks = []

        # --- Safeguard against gaps ---
        all_indices_in_plan = set()
        for topic in merge_plan:
            start, end = topic["chunk_index_range"]
            all_indices_in_plan.update(range(start, end + 1))

        expected_indices = set(range(1, len(blocks) + 1))
        gap_indices = expected_indices - all_indices_in_plan

        if gap_indices:
            logger.info(
                f"Warning: LLM plan has gaps. Missing indices: {sorted(list(gap_indices))}. Creating standalone blocks for them."
            )
            for gap_idx in sorted(list(gap_indices)):
                # Add the missing chunk as its own topic group to the plan
                original_block = blocks[gap_idx - 1]
                merge_plan.append(
                    {
                        "new_title": original_block.name,
                        "chunk_index_range": [gap_idx, gap_idx],
                    }
                )

        # Sort the plan by the first chunk index in each group to maintain document order
        merge_plan.sort(key=lambda x: x["chunk_index_range"][0])
        # --- End Safeguard ---

        for topic_group in merge_plan:
            # Extract range and convert to indices list
            start, end = topic_group["chunk_index_range"]
            chunk_indices = list(range(start, end + 1))
            indices_to_merge = [
                i - 1 for i in chunk_indices
            ]  # Convert to 0-based index
            blocks_to_merge = [blocks[i] for i in indices_to_merge]

            if not blocks_to_merge:
                continue

            # Get the new title directly from the unified plan
            new_title = topic_group["new_title"].strip().strip('"')

            if len(blocks_to_merge) == 1:
                # If a block is standalone, update its title and add it.
                block = blocks_to_merge[0]
                block.name = new_title
                final_blocks.append(block)
                continue

            # This is a group to merge
            merged_content = "\n\n".join([b.content for b in blocks_to_merge])

            # Safeguard: Check size before confirming merge, in case the LLM makes a mistake.
            if self._estimate_tokens(merged_content) > max_tokens:
                logger.warn(
                    f"Warning: Proposed merge for '{new_title}' exceeds max_tokens. Keeping chunks separate."
                )
                final_blocks.extend(blocks_to_merge)
                continue

            merged_block = Block(
                name=new_title,
                content=merged_content,
                position=blocks_to_merge[0].position,  # Use first block's position
            )
            final_blocks.append(merged_block)
            logger.info(
                f"[Thematic Merge] Processed group '{new_title}' using chunk range [{start}-{end}]"
            )

        return final_blocks

    def _split_content_by_heading(
        self, content: str, level: int, preface_threshold: int = 512
    ) -> List[dict]:
        """
        Splits content by a specific heading level using a regex-based approach.

        Returns a list of dictionaries, where each dict has "title" and "content".
        The content *before* the first heading is associated with the first heading block,
        unless it's too long (> preface_threshold tokens), in which case it becomes a separate chunk.

        Args:
            content: The content to split
            level: The heading level to split by (1 for #, 2 for ##, etc.)
            preface_threshold: Token threshold for creating separate preface chunk (default: 1024)
        """
        heading_prefix = "#" * level + " "
        # This regex splits the text by headings, but keeps the heading line (the delimiter)
        # as part of the *following* text block, which is exactly what we want.
        # It looks for a newline character followed by the heading prefix.
        splitter = re.compile(f"(^({re.escape(heading_prefix)}.*)$)", re.MULTILINE)

        parts = splitter.split(content)

        # Corner Case 1: No headings of this level found at all.
        # The entire content is preface.
        if len(parts) == 1:
            preface_content = content.strip()
            if not preface_content:
                return []

            # Create a single chunk. Use the first non-empty line as a heuristic title.
            lines = preface_content.split("\n")
            first_line = next((line for line in lines if line.strip()), "")
            title = (first_line[:75] + "...") if len(first_line) > 75 else first_line
            return [{"title": title, "content": preface_content}]

        chunks = []

        # The first part is the content before any headings of this level.
        preface_content = parts[0].strip()

        # The rest of the parts come in triplets.
        for i in range(1, len(parts), 3):
            title_line = parts[i]
            title = title_line.lstrip("# ").strip()

            # Corner Case 2: Handle trailing heading that has no content after it.
            if i + 2 < len(parts):
                chunk_content = parts[i + 2].strip()
                full_chunk_content = (title_line + "\n" + chunk_content).strip()
            else:
                full_chunk_content = title_line.strip()

            chunks.append({"title": title, "content": full_chunk_content})

        # Handle preface content based on its size
        if preface_content and chunks:
            preface_tokens = self._estimate_tokens(preface_content)

            if preface_tokens > preface_threshold:
                # Create a separate chunk for long preface content
                lines = preface_content.split("\n")
                first_line = next((line for line in lines if line.strip()), "")
                preface_title = (
                    (first_line[:75] + "...") if len(first_line) > 75 else first_line
                )
                if not preface_title:
                    preface_title = "Document Introduction"

                preface_chunk = {"title": preface_title, "content": preface_content}
                # Insert preface chunk at the beginning
                chunks.insert(0, preface_chunk)
                logger.info(
                    f"[Split] Created separate preface chunk '{preface_title}' ({preface_tokens} tokens)"
                )
            else:
                # Merge short preface with first chunk (original behavior)
                chunks[0]["content"] = preface_content + "\n\n" + chunks[0]["content"]

        return chunks

    def _has_lower_level_headings(self, content: str, current_level: int) -> bool:
        """Checks if the content contains any headings of a lower level."""
        # Pattern to match headings with more '#' symbols than current_level
        # e.g., if current_level is 2 (##), it looks for ###, ####, etc.
        pattern = re.compile(f"^#{'{'}{current_level + 1},{'}'} .*", re.MULTILINE)
        return bool(pattern.search(content))

    def _estimate_tokens(self, text: str) -> int:
        """Simple token estimation: roughly 4 characters per token"""
        return calculate_tokens(text)

    def _simple_split(
        self, content: str, base_title: str, position_offset: int, target_tokens: int
    ) -> List[Block]:
        """Simple fallback splitting by paragraphs and sentences"""
        lines = content.split("\n")
        chunks = []
        current_chunk = []
        current_tokens = 0

        for line in lines:
            line_tokens = self._estimate_tokens(line)

            if current_tokens + line_tokens > target_tokens and current_chunk:
                # Save current chunk
                chunk_content = "\n".join(current_chunk)
                chunks.append(chunk_content)
                current_chunk = [line]
                current_tokens = line_tokens
            else:
                current_chunk.append(line)
                current_tokens += line_tokens

        # Save last chunk
        if current_chunk:
            chunk_content = "\n".join(current_chunk)
            chunks.append(chunk_content)

        # Convert to Block objects
        blocks = []
        for i, chunk_content in enumerate(chunks):
            title = f"{base_title} - Part {i+1}" if base_title else f"Part {i+1}"
            blocks.append(
                Block(name=title, content=chunk_content, position=position_offset + i)
            )

        return blocks
