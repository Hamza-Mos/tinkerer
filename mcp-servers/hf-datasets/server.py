"""HuggingFace Dataset Search MCP Server.

Provides a single powerful tool for searching datasets on HuggingFace Hub.
Claude Code can write Python directly for loading/filtering - this server
only wraps the complex HF Hub search API.

pip install -r requirements.txt

Environment Variables:
- HF_TOKEN (optional): HuggingFace API token for accessing gated datasets.
  Get token at: https://huggingface.co/settings/tokens
"""

import os
import sys
from mcp.server.fastmcp import FastMCP
from huggingface_hub import HfApi
from typing import Optional


def validate_hf_token() -> None:
    """Validate HF_TOKEN environment variable at startup.

    Issues a warning if not set, since some datasets require authentication.
    This is a non-blocking warning - the server will still start.
    """
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print(
            "WARNING: HF_TOKEN not set. Some gated datasets may not be accessible.\n"
            "  Get a token at: https://huggingface.co/settings/tokens\n"
            "  Then set: export HF_TOKEN='your-token'\n",
            file=sys.stderr,
        )


# Validate environment at module load
validate_hf_token()

mcp = FastMCP("hf-datasets")
api = HfApi()


@mcp.tool()
def search_datasets(
    query: str,
    task_type: Optional[str] = None,
    language: Optional[str] = None,
    size_category: Optional[str] = None,
    max_results: int = 10,
) -> str:
    """Search HuggingFace Hub for datasets matching your criteria.

    Input validation:
    - query: Required, cannot be empty
    - max_results: Must be between 1 and 100

    Use this tool to find datasets for fine-tuning. Results are sorted by
    download count (most popular first).

    ARGUMENTS:
    - query (required): Search terms. Examples:
        - "poetry" - find poetry datasets
        - "sentiment analysis" - find sentiment datasets
        - "instruction following" - find instruction datasets
        - "code python" - find Python code datasets

    - task_type (optional): Filter by ML task. Common values:
        - "text-generation" - for language model training
        - "text-classification" - for classification tasks
        - "question-answering" - for QA datasets
        - "summarization" - for summarization tasks
        - "translation" - for translation datasets

    - language (optional): ISO language code. Examples:
        - "en" - English
        - "es" - Spanish
        - "zh" - Chinese
        - "fr" - French

    - size_category (optional): Dataset size range. Values:
        - "n<1K" - less than 1,000 examples
        - "1K<n<10K" - 1,000 to 10,000 examples
        - "10K<n<100K" - 10,000 to 100,000 examples
        - "100K<n<1M" - 100,000 to 1 million examples

    - max_results (optional): Number of results to return (default: 10)

    RETURNS:
    A formatted list of matching datasets with metadata including:
    - Dataset ID (use this for loading)
    - Download count
    - Tags (task type, language, etc.)
    - Description preview

    AFTER FINDING A DATASET:
    Write Python directly to load and process it:

    ```python
    from datasets import load_dataset

    # Safe defaults for loading
    ds = load_dataset("dataset_id", split="train", streaming=True, trust_remote_code=False)

    # Filter examples as needed
    examples = [ex for ex in ds.take(100) if your_filter(ex)]
    ```

    TIPS:
    - Start with a broad search, then filter
    - Popular datasets (high downloads) are usually higher quality
    - Use streaming=True when loading to avoid memory issues
    """
    # Input validation
    if not query or not query.strip():
        return "Error: query parameter is required and cannot be empty"
    if max_results < 1 or max_results > 100:
        return "Error: max_results must be between 1 and 100"
    query = query.strip()

    try:
        # Build filter kwargs for list_datasets
        kwargs = {"search": query, "sort": "downloads", "limit": max_results, "full": True}
        if task_type:
            kwargs["task_categories"] = task_type
        if language:
            kwargs["language"] = language
        if size_category:
            kwargs["size_categories"] = size_category

        results = list(api.list_datasets(**kwargs))

        output = []
        for ds in results:
            downloads = getattr(ds, "downloads", 0)
            tags = getattr(ds, "tags", [])
            description = getattr(ds, "description", "") or ""

            # Parse tags into categories
            task_tags = []
            lang_tags = []
            other_tags = []
            for tag in tags:
                if ":" in tag:
                    cat, val = tag.split(":", 1)
                    if cat == "task_categories":
                        task_tags.append(val)
                    elif cat == "language":
                        lang_tags.append(val)
                else:
                    other_tags.append(tag)

            # Build rich metadata string
            meta_parts = []
            if downloads:
                meta_parts.append(f"Downloads: {downloads:,}")
            if task_tags:
                meta_parts.append(f"Tasks: {', '.join(task_tags[:2])}")
            if lang_tags:
                meta_parts.append(f"Lang: {', '.join(lang_tags[:3])}")

            # Truncate description
            desc_preview = description[:150].replace("\n", " ").strip()
            if len(description) > 150:
                desc_preview += "..."

            dataset_id = getattr(ds, "id", "unknown")
            entry = f"- **{dataset_id}**\n  {' | '.join(meta_parts)}"
            if desc_preview:
                entry += f"\n  {desc_preview}"
            output.append(entry)

        if not output:
            return f"No datasets found for query '{query}'. Try broader search terms."

        return f"Found {len(output)} datasets:\n\n" + "\n\n".join(output)

    except Exception as e:
        # Check for HfHubHTTPError specifically for better error messages
        error_type = type(e).__name__
        if "HfHubHTTPError" in error_type or "HTTPError" in error_type:
            return f"HuggingFace API error: {e}"
        return f"Unexpected error: {error_type}: {e}\n\nTry a simpler query."


if __name__ == "__main__":
    mcp.run(transport="stdio")
