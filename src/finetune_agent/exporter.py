"""Dataset exporter module for various JSONL formats.

Exports datasets to common fine-tuning formats:
- Instruct JSONL: {"instruction": "...", "input": "...", "output": "...", "metadata": {...}}
- Chat JSONL: {"messages": [...], "metadata": {...}}
- Simple QA JSONL: {"question": "...", "answer": "...", "metadata": {...}}

Also includes golden set sampling for curated evaluation sets.
"""

import json
from pathlib import Path
from typing import Any

from finetune_agent.schemas import Dataset, DatasetOutput, QAPair


# =============================================================================
# Export Functions
# =============================================================================

def export_qa_jsonl(
    dataset_output: DatasetOutput,
    output_path: str | Path,
    include_metadata: bool = True,
) -> int:
    """Export dataset to simple Q&A JSONL format.
    
    Format: {"question": "...", "answer": "...", "metadata": {...}}
    
    Args:
        dataset_output: The dataset to export
        output_path: Path to write the JSONL file
        include_metadata: Whether to include metadata in output
        
    Returns:
        Number of items exported
    """
    output_path = Path(output_path)
    count = 0
    
    with open(output_path, "w", encoding="utf-8") as f:
        for dataset in dataset_output.datasets:
            for item in dataset.items:
                record = {
                    "question": item.question,
                    "answer": item.answer,
                }
                if include_metadata:
                    # Add dataset type to metadata
                    metadata = {**item.metadata, "dataset_type": dataset.type}
                    record["metadata"] = metadata
                
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
    
    return count


def export_instruct_jsonl(
    dataset_output: DatasetOutput,
    output_path: str | Path,
    system_prompt: str = "",
    include_metadata: bool = True,
) -> int:
    """Export dataset to instruction-following JSONL format.
    
    Format: {"instruction": "...", "input": "...", "output": "...", "metadata": {...}}
    
    This format is commonly used for instruction-tuning models like Alpaca.
    
    Args:
        dataset_output: The dataset to export
        output_path: Path to write the JSONL file
        system_prompt: Optional system prompt to include as instruction prefix
        include_metadata: Whether to include metadata in output
        
    Returns:
        Number of items exported
    """
    output_path = Path(output_path)
    count = 0
    
    with open(output_path, "w", encoding="utf-8") as f:
        for dataset in dataset_output.datasets:
            for item in dataset.items:
                # For instruction format, the question becomes the instruction
                # and the answer becomes the output
                instruction = item.question
                if system_prompt:
                    instruction = f"{system_prompt}\n\n{instruction}"
                
                record = {
                    "instruction": instruction,
                    "input": "",  # Could be extended to support context
                    "output": item.answer,
                }
                if include_metadata:
                    metadata = {**item.metadata, "dataset_type": dataset.type}
                    record["metadata"] = metadata
                
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
    
    return count


def export_chat_jsonl(
    dataset_output: DatasetOutput,
    output_path: str | Path,
    system_prompt: str = "You are a helpful assistant.",
    include_metadata: bool = True,
) -> int:
    """Export dataset to chat format JSONL.
    
    Format: {"messages": [{"role": "system", "content": "..."}, 
                          {"role": "user", "content": "..."}, 
                          {"role": "assistant", "content": "..."}], 
             "metadata": {...}}
    
    This format is commonly used for chat fine-tuning (OpenAI, Llama chat, etc.).
    
    Args:
        dataset_output: The dataset to export
        output_path: Path to write the JSONL file
        system_prompt: System message to include in each conversation
        include_metadata: Whether to include metadata in output
        
    Returns:
        Number of items exported
    """
    output_path = Path(output_path)
    count = 0
    
    with open(output_path, "w", encoding="utf-8") as f:
        for dataset in dataset_output.datasets:
            for item in dataset.items:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": item.question},
                    {"role": "assistant", "content": item.answer},
                ]
                
                record = {"messages": messages}
                if include_metadata:
                    metadata = {**item.metadata, "dataset_type": dataset.type}
                    record["metadata"] = metadata
                
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
    
    return count


# =============================================================================
# Golden Set Sampling
# =============================================================================

def _calculate_uniqueness_score(item: QAPair, all_items: list[QAPair]) -> float:
    """Calculate uniqueness score for an item compared to all items.
    
    Uses a simple character-based diversity measure.
    """
    item_text = f"{item.question} {item.answer}"
    item_chars = set(item_text.lower())
    
    total_similarity = 0
    for other in all_items:
        if other is item:
            continue
        other_text = f"{other.question} {other.answer}"
        other_chars = set(other_text.lower())
        
        intersection = len(item_chars & other_chars)
        union = len(item_chars | other_chars)
        similarity = intersection / union if union > 0 else 0
        total_similarity += similarity
    
    # Higher diversity = lower average similarity
    avg_similarity = total_similarity / max(1, len(all_items) - 1)
    return (1 - avg_similarity) * 100


def _get_difficulty_rank(item: QAPair) -> int:
    """Get difficulty rank (higher = harder)."""
    difficulty = item.metadata.get("difficulty", "medium")
    if isinstance(difficulty, str):
        difficulty = difficulty.lower()
    
    rank_map = {"easy": 1, "medium": 2, "hard": 3}
    return rank_map.get(difficulty, 2)


def _get_code_density(item: QAPair) -> float:
    """Calculate code density in the answer."""
    answer = item.answer
    
    # Count code block characters
    code_chars = 0
    in_code_block = False
    
    lines = answer.split("\n")
    for line in lines:
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            code_chars += len(line)
    
    # Also count inline code
    inline_code = len([c for c in answer.split("`") if len(c) > 3])
    
    total = len(answer)
    if total == 0:
        return 0
    
    return (code_chars + inline_code * 10) / total


def sample_golden_set(
    dataset_output: DatasetOutput,
    output_path: str | Path,
    top_unique: int = 10,
    top_hard: int = 10,
    top_code_heavy: int = 10,
) -> dict[str, int]:
    """Create a golden set of the best items from the dataset.
    
    Samples:
    - Top N most unique items (diverse vocabulary/content)
    - Top N hardest items
    - Top N most code-heavy items
    
    Avoids duplicates across categories.
    
    Args:
        dataset_output: The dataset to sample from
        output_path: Path to write the golden set JSONL
        top_unique: Number of most unique items to include
        top_hard: Number of hardest items to include
        top_code_heavy: Number of most code-heavy items to include
        
    Returns:
        Dictionary with counts per category
    """
    output_path = Path(output_path)
    
    # Collect all items with their source info
    all_items: list[tuple[QAPair, str]] = []
    for dataset in dataset_output.datasets:
        for item in dataset.items:
            all_items.append((item, dataset.type))
    
    if not all_items:
        # Write empty file
        with open(output_path, "w", encoding="utf-8") as f:
            pass
        return {"unique": 0, "hard": 0, "code_heavy": 0, "total": 0}
    
    # Extract just items for scoring
    items_only = [item for item, _ in all_items]
    
    # Calculate scores for all items
    item_scores: list[dict[str, Any]] = []
    for item, dtype in all_items:
        item_scores.append({
            "item": item,
            "dataset_type": dtype,
            "uniqueness": _calculate_uniqueness_score(item, items_only),
            "difficulty_rank": _get_difficulty_rank(item),
            "code_density": _get_code_density(item),
        })
    
    selected_ids: set[str] = set()
    golden_items: list[dict[str, Any]] = []
    
    def add_to_golden(scored_item: dict, category: str) -> bool:
        """Add item to golden set if not already selected."""
        item = scored_item["item"]
        item_id = item.metadata.get("id", f"{item.question[:50]}_{item.answer[:50]}")
        
        if item_id in selected_ids:
            return False
        
        selected_ids.add(item_id)
        golden_items.append({
            "item": item,
            "dataset_type": scored_item["dataset_type"],
            "category": category,
        })
        return True
    
    counts = {"unique": 0, "hard": 0, "code_heavy": 0}
    
    # Top unique items
    sorted_by_unique = sorted(item_scores, key=lambda x: x["uniqueness"], reverse=True)
    for scored in sorted_by_unique:
        if counts["unique"] >= top_unique:
            break
        if add_to_golden(scored, "most_unique"):
            counts["unique"] += 1
    
    # Top hardest items
    sorted_by_hard = sorted(
        item_scores, 
        key=lambda x: (x["difficulty_rank"], len(x["item"].answer)),
        reverse=True
    )
    for scored in sorted_by_hard:
        if counts["hard"] >= top_hard:
            break
        if add_to_golden(scored, "hardest"):
            counts["hard"] += 1
    
    # Top code-heavy items
    sorted_by_code = sorted(item_scores, key=lambda x: x["code_density"], reverse=True)
    for scored in sorted_by_code:
        if counts["code_heavy"] >= top_code_heavy:
            break
        if scored["code_density"] > 0:  # Only include items that actually have code
            if add_to_golden(scored, "code_heavy"):
                counts["code_heavy"] += 1
    
    # Write golden set
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in golden_items:
            item = entry["item"]
            record = {
                "question": item.question,
                "answer": item.answer,
                "metadata": {
                    **item.metadata,
                    "dataset_type": entry["dataset_type"],
                    "golden_category": entry["category"],
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    counts["total"] = len(golden_items)
    return counts


# =============================================================================
# Utility Functions
# =============================================================================

def export_all_formats(
    dataset_output: DatasetOutput,
    output_dir: str | Path,
    system_prompt: str = "You are a helpful assistant.",
) -> dict[str, int]:
    """Export dataset to all supported formats.
    
    Creates:
    - dataset_qa.jsonl
    - dataset_instruct.jsonl
    - dataset_chat.jsonl
    - golden_set.jsonl
    
    Args:
        dataset_output: The dataset to export
        output_dir: Directory to write files to
        system_prompt: System prompt for chat/instruct formats
        
    Returns:
        Dictionary with counts for each export type
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    counts = {}
    
    # Export QA format
    counts["qa"] = export_qa_jsonl(
        dataset_output,
        output_dir / "dataset_qa.jsonl",
    )
    
    # Export instruct format
    counts["instruct"] = export_instruct_jsonl(
        dataset_output,
        output_dir / "dataset_instruct.jsonl",
        system_prompt=system_prompt,
    )
    
    # Export chat format
    counts["chat"] = export_chat_jsonl(
        dataset_output,
        output_dir / "dataset_chat.jsonl",
        system_prompt=system_prompt,
    )
    
    # Create golden set
    golden_counts = sample_golden_set(
        dataset_output,
        output_dir / "golden_set.jsonl",
    )
    counts["golden_set"] = golden_counts["total"]
    counts["golden_details"] = golden_counts
    
    return counts
