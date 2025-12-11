import json
from typing import List


def extract_levels(reasoning_chain: List[dict]) -> List[str]:
    """Return ordered unique levels from a reasoning_chain list.

    The input is a list of steps, each with a 'level' field. We preserve the
    original order (typically continental → national → city → local) and drop
    duplicates if any.
    """
    seen_levels = set()
    ordered_levels: List[str] = []
    for step in reasoning_chain or []:
        level = step.get("level")
        if level is None:
            continue
        if level not in seen_levels:
            seen_levels.add(level)
            ordered_levels.append(level)
    return ordered_levels


def filter_file(input_path: str, output_path: str) -> None:
    """Read JSONL from input_path, write filtered JSONL to output_path.

    Each output line contains only 'image_path' and 'levels' fields.
    """
    with open(input_path, "r", encoding="utf-8") as fin, open(
        output_path, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            image_path = record.get("image_path")
            reasoning_chain = (
                (record.get("location_analysis") or {}).get("reasoning_chain")
            )
            levels = extract_levels(reasoning_chain if isinstance(reasoning_chain, list) else [])

            out_obj = {"image_path": image_path, "levels": levels}
            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    # Paths based on the repository layout
    INPUT = "data/json/location_analysis_fixed.jsonl"
    OUTPUT = "data/json/location_levels.jsonl"
    filter_file(INPUT, OUTPUT)




