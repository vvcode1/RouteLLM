from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_MODEL = os.environ.get(
    "ROUTE_LLM_BASE_MODEL",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
)
DEFAULT_ADAPTER = os.environ.get(
    "ROUTE_LLM_ADAPTER",
    str(REPO_ROOT / "models" / "route-llm-llama3.1-8b-instruct-lora"),
)
DEFAULT_RELATIONSHIPS = os.environ.get(
    "ROUTE_LLM_AS_RELATIONSHIPS",
    str(REPO_ROOT / "data" / "as-relationships" / "20241201.as-rel.txt"),
)


def rel_code_to_text(as1: str, as2: str, rel: str) -> str:
    """Convert a CAIDA AS relationship code to natural-language text."""
    rel = str(rel).strip()
    if rel == "-1":
        return f"AS{as1} is the provider of AS{as2}. AS{as2} is the customer of AS{as1}."
    if rel == "0":
        return f"AS{as1} and AS{as2} are peers."
    if rel == "1":
        return f"AS{as1} and AS{as2} have a sibling relationship."
    return f"The relationship between AS{as1} and AS{as2} is unknown."


def load_as_relationships(
    rel_file: str | Path,
) -> Tuple[List[Dict[str, str]], Dict[str, List[str]]]:
    """Load CAIDA-style ``AS1|AS2|relationship|source`` records."""
    records: List[Dict[str, str]] = []
    asn_to_knowledge: Dict[str, List[str]] = {}

    with Path(rel_file).open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split("|")
            if len(parts) < 3:
                continue

            as1, as2, rel = (part.strip() for part in parts[:3])
            relation_text = rel_code_to_text(as1, as2, rel)
            records.append({"as1": as1, "as2": as2, "rel": rel, "text": relation_text})
            asn_to_knowledge.setdefault(as1, []).append(relation_text)
            asn_to_knowledge.setdefault(as2, []).append(relation_text)

    return records, asn_to_knowledge


def extract_asns_from_text(text: str) -> List[str]:
    """Extract explicitly marked ASNs and numeric AS-path components."""
    asns = re.findall(r"\bAS(\d+)\b", text, flags=re.IGNORECASE)
    as_path_matches = re.findall(
        r"(?:AS[ _-]?path)\s*[:=]\s*([\d\s]+)",
        text,
        flags=re.IGNORECASE,
    )
    for match in as_path_matches:
        asns.extend(re.findall(r"\d{1,10}", match))
    return list(dict.fromkeys(asns))


def extract_as_path_from_update(update: Dict) -> List[str]:
    """Extract ASNs from common structured and free-text update fields."""
    asns: List[str] = []
    as_path = update.get("as_path")
    if isinstance(as_path, list):
        asns.extend(str(value) for value in as_path)
    elif isinstance(as_path, str):
        asns.extend(re.findall(r"\d+", as_path))

    for field in ("bgp_update", "text"):
        value = update.get(field)
        if isinstance(value, str):
            asns.extend(extract_asns_from_text(value))
    return list(dict.fromkeys(asns))


def retrieve_as_relationship_knowledge(
    update: Dict,
    asn_to_knowledge: Dict[str, List[str]],
    top_k: int = 8,
) -> List[str]:
    """Retrieve relationship statements for ASNs present in an update."""
    retrieved: List[str] = []
    seen = set()
    for asn in extract_as_path_from_update(update):
        for item in asn_to_knowledge.get(asn, []):
            if item not in seen:
                seen.add(item)
                retrieved.append(item)
            if len(retrieved) >= top_k:
                return retrieved
    return retrieved


def build_user_prompt(update: Dict, retrieved_knowledge: List[str]) -> str:
    """Combine a BGP update and retrieved knowledge into a model prompt."""
    update_text = json.dumps(update, ensure_ascii=False, indent=2)
    if retrieved_knowledge:
        knowledge_block = "\n".join(
            f"{index}. {item}" for index, item in enumerate(retrieved_knowledge, 1)
        )
    else:
        knowledge_block = "No relevant AS relationship knowledge was retrieved."

    return (
        f"Please analyze the BGP routing update following <BGP Update:>{update_text}\n"
        "If an anomaly is detected, respond with exactly one of the following "
        "anomaly types: prefix hijack, route leak, bgp hijack, or route hijack. "
        "If no anomaly is detected, respond with: none. Only return one of the "
        "five exact strings. Do not include any additional explanation or variation.\n"
        f"Retrieved AS relationship knowledge:\n{knowledge_block}"
    )


def run_inference(
    update: Dict,
    base_model: str,
    adapter: str,
    relationship_file: str,
    top_k: int,
    max_new_tokens: int,
) -> Tuple[str, List[str]]:
    """Load the base model and adapter, then classify one BGP update."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _, asn_to_knowledge = load_as_relationships(relationship_file)
    knowledge = retrieve_as_relationship_knowledge(update, asn_to_knowledge, top_k)
    prompt = build_user_prompt(update, knowledge)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype="auto",
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant specialized in BGP routing analysis.",
        },
        {"role": "user", "content": prompt},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    response = tokenizer.decode(outputs[0][input_ids.shape[-1] :], skip_special_tokens=True)
    return response.strip(), knowledge


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify one BGP update with ROUTE LLM")
    parser.add_argument("--input", required=True, help="JSON file containing one BGP update")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER)
    parser.add_argument("--as-relationships", default=DEFAULT_RELATIONSHIPS)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.input).open("r", encoding="utf-8") as file_obj:
        update = json.load(file_obj)
    if not isinstance(update, dict):
        raise ValueError("The input JSON must contain one object.")

    response, knowledge = run_inference(
        update=update,
        base_model=args.base_model,
        adapter=args.adapter,
        relationship_file=args.as_relationships,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
    )
    print(json.dumps({"prediction": response, "retrieved_knowledge": knowledge}, indent=2))


if __name__ == "__main__":
    main()
