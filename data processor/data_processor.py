from __future__ import annotations
import csv
import json
import random
import re
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional


INSTRUCTION_TEMPLATES = [
    "Analyze the BGP update after <BGP Update:>. Decide whether it is normal or anomalous. If anomalous, classify it as prefix_hijack, path_hijack, route_leak, or unknown.",
    "You are a routing security analyst. Read the content after <BGP Update:> and judge whether the BGP announcement is normal. If it is abnormal, return the anomaly type: prefix_hijack, path_hijack, route_leak, or unknown.",
    "Inspect the BGP routing message that follows <BGP Update:>. Determine whether it violates expected inter-domain routing behavior. Output normal or anomalous, and provide anomaly_type if applicable.",
    "Please examine the BGP update below. Use the routing fields to decide whether this update is benign. If not benign, label the anomaly type as prefix_hijack, path_hijack, route_leak, or unknown.",
    "Given the BGP update after <BGP Update:>, identify whether this update is normal. If an anomaly exists, identify the category and briefly explain why.",
]
@dataclass
class BGPUpdate:
    timestamp: str = ""
    type: str = ""
    peer_ip: str = ""
    peer_asn: str = ""
    prefix: str = ""
    as_path: str = ""
    origin: str = ""
    next_hop: str = ""
    communities: str = ""
    med: str = ""
    local_pref: str = ""
    raw_text: str = ""

    def to_natural_language(self) -> str:
        parts = [
            f"TIME: {self.timestamp}" if self.timestamp else None,
            f"TYPE: {self.type}" if self.type else None,
            f"FROM: {self.peer_ip} AS{self.peer_asn}" if self.peer_ip or self.peer_asn else None,
            f"PREFIX: {self.prefix}" if self.prefix else None,
            f"ASPATH: {self.as_path}" if self.as_path else None,
            f"ORIGIN: {self.origin}" if self.origin else None,
            f"NEXT_HOP: {self.next_hop}" if self.next_hop else None,
            f"COMMUNITIES: {self.communities}" if self.communities else None,
            f"MED: {self.med}" if self.med else None,
            f"LOCAL_PREF: {self.local_pref}" if self.local_pref else None,
        ]
        return ", ".join(p for p in parts if p)


class DataProcessor:
    """
    A lightweight implementation of the paper's Data Processor.

    Responsibilities:
    1. Convert raw MRT/BGP data to human-readable text with bgpdump.
    2. Parse bgpdump output into structured BGP updates.
    3. Convert AS relationship tuples into natural-language knowledge entries.
    4. Generate instruction variants used during training/inference.
    5. Build JSONL samples for downstream fine-tuning or inference.
    """

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)

    def run_bgpdump(self, mrt_file: str | Path, bgpdump_bin: str = "bgpdump") -> str:
        """Run bgpdump on a MRT file and return stdout as text."""
        mrt_file = str(mrt_file)
        result = subprocess.run(
            [bgpdump_bin, "-m", mrt_file],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout

    def parse_bgpdump_line(self, line: str) -> Optional[BGPUpdate]:
        """
        Parse one bgpdump -m line.

        The -m format is pipe-separated and often looks like:
        BGP4MP|1725328063|A|192.0.2.1|64500|203.0.113.0/24|64500 64496|IGP|192.0.2.1|0|0|64500:100|NAG||
        """
        line = line.strip()
        if not line or "|" not in line:
            return None

        fields = line.split("|")
        if len(fields) < 8:
            return None

        # Conservative parser: different collectors may export slightly different lengths.
        update = BGPUpdate(
            type=fields[2] if len(fields) > 2 else "",
            timestamp=fields[1] if len(fields) > 1 else "",
            peer_ip=fields[3] if len(fields) > 3 else "",
            peer_asn=fields[4] if len(fields) > 4 else "",
            prefix=fields[5] if len(fields) > 5 else "",
            as_path=fields[6] if len(fields) > 6 else "",
            origin=fields[7] if len(fields) > 7 else "",
            next_hop=fields[8] if len(fields) > 8 else "",
            local_pref=fields[9] if len(fields) > 9 else "",
            med=fields[10] if len(fields) > 10 else "",
            communities=fields[11] if len(fields) > 11 else "",
            raw_text=line,
        )
        return update

    def parse_bgpdump_text(self, text: str) -> List[BGPUpdate]:
        updates: List[BGPUpdate] = []
        for line in text.splitlines():
            item = self.parse_bgpdump_line(line)
            if item:
                updates.append(item)
        return updates

    def as_relationship_to_text(self, as1: str, as2: str, rel: str) -> str:
        rel = rel.strip().lower()
        if rel in {"-1", "p2c"}:
            return f"AS{as1} is a provider AS of AS{as2}."
        if rel in {"1", "c2p"}:
            return f"AS{as1} is a customer AS of AS{as2}."
        if rel in {"0", "p2p"}:
            return f"AS{as1} is a peer AS of AS{as2}."
        if rel in {"s2s", "2"}:
            return f"AS{as1} is a sibling AS of AS{as2}."
        return f"AS{as1} has relationship {rel} with AS{as2}."

    def load_as_relationships(self, rel_file: str | Path) -> List[Dict[str, str]]:
        """
        Load CAIDA-style relationship lines: AS1|AS2|RELATIONSHIP|SOURCE
        Comments beginning with # are ignored.
        """
        entries: List[Dict[str, str]] = []
        with open(rel_file, "r", encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw or raw.startswith("#"):
                    continue
                parts = raw.split("|")
                if len(parts) < 3:
                    continue
                as1, as2, rel = parts[:3]
                source = parts[3] if len(parts) > 3 else ""
                entries.append(
                    {
                        "as1": as1,
                        "as2": as2,
                        "relationship": rel,
                        "source": source,
                        "text": self.as_relationship_to_text(as1, as2, rel),
                    }
                )
        return entries

    def choose_instruction(self) -> str:
        return self.rng.choice(INSTRUCTION_TEMPLATES)

    def build_inference_prompt(self, update: BGPUpdate) -> str:
        inst = self.choose_instruction()
        return f"{inst}\n\n<BGP Update:> {update.to_natural_language()}"

    def build_training_record(
        self,
        update: BGPUpdate,
        label: str,
        anomaly_type: str = "none",
        explanation: str = "",
    ) -> Dict[str, str]:
        prompt = self.build_inference_prompt(update)
        answer = {
            "label": label,
            "anomaly_type": anomaly_type,
            "explanation": explanation,
        }
        return {
            "instruction": prompt,
            "output": json.dumps(answer, ensure_ascii=False),
        }

    def write_jsonl(self, records: Iterable[Dict[str, str]], output_file: str | Path) -> None:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def write_relationship_knowledge_jsonl(
        self, rel_file: str | Path, output_file: str | Path
    ) -> None:
        entries = self.load_as_relationships(rel_file)
        rows = [{"text": e["text"], "metadata": {k: v for k, v in e.items() if k != "text"}} for e in entries]
        self.write_jsonl(rows, output_file)


def demo() -> None:
    dp = DataProcessor(seed=7)
    example_line = (
        "BGP4MP|1725328063|A|2406:840:eb83::1|140731|203.0.113.0/24|140731 137990|IGP|2406:840:eb83::1|100|0|140731:100|NAG||"
    )
    update = dp.parse_bgpdump_line(example_line)
    assert update is not None
    print("Natural language update:\n", update.to_natural_language())
    print("\nInference prompt:\n", dp.build_inference_prompt(update))
    print("\nAS relationship text:\n", dp.as_relationship_to_text("3356", "15169", "-1"))


if __name__ == "__main__":
    demo()
