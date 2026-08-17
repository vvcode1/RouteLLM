from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI


SCRIPT_DIR = Path(__file__).resolve().parent
SENTENCE_BOUNDARY = re.compile(r"([.!?])\s+")


def normalize_email_text(text: str) -> str:
    """Normalize line endings and whitespace while preserving paragraphs."""
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"\n{2,}", " <PARA> ", normalized)
    normalized = normalized.replace("\n", " ").replace(" <PARA> ", "\n\n")
    return re.sub(r"[ \t]+", " ", normalized).strip()


def split_long_text(text: str, max_chars: int = 45_000) -> list[str]:
    """Split a long message near paragraph or sentence boundaries."""
    if len(text) <= max_chars:
        return [text]

    parts: list[str] = []
    remainder = text
    while len(remainder) > max_chars:
        search_start = max(0, max_chars - 5_000)
        split_index = remainder.rfind("\n\n", search_start, max_chars + 1)
        if split_index == -1:
            sentence_matches = list(
                SENTENCE_BOUNDARY.finditer(remainder, search_start, max_chars + 1)
            )
            split_index = sentence_matches[-1].end() if sentence_matches else max_chars
        part = remainder[:split_index].strip()
        if part:
            parts.append(part)
        remainder = remainder[split_index:].strip()
    if remainder:
        parts.append(remainder)
    return parts


def clean_response_content(response_content: str) -> str:
    """Remove optional Markdown fences from an otherwise raw model response."""
    content = (response_content or "").strip()
    return re.sub(r"```json|```", "", content, flags=re.IGNORECASE).strip()


def build_prompt(email_content: str, email_date: str) -> str:
    return f"""
You are a BGP security analyst. Your task is to analyze the given email and extract structured data about specific BGP security incidents.

---

### Email Content:
{email_content}

### Email Date:
{email_date}

---

### Instructions:

1. **Incident Determination**
   - Determine if the email describes at least one specific BGP security incident involving route prefixes.
   - An incident must explicitly mention at least one IP prefix in CIDR format (for example, "192.0.2.0/24" or "2001:db8::/32").
   - If no such incident is described, return:
     [
         {{
             "IsIncident": false,
             "Reason": "<brief reason why this is not an incident>"
         }}
     ]

2. **For each detected incident, extract the following fields:**
   - **IsIncident**: (boolean) true
   - **Time**: (string) "YYYY-MM-DD HH:MM:SS"
     - Use the time mentioned in the email.
     - If only a date or date range is given, choose a representative datetime.
     - If no time is mentioned, return the Email Date.
   - **Attacker**: (string or array) AS in "AS1234" form. If multiple attackers exist, use an array. If unknown, use "".
   - **Victim**: (string or array) AS in "AS5678" form. If multiple victims exist, use an array. If unknown, use "".
   - **Prefixes**: (array) The specific route prefixes involved in the incident.
   - **Description**: (string) The exact sentence or paragraph that supports the incident determination.
   - **Reason**: (string) A brief explanation of the evidence.

3. **Output Requirements**
   - Return only a valid JSON array.
   - Each object must contain all fields above in the same order, even if some values are empty.
   - Do not include explanations or Markdown.
   - If multiple incidents are found, sort them by "Time" ascending.
""".strip()


def call_llm(
    client: OpenAI,
    prompt: str,
    model: str,
    timeout_seconds: int,
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        timeout=timeout_seconds,
    )
    return (response.choices[0].message.content or "").strip()


def analyze_email(
    client: OpenAI,
    email_id: str,
    email_content: str,
    email_date: str,
    model: str,
    timeout_seconds: int,
    max_chars: int,
) -> list[str]:
    normalized = normalize_email_text(email_content or "")
    parts = split_long_text(normalized, max_chars=max_chars)
    raw_responses: list[str] = []

    for index, part in enumerate(parts, start=1):
        if len(parts) > 1:
            print(
                f"Splitting Email_ID={email_id}: part {index}/{len(parts)} "
                f"({len(part)} characters)",
                flush=True,
            )
        try:
            raw = call_llm(client, build_prompt(part, str(email_date)), model, timeout_seconds)
            raw_responses.append(clean_response_content(raw))
        except Exception as exc:
            print(f"LLM error for Email_ID={email_id}, part={index}: {exc}", flush=True)
            raw_responses.append(f"__ERROR__: {exc}")
    return raw_responses


def process_emails(args: argparse.Namespace) -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    client = OpenAI()
    matching_emails = pd.read_csv(args.input, dtype=str)
    csv_fields = ["Email_ID", "Part", "Model", "RawResponse"]
    output_csv = Path(args.output_csv)
    output_jsonl = Path(args.output_jsonl)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_csv.exists() or output_csv.stat().st_size == 0
    start_time = time.perf_counter()

    with output_csv.open("a", newline="", encoding="utf-8") as csv_file, output_jsonl.open(
        "a", encoding="utf-8"
    ) as jsonl_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
        if write_header:
            writer.writeheader()

        total = len(matching_emails)
        for row_number, (_, row) in enumerate(matching_emails.iterrows(), start=1):
            email_id = row.get("Email_ID", "")
            responses = analyze_email(
                client=client,
                email_id=email_id,
                email_content=row.get("Email_Content", ""),
                email_date=row.get("Date", ""),
                model=args.model,
                timeout_seconds=args.timeout,
                max_chars=args.max_chars,
            )
            for part_number, raw_response in enumerate(responses, start=1):
                result = {
                    "Email_ID": email_id,
                    "Part": part_number,
                    "Model": args.model,
                    "RawResponse": raw_response,
                }
                writer.writerow(result)
                jsonl_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            csv_file.flush()
            jsonl_file.flush()
            print(f"[{row_number}/{total}] Email_ID={email_id} parts={len(responses)}", flush=True)
            if args.delay > 0:
                time.sleep(args.delay)

    elapsed = time.perf_counter() - start_time
    print(
        f"Analysis complete in {elapsed:.1f}s. CSV -> {output_csv}, "
        f"JSONL -> {output_jsonl}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract BGP incidents from candidate emails")
    parser.add_argument("--input", default=str(SCRIPT_DIR / "matching_emails.csv"))
    parser.add_argument("--output-csv", default=str(SCRIPT_DIR / "gpt3_5_turbo.csv"))
    parser.add_argument("--output-jsonl", default=str(SCRIPT_DIR / "gpt3_5_turbo.jsonl"))
    parser.add_argument("--model", default=os.environ.get("ROUTE_LLM_EXTRACTION_MODEL", "o4-mini"))
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--max-chars", type=int, default=45_000)
    parser.add_argument("--delay", type=float, default=0.5)
    return parser.parse_args()


if __name__ == "__main__":
    process_emails(parse_args())
