import json
import pandas as pd
import re

def extract_incidents():
    # Fixed input and output file names
    input_csv = "gpt3_5_turbo.csv"
    output_csv = "incidents_only.csv"

    def clean_json_array_string(s: str) -> str:
        """Remove ```json fences and extract JSON array portion if possible"""
        if not s:
            return "[]"
        s = str(s).strip()
        s = re.sub(r"```json|```", "", s, flags=re.IGNORECASE).strip()
        if '[' in s and ']' in s:
            start = s.find('[')
            end = s.rfind(']')
            if start >= 0 and end > start:
                return s[start:end+1].strip()
        return s

    def to_str_or_join(value):
        """Convert Attacker/Victim to string format"""
        if isinstance(value, list):
            return ";".join(str(x) for x in value if str(x).strip())
        if isinstance(value, str):
            return value
        return ""

    def prefixes_to_str(prefixes):
        """Convert Prefixes to string format"""
        if isinstance(prefixes, list):
            return ";".join(str(x) for x in prefixes if str(x).strip())
        if isinstance(prefixes, str):
            return prefixes
        return ""

    # Read CSV file
    df = pd.read_csv(input_csv, dtype=str, keep_default_na=False)

    out_rows = []
    for _, row in df.iterrows():
        email_id = row.get("Email_ID", "")
        part = row.get("Part", "")
        raw = row.get("RawResponse", "")

        raw_clean = clean_json_array_string(raw)

        try:
            data = json.loads(raw_clean)
            if isinstance(data, dict):
                data = [data]
        except json.JSONDecodeError:
            continue

        if not isinstance(data, list):
            continue

        for obj in data:
            if not isinstance(obj, dict):
                continue
            if str(obj.get("IsIncident", "")).lower() != "true":
                continue

            out_rows.append({
                "Email_ID": email_id,
                "Part": part,
                "Time": obj.get("Time", ""),
                "Attacker": to_str_or_join(obj.get("Attacker", "")),
                "Victim": to_str_or_join(obj.get("Victim", "")),
                "Prefixes": prefixes_to_str(obj.get("Prefixes", [])),
                "Description": obj.get("Description", ""),
                "Reason": obj.get("Reason", "")
            })

    # Save results (including Reason field)
    pd.DataFrame(
        out_rows,
        columns=["Email_ID", "Part", "Time", "Attacker", "Victim", "Prefixes", "Description", "Reason"]
    ).to_csv(output_csv, index=False, encoding="utf-8")

    print(f"Generated {output_csv} with {len(out_rows)} records.")
    return output_csv

def deduplicate_incidents():
    """
    Remove duplicate incidents based on Attacker, Victim, and Prefixes combination.
    Keeps the first occurrence of each unique combination.
    """
    # Fixed file names
    input_csv = "incidents_only.csv"     # Contains: Email_ID, Part, Time, Attacker, Victim, Prefixes, Description, Reason
    output_csv = "incidents_dedup.csv"   # Deduplicated output

    # Read input file
    df = pd.read_csv(input_csv, dtype=str, keep_default_na=False)

    # Light normalization (remove whitespace): avoid same values being treated as different due to extra spaces
    for col in ["Attacker", "Victim", "Prefixes"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Deduplicate: by Attacker, Victim, Prefixes combination, keep first occurrence
    dedup_df = df.drop_duplicates(subset=["Attacker", "Victim", "Prefixes"], keep="first")

    # Save (maintain original column order, will automatically export existing columns if some are missing)
    dedup_df.to_csv(output_csv, index=False, encoding="utf-8")

    print(f"Input: {input_csv}  Total {len(df)} rows")
    print(f"Output: {output_csv}  After deduplication {len(dedup_df)} rows (kept first by Attacker, Victim, Prefixes)")

    return output_csv

def main():
    print("Step 1: Extracting BGP incidents from analysis results...")
    extract_incidents()

    print("\nStep 2: Deduplicating incidents...")
    final_output = deduplicate_incidents()

    print(f"\nProcessing complete. Final output: {final_output}")

if __name__ == "__main__":
    main()