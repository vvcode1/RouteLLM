import re
import pandas as pd
from pathlib import Path

# ==== 0. Input Configuration ====
INPUT_CSV = "./nanog_email.csv"
OUT_CSV   = "./matching_emails.csv"
OUT_TXT   = "./matching_emails.txt"

PROGRESS_INTERVAL = 1000

# Your keyword list (keeping original)
keywords = [
    # Route leaks
    'route leak', 'routing leak', 'bgp leak',
    'type-1 route leak', 'type-2 route leak', 'type-3 route leak',
    'type-4 route leak', 'type-5 route leak', 'type-6 route leak', 'type-7 route leak',
    'valley-free violation', 'policy violation', 'route propagation error',
    'accidental route leak', 'route redistribution issue',

    # Hijacks
    'route hijack', 'routing hijack', 'prefix hijack', 'origin hijack', 'bgp hijack',
    'as hijack', 'asn hijack', 'bgp prefix hijack', 'bgp route hijack',
    'as origin spoofing', 'unauthorized prefix announcement', 'origin-as hijack',
    'inter-as hijack', 'hijacked route', 'hijacked prefix', 'route usurpation',

    # Injection & Spoofing
    'route injection', 'prefix injection', 'subprefix hijack',
    'prefix misannouncement', 'route misannouncement',
    'bogus route', 'bogus prefix', 'ghost route', 'fake prefix',
    'unauthorized announcement', 'prefix spoofing', 'illegitimate prefix',

    # Path Manipulation
    'as path manipulation', 'as path prepending attack', 'as path prepending',
    'as path poisoning', 'as path spoofing', 'as path hiding',
    'false as path', 'artificial as path', 'as path forgery', 'path forgery',

    # BGP Misconfigurations
    'bgp misconfiguration', 'bgp configuration error', 'route flapping',
    'bgp session reset', 'prefix loop', 'routing table overflow',
    'policy misconfiguration', 'route reflector misbehavior',

    # General attack types
    'bgp attack', 'route manipulation', 'routing attack',
    'invalid route announcement', 'unauthorized route announcement',
    'routing anomaly', 'routing incident', 'routing security event',
    'network instability', 'suspicious bgp update', 'malicious bgp update',
    'network attack', 'control plane attack', 'interdomain routing threat'
]

# ==== Utility function: Normalize email text ====
def normalize_email_text(text: str) -> str:
    if not text:
        return ""
    t = text.replace('\r\n', '\n').replace('\r', '\n')
    # Mark 2+ newlines as paragraph separators
    t = re.sub(r'\n{2,}', ' <PARA> ', t)
    # Merge remaining single newlines (soft wraps) into spaces
    t = t.replace('\n', ' ')
    # Restore paragraph separators as double newlines for paragraph/sentence splitting
    t = t.replace(' <PARA> ', '\n\n')
    # Compress extra whitespace
    t = re.sub(r'[ \t]+', ' ', t)
    # Remove leading/trailing whitespace
    t = t.strip()
    return t

# Sentence splitting (period/question/exclamation mark or blank lines)
SENT_SPLIT = re.compile(r'(?<=[\.\?!])\s+|\n{2,}')

# ==== Read input data ====
if not Path(INPUT_CSV).exists():
    raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

df = pd.read_csv(INPUT_CSV, dtype=str)

required = ['Email_ID', 'Date', 'Email_Title', 'Author', 'Email_Address', 'Email_Content']
for c in required:
    if c not in df.columns:
        raise ValueError(f"Missing required column: {c}")

# Fill null values
df['Email_Content'] = df['Email_Content'].fillna("").astype(str)
df['Email_Title']   = df['Email_Title'].fillna("").astype(str)

def match_and_record_keywords_substring(row, keywords):
    # Normalize content, preserve paragraphs, remove soft wraps
    content_norm = normalize_email_text(row['Email_Content'])
    title_norm   = row['Email_Title'].strip()

    content_low = content_norm.lower()
    title_low   = title_norm.lower()

    matched_keywords = [kw for kw in keywords if kw in content_low or kw in title_low]

    matched_sentences = []
    if matched_keywords and content_norm:
        # Split by punctuation or blank lines; sentences are now merged due to soft wrap removal
        sentences = [s.strip() for s in SENT_SPLIT.split(content_norm) if s and s.strip()]
        seen = set()
        for s in sentences:
            s_low = s.lower()
            if any(kw in s_low for kw in matched_keywords):
                if s not in seen:
                    matched_sentences.append(s)
                    seen.add(s)

    return matched_keywords, matched_sentences

# ==== Processing with progress tracking ====
total_rows = len(df)
print(f"Processing {total_rows} emails with substring matching + newline normalization...")

matched_keywords_list  = []
matched_sentences_list = []

for idx, row in df.iterrows():
    mk, ms = match_and_record_keywords_substring(row, keywords)
    matched_keywords_list.append(mk)
    matched_sentences_list.append(ms)

    if (idx + 1) % PROGRESS_INTERVAL == 0:
        print(f"Processed {idx + 1}/{total_rows} rows...")

df['Matched_Keywords']  = matched_keywords_list
df['Matched_Sentences'] = matched_sentences_list

# ==== Filter matching emails ====
matching_emails = df[df['Matched_Keywords'].apply(lambda x: len(x) > 0)]

# ==== Output results ====
if matching_emails.empty:
    print("No matching emails found...")
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("Total matching emails: 0\n")
else:
    matching_emails.to_csv(OUT_CSV, index=False, encoding="utf-8")

    keyword_counts = {
        kw: matching_emails['Matched_Keywords'].apply(lambda lst: kw in lst).sum()
        for kw in keywords
    }
    total_matching_emails = len(matching_emails)

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(f"Total matching emails: {total_matching_emails}\n\n")
        f.write("Keyword hit counts:\n")
        for kw, cnt in sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True):
            f.write(f"- {kw}: {cnt}\n")

        f.write("\nExamples (up to 20):\n")
        for _, r in matching_emails.head(20).iterrows():
            f.write(f"\n[Email_ID: {r.get('Email_ID','')}] {r.get('Email_Title','')}\n")
            f.write(f"Date: {r.get('Date','')} | Author: {r.get('Author','')} <{r.get('Email_Address','')}>\n")
            f.write("Matched Keywords: " + ", ".join(r['Matched_Keywords']) + "\n")
            if r['Matched_Sentences']:
                f.write("Matched Sentences:\n")
                for s in r['Matched_Sentences'][:5]:
                    f.write(f"  - {s}\n")

print("Processing complete.")