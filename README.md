# ROUTELLM

ROUTELLM is a research prototype for discovering real-world Border Gateway
Protocol (BGP) incidents and detecting routing anomalies with a
routing-domain-adapted large language model.


## Repository structure

| Path | Purpose |
| --- | --- |
| `mail_scraper/` | Collects mailing-list messages and routing-security blog posts from NANOG, AusNOG, RIPE NCC, NZNOG, ITNOG, SAFNOG, SGNOG, APNIC, MANRS, Cloudflare, and related sources. |
| `event_collector/` | Filters messages with routing-security keywords, invokes an LLM to extract structured incidents, deduplicates results, and completes missing prefix/ASN fields using IRR data. |
| `event_collector/events/event.csv` | Anomaly-event dataset extracted and normalized by the event-collection pipeline. |
| `bgp_fetcher/` | Downloads RouteViews updates around each event and selects messages related to the reported prefix and ASNs. |
| `data processor/` | Parses `bgpdump` output, converts BGP updates and AS relationships into text, and builds records for training or inference. |
| `routing_adapter/` | Contains SentencePiece tokenizer experiments and full/LoRA fine-tuning scripts. |
| `inference/` | Loads a Llama 3.1 base model, a PEFT/LoRA adapter, and AS-relationship evidence for anomaly classification. |

## Installation

### Environment 

The experiments were conducted with Python 3.12, PyTorch 2.1.2, CUDA 12.1, and Ubuntu 22.04 LTS.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Full-model fine-tuning additionally uses FlashAttention 2. Install a version
compatible with the local CUDA and PyTorch environment separately:

```bash
python -m pip install flash-attn --no-build-isolation
```

### External tools

Parts of the pipeline also require:

- `curl`, `wget`, and `bzip2` for downloading routing archives and IRR data;
- `bgpdump`, or the compatible `bgpd` binary used by
  `bgp_fetcher/bgp_anomaly_collector.py`, for parsing MRT/BGP archives;
- access to the Llama-3.1-8B-Instruct base model for training and inference;
- an LLM API credential for the incident-extraction stage.

Never commit API credentials. Store them in environment variables such as
`OPENAI_API_KEY`. `OPENAI_BASE_URL` may also be set when using a compatible
provider endpoint.

Model and routing-data locations can be configured without editing source code:

```bash
export ROUTE_LLM_BASE_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
export ROUTE_LLM_ADAPTER="models/route-llm-llama3.1-8b-instruct-lora"
export ROUTE_LLM_AS_RELATIONSHIPS="data/as-relationships/20241201.as-rel.txt"
```

## Pipeline

The commands below describe the current processing order. Several research
scripts still use constants and relative paths declared near the top of each
file. Check those values before running a stage, and run source-specific
scrapers from their own directories unless their paths have been updated.

### 1. Collect mailing-list and blog data

There are two kinds of collectors:

- `*_mbox.py` and some `*_scraper.py` scripts convert an existing mbox archive
  into CSV;
- web scrapers download public archive pages directly and append messages to a
  source-specific CSV.

For example, the NANOG converter expects its mbox input path and output path to
be configured at the top of the script:

```bash
cd mail_scraper/nanog
python nanog_scraper.py
```

Other source directories follow the same pattern. The normalized email schema
used by later stages is:

```text
Email_ID, Date, Email_Title, Author, Email_Address, Email_Content
```

Thread-aware web collectors may additionally produce `Thread_ID` and `Level`.

### 2. Filter candidate messages

`event_collector/keyword_match.py` searches titles and message bodies for BGP
security terminology. Its current default input is `nanog_email.csv` in the
working directory.

```bash
cd event_collector
python keyword_match.py
```

The script produces `matching_emails.csv` and a human-readable
`matching_emails.txt`.

### 3. Extract and normalize incidents with an LLM

`event_collector/llm_process.py` sends candidate messages to an LLM and stores
the raw structured responses. `event_collector/event_extractor.py` parses those
responses and removes duplicates.

```bash
export OPENAI_API_KEY="your-api-key"
python llm_process.py \
  --input matching_emails.csv
python event_extractor.py
```

Use `python llm_process.py --help` to configure the model,
input, output, timeout, and request delay. 

To complete missing prefix and victim-AS fields with IRR data:

```bash
python irr_completer.py \
  --input incidents_dedup.csv \
  --output incidents_completed.csv \
  --db-dir ../data/irr
```

This step downloads public IRR databases unless `--skip-download` is supplied.
Human validation is required before extracted incidents are treated as ground
truth.

### 4. Download BGP updates associated with verified events

The BGP collector expects a CSV with these columns:

```text
Time, Victim, Attacker, Prefix, Category
```

Use one row per prefix when an event contains multiple prefixes.

```bash
cd ..
python bgp_fetcher/bgp_anomaly_collector.py \
  --events /path/to/verified_events.csv \
  --collector route-views4 \
  --output data/bgp_results \
  --bgpd /path/to/bgpd
```

For each event, the script downloads RouteViews updates from the 12 hours before
and after the reported time, filters them by prefix and AS information, and
writes matching messages to an event-specific CSV file.

### 5. Prepare model data

`data processor/data_processor.py` provides the `DataProcessor` class for:

- parsing pipe-separated `bgpdump -m` output;
- converting BGP fields to readable text;
- converting CAIDA-style AS relationships to natural-language statements;
- producing JSONL records for fine-tuning and inference.

Training JSON/JSONL records used by the current scripts contain the fields:

```json
{
  "instruction": "classification instruction",
  "input": "<BGP Update:> ...",
  "output": "expected label or structured response"
}
```

### 6. Train the tokenizer and routing adapter

Tokenizer experiment:

```bash
python routing_adapter/custom_tokenizer_trainer.py \
  --model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
  --data_path /path/to/training.jsonl \
  --output_dir outputs/tokenizer
```

The training scripts currently read their model path, dataset directory,
output directory, and class-ratio experiments from an internal `config`
dictionary. Update that configuration before starting a run.

```bash
cd routing_adapter
python multi_ratio_lora_training.py

# Optional and substantially more expensive:
python multi_ratio_full_finetuning_training.py
```

### 7. Run inference

Create a JSON file containing one BGP update, for example:

```json
{
  "prefix": "203.0.113.0/24",
  "as_path": [64512, 3356, 64496],
  "timestamp": "2025-04-01T12:00:00Z",
  "type": "A"
}
```

Then provide a CAIDA-style AS-relationship file and run:

```bash
python inference/inference.py \
  --input /path/to/update.json \
  --as-relationships /path/to/as-rel.txt
```

The base model is not included in this repository. Access to
`meta-llama/Meta-Llama-3.1-8B-Instruct` or a compatible local copy is required.


## License

The repository's original code and included LoRA adapter are licensed under the
[Apache License 2.0](LICENSE). The Meta Llama 3.1 base model is not included and
is governed by Meta's separate license and acceptable-use terms.

## Citation

Citation metadata will be added after the paper's final bibliographic details
are available.
