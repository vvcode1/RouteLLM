import os
import sys
import gzip
import re
import ipaddress
import pandas as pd
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import click
from tqdm import tqdm
import requests
from urllib.parse import urlparse

class IRRDataCompleter:
    """Class to handle IRR data downloading, parsing, and event completion."""

    IRR_URLS = [
        'ftp://ftp.radb.net/radb/dbase/altdb.db.gz',
        'ftp://ftp.radb.net/radb/dbase/arin.db.gz',
        'ftp://ftp.radb.net/radb/dbase/bboi.db.gz',
        'ftp://ftp.radb.net/radb/dbase/bell.db.gz',
        'ftp://ftp.radb.net/radb/dbase/canarie.db.gz',
        'ftp://ftp.radb.net/radb/dbase/easynet.db.gz',
        # 'ftp://ftp.bgp.net.br/host.db.gz',
        'ftp://ftp.radb.net/radb/dbase/jpirr.db.gz',
        # 'ftp://ftp.bgp.net.br/level3.db.gz',
        'ftp://ftp.radb.net/radb/dbase/nestegg.db.gz',
        'ftp://ftp.radb.net/radb/dbase/nttcom.db.gz',
        'ftp://ftp.radb.net/radb/dbase/openface.db.gz',
        'ftp://ftp.radb.net/radb/dbase/panix.db.gz',
        # 'ftp://ftp.radb.net/radb/dbase/radb.db.gz',
        # 'ftp://ftp.radb.net/radb/dbase/reach.db.gz',
        'ftp://ftp.radb.net/radb/dbase/rgnet.db.gz',
        'ftp://ftp.radb.net/radb/dbase/rogers.db.gz',
        # 'ftp://ftp.radb.net/radb/dbase/tc.db.gz',
        # 'ftp://ftp.lacnic.net/lacnic/irr/lacnic.db.gz',
        # 'ftp://ftp.afrinic.net/pub/dbase/afrinic.db.gz',
        'ftp://ftp.apnic.net/pub/apnic/whois/apnic.db.route.gz',
        # 'ftp://ftp.apnic.net/pub/apnic/whois/apnic.db.route6.gz',
        'ftp://ftp.apnic.net/pub/apnic/whois/apnic.db.aut-num.gz',
        'ftp://ftp.ripe.net/ripe/dbase/ripe.db.gz',
        # 'ftp://irr-mirror.idnic.net/idnic.db.gz',
        'ftp://ftp.bgp.net.br/wcgdb.db.gz',
    ]

    # IRR database encodings can be inconsistent
    ENCODING_LIST = ['utf-8', 'Windows-1252', 'Windows-1254', 'ISO-8859-1', 'ISO-2022-JP']

    def __init__(self, db_dir=None):
        """Initialize IRR data completer."""
        self.db_dir = Path(db_dir) if db_dir else Path.home() / '.irr'
        self.db_dir.mkdir(exist_ok=True)

        # Data structures to store IRR information
        self.as_to_prefixes = {}  # AS -> set of prefixes
        self.prefix_to_as = {}    # prefix -> AS
        self.as_names = {}        # AS -> organization name

    def download_irr_db(self, url):
        """Download a single IRR database file."""
        try:
            filename = Path(url).name
            gz_path = self.db_dir / filename
            db_path = gz_path.with_suffix('')  # Remove .gz extension

            # Skip if already downloaded and extracted
            if db_path.exists():
                print(f"Skipping {filename} (already exists)")
                return db_path

            print(f"Downloading {filename}...")

            # Use wget for FTP downloads (more reliable than requests for FTP)
            result = subprocess.run([
                'wget', '-q', '--timeout=30', '--tries=3',
                '-O', str(gz_path), url
            ], capture_output=True)

            if result.returncode != 0:
                print(f"Failed to download {url}: {result.stderr.decode()}")
                return None

            # Decompress
            with gzip.open(gz_path, 'rb') as f_in:
                with open(db_path, 'wb') as f_out:
                    f_out.write(f_in.read())

            # Remove compressed file to save space
            gz_path.unlink()

            print(f"Downloaded and extracted: {filename}")
            return db_path

        except Exception as e:
            print(f"Error downloading {url}: {e}")
            return None

    def sync_irr_dbs(self, max_workers=4):
        """Download all IRR databases concurrently."""
        print("Synchronizing IRR databases from Internet...")

        # Change to IRR directory
        original_cwd = os.getcwd()
        os.chdir(self.db_dir)

        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit download tasks
                future_to_url = {
                    executor.submit(self.download_irr_db, url): url
                    for url in self.IRR_URLS
                }

                # Process completed downloads
                db_files = []
                for future in tqdm(as_completed(future_to_url),
                                 total=len(future_to_url),
                                 desc="Downloading IRR databases"):
                    url = future_to_url[future]
                    try:
                        db_path = future.result()
                        if db_path:
                            db_files.append(db_path)
                    except Exception as e:
                        print(f"Error processing {url}: {e}")

                print(f"Successfully downloaded {len(db_files)} IRR databases")
                return db_files

        finally:
            os.chdir(original_cwd)

    def read_file_with_encoding(self, file_path):
        """Read file with multiple encoding attempts."""
        for encoding in self.ENCODING_LIST:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read(), encoding
            except (UnicodeDecodeError, UnicodeError):
                continue

        # If all encodings fail, read as binary and ignore errors
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read(), 'utf-8-ignore'

    def parse_irr_record(self, record_text):
        """Parse a single IRR record and extract relevant information."""
        lines = record_text.strip().split('\n')
        record = {}
        current_field = None

        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            if ':' in line:
                field, value = line.split(':', 1)
                field = field.strip().lower()
                value = value.strip()
                current_field = field

                if field in record:
                    if isinstance(record[field], list):
                        record[field].append(value)
                    else:
                        record[field] = [record[field], value]
                else:
                    record[field] = value
            elif current_field and line.startswith(' '):
                # Continuation of previous field
                continuation = line.strip()
                if isinstance(record[current_field], list):
                    record[current_field][-1] += ' ' + continuation
                else:
                    record[current_field] += ' ' + continuation

        return record

    def extract_as_number(self, as_string):
        """Extract AS number from various formats."""
        if not as_string:
            return None

        # Handle AS1234, as1234, 1234 formats
        as_match = re.search(r'(?:AS|as)?(\d+)', str(as_string))
        if as_match:
            return int(as_match.group(1))
        return None

    def parse_irr_databases(self, db_files):
        """Parse IRR database files and build lookup tables."""
        print("Parsing IRR databases...")

        total_records = 0
        route_records = 0
        aut_num_records = 0

        for db_file in tqdm(db_files, desc="Processing IRR files"):
            try:
                content, encoding = self.read_file_with_encoding(db_file)

                # Split into records (separated by empty lines)
                records = re.split(r'\n\s*\n', content)

                for record_text in records:
                    if not record_text.strip():
                        continue

                    total_records += 1
                    record = self.parse_irr_record(record_text)

                    # Process route objects (IPv4 and IPv6)
                    if 'route' in record or 'route6' in record:
                        route_records += 1
                        prefix = record.get('route') or record.get('route6')
                        origin = record.get('origin')

                        if prefix and origin:
                            as_num = self.extract_as_number(origin)
                            if as_num:
                                # Store AS -> prefixes mapping
                                if as_num not in self.as_to_prefixes:
                                    self.as_to_prefixes[as_num] = set()
                                self.as_to_prefixes[as_num].add(prefix)

                                # Store prefix -> AS mapping
                                self.prefix_to_as[prefix] = as_num

                    # Process aut-num objects
                    elif 'aut-num' in record:
                        aut_num_records += 1
                        as_string = record.get('aut-num')
                        as_name = record.get('as-name', '') or record.get('descr', '')

                        if as_string:
                            as_num = self.extract_as_number(as_string)
                            if as_num and as_name:
                                self.as_names[as_num] = as_name

            except Exception as e:
                print(f"Error processing {db_file}: {e}")
                continue

        print(f"Parsing complete:")
        print(f"  Total records processed: {total_records}")
        print(f"  Route records: {route_records}")
        print(f"  aut-num records: {aut_num_records}")
        print(f"  Unique AS numbers with prefixes: {len(self.as_to_prefixes)}")
        print(f"  Unique prefix-to-AS mappings: {len(self.prefix_to_as)}")

    def find_as_for_prefix(self, prefix):
        """Find AS number for a given prefix."""
        if not prefix:
            return None

        # Try exact match first
        if prefix in self.prefix_to_as:
            return self.prefix_to_as[prefix]

        # Try to find containing prefix
        try:
            target_network = ipaddress.ip_network(prefix, strict=False)
            best_match_as = None
            best_match_len = -1

            for stored_prefix, as_num in self.prefix_to_as.items():
                try:
                    stored_network = ipaddress.ip_network(stored_prefix, strict=False)

                    # Check if stored prefix contains target prefix
                    if target_network.subnet_of(stored_network):
                        if stored_network.prefixlen > best_match_len:
                            best_match_as = as_num
                            best_match_len = stored_network.prefixlen

                except (ipaddress.AddressValueError, ValueError):
                    continue

            return best_match_as

        except (ipaddress.AddressValueError, ValueError):
            return None

    def find_prefixes_for_as(self, as_number):
        """Find all prefixes for a given AS number."""
        if not as_number:
            return []

        try:
            as_num = int(str(as_number).replace('AS', '').replace('as', ''))
            return list(self.as_to_prefixes.get(as_num, []))
        except (ValueError, TypeError):
            return []

    def complete_event_data(self, events_df):
        """Complete missing fields in events DataFrame."""
        print("Completing missing event data using IRR information...")

        completed_rows = []

        for idx, row in tqdm(events_df.iterrows(), total=len(events_df), desc="Processing events"):
            victim_as = row.get('Victim', '')
            attacker_as = row.get('Attacker', '')
            prefix = row.get('Prefix', '')

            # Convert to string and clean
            victim_as = str(victim_as).strip() if pd.notna(victim_as) else ''
            attacker_as = str(attacker_as).strip() if pd.notna(attacker_as) else ''
            prefix = str(prefix).strip() if pd.notna(prefix) else ''

            # Case 1: Missing Victim AS, have Prefix
            if (not victim_as or victim_as == 'nan') and prefix:
                found_as = self.find_as_for_prefix(prefix)
                if found_as:
                    row = row.copy()
                    row['Victim'] = str(found_as)
                    print(f"Filled Victim AS {found_as} for prefix {prefix}")

            # Case 2: Missing Prefix, have Victim AS
            if (not prefix or prefix == 'nan') and victim_as:
                try:
                    as_num = int(str(victim_as).replace('AS', '').replace('as', ''))
                    prefixes = self.find_prefixes_for_as(as_num)

                    if prefixes:
                        if len(prefixes) == 1:
                            # Single prefix - update current row
                            row = row.copy()
                            row['Prefix'] = prefixes[0]
                            print(f"Filled prefix {prefixes[0]} for AS {as_num}")
                        else:
                            # Multiple prefixes - create separate rows
                            print(f"Expanding AS {as_num} into {len(prefixes)} prefix entries")
                            for prefix_item in prefixes:
                                new_row = row.copy()
                                new_row['Prefix'] = prefix_item
                                completed_rows.append(new_row)
                            continue  # Skip adding original row

                except (ValueError, TypeError):
                    pass

            # Case 3: Have both Victim AS and Prefix, but prefix might be multiple
            elif victim_as and prefix and ',' in prefix:
                # Split multiple prefixes and create separate rows
                prefixes = [p.strip() for p in prefix.split(',') if p.strip()]
                print(f"Expanding {len(prefixes)} prefixes for AS {victim_as}")
                for prefix_item in prefixes:
                    new_row = row.copy()
                    new_row['Prefix'] = prefix_item
                    completed_rows.append(new_row)
                continue  # Skip adding original row

            # Add row (original or modified)
            completed_rows.append(row)

        completed_df = pd.DataFrame(completed_rows)
        print(f"Completion summary:")
        print(f"  Original rows: {len(events_df)}")
        print(f"  Completed rows: {len(completed_df)}")
        print(f"  Rows added due to prefix expansion: {len(completed_df) - len(events_df)}")

        return completed_df

@click.command()
@click.option("--input", "-i", type=str, required=True, help="Input CSV file with BGP anomaly events")
@click.option("--output", "-o", type=str, required=True, help="Output CSV file with completed data")
@click.option("--db-dir", type=str, default=None, help="Directory to store IRR databases")
@click.option("--skip-download", is_flag=True, help="Skip IRR database download (use existing)")
@click.option("--max-workers", type=int, default=4, help="Max workers for concurrent downloads")
def main(input, output, db_dir, skip_download, max_workers):
    """Complete missing BGP anomaly event data using IRR databases."""

    # Validate input file
    input_path = Path(input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input}")
        sys.exit(1)

    # Initialize completer
    completer = IRRDataCompleter(db_dir)

    # Download and sync IRR databases unless skipped
    if not skip_download:
        db_files = completer.sync_irr_dbs(max_workers)
        if not db_files:
            print("Error: Failed to download IRR databases")
            sys.exit(1)
    else:
        # Find existing database files
        db_files = list(completer.db_dir.glob("*.db"))
        if not db_files:
            print(f"Error: No IRR database files found in {completer.db_dir}")
            print("Run without --skip-download to download databases first")
            sys.exit(1)

    # Parse IRR databases
    completer.parse_irr_databases(db_files)

    # Load events CSV
    try:
        events_df = pd.read_csv(input_path)
        print(f"Loaded {len(events_df)} events from {input}")
    except Exception as e:
        print(f"Error loading events CSV: {e}")
        sys.exit(1)

    # Complete missing data
    completed_df = completer.complete_event_data(events_df)

    # Save results
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed_df.to_csv(output_path, index=False)

    print(f"Completed data saved to: {output}")

if __name__ == "__main__":
    main()