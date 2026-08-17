from pathlib import Path
from io import StringIO
from urllib.parse import urljoin
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np
import subprocess
import re
import json
import click
import os
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_DIR = SCRIPT_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Remove incomplete cache files for current month
current_ym = datetime.now().strftime("%Y.%m")
for cache_file in CACHE_DIR.glob(f"*{current_ym}*"):
    cache_file.unlink()

def get_all_collectors(url_index="http://routeviews.org/"):
    """Get available collectors from RouteViews."""
    cache_path = CACHE_DIR / f"collectors2url.{url_index.replace('/', '+')}"
    if cache_path.exists():
        try:
            return json.load(open(cache_path, "r"))
        except:
            pass

    res = subprocess.check_output(["curl", "-s", url_index]).decode()
    res = re.sub(r"\s\s+", " ", res.replace("\n", " "))
    collectors2url = {}

    for a, b in re.findall(r'\<A HREF="(.+?)"\>.+?\([\w\s]+, from (.+?)\)', res):
        collector_name = b.split(".")[-3]
        if collector_name in collectors2url:
            idx = 2
            while f"{collector_name}{idx}" in collectors2url:
                idx += 1
            collector_name = f"{collector_name}{idx}"
        collectors2url[collector_name] = urljoin(url_index, a) + "/"

    json.dump(collectors2url, open(cache_path, "w"), indent=2)
    return collectors2url

def get_archive_list(collector, collectors2url, dtime1, dtime2):
    """Get list of BGP update archives for the specified time range."""
    if collector not in collectors2url:
        return []

    def pull_list(ym):
        target_url = urljoin(collectors2url[collector], f"{ym}/UPDATES") + "/"
        cache_path = CACHE_DIR / f"archive_list.{target_url.replace('/', '+')}"
        if cache_path.exists():
            try:
                return target_url, json.load(open(cache_path, "r"))
            except:
                pass

        res = subprocess.check_output(["curl", "-s", target_url]).decode()
        archive_list = re.findall(
            r'\<a href="(.+?(\d{4}).??(\d{2}).??(\d{2}).??(\d{4}).*?\.bz2)"\>', res)

        json.dump(archive_list, open(cache_path, "w"), indent=2)
        return target_url, archive_list

    ym1 = dtime1.strftime("%Y.%m")
    ym2 = dtime2.strftime("%Y.%m")
    target_url1, archive_list1 = pull_list(ym1)
    target_url2, archive_list2 = pull_list(ym2)

    if not archive_list1 or not archive_list2:
        print(f"Failed to get archive list for {collector}: {dtime1} to {dtime2}")
        return []

    time_list1 = ["".join(i[1:]) for i in archive_list1]
    time_list2 = ["".join(i[1:]) for i in archive_list2]
    t1 = dtime1.strftime("%Y%m%d%H%M")
    t2 = dtime2.strftime("%Y%m%d%H%M")
    idx1 = np.searchsorted(time_list1, t1, side="left")
    idx2 = np.searchsorted(time_list2, t2, side="right")

    if time_list1 == time_list2:
        data = [urljoin(target_url1, i[0]) for i in archive_list1[idx1:idx2]]
    else:
        data = [urljoin(target_url1, i[0]) for i in archive_list1[idx1:]]

        current_month = datetime(dtime1.year, dtime1.month, 1)
        current_month += relativedelta(months=1)
        upper_bound = datetime(dtime2.year, dtime2.month, 1)

        while current_month < upper_bound:
            cur_ym = current_month.strftime("%Y.%m")
            cur_target_url, cur_archive_list = pull_list(cur_ym)
            data += [urljoin(cur_target_url, i[0]) for i in cur_archive_list]
            current_month += relativedelta(months=1)

        data += [urljoin(target_url2, i[0]) for i in archive_list2[:idx2]]

    return data

def download_data(url, collector):
    """Download and decompress BGP update file."""
    fname = url.split("/")[-1].strip()
    outpath = SCRIPT_DIR / "updates" / collector / fname
    fpath = outpath.with_suffix("")

    if fpath.exists():
        return fpath

    outpath.parent.mkdir(exist_ok=True, parents=True)
    try:
        subprocess.run(["curl", "-s", url, "--output", str(outpath)], check=True)
        subprocess.run(["bzip2", "-d", str(outpath)], check=True)
        print(f"Downloaded updates for {collector}: {outpath.stem}")
        return fpath
    except subprocess.CalledProcessError as e:
        print(f"Error downloading {url}: {e}")
        return None

def load_updates_to_df(fpath, bgpd=SCRIPT_DIR/"bgpd"):
    """Parse BGP updates file using bgpd and return DataFrame."""
    if not fpath or not fpath.exists():
        return pd.DataFrame()

    try:
        res = subprocess.check_output([str(bgpd), "-q", "-m", "-u", str(fpath)]).decode()
        fmt = "type|timestamp|A/W|peer-ip|peer-asn|prefix|as-path|origin-protocol|next-hop|local-pref|MED|community|atomic-agg|aggregator|unknown-field-1|unknown-field-2"
        cols = fmt.split("|")
        df = pd.read_csv(StringIO(res), sep="|", names=cols, usecols=cols[:-2], dtype=str, keep_default_na=False)
        return df
    except subprocess.CalledProcessError as e:
        print(f"Error parsing {fpath}: {e}")
        return pd.DataFrame()

def parse_event_time(time_str):
    formats = [
        "%Y%m%d%H%M"
    ]

    for fmt in formats:
        try:
            return datetime.strptime(str(time_str).strip(), fmt)
        except ValueError:
            continue

    # Try pandas parsing as fallback
    try:
        return pd.to_datetime(time_str)
    except:
        raise ValueError(f"Unable to parse time format: {time_str}")

def parse_as_path(as_path_str):
    """Parse AS path string and return list of AS numbers."""
    if not as_path_str or pd.isna(as_path_str):
        return []

    # Handle AS path prepending, AS sets, and confederations
    # Remove AS sets (enclosed in {}) and confederations (enclosed in ())
    cleaned_path = re.sub(r'[{}()]', '', str(as_path_str))

    # Split by space and filter out empty strings
    as_list = [as_num.strip() for as_num in cleaned_path.split() if as_num.strip()]

    return as_list

def check_as_in_path(as_path_str, target_as):
    """Check if target AS appears in the BGP AS path."""
    as_path = parse_as_path(as_path_str)
    target_as_str = str(target_as).strip()
    return target_as_str in as_path

def filter_bgp_updates(df, victim_as, attacker_as, prefix):
    if df.empty:
        return df

    # Convert AS numbers to strings and handle different formats
    victim_as = str(victim_as).strip() if pd.notna(victim_as) else ""
    attacker_as = str(attacker_as).strip() if pd.notna(attacker_as) else ""
    prefix = str(prefix).strip() if pd.notna(prefix) else ""

    print(f"Filtering BGP updates - Victim: {victim_as}, Attacker: {attacker_as}, Prefix: {prefix}")

    # Create filter conditions
    conditions = []

    if prefix:
        # Support both exact prefix match and subnet containment
        prefix_escaped = prefix.replace('.', r'\.').replace('/', r'\/')
        prefix_condition = df['prefix'].str.contains(f'^{prefix_escaped}', regex=True, na=False)
        conditions.append(prefix_condition)
        print(f"Prefix filter applied: {prefix}")

    if victim_as and victim_as != "nan":
        def check_victim_as(as_path):
            as_list = parse_as_path(as_path)
            if not as_list:
                return False
            # For most anomalies, victim AS should be the origin (last AS in path)
            return as_list[-1] == victim_as

        victim_condition = df['as-path'].apply(check_victim_as)
        conditions.append(victim_condition)
        print(f"Victim AS filter applied: {victim_as} (checking origin position)")

    if attacker_as and attacker_as != "nan":
        def check_attacker_as(as_path):
            return check_as_in_path(as_path, attacker_as)

        attacker_condition = df['as-path'].apply(check_attacker_as)
        conditions.append(attacker_condition)
        print(f"Attacker AS filter applied: {attacker_as} (checking entire path)")

    # Apply all conditions
    if conditions:
        final_condition = conditions[0]
        for condition in conditions[1:]:
            final_condition = final_condition & condition

        filtered_df = df[final_condition].copy()

        # Add AS path analysis information for reference
        if not filtered_df.empty:
            filtered_df['as_path_parsed'] = filtered_df['as-path'].apply(parse_as_path)
            filtered_df['path_length'] = filtered_df['as_path_parsed'].apply(len)

        print(f"Found {len(filtered_df)} BGP updates matching criteria")
        return filtered_df

    return df

def process_single_event(event_row, collector, collectors2url, bgpd, output_dir, num_workers=1):
    """Process a single security event."""
    try:
        event_time = parse_event_time(event_row['Time'])
        victim_as = event_row.get('Victim', '')
        attacker_as = event_row.get('Attacker', '')
        prefix = event_row.get('Prefix', '')
        category = event_row.get('Category', '')

        # Calculate time window (±12 hours)
        start_time = event_time - timedelta(hours=12)
        end_time = event_time + timedelta(hours=12)

        print(f"\nProcessing event: {event_time}")
        print(f"Victim AS: {victim_as}, Attacker AS: {attacker_as}, Prefix: {prefix}")
        print(f"Time window: {start_time} to {end_time}")

        # Get archive list for this time window
        data_urls = get_archive_list(collector, collectors2url, start_time, end_time)
        if not data_urls:
            print(f"No BGP update files found for time window")
            return

        print(f"Found {len(data_urls)} BGP update files")

        # Download files
        def download_job(url):
            return download_data(url, collector)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            file_paths = list(executor.map(download_job, data_urls))

        # Filter out failed downloads
        file_paths = [fp for fp in file_paths if fp is not None]

        if not file_paths:
            print(f"Failed to download any BGP update files")
            return

        # Process each file and collect matching updates
        all_matching_updates = []

        for fpath in file_paths:
            print(f"Processing file: {fpath.name}")
            df = load_updates_to_df(fpath, bgpd)

            if not df.empty:
                # Filter updates based on event criteria
                filtered_df = filter_bgp_updates(df, victim_as, attacker_as, prefix)

                if not filtered_df.empty:
                    # Add event metadata
                    filtered_df['event_time'] = event_time.strftime("%Y-%m-%d %H:%M:%S")
                    filtered_df['event_victim'] = victim_as
                    filtered_df['event_attacker'] = attacker_as
                    filtered_df['event_prefix'] = prefix
                    filtered_df['event_category'] = category
                    filtered_df['file_source'] = fpath.name

                    all_matching_updates.append(filtered_df)
                    print(f"Found {len(filtered_df)} matching updates in {fpath.name}")

        # Save results if any matches found
        if all_matching_updates:
            combined_df = pd.concat(all_matching_updates, ignore_index=True)

            # Create output filename based on event details
            event_time_str = event_time.strftime("%Y%m%d_%H%M%S")
            safe_prefix = prefix.replace('/', '_').replace('.', '_') if prefix else 'no_prefix'
            output_filename = f"event_{event_time_str}_{victim_as}_{attacker_as}_{safe_prefix}.csv"
            output_path = Path(output_dir) / output_filename

            combined_df.to_csv(output_path, index=False)
            print(f"Saved {len(combined_df)} matching BGP updates to {output_path}")
        else:
            print(f"No matching BGP updates found for this event")

    except Exception as e:
        print(f"Error processing event: {e}")

@click.command()
@click.option("--events", type=str, required=True, help="CSV file containing security events")
@click.option("--collector", type=str, default="route-views4", help="RouteViews collector name")
@click.option("--output", type=str, default="bgp_results/", help="Output directory for results")
@click.option("--num-workers", type=int, default=4, help="Number of download workers")
@click.option("--bgpd", type=str, default=None, help="Path to bgpd binary")
def main(events, collector, output, num_workers, bgpd):
    """Process security events and collect related BGP updates."""

    # Validate inputs
    events_path = Path(events)
    if not events_path.exists():
        print(f"Error: Events file not found: {events}")
        sys.exit(1)

    # Set bgpd path
    if bgpd:
        bgpd_path = Path(bgpd)
    else:
        bgpd_path = SCRIPT_DIR / "bgpd"

    if not bgpd_path.exists():
        print(f"Error: bgpd binary not found: {bgpd_path}")
        print("Please ensure bgpd is available or specify path with --bgpd")
        sys.exit(1)

    # Create output directory
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load events CSV
    try:
        events_df = pd.read_csv(events_path)
        print(f"Loaded {len(events_df)} events from {events}")
    except Exception as e:
        print(f"Error loading events CSV: {e}")
        sys.exit(1)

    # Validate CSV format
    required_columns = ['Time', 'Victim', 'Attacker', 'Prefix', 'Category']
    missing_columns = [col for col in required_columns if col not in events_df.columns]
    if missing_columns:
        print(f"Error: Missing required columns in events CSV: {missing_columns}")
        print(f"Available columns: {list(events_df.columns)}")
        sys.exit(1)

    # Get collectors
    print("Fetching RouteViews collectors...")
    collectors2url = get_all_collectors()

    if collector not in collectors2url:
        print(f"Error: Collector '{collector}' not found")
        print(f"Available collectors: {list(collectors2url.keys())}")
        sys.exit(1)

    print(f"Using collector: {collector}")
    print(f"Collector URL: {collectors2url[collector]}")

    # Process each event
    for idx, event_row in events_df.iterrows():
        print(f"\n{'='*60}")
        print(f"Processing event {idx+1}/{len(events_df)}")
        print(f"{'='*60}")

        process_single_event(
            event_row,
            collector,
            collectors2url,
            bgpd_path,
            output_dir,
            num_workers
        )

    print(f"\n{'='*60}")
    print("All events processed!")
    print(f"Results saved in: {output_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()