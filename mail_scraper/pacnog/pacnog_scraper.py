import requests
import csv
from datetime import datetime
import os
# For PACNOG
# base_url = "https://orbit.apnic.net/api/v1/mailing-list/pacnog@pacnog.org/recent-threads?limit=5"
# For INNOG
base_url = "https://orbit.apnic.net/api/v1/mailing-list/innog@innog.net/recent-threads?limit=5"
headers = {
    'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0'
}

output_file = "../data/pacnog/pacnog_email.csv"
output_dir = os.path.dirname(output_file)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def fetch_and_save_to_csv():
    offset = 0
    max_offset = 410  #  max offset (https://orbit.apnic.net/api/v1/mailing-list/pacnog@pacnog.org/recent-threads?limit=5&offset=2234)
    step = 5
    email_id = 1

    file_exists = os.path.exists(output_file)

    with open(output_file, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        if not file_exists:
            writer.writerow(['Email_ID', 'Thread_ID', 'Level', 'Date', 'Email_Title', 'Author', 'Email_Address', 'Email_Content'])

        while offset <= max_offset:
            url = f"{base_url}&offset={offset}" if offset > 0 else base_url
            print(f"Requesting: {url}")

            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])

                for result in results:
                    starting_email = result.get("starting_email", {})
                    date_active = result.get("date_active", "")
                    subject = starting_email.get("subject", "")
                    sender = starting_email.get("sender", {})
                    author = sender.get("name", "")
                    email_address = sender.get("name", "")
                    content = starting_email.get("content", "")

                    formatted_date = format_date(date_active)

                    writer.writerow([email_id, "NULL", "NULL", formatted_date, subject, author, email_address, content])
                    email_id += 1

                print(f"Offset {offset} processing completed...")
            else:
                print(f"Request Failed: Offset {offset} ... {response.status_code}")
                break

            offset += step

    print(f"All data has been saved to {output_file}")

def format_date(date_str):
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        print(f"Error: {e}")
        return date_str

fetch_and_save_to_csv()