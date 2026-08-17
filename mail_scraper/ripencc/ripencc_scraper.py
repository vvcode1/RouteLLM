import os
import re
import csv
from email.parser import BytesParser
from datetime import datetime
from email import policy

mbox_file_path = "../data/ripencc/routing-wg@ripe.net.mbox"

output_csv_path = "../data/ripencc/ripe_ncc_email.csv"

base_dir = "../data/ripencc"
os.makedirs(base_dir, exist_ok=True)


def extract_email_content(msg):
    content = ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                content += part.get_payload(decode=True).decode(part.get_content_charset('utf-8'), errors="ignore")
    else:
        content = msg.get_payload(decode=True).decode(msg.get_content_charset('utf-8'), errors="ignore")
    return content.strip()


def parse_mbox_to_csv(mbox_path, output_path):
    email_id_counter = 1
    with open(mbox_path, "rb") as mbox_file, open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Email_ID", "Date", "Email_Title", "Author", "Email_Address", "Email_Content"])

        current_email_data = b""
        for line in mbox_file:
            if line.startswith(b"From "):
                if current_email_data:
                    process_email(writer, current_email_data, email_id_counter)
                    email_id_counter += 1
                    current_email_data = b""
                current_email_data += line
            else:
                current_email_data += line

        if current_email_data:
            process_email(writer, current_email_data, email_id_counter)

    print(f"Parsing complete, CSV saved to {os.path.abspath(output_path)}")



def process_email(writer, email_data, email_id_counter):
    try:
        msg = BytesParser(policy=policy.default).parsebytes(email_data)

        email_date = msg.get("Date", "").strip()
        try:
            parsed_date = datetime.strptime(email_date, "%a, %d %b %Y %H:%M:%S %z")
            formatted_date = parsed_date.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            formatted_date = email_date

        subject = msg.get("Subject", "").strip()
        from_field = msg.get("From", "")
        email_match = re.match(r'(.*)<(.*)>', from_field)

        if email_match:
            author = email_match.group(1).strip()
            email_address = email_match.group(2).strip()
        else:
            author = from_field.strip()
            email_address = ""

        content = extract_email_content(msg)

        writer.writerow([email_id_counter, formatted_date, subject, author, email_address, content])

    except Exception as e:
        print(f"Skipping an email. Error: {e}")


if __name__ == "__main__":
    parse_mbox_to_csv(mbox_file_path, output_csv_path)
