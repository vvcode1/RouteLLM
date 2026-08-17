"""
NZNOG MBOX Email Parser

This script parses NZNOG (New Zealand Network Operators' Group) mailing list MBOX format
email archives and converts them to CSV format. MBOX files contain email messages in a
standard Unix mailbox format.

Prerequisites:
1. Download the NZNOG MBOX file from the mailing list archive
2. Place the MBOX file at the specified path: ../data/nznog/nznog@lists.nznog.org.mbox
3. Run this script to convert emails to CSV format

Usage:
    python nznog_mbox_parser.py

The script will create a CSV file with the following columns:
- Email_ID: Sequential identifier
- Date: Email timestamp in YYYY-MM-DD HH:MM:SS format
- Email_Title: Subject line
- Author: Sender name
- Email_Address: Sender email address
- Email_Content: Message body content
"""

import os
import re
import csv
from email.parser import BytesParser
from datetime import datetime
from email import policy

# Define NZNOG MBOX file path
mbox_file_path = "../data/nznog/nznog@lists.nznog.org.mbox"
# Define output CSV path
output_csv_path = "../data/nznog/nznog_email.csv"

# Ensure output directory exists
base_dir = "../data/nznog"
os.makedirs(base_dir, exist_ok=True)

def extract_email_content(msg):
    """
    Extract plain text content from email message.
    Handles both multipart and single-part messages.

    Args:
        msg: Parsed email message object

    Returns:
        String containing the email content
    """
    content = ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                content += part.get_payload(decode=True).decode(part.get_content_charset('utf-8'), errors="ignore")
    else:
        content = msg.get_payload(decode=True).decode(msg.get_content_charset('utf-8'), errors="ignore")
    return content.strip()

def parse_mbox_to_csv(mbox_path, output_path):
    """
    Parse NZNOG MBOX file and convert to CSV format.

    Args:
        mbox_path: Path to the input MBOX file
        output_path: Path for the output CSV file
    """
    # Check if MBOX file exists
    if not os.path.exists(mbox_path):
        print(f"Error: NZNOG MBOX file not found at {os.path.abspath(mbox_path)}")
        print("Please download the NZNOG MBOX file and place it at the specified path.")
        print("Instructions:")
        print("1. Visit the NZNOG mailing list archives")
        print("2. Download the MBOX archive file")
        print("3. Save it as 'nznog@lists.nznog.org.mbox' in the '../data/nznog/' directory")
        print("4. Run this script again")
        return

    email_id_counter = 1
    processed_emails = 0

    with open(mbox_path, "rb") as mbox_file, open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        # Write CSV header
        writer.writerow(["Email_ID", "Date", "Email_Title", "Author", "Email_Address", "Email_Content"])

        current_email_data = b""

        print(f"Processing NZNOG MBOX file: {os.path.abspath(mbox_path)}")

        for line in mbox_file:
            # MBOX format: each email starts with "From " line
            if line.startswith(b"From "):
                if current_email_data:
                    if process_email(writer, current_email_data, email_id_counter):
                        processed_emails += 1
                    email_id_counter += 1
                    current_email_data = b""
                current_email_data += line
            else:
                current_email_data += line

        # Process the last email
        if current_email_data:
            if process_email(writer, current_email_data, email_id_counter):
                processed_emails += 1

    print(f"Parsing complete!")
    print(f"Total emails processed: {processed_emails}")
    print(f"CSV saved to: {os.path.abspath(output_path)}")

def process_email(writer, email_data, email_id_counter):
    """
    Process individual email and write to CSV.

    Args:
        writer: CSV writer object
        email_data: Raw email data bytes
        email_id_counter: Sequential email ID

    Returns:
        Boolean indicating if email was successfully processed
    """
    try:
        # Parse email using BytesParser
        msg = BytesParser(policy=policy.default).parsebytes(email_data)

        # Extract and format date
        email_date = msg.get("Date", "").strip()
        try:
            parsed_date = datetime.strptime(email_date, "%a, %d %b %Y %H:%M:%S %z")
            formatted_date = parsed_date.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            # If date parsing fails, keep original format
            formatted_date = email_date

        # Extract subject
        subject = msg.get("Subject", "").strip()

        # Parse From field to extract author name and email address
        from_field = msg.get("From", "")
        email_match = re.match(r'(.*)<(.*)>', from_field)
        if email_match:
            author = email_match.group(1).strip()
            email_address = email_match.group(2).strip()
        else:
            author = from_field.strip()
            email_address = ""

        # Extract email content
        content = extract_email_content(msg)

        # Write to CSV
        writer.writerow([email_id_counter, formatted_date, subject, author, email_address, content])

        # Print progress every 500 emails
        if email_id_counter % 500 == 0:
            print(f"Processed {email_id_counter} emails...")

        return True

    except Exception as e:
        print(f"Error processing email {email_id_counter}: {e}")
        return False

if __name__ == "__main__":
    print("NZNOG MBOX Email Parser")
    print("=" * 50)
    print("This script converts NZNOG mailing list MBOX archives to CSV.")
    print()

    parse_mbox_to_csv(mbox_file_path, output_csv_path)