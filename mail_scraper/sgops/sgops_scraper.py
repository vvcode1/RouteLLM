import os
import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin
import numpy as np
import pandas as pd
from datetime import datetime
import time
import logging
import locale
import cloudscraper
from dateutil import parser

logger = logging.getLogger(__name__)

# nanog_url = "https://mailman.nanog.org/pipermail/nanog/"
# afnog_url = "https://afnog.org/pipermail/afnog/"
ausnog_url = "https://list.sgnog.net/pipermail/sgops/"
# sanog_url = "https://lists.sanog.org/pipermail/sanog/"
# lacnog_url = "https://mail.lacnic.net/pipermail/lacnog/"


session = requests.Session()

# headers = {
#     'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
# }

headers = {
    'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0'
}

# Send a GET request to the URL
# response = requests.get(nanog_url, headers=headers)

# Use cloudscraper to avoid Clsoudflare anti-crawl
scraper = cloudscraper.create_scraper()
response = scraper.get(ausnog_url)
# print(response.text)

soup = BeautifulSoup(response.content, 'html.parser')

links = soup.find_all('a', href=True)
# year_month = soup.find_all('td')

full_thread_links = [link['href'] for link in links if 'thread.html' in link['href']]
# element = '2000-January/thread.html'

# Define the start and end points
start_period = '2014-January/thread.html'
end_period = '2025-July/thread.html'

current_folder_index = 0

def parse_period(link):
    try:
        period = link.split('/')[0]  # Extract 'YYYY-Month'
        return datetime.strptime(period, '%Y-%B')
    except ValueError:
        return None

start_date = parse_period(start_period)
end_date = parse_period(end_period)

if not start_date or not end_date:
    raise ValueError("Invalid start or end period format.")

# Extract valid links within the date range
thread_links = [
    link for link in full_thread_links
    if start_date <= parse_period(link) <= end_date
]

print(f"Extracted thread links from {start_period} to {end_period}:")
print(thread_links)

previous_content = None

email_store_path = '../data/sgops/sgops_email.csv'
df = pd.DataFrame(columns=['Email_ID', 'Thread_ID', 'Level', 'Date', 'Email_Title', 'Author', 'Email_Address', 'Email_Content'])

# df = pd.read_csv(email_store_path)
# access thread.html
for link in thread_links:
    thread_url = urljoin(ausnog_url, link)
    logger.info(f"Accessing: {thread_url}")

    # thread_response = requests.get(thread_url, headers=headers)
    thread_response = scraper.get(thread_url)

    # print(thread_response.text)
    thread_soup = BeautifulSoup(thread_response.content, 'html.parser')

    # Find all thread links in the page
    links = thread_soup.find_all('a', href=True)

    thread_links = [link for link in links if link['href'].endswith('.html')]

    # Track the previous content
    previous_title = None

    # if df.empty:
    #     thread_id = 0
    #     level = 0
    # else:
    #     thread_id = df.iloc[-1]['Thread_ID']
    #     level = df.iloc[-1]['Level']

    # print(f"Thread ID: {thread_id}, Level: {level}")
    # Access each email page
    for each_link in thread_links:
        # time.sleep(1)
        full_url = urljoin(thread_url, each_link['href'])
        logger.info(f"Accessing: {full_url}")
        print(f"Accessing: {full_url}")

        email_id = int(each_link['href'].split('.', 1)[0])
        #if email_id in df['Email_ID']:
        if np.any(df['Email_ID'] == email_id):
            print('The email was already scrapsed!')
            logger.info('The email was already scrapsed!')
            continue

        try:
            # full_response = requests.get(full_url, headers=headers)
            full_response = scraper.get(full_url)
        except requests.exceptions.RequestException as e:
            print(e)
            continue

        full_soup = BeautifulSoup(full_response.content, 'html.parser')

        # Extract the content of the thread title
        title = each_link.text

        # Check if content is the same as the previous one
        if previous_title and title == previous_title:
            logger.info(f"Content is the same as previous page.")
            level += 1
        else:
            logger.info(f"Content is different or this is the first page.")
            previous_title = title  # Update the previous content
            current_folder_index += 1
            level = 0

        # if not os.path.exists(str(thread_id)):
        #     os.makedirs(str(thread_id))
        # with open(str(thread_id)+"/"+f"{each_link['href']}", "w", encoding="utf-8") as file:
        #     file.write(full_response.text)

        base_dir = "../data/sgops"
        os.makedirs(base_dir, exist_ok=True)
        data_dir = os.path.join(base_dir, "raw_html")
        os.makedirs(data_dir, exist_ok=True)

        thread_dir = os.path.join(data_dir, str(current_folder_index))
        if not os.path.exists(thread_dir):
            os.makedirs(thread_dir)

        file_path = os.path.join(thread_dir, f"{each_link['href']}")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(f"<!-- URL: {full_url} -->\n")
            file.write(full_response.text)
        try:
            page_title = full_soup.find('h1').text
        except AttributeError as e:
            logger.info(f"An error occurred: {e}")
            page_title = None

        # try:
        #     datetime_obj = datetime.strptime(full_soup.find('i').text, "%a %b %d %H:%M:%S %Z %Y")
        #     timestamp = datetime_obj.strftime("%Y-%m-%d %H:%M:%S")
        # except AttributeError as e:
        #     logger.info(f"An error occurred: {e}")
        #     timestamp = None

        try:
            time_text = full_soup.find('i').text.strip()
            tz_mapping = {
                "AEDT": "+1100",
                "AEST": "+1000",
            }
            for tz_name, tz_offset in tz_mapping.items():
                if tz_name in time_text:
                    time_text = time_text.replace(tz_name, tz_offset)
                    break
            datetime_obj = datetime.strptime(time_text, "%a %b %d %H:%M:%S %z %Y")
            timestamp = datetime_obj.strftime("%Y-%m-%d %H:%M:%S")
        except AttributeError as e:
            logger.info(f"An error occurred: {e}")
            timestamp = None
        except ValueError as e:
            logger.info(f"Date parsing failed: {e}")
            timestamp = None
        # try: # For LACNOG

        #     time_text = full_soup.find('i').text.strip()
        #     days_mapping = {"Lun": "Mon", "Mar": "Tue", "Mie": "Wed", "Jue": "Thu", "Vie": "Fri", "Sab": "Sat", "Dom": "Sun"}
        #     months_mapping = {"Ene": "Jan", "Feb": "Feb", "Mar": "Mar", "Abr": "Apr", "Mayo": "May", "Jun": "Jun",
        #                     "Jul": "Jul", "Ago": "Aug", "Sep": "Sep", "Oct": "Oct", "Nov": "Nov", "Dic": "Dec"}

        #     for spanish, english in days_mapping.items():
        #         time_text = time_text.replace(spanish, english)

        #     for spanish, english in months_mapping.items():
        #         time_text = time_text.replace(spanish, english)

        #     timezone_mapping = {
        #         "BRT": "-0300",  # Brasília Time
        #         "BRST": "-0200",  # Brasília Summer Time
        #     }

        #     for tz_name, tz_offset in timezone_mapping.items():
        #         if tz_name in time_text:
        #             time_text = time_text.replace(tz_name, tz_offset)
        #             break

        #     if " -03" in time_text:
        #         time_text = time_text.replace(" -03", " -0300")

        #     time_format = "%a %b %d %H:%M:%S %z %Y"

        #     parsed_date = datetime.strptime(time_text, time_format)

        #     output_format = "%Y-%m-%d %H:%M:%S"

        #     timestamp = parsed_date.strftime(output_format)
        # except Exception:
        #     timestamp = None

        try:
            author = full_soup.find('b').text
        except AttributeError as e:
            logger.info(f"An error occurred: {e}")
            author = None

        try:
            original_email_content = full_soup.find('pre').text
            email_content = original_email_content.split('-------------- next part --------------', 1)[0]
        except AttributeError as e:
            logger.info(f"An error occurred: {e}")
            email_content = None

        try:
            email_addr = full_soup.find('a').text
        except AttributeError as e:
            logger.info(f"An error occurred: {e}")
            email_addr = None

        new_data = {
            'Email_ID': email_id,
            'Thread_ID': str(current_folder_index),
            'Level': level,
            'Date': timestamp,
            'Email_Title': page_title,
            'Author': author,
            'Email_Address': email_addr,
            'Email_Content': email_content,
            'URL': full_url
        }
        new_df = pd.DataFrame([new_data])
        new_df.to_csv(email_store_path, mode='a', header=False, index=False)
logger.info(len(df))
print(len(df))

print("All thread.html pages have been accessed.")