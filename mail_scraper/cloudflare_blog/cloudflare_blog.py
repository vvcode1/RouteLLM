# import cloudscraper
# scraper = cloudscraper.create_scraper()
# resp = scraper.get("https://blog.cloudflare.com/celebrate-micro-small-and-medium-sized-enterprises-day-with-cloudflare")
# print(resp.text)

import cloudscraper
import re
import html
import json
from bs4 import BeautifulSoup
from dateutil import parser
import pytz
import csv
import time

def parse_blog(url):
    scraper = cloudscraper.create_scraper()
    resp = scraper.get(url)
    html_text = resp.text
    soup = BeautifulSoup(html_text, "lxml")

    # 1. Extract title
    title = ""
    meta_title = soup.find("meta", attrs={"name": "title"})
    if meta_title:
        title = meta_title.get("content", "").strip()
    if not title:
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.text.strip()

    # 2. Extract author(s)
    authors = []
    meta_author = soup.find("meta", attrs={"name": "twitter:data1"})
    if meta_author:
        authors = [meta_author.get("content", "")]
    authors = ", ".join(authors)

    # 3. Extract publication time
    pub_time = ""
    meta_time = soup.find("meta", attrs={"property": "article:published_time"})
    if meta_time:
        time_str = meta_time.get("content", "")
        try:
            dt = parser.parse(time_str)
            dt_utc = dt.astimezone(pytz.utc)
            pub_time = dt_utc.strftime("%Y-%m-%d %H:%M:%S")
        except Exception as e:
            print("Time parsing error:", e)

    # 4. Extract main content
    pattern = r'<astro-island[^>]+component-export="PostContent"[^>]+props="([^"]+)"'
    match = re.search(pattern, html_text)
    content = ""
    if match:
        props_html = match.group(1)
        props_json = html.unescape(props_html)
        try:
            props = json.loads(props_json)
            if "html" in props:
                html_body = props["html"]
                soup2 = BeautifulSoup(html_body, "lxml")
                pieces = []
                for tag in soup2.find_all(['h1','h2','h3','h4','h5','h6','p','li']):
                    txt = tag.get_text(" ", strip=True)
                    if txt:
                        pieces.append(txt)
                content = " ".join(pieces)
            elif "content" in props:
                soup2 = BeautifulSoup(props["content"], "lxml")
                content = soup2.get_text(separator=" ")
        except Exception as e:
            print("Content parsing failed:", e)

    # Fallback content extraction method
    if not content:
        content_div = soup.find("div", class_="post-content")
        if content_div:
            content = content_div.get_text(separator=" ").strip()

    # Clean up content formatting
    content = re.sub(r"\s+", " ", content)
    content = content.replace("Â", "").strip()

    # Extract Blog ID from URL
    blog_id = ""
    # Usually the last part of the URL
    blog_id = url.rstrip('/').split('/')[-1]

    return {
        "Blog_ID": blog_id,
        "Date": pub_time,
        "Blog_Title": title,
        "Author": authors,
        "Blog_Content": content,
        "Blog_Url": url
    }

def get_all_blog_links(page_url):
    # Scrape all blog links from one page
    scraper = cloudscraper.create_scraper()
    resp = scraper.get(page_url)
    html_text = resp.text

    # Use regex to directly find props
    pattern = r'<astro-island[^>]+component-export="PostCard"[^>]+props="([^"]+)"'
    matches = re.findall(pattern, html_text)
    links = []

    for m in matches:
        try:
            m_json = html.unescape(m)
            data = json.loads(m_json)
            post = data['post'][1]
            slug = post['slug'][1]
            links.append(f"https://blog.cloudflare.com/{slug}")
        except Exception as e:
            print("Link parsing error:", e)

    return links

def save_to_csv(blog_data_list, filename="cloudflare_blogs.csv"):
    """
    Save blog data to CSV file.

    Args:
        blog_data_list: List of blog data dictionaries
        filename: Output CSV filename
    """
    keys = ["Blog_ID", "Date", "Blog_Title", "Author", "Blog_Content", "Blog_Url"]
    with open(filename, "w", newline='', encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for idx, row in enumerate(blog_data_list, start=1):
            row['Blog_ID'] = idx  # Auto-increment ID
            writer.writerow(row)

def main():
    total_pages = 162
    all_blog_links = []

    print("Starting to collect all blog post links...")
    for page_num in range(1, total_pages+1):
        page_url = f"https://blog.cloudflare.com/page/{page_num}/"
        print(f"Collecting page {page_num}: {page_url}")
        links = get_all_blog_links(page_url)
        all_blog_links.extend(links)
        time.sleep(1)  # Rate limiting

    print(f"Found {len(all_blog_links)} articles total. Starting content extraction...")
    all_blogs = []

    for idx, url in enumerate(all_blog_links):
        print(f"[{idx+1}/{len(all_blog_links)}] Parsing {url}")
        try:
            info = parse_blog(url)
            # ID will be set automatically when writing to CSV
            all_blogs.append(info)
            time.sleep(1)  # Rate limiting
        except Exception as e:
            print("Parsing failed:", e)
            continue

    print(f"Saving to CSV, total {len(all_blogs)} articles...")
    save_to_csv(all_blogs, filename="cloudflare_blogs.csv")
    print("Processing complete!")

if __name__ == "__main__":
    main()