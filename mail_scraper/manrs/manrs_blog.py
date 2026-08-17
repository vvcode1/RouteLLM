import requests
import re
import csv
import time
import html
from bs4 import BeautifulSoup
from dateutil import parser
import pytz

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
}

def get_all_blog_links(page_url):
    """Get all MANRS blog URLs from a single page"""
    resp = requests.get(page_url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")
    links = []
    for article in soup.find_all("article", class_="site-card"):
        a_tag = article.find("a", class_="card-link")
        if a_tag and a_tag['href']:
            links.append(a_tag['href'])
    return links

def parse_blog(url):
    """Parse detailed content from a single blog post"""
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    html_text = resp.text
    soup = BeautifulSoup(html_text, "lxml")

    # 1. Extract title
    title = ""
    h1 = soup.find("h1", class_="entry-title")
    if h1:
        title = h1.get_text(strip=True)
    else:
        meta_title = soup.find("meta", property="og:title")
        if meta_title:
            title = meta_title.get("content", "").strip()

    # 2. Extract author
    author = ""
    post_meta = soup.find("ul", class_="post-meta")
    if post_meta:
        author_li = post_meta.find("li", class_="post-author")
        if author_li:
            author = author_li.get_text(strip=True).replace("By ", "")
    if not author:
        meta_author = soup.find("meta", attrs={"name": "author"})
        if meta_author:
            author = meta_author.get("content", "").strip()

    # 3. Extract publication time from meta tags
    pub_time = ""
    meta_time = soup.find("meta", property="article:published_time")
    if meta_time:
        time_str = meta_time.get("content", "")
        try:
            dt = parser.parse(time_str)
            dt_utc = dt.astimezone(pytz.utc)
            pub_time = dt_utc.strftime("%Y-%m-%d %H:%M:%S")
        except Exception as e:
            print("Time parsing error:", e)

    # 4. Extract main content
    content = ""
    content_div = soup.find("div", class_="entry-content")
    if content_div:
        pieces = []
        for tag in content_div.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li']):
            txt = tag.get_text(" ", strip=True)
            if txt:
                pieces.append(txt)
        content = " ".join(pieces)

    # Clean up content formatting
    content = re.sub(r"\s+", " ", content)
    content = content.strip()

    # Extract Blog ID from URL
    blog_id = url.rstrip('/').split('/')[-1]

    return {
        "Blog_ID": blog_id,
        "Date": pub_time,
        "Blog_Title": title,
        "Author": author,
        "Blog_Content": content,
        "Blog_Url": url
    }

def save_to_csv(blog_data_list, filename="manrs_blogs.csv", write_header=False):
    """
    Save blog data to CSV file.

    Args:
        blog_data_list: List of blog data dictionaries
        filename: Output CSV filename
        write_header: Whether to write CSV header
    """
    keys = ["Blog_ID", "Date", "Blog_Title", "Author", "Blog_Content", "Blog_Url"]
    mode = "a" if not write_header else "w"
    with open(filename, mode, newline='', encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        if write_header:
            writer.writeheader()
        for row in blog_data_list:
            writer.writerow(row)

def main():
    """
    Main function to scrape MANRS blog posts.
    Collects all blog links first, then scrapes content with buffered writing.
    """
    total_pages = 23
    all_blog_links = []

    print("Starting to collect all blog post links...")
    for page_num in range(1, total_pages+1):
        page_url = f"https://manrs.org/blog/page/{page_num}/"
        print(f"Collecting page {page_num}: {page_url}")
        links = get_all_blog_links(page_url)
        all_blog_links.extend(links)
        time.sleep(1)  # Rate limiting

    print(f"Found {len(all_blog_links)} articles total. Starting content extraction...")

    # Use buffered writing for better performance
    buffer = []
    csv_file = "manrs_blogs.csv"
    blog_id_counter = 1

    # Initialize CSV file with header
    save_to_csv([], filename=csv_file, write_header=True)

    for idx, url in enumerate(all_blog_links):
        print(f"[{idx+1}/{len(all_blog_links)}] Parsing {url}")
        try:
            info = parse_blog(url)
            info["Blog_ID"] = blog_id_counter
            blog_id_counter += 1
            buffer.append(info)

            # Write buffer to file every 10 articles
            if len(buffer) >= 10:
                save_to_csv(buffer, filename=csv_file, write_header=False)
                print(f"Written {blog_id_counter-1} entries")
                buffer.clear()

            time.sleep(1)  # Rate limiting
        except Exception as e:
            print("Parsing failed:", e)
            continue

    # Write remaining buffer content
    if buffer:
        save_to_csv(buffer, filename=csv_file, write_header=False)
        print(f"Written {blog_id_counter-1} entries (all complete)")

    print("Processing complete!")

if __name__ == "__main__":
    main()
