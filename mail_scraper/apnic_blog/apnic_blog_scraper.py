import cloudscraper
import csv
import os
from bs4 import BeautifulSoup
from dateutil import parser as dtparser
import pytz
import time

def parse_blog(url):
    scraper = cloudscraper.create_scraper()
    resp = scraper.get(url)
    html_text = resp.text
    soup = BeautifulSoup(html_text, "lxml")

    # Extract title
    title_tag = soup.find("h1", class_="entry-title")
    title = title_tag.text.strip() if title_tag else ""

    # Extract author
    author = ""
    p_meta = soup.find("p", class_="meta-author-and-date")
    if p_meta:
        author_tag = p_meta.find("a", rel="author")
        if author_tag:
            author = author_tag.text.strip()

    # Extract publication time (using og:updated_time)
    pub_time = ""
    meta_time = soup.find("meta", attrs={"property": "og:updated_time"})
    if meta_time:
        time_str = meta_time.get("content", "")
        try:
            dt = dtparser.parse(time_str)
            dt_utc = dt.astimezone(pytz.UTC)
            pub_time = dt_utc.strftime("%Y-%m-%d %H:%M:%S")
        except Exception as e:
            print(f"Time parsing error: {e} ({time_str})")

    # Extract main content
    content = ""
    content_div = soup.find("div", id="article-content", class_="entry-content")
    if content_div:
        paragraphs = []
        for tag in content_div.find_all(['p', 'li', 'h2', 'h3']):
            txt = tag.get_text(" ", strip=True)
            if txt:
                paragraphs.append(txt)
        content = "\n".join(paragraphs).strip()
    else:
        content = "(Content not found)"

    blog_id = url.rstrip('/').split('/')[-1]
    return {
        "Blog_ID": blog_id,
        "Date": pub_time,
        "Blog_Title": title,
        "Author": author,
        "Blog_Content": content,
        "Blog_Url": url
    }

def get_all_blog_links(page_url):
    scraper = cloudscraper.create_scraper()
    resp = scraper.get(page_url)
    html_text = resp.text
    soup = BeautifulSoup(html_text, "lxml")

    blogs = []
    for article in soup.find_all("article"):
        h3 = article.find("h3", class_="entry-title")
        if not h3:
            continue
        a = h3.find("a")
        if not a:
            continue

        title = a.text.strip()
        url = a['href'].strip()

        p = article.find("p", class_="meta-author-and-date")
        author = ""
        if p:
            author_tag = p.find("a")
            if author_tag:
                author = author_tag.text.strip()

        blogs.append({
            "Blog_Title": title,
            "Blog_Url": url,
            "Author": author,
        })

    return blogs

def write_csv(rows, filename, write_header=False):
    """
    Write blog data to CSV file.

    Args:
        rows: List of dictionaries containing blog data
        filename: Output CSV filename
        write_header: Whether to write CSV header
    """
    keys = ["Blog_ID", "Date", "Blog_Title", "Author", "Blog_Content", "Blog_Url"]
    mode = "w" if write_header else "a"
    with open(filename, mode, newline='', encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)

def main():
    """
    Main function to scrape APNIC blog posts.
    Iterates through specified pages and extracts blog post information.
    """
    total_pages = 2  # Adjust based on actual number of pages
    filename = "apnic_blogs.csv"

    # Remove existing file if it exists to start fresh
    if os.path.exists(filename):
        os.remove(filename)

    first_page = True

    for page_num in range(1, total_pages + 1):
        page_url = f"https://blog.apnic.net/page/{page_num}/"
        print(f"Collecting page {page_num}: {page_url}")

        blog_list = get_all_blog_links(page_url)
        page_blogs = []

        for idx, blog in enumerate(blog_list):
            url = blog["Blog_Url"]
            print(f"  Parsing {url}")

            try:
                info = parse_blog(url)
                page_blogs.append(info)
                time.sleep(1)  # Rate limiting to be respectful to the server
            except Exception as e:
                print(f"Parsing failed: {e}")
                continue

        # Write CSV for each page
        write_csv(page_blogs, filename, write_header=first_page)
        first_page = False  # Only write header for the first page
        print(f"Page {page_num} written: {len(page_blogs)} entries")

if __name__ == "__main__":
    main()
