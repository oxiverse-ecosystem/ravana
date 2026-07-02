#!/usr/bin/env python3
"""Expand teen_seeds.txt with real web content via IntentForge gateway."""
import sys, os, re, json, time, urllib.request
from collections import Counter

_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
corpus_path = os.path.join(_proj_root, "data", "corpora", "teen_seeds.txt")

TOPICS = [
    "science technology", "nature environment", "art music", "space astronomy",
    "history culture", "health medicine", "ocean marine life", "animals wildlife",
    "sports fitness", "food cooking", "travel geography", "psychology mind",
    "physics mathematics", "computer programming", "philosophy ethics",
    "climate weather", "energy renewable", "education learning",
    "invention discovery", "mythology folklore",
]

HEADERS = {'User-Agent': 'Mozilla/5.0 RAVANA/1.0'}

def fetch_search_results(topic, max_articles=5):
    url = f"http://localhost:4000/search?q={urllib.parse.quote(topic)}"
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        resp = urllib.request.urlopen(req, timeout=15)
        data = json.loads(resp.read())
        results = data.get('results', [])
        return [r['url'] for r in results[:max_articles] if r.get('url')]
    except Exception as e:
        print(f"  Search failed for '{topic}': {e}")
        return []

def fetch_article_text(url):
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        resp = urllib.request.urlopen(req, timeout=10)
        html = resp.read().decode('utf-8', errors='replace')
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
        for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            tag.decompose()
        main = soup.find('article') or soup.find('main') or soup.find('body')
        text = main.get_text(separator=' ', strip=True) if main else soup.get_text(separator=' ', strip=True)
        return re.sub(r'\s+', ' ', text)
    except Exception:
        return ""

def extract_sentences(text):
    sentences = []
    for s in re.split(r'[.!?]+', text):
        s = s.strip()
        if len(s) < 20 or len(s) > 300:
            continue
        words = s.split()
        if len(words) < 5 or len(words) > 50:
            continue
        if any(w in s.lower() for w in ['cookie', 'privacy', 'sign up', 'subscribe', 'click here']):
            continue
        s_clean = s[0].upper() + s[1:] if s else s
        sentences.append(s_clean)
    return sentences

def main():
    print("=" * 60)
    print("Expanding teen_seeds.txt with web content")
    print("=" * 60)

    existing = set()
    if os.path.exists(corpus_path):
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    existing.add(line.lower())
        print(f"Existing corpus: {len(existing)} sentences")

    new_sentences = []
    seen_lower = existing.copy()

    for topic in TOPICS:
        print(f"\n[{topic}] Searching...", end=" ", flush=True)
        urls = fetch_search_results(topic)
        if not urls:
            print("no results")
            continue
        print(f"{len(urls)} articles")
        for url in urls:
            text = fetch_article_text(url)
            if not text:
                continue
            sents = extract_sentences(text)
            for s in sents:
                if s.lower() not in seen_lower:
                    new_sentences.append(s)
                    seen_lower.add(s.lower())
            print(f"  {len(sents)} sentences from {url.split('/')[:3]}")
            time.sleep(0.5)

    if not new_sentences:
        print("\nNo new sentences found.")
        return

    print(f"\n{'='*60}")
    print(f"Adding {len(new_sentences)} new sentences to corpus")
    print(f"{'='*60}")

    with open(corpus_path, "a", encoding="utf-8") as f:
        for s in new_sentences:
            f.write(s + "\n")

    total = len(existing) + len(new_sentences)
    print(f"Corpus now has {total} sentences (was {len(existing)})")

if __name__ == "__main__":
    import urllib.parse
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        print("BeautifulSoup required. Install: pip install beautifulsoup4")
        sys.exit(1)
    main()
