#!/usr/bin/env python3
"""
Gather Teen Seeds — Fetch natural English conversational sentences from the web
================================================================================
Fetches teen-level conversational English from multiple free sources:
1. Simple Wikipedia API (easy-to-read encyclopedia sentences)
2. Quotable API (inspirational/fun quotes suitable for teens)
3. Gutenberg Poetry / public domain short texts
4. Zen Quotes API (motivational/simple language quotes)

Output: data/corpora/teen_seeds.txt — one sentence per line, cleaned and deduplicated.

Usage:
    python scripts/gather_teen_seeds.py        # Fetch from all sources
    python scripts/gather_teen_seeds.py --force  # Re-fetch even if file exists
"""

import sys
import os
import json
import urllib.request
import urllib.error
import urllib.parse
import re
import time
import random
from typing import List, Set

_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(_proj_root, "data", "corpora", "teen_seeds.txt")

# ─── Rate limiting ───
MIN_DELAY = 0.2  # seconds between API calls (reduced from 0.5 for speed)
TIMEOUT_SEC = 10 # default timeout per request

# ─── Source 1: Simple Wikipedia API ───

SIMPLE_WIKI_CATEGORIES = [
    "Animals", "Food", "Music", "Sports", "Science", "Technology",
    "History", "Geography", "Art", "Literature", "Mathematics", "Nature",
    "Health", "People", "Plants", "Weather", "Space", "Education",
    "Family", "Friends", "School", "Hobbies", "Games", "Movies",
    "Computers", "Internet", "Transport", "Energy", "Environment",
]

SIMPLE_WIKI_ARTICLES = [
    "Dog", "Cat", "Bird", "Fish", "Horse", "Elephant", "Lion", "Tiger",
    "Bear", "Whale", "Dolphin", "Eagle", "Butterfly", "Bee", "Ant",
    "Sun", "Moon", "Earth", "Water", "Fire", "Air", "Tree", "Flower",
    "Rain", "Snow", "Wind", "Cloud", "Ocean", "River", "Mountain",
    "Love", "Friendship", "Family", "Music", "Art", "Book", "School",
    "Teacher", "Student", "Science", "Sports", "Game", "Food", "Health",
    "Computer", "Internet", "Car", "Train", "Plane", "Boat", "Bicycle",
    "House", "City", "Country", "World", "Time", "Number", "Color",
    "Human", "Brain", "Heart", "Eye", "Hand", "Smile", "Laughter",
    "Dream", "Hope", "Courage", "Kindness", "Honesty", "Respect",
    "Friend", "Happy", "Sad", "Anger", "Fear", "Trust", "Belief",
    "Freedom", "Peace", "Nature", "Animal", "Plant", "Music",
    "Language", "Story", "Poem", "Song", "Dance", "Film", "Photo",
    "Building", "Bridge", "Garden", "Park", "Beach", "Forest", "Desert",
    "Island", "Valley", "Star", "Planet", "Rainbow", "Lightning",
    "Thunder", "Ice", "Wind", "Fire", "Mountain", "Lake", "Sea",
]

# ─── Source 2: Hacker News API (no key required) ───
# Fetch top HN stories + their top-level comments for tech discussion data
HN_TOP_STORIES = "https://hacker-news.firebaseio.com/v0/topstories.json"
HN_ITEM = "https://hacker-news.firebaseio.com/v0/item/{}.json"
HN_MAX_STORIES = 5      # how many top stories to fetch (reduced from 8)
HN_MAX_COMMENTS = 3      # how many comments per story (reduced from 5)
HN_FETCH_INTERVAL = 0.15 # seconds between HN API calls (reduced from 0.3)

# ─── Source 3: Open Library API (no key required) ───
# Fetch book descriptions across a range of teen-friendly topics
OPENLIB_SEARCH = "https://openlibrary.org/search.json?q={}&limit=5&fields=key,title,author_name,first_sentence,subject"
OPENLIB_WORKS = "https://openlibrary.org{}.json"
OPENLIB_TOPICS = [
    "adventure", "friendship", "science fiction", "fantasy",
    "mystery", "space exploration", "animals",
    "technology", "nature",
]

# ─── Source 4: WordPress Blog APIs (public REST endpoints) ───
# Popular blogs built on WordPress that expose a public REST API
WORDPRESS_BLOGS = [
    {
        "name": "TED Blog",
        "url": "https://blog.ted.com/wp-json/wp/v2/posts",
        "per_page": 2,
        "max_chars": 400,
    },
    {
        "name": "TechCrunch",
        "url": "https://techcrunch.com/wp-json/wp/v2/posts",
        "per_page": 2,
        "max_chars": 400,
    },
]

# ─── Source 5: ZenQuotes API (free, no key) ───

ZENQUOTES_API = "https://zenquotes.io/api/quotes/"

# ─── Source 6: Random User Generator (generates profile descriptions) ───

RANDOM_USER_API = "https://randomuser.me/api/?nat=us,gb"

# ─── Helpers ───

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                  'RAVANA-TeenSeedGatherer/1.0'
}


def _fetch(url: str, timeout: int = TIMEOUT_SEC) -> str:
    """Fetch a URL with rate limiting and retry."""
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        data = resp.read().decode('utf-8', errors='replace')
        time.sleep(MIN_DELAY)  # rate limit
        return data
    except Exception as e:
        print(f"  [warn] Failed to fetch {url}: {e}")
        time.sleep(MIN_DELAY)
        return ""


def _fetch_hn(url: str, timeout: int = TIMEOUT_SEC) -> str:
    """Fetch Hacker News API with HN-specific rate limiting."""
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        data = resp.read().decode('utf-8', errors='replace')
        time.sleep(HN_FETCH_INTERVAL)
        return data
    except Exception as e:
        print(f"  [warn] HN fetch failed: {e}")
        time.sleep(HN_FETCH_INTERVAL)
        return ""


def _clean_sentence(s: str) -> str:
    """Clean and normalize a sentence."""
    s = s.strip()
    # Remove markdown/HTML artifacts
    s = re.sub(r'<[^>]+>', '', s)
    s = re.sub(r'\[\d+\]', '', s)
    s = re.sub(r'\([^)]*\)', '', s)
    # Remove non-ASCII but keep basic punctuation
    s = re.sub(r'[^\x20-\x7E\s\'\"]', '', s)
    # Normalize whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    # Must start with capital letter
    if not s or not s[0].isupper():
        return ""
    # Must end with sentence-ending punctuation
    if not s.endswith(('.', '!', '?')):
        s += '.'
    # Minimum length
    if len(s.split()) < 3 or len(s.split()) > 30:
        return ""
    # Remove sentences with too many special characters
    special_count = sum(1 for c in s if c in '#@$%^&*_=+[]{}|\\:;"<>/')
    if special_count > 2:
        return ""
    return s


def _split_into_sentences(text: str) -> List[str]:
    """Split text into clean sentences."""
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    cleaned = []
    for sent in sentences:
        s = _clean_sentence(sent)
        if s:
            cleaned.append(s)
    return cleaned


def _strip_html(text: str) -> str:
    """Strip HTML tags and decode common entities from WordPress content."""
    text = re.sub(r'<[^>]+>', '', text)
    text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    text = text.replace('&quot;', '"').replace('&#039;', "'")
    text = text.replace('&nbsp;', ' ').replace('&#8217;', "'")
    text = text.replace('&#8211;', '-').replace('&#8212;', '--')
    text = re.sub(r'&[a-zA-Z]+;', ' ', text)
    return text.strip()


# ─── Source Fetchers ───

def fetch_simple_wikipedia() -> List[str]:
    """Fetch article summaries from Simple Wikipedia API."""
    sentences = []
    articles = SIMPLE_WIKI_ARTICLES[:30]  # Limit to 30 articles per run
    random.shuffle(articles)

    for i, title in enumerate(articles):
        if i > 0 and i % 5 == 0:
            print(f"    ...fetched {i}/{len(articles)} Simple Wikipedia articles", flush=True)

        url = (
            "https://simple.wikipedia.org/w/api.php"
            f"?action=query&prop=extracts&exintro=true&explaintext=true"
            f"&titles={urllib.parse.quote(title)}&format=json"
            "&exsentences=5&redirects=1"
        )
        data = _fetch(url)
        if not data:
            continue

        try:
            parsed = json.loads(data)
            pages = parsed.get('query', {}).get('pages', {})
            for page_id, page in pages.items():
                if page_id == '-1':
                    continue
                extract = page.get('extract', '')
                if extract:
                    sents = _split_into_sentences(extract)
                    sentences.extend(sents)
        except (json.JSONDecodeError, KeyError):
            continue

    print(f"  [Simple Wikipedia] Collected {len(sentences)} sentences", flush=True)
    return sentences


def fetch_hacker_news() -> List[str]:
    """Fetch top Hacker News stories and their top comments."""
    sentences = []
    seen_texts: Set[str] = set()

    # Get top story IDs
    data = _fetch_hn(HN_TOP_STORIES)
    if not data:
        print(f"  [Hacker News] Failed to get top stories list", flush=True)
        return sentences

    try:
        story_ids = json.loads(data)[:HN_MAX_STORIES]
    except (json.JSONDecodeError, TypeError):
        return sentences

    for sid in story_ids:
        data = _fetch_hn(HN_ITEM.format(sid))
        if not data:
            continue
        try:
            story = json.loads(data)
            if story.get('type') != 'story' or story.get('dead') or story.get('deleted'):
                continue

            title = story.get('title', '')
            url = story.get('url', '')

            # Extract meaningful sentences from the title
            if title and title not in seen_texts:
                seen_texts.add(title)
                s = _clean_sentence(title)
                if s:
                    sentences.append(s)

            # Fetch top-level comments
            kids = story.get('kids', [])[:HN_MAX_COMMENTS]
            for cid in kids:
                cdata = _fetch_hn(HN_ITEM.format(cid))
                if not cdata:
                    continue
                try:
                    comment = json.loads(cdata)
                    if comment.get('dead') or comment.get('deleted'):
                        continue
                    text = comment.get('text', '')
                    if text and text not in seen_texts:
                        seen_texts.add(text)
                        # Strip HTML from HN comments
                        clean_text = re.sub(r'<[^>]+>', '', text)
                        # Split into sentences
                        for part in re.split(r'(?<=[.!?])\s+', clean_text):
                            s = _clean_sentence(part)
                            if s and s not in seen_texts:
                                seen_texts.add(s)
                                sentences.append(s)
                except (json.JSONDecodeError, TypeError):
                    continue
        except (json.JSONDecodeError, TypeError):
            continue

    print(f"  [Hacker News] Collected {len(sentences)} sentences", flush=True)
    return sentences


def fetch_open_library() -> List[str]:
    """Fetch book descriptions from Open Library API."""
    sentences = []
    seen_texts: Set[str] = set()

    for topic in OPENLIB_TOPICS:
        url = OPENLIB_SEARCH.format(urllib.parse.quote(topic))
        data = _fetch(url)
        if not data:
            continue

        try:
            parsed = json.loads(data)
            docs = parsed.get('docs', [])
            for doc in docs:
                # Try to get description from the work endpoint
                key = doc.get('key', '')
                if key and key not in seen_texts:
                    seen_texts.add(key)
                    work_url = OPENLIB_WORKS.format(key)
                    work_data = _fetch(work_url)
                    if work_data:
                        try:
                            work = json.loads(work_data)
                            desc = work.get('description', '')
                            if isinstance(desc, dict):
                                desc = desc.get('value', '')
                            if desc and isinstance(desc, str):
                                sents = _split_into_sentences(desc[:500])
                                for s in sents:
                                    if s not in seen_texts:
                                        seen_texts.add(s)
                                        sentences.append(s)
                        except (json.JSONDecodeError, TypeError):
                            pass

                # Also use the first_sentence field if available
                first_sent = doc.get('first_sentence', '')
                if isinstance(first_sent, dict):
                    first_sent = first_sent.get('value', '')
                if first_sent and isinstance(first_sent, str) and first_sent not in seen_texts:
                    seen_texts.add(first_sent)
                    s = _clean_sentence(first_sent)
                    if s:
                        sentences.append(s)

                # Generate a sentence from the title
                title = doc.get('title', '')
                author = doc.get('author_name', ['unknown'])[0] if doc.get('author_name') else 'an author'
                if title and title not in seen_texts:
                    seen_texts.add(title)
                    book_sent = f"The book {title} by {author} explores the theme of {topic}."
                    s = _clean_sentence(book_sent)
                    if s:
                        sentences.append(s)
        except (json.JSONDecodeError, TypeError):
            continue

    print(f"  [Open Library] Collected {len(sentences)} sentences", flush=True)
    return sentences


def fetch_wordpress_blogs() -> List[str]:
    """Fetch article excerpts from popular WordPress blogs."""
    sentences = []
    seen_texts: Set[str] = set()

    for blog in WORDPRESS_BLOGS:
        name = blog["name"]
        url = blog["url"]
        per_page = blog["per_page"]
        max_chars = blog["max_chars"]

        fetch_url = f"{url}?per_page={per_page}&_embed=1"
        data = _fetch(fetch_url, timeout=20)
        if not data:
            print(f"  [WordPress/{name}] Failed to fetch articles", flush=True)
            continue

        try:
            articles = json.loads(data)
            if isinstance(articles, dict):
                # Some APIs wrap results
                articles = articles.get('posts', articles.get('results', [articles]))

            for article in articles[:per_page]:
                # Extract excerpt
                excerpt_raw = article.get('excerpt', {}).get('rendered', '')
                content_raw = article.get('content', {}).get('rendered', '')

                # Use excerpt first (it's shorter/cleaner), fall back to content
                text = _strip_html(excerpt_raw) if excerpt_raw else _strip_html(content_raw)
                if not text:
                    continue

                text = text[:max_chars]
                text_sents = _split_into_sentences(text)
                for s in text_sents:
                    if s not in seen_texts:
                        seen_texts.add(s)
                        sentences.append(s)

                # Also use the title
                title = article.get('title', {}).get('rendered', '')
                if title and title not in seen_texts:
                    seen_texts.add(title)
                    s = _clean_sentence(f"In a recent article, {title} was discussed.")
                    if s:
                        sentences.append(s)
        except (json.JSONDecodeError, TypeError) as e:
            print(f"  [WordPress/{name}] Parse error: {e}", flush=True)
            continue

    print(f"  [WordPress Blogs] Collected {len(sentences)} sentences", flush=True)
    return sentences


def fetch_zenquotes() -> List[str]:
    """Fetch quotes from ZenQuotes API."""
    sentences = []

    data = _fetch(ZENQUOTES_API)
    if not data:
        return sentences

    try:
        quotes = json.loads(data)
        if isinstance(quotes, list):
            for q in quotes[:50]:
                text = q.get('q', '').strip()
                author = q.get('a', '').strip()
                if text:
                    s = _clean_sentence(text)
                    if s:
                        sentences.append(s)
                    if author:
                        ref = f"{author} said that {text[0].lower() + text[1:]}"
                        s = _clean_sentence(ref)
                        if s:
                            sentences.append(s)
    except (json.JSONDecodeError, TypeError):
        pass

    print(f"  [ZenQuotes] Collected {len(sentences)} sentences", flush=True)
    return sentences


def fetch_random_user_profiles() -> List[str]:
    """Fetch random user profiles and generate descriptive sentences."""
    sentences = []

    for i in range(10):
        data = _fetch(RANDOM_USER_API)
        if not data:
            continue

        try:
            parsed = json.loads(data)
            results = parsed.get('results', [])
            for user in results:
                name = user.get('name', {})
                first = name.get('first', '')
                last = name.get('last', '')
                gender = user.get('gender', '').capitalize()
                dob = user.get('dob', {})
                age = dob.get('age', 20)
                nat = user.get('nat', 'US')
                email = user.get('email', '')
                username = email.split('@')[0] if email else first.lower()

                sentences.append(
                    f"My name is {first} {last} and I am {age} years old."
                )
                sentences.append(
                    f"I live in a country called {nat} and I speak English."
                )
                sentences.append(
                    f"People describe me as a {gender.lower()} who loves learning new things."
                )
        except (json.JSONDecodeError, KeyError):
            continue

    print(f"  [RandomUser] Generated {len(sentences)} sentences", flush=True)
    return sentences


# ─── Teen-Level Conversational Templates ───

TEEN_CONVERSATION_TEMPLATES = [
    # Greetings & Social
    "Hello, how are you doing today?",
    "It is nice to meet you.",
    "How have you been lately?",
    "I am doing great, thanks for asking.",
    "What do you like to do for fun?",
    "I really enjoy spending time with my friends.",
    "That sounds like a lot of fun.",
    "Let me tell you about my day.",
    "Have you heard about the new movie?",
    "I think that is a really cool idea.",

    # Feelings & Emotions
    "I feel really happy when I see my friends.",
    "Sometimes I get nervous before a big test.",
    "It is okay to feel sad sometimes.",
    "I am excited about the weekend.",
    "You should never be afraid to ask for help.",
    "It makes me angry when people are unfair.",
    "I feel grateful for everything I have.",
    "Being kind to others makes me feel good.",
    "I am curious about how things work.",
    "That is really disappointing to hear.",

    # Opinions & Thoughts
    "I think that everyone deserves respect.",
    "In my opinion, honesty is the best policy.",
    "I believe that hard work pays off.",
    "Personally, I think music brings people together.",
    "To be honest, I really enjoy reading books.",
    "I feel like we should help each other more.",
    "As far as I know, practice makes perfect.",
    "I guess it depends on the situation.",
    "The way I see it, every problem has a solution.",
    "I am pretty sure that things will get better.",

    # Relationships
    "A true friend is someone who always listens.",
    "Family is really important to me.",
    "My best friend and I have known each other for years.",
    "Sometimes friends disagree but that is okay.",
    "I love spending time with my family.",
    "Trust is the foundation of any relationship.",
    "Good communication is key to strong friendships.",
    "My parents always support me no matter what.",
    "I learned a lot from my older sibling.",
    "Friendship means being there for each other.",

    # School & Learning
    "I have a lot to learn and that is exciting.",
    "Science class is really interesting this year.",
    "My favorite subject in school is history.",
    "I am learning how to play the guitar.",
    "Education opens doors to many opportunities.",
    "I really admire my teacher for being so patient.",
    "Studying with friends makes learning more fun.",
    "I want to learn more about space exploration.",
    "Math can be challenging but it is worth it.",
    "I enjoy writing stories for my English class.",

    # Daily Life
    "I woke up early this morning to go for a walk.",
    "Breakfast is my favorite meal of the day.",
    "I like to listen to music while I do my homework.",
    "After school, I usually hang out with my friends.",
    "I try to read at least one book every month.",
    "My favorite hobby is drawing and painting.",
    "I enjoy playing sports with my friends.",
    "Sometimes I just like to sit and think.",
    "I am trying to learn a new language.",
    "Every day is a new adventure.",

    # Nature & World
    "I love going for walks in the park.",
    "The sunset was absolutely beautiful today.",
    "Nature is full of amazing things to discover.",
    "Taking care of the environment is everyone's responsibility.",
    "I saw a beautiful rainbow after the rain.",
    "The stars are so bright tonight.",
    "Spring is my favorite season of the year.",
    "I love the sound of rain on the roof.",
    "Animals are such fascinating creatures.",
    "Planting a tree is a great way to help the planet.",

    # Motivation & Growth
    "Never give up on your dreams.",
    "Every mistake is a chance to learn something new.",
    "You are capable of amazing things.",
    "Small steps lead to big changes over time.",
    "Believe in yourself and anything is possible.",
    "The future is full of possibilities.",
    "It is never too late to start something new.",
    "Challenges make us stronger in the long run.",
    "Every day is a fresh start.",
    "You can achieve anything if you work hard enough.",

    # Technology & Modern Life
    "I use my computer to learn new things online.",
    "Technology changes the way we communicate.",
    "The internet is a powerful tool for learning.",
    "I love discovering new music on streaming services.",
    "Social media helps me stay connected with friends.",
    "Video games are a fun way to relax.",
    "I am learning how to code in my free time.",
    "Smartphones make it easy to stay in touch.",
    "Artificial intelligence is changing the world.",
    "It is important to take breaks from screens.",

    # Philosophical / Deep
    "What is the meaning of happiness?",
    "I wonder why people dream when they sleep.",
    "Sometimes the simplest things bring the most joy.",
    "Why do we remember the past but not the future?",
    "Curiosity is what drives human progress.",
    "Everything in nature follows a pattern.",
    "Time is the most valuable thing we have.",
    "Change is the only constant in life.",
    "The more you learn, the more you realize you do not know.",
    "We are all connected in ways we cannot see.",

    # Questions
    "What do you think about artificial intelligence?",
    "Have you ever wondered why the sky is blue?",
    "What is your favorite thing to do on weekends?",
    "Do you believe in the power of positive thinking?",
    "What does success mean to you?",
    "How do you deal with stress and anxiety?",
    "What is the most important lesson life has taught you?",
    "If you could travel anywhere, where would you go?",
    "What makes a person truly happy?",
    "How can we make the world a better place?",
]


# ─── Main Pipeline ───

def gather_all_sentences() -> Set[str]:
    """Gather sentences from all sources and return a deduplicated set."""
    all_sentences: Set[str] = set()

    print("\n[1/6] Fetching from Simple Wikipedia...", flush=True)
    try:
        sents = fetch_simple_wikipedia()
        all_sentences.update(sents)
    except Exception as e:
        print(f"  [warn] Simple Wikipedia failed: {e}", flush=True)

    print("\n[2/6] Fetching from Hacker News (tech discussions)...", flush=True)
    try:
        sents = fetch_hacker_news()
        all_sentences.update(sents)
    except Exception as e:
        print(f"  [warn] Hacker News API failed: {e}", flush=True)

    print("\n[3/6] Fetching book descriptions from Open Library...", flush=True)
    try:
        sents = fetch_open_library()
        all_sentences.update(sents)
    except Exception as e:
        print(f"  [warn] Open Library API failed: {e}", flush=True)

    print("\n[4/6] Fetching from WordPress blogs...", flush=True)
    try:
        sents = fetch_wordpress_blogs()
        all_sentences.update(sents)
    except Exception as e:
        print(f"  [warn] WordPress blogs failed: {e}", flush=True)

    print("\n[5/6] Fetching quotes and user descriptions...", flush=True)
    try:
        sents = fetch_zenquotes()
        all_sentences.update(sents)
    except Exception as e:
        print(f"  [warn] ZenQuotes API failed: {e}", flush=True)

    try:
        sents = fetch_random_user_profiles()
        all_sentences.update(sents)
    except Exception as e:
        print(f"  [warn] RandomUser API failed: {e}", flush=True)

    print("\n[6/6] Adding teen conversation templates...", flush=True)
    for sent in TEEN_CONVERSATION_TEMPLATES:
        s = _clean_sentence(sent)
        if s:
            all_sentences.add(s)

    return all_sentences


def save_sentences(sentences: Set[str], path: str):
    """Save sentences to file, sorted, one per line."""
    sorted_sents = sorted(sentences)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for sent in sorted_sents:
            f.write(sent + "\n")

    print(f"\n  Saved {len(sorted_sents)} sentences to {path}", flush=True)
    print(f"  Total characters: {sum(len(s) for s in sorted_sents)}", flush=True)
    print(f"  Total words: {sum(len(s.split()) for s in sorted_sents)}", flush=True)

    # Quick stats
    word_counts = [len(s.split()) for s in sorted_sents]
    avg_len = sum(word_counts) / len(word_counts) if word_counts else 0
    print(f"  Avg sentence length: {avg_len:.1f} words", flush=True)
    print(f"  Shortest: {min(word_counts)} words, Longest: {max(word_counts)} words", flush=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Gather teen-level conversational English sentences from the web"
    )
    parser.add_argument("--force", action="store_true",
                        help="Re-fetch even if output file already exists")
    parser.add_argument("--output", default=OUTPUT_PATH,
                        help=f"Output path (default: {OUTPUT_PATH})")
    args = parser.parse_args()

    # Check if already exists
    if os.path.exists(args.output) and not args.force:
        with open(args.output, "r", encoding="utf-8") as f:
            existing = [l.strip() for l in f if l.strip()]
        print(f"  teen_seeds.txt already exists with {len(existing)} sentences.", flush=True)
        print(f"  Use --force to re-fetch from the internet.", flush=True)
        return

    print("=" * 60)
    print("  RAVANA Teen Seeds Gatherer")
    print("  Gathering natural English sentences from the internet")
    print("=" * 60)

    sentences = gather_all_sentences()

    if not sentences:
        print("\n  [warn] No sentences gathered from online sources.", flush=True)
        print("  Falling back to built-in conversation templates only.", flush=True)
        sentences = set()
        for sent in TEEN_CONVERSATION_TEMPLATES:
            s = _clean_sentence(sent)
            if s:
                sentences.add(s)

    save_sentences(sentences, args.output)

    print("\n" + "=" * 60)
    print("  Done! The decoder can now train on real English sentences.")
    print("  Run: python scripts/train.py --mode phase2")
    print("=" * 60)


if __name__ == "__main__":
    main()
