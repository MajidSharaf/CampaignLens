"""
Stage 0 - YouTube Comment Scraper
Hits the YouTube Data API v3, pulls comments from a list of video URLs,
and writes comments.csv. Paste full YouTube URLs - video IDs are
extracted automatically, no manual lookup needed.

Input:  YouTube video URLs (via --urls, --urls-file, or interactive prompt)
        + YOUTUBE_API_KEY env var
Output: comments.csv

Usage:
    python3 0_scraper.py
    python3 0_scraper.py --urls "https://youtube.com/watch?v=abc,https://youtu.be/xyz"
    python3 0_scraper.py --urls-file my_urls.txt
    python3 0_scraper.py --input-csv path/to/existing_comments.csv
"""

import argparse
import os
import re
import sys

import pandas as pd
import googleapiclient.discovery
from dotenv import load_dotenv


# ---------- video id extraction ----------

def extractVideoId(url_or_id):
    """
    Accepts a full YouTube URL in any common format, or a bare video ID,
    and returns just the 11-character video ID.
    """
    url_or_id = url_or_id.strip()

    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",       # watch?v=ID or /ID
        r"youtu\.be\/([0-9A-Za-z_-]{11})",        # youtu.be/ID
        r"embed\/([0-9A-Za-z_-]{11})",            # embed/ID
        r"shorts\/([0-9A-Za-z_-]{11})",           # shorts/ID
    ]

    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)

    # fall back: maybe it's already a bare 11-char ID
    if re.fullmatch(r"[0-9A-Za-z_-]{11}", url_or_id):
        return url_or_id

    raise ValueError(f"Could not extract a video ID from: {url_or_id}")


# ---------- scraping ----------

def getComments(youtube, video_id):
    rows = []
    request = youtube.commentThreads().list(
        part="snippet", videoId=video_id, maxResults=100, textFormat="plainText"
    )

    while request:
        response = request.execute()
        for item in response["items"]:
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            rows.append({
                "author":     snippet["authorDisplayName"],
                "updated_at": snippet["publishedAt"],
                "like_count": snippet["likeCount"],
                "text":       snippet["textOriginal"],
                "video_id":   snippet["videoId"],
                "public":     item["snippet"]["isPublic"],
                "language":   snippet.get("authorChannelId", {}).get("value", None),
            })
        request = youtube.commentThreads().list_next(request, response)

    return pd.DataFrame(rows)


def filterEnglish(df):
    """
    Lightweight English filter: keeps rows where the text is mostly ASCII.
    The YouTube API doesn't reliably return a language field per comment,
    so this heuristic stands in for true language detection.
    """
    def isMostlyAscii(text):
        if not isinstance(text, str) or not text.strip():
            return False
        ascii_ratio = sum(c.isascii() for c in text) / len(text)
        return ascii_ratio >= 0.85

    return df[df["text"].apply(isMostlyAscii)].reset_index(drop=True)


def runScraper(video_urls, output_path="comments.csv"):
    load_dotenv()
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "YOUTUBE_API_KEY not found. Add it to a .env file in this folder:\n"
            "YOUTUBE_API_KEY=your_key_here"
        )

    youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=api_key)

    video_ids = [extractVideoId(u) for u in video_urls]
    print(f"Scraping {len(video_ids)} videos: {video_ids}")

    all_dfs = []
    for vid in video_ids:
        print(f"  Fetching comments for {vid}...")
        df = getComments(youtube, vid)
        print(f"    -> {len(df)} comments")
        all_dfs.append(df)

    all_comments = pd.concat(all_dfs, ignore_index=True)
    all_comments = filterEnglish(all_comments)
    all_comments = all_comments.drop(columns=["language"], errors="ignore")

    all_comments.to_csv(output_path, index=False)
    print(f"Saved {len(all_comments)} rows to {output_path}")
    return all_comments


# ---------- entry point ----------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scrape YouTube comments. Paste video URLs directly - "
                     "the video ID is extracted automatically, no manual lookup needed."
    )
    parser.add_argument(
        "--urls",
        help="Comma-separated YouTube video URLs (any format - full link, "
             "youtu.be short link, shorts link, or bare video ID). "
             "Example: --urls \"https://youtube.com/watch?v=abc,https://youtu.be/xyz\""
    )
    parser.add_argument(
        "--urls-file",
        help="Path to a text file with one YouTube URL per line, as an "
             "alternative to --urls for long lists"
    )
    parser.add_argument(
        "--input-csv",
        help="If provided, skip the API call entirely and use this CSV as-is "
             "(must already have: video_id, author, updated_at, like_count, text)"
    )
    parser.add_argument(
        "--output",
        default="comments.csv",
        help="Output path for the scraped CSV (default: comments.csv)"
    )
    args = parser.parse_args()

    if args.input_csv:
        if not os.path.exists(args.input_csv):
            print(f"Error: {args.input_csv} not found", file=sys.stderr)
            sys.exit(1)
        df = pd.read_csv(args.input_csv)
        required = ["video_id", "author", "updated_at", "like_count", "text"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"Error: input CSV is missing columns: {missing}", file=sys.stderr)
            sys.exit(1)
        df.to_csv(args.output, index=False)
        print(f"Used provided CSV as-is. Saved {len(df)} rows to {args.output}")

    elif args.urls:
        urls = [u.strip() for u in args.urls.split(",") if u.strip()]
        runScraper(urls, output_path=args.output)

    elif args.urls_file:
        if not os.path.exists(args.urls_file):
            print(f"Error: {args.urls_file} not found", file=sys.stderr)
            sys.exit(1)
        with open(args.urls_file, "r") as f:
            urls = [line.strip() for line in f if line.strip()]
        runScraper(urls, output_path=args.output)

    else:
        urls = []
        print("Paste YouTube URLs one at a time. Press Enter on an empty line when done.")
        while True:
            line = input("URL: ").strip()
            if not line:
                break
            urls.append(line)

        if not urls:
            print("No URLs provided. Exiting.", file=sys.stderr)
            sys.exit(1)

        runScraper(urls, output_path=args.output)
