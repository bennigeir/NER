import feedparser



VISIR_URL = 'https://www.visir.is/rss/allt'

NewsFeed = feedparser.parse(VISIR_URL)
entries = NewsFeed.entries

for entry in entries:
    title = entry.title
    summary = entry.summary
    published_parsed = entry.published_parsed