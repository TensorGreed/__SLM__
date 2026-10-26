"""Document sources — pull documents from a website or a git repository.

``web_crawl`` and ``git_repo`` are pure source readers (network / ``git`` CLI,
no database); ``importer`` turns what they find into ``RawDocument`` rows via
the ordinary ingestion path, so cleaning, the passage index and the rest of the
pipeline treat them like uploads.
"""
