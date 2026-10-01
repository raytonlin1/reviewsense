"""ReviewSense: conversational review intelligence.

Process-wide runtime settings, applied before any ML library is imported:
  * TOKENIZERS_PARALLELISM=false: Hugging Face fast tokenizers warn (and can deadlock) when a process forks after
    they have used threads, which happens with data-loader workers and web-server workers.
"""
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
