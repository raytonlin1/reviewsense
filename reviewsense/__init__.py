"""ReviewSense: conversational review intelligence.

Process-wide settings, applied before any ML library is imported:
  * TOKENIZERS_PARALLELISM=false: Hugging Face fast tokenizers warn (and can deadlock) when a process forks after
    they have used threads, which happens with data-loader workers and web-server workers.
  * HAYSTACK_TELEMETRY_ENABLED=False: don't send usage analytics from a service by default.
"""
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HAYSTACK_TELEMETRY_ENABLED", "False")
