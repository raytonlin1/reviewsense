"""ReviewSense: conversational review intelligence.

Process-wide runtime settings, applied before any ML library is imported:
  * TOKENIZERS_PARALLELISM=false: Hugging Face fast tokenizers warn (and can deadlock) when a process forks after
    they have used threads, which happens with data-loader workers and web-server workers.
  * OMP_NUM_THREADS=1, then torch threads restored (Part 7): on macOS the pip packages of faiss-cpu and torch each
    bundle their own copy of the OpenMP threading library (libomp). Two active OpenMP thread pools crash FAISS
    search (segmentation fault). Starting OpenMP single-threaded keeps FAISS's copy idle, and
    torch.set_num_threads() gives PyTorch its full thread pool back. Set OMP_NUM_THREADS yourself to override.
  * HAYSTACK_TELEMETRY_ENABLED=False (Part 7): don't send usage analytics from a service by default.
"""
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HAYSTACK_TELEMETRY_ENABLED", "False")
_user_set_omp = "OMP_NUM_THREADS" in os.environ
os.environ.setdefault("OMP_NUM_THREADS", "1")

if not _user_set_omp:
    import torch

    torch.set_num_threads(os.cpu_count() or 1)
