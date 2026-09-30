"""ReviewSense: conversational review intelligence.

Process-wide runtime settings, applied before any ML library is imported:
  * OMP_NUM_THREADS=1 then torch threads restored: pip wheels of faiss-cpu and torch each bundle their own
    libomp on macOS; two active OpenMP thread pools segfault inside FAISS search. Starting OpenMP single-threaded
    keeps FAISS's copy idle while torch.set_num_threads() gives PyTorch its full thread pool back.
    (Set OMP_NUM_THREADS yourself to override.)
  * Haystack telemetry off: don't send usage analytics from a production service by default.
"""
import os

_user_set_omp = "OMP_NUM_THREADS" in os.environ
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("HAYSTACK_TELEMETRY_ENABLED", "False")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

if not _user_set_omp:
    import torch

    torch.set_num_threads(os.cpu_count() or 1)