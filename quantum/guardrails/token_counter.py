"""
Local Token Counter for Quantum Guardrails.
Computes token statistics (min, max, mean, p95, p99) entirely on-device.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List


class LocalTokenCounter:
    def __init__(self, encoding_name: str = "cl100k_base"):
        self.encoding_name = encoding_name
        self._encoder = None
        self._init_encoder()

    def _init_encoder(self) -> None:
        try:
            import tiktoken
            self._encoder = tiktoken.get_encoding(self.encoding_name)
        except Exception:
            self._encoder = None

    def count_tokens(self, text: str) -> int:
        """Count tokens in string."""
        if not text:
            return 0
        if self._encoder is not None:
            try:
                return len(self._encoder.encode(text))
            except Exception:
                pass
        # Fallback: approximation ~ 1.3 tokens per whitespace word or len/4
        words = text.split()
        return max(1, int(len(words) * 1.33))

    def compute_distribution(self, texts: List[str]) -> Dict[str, Any]:
        """
        Calculates distribution metrics on an array of texts.
        Returns aggregate numbers only.
        """
        if not texts:
            return {
                "count": 0,
                "min": 0,
                "max": 0,
                "mean": 0.0,
                "median": 0,
                "p50": 0,
                "p90": 0,
                "p95": 0,
                "p99": 0,
                "buckets": {},
                "recommended_max_seq_length": 512,
            }

        counts = [self.count_tokens(t) for t in texts]
        counts.sort()

        total = len(counts)
        min_tok = counts[0]
        max_tok = counts[-1]
        mean_tok = round(sum(counts) / total, 1)

        def percentile(p: float) -> int:
            idx = int(math.ceil((p / 100.0) * total)) - 1
            idx = max(0, min(total - 1, idx))
            return counts[idx]

        p50 = percentile(50)
        p90 = percentile(90)
        p95 = percentile(95)
        p99 = percentile(99)

        # Bucket distribution
        buckets = {
            "<= 256": sum(1 for c in counts if c <= 256),
            "257 - 512": sum(1 for c in counts if 256 < c <= 512),
            "513 - 1024": sum(1 for c in counts if 512 < c <= 1024),
            "1025 - 2048": sum(1 for c in counts if 1024 < c <= 2048),
            "2049 - 4096": sum(1 for c in counts if 2048 < c <= 4096),
            "> 4096": sum(1 for c in counts if c > 4096),
        }

        # Recommend standard max_seq_length covering p99
        if p99 <= 512:
            recommended_max = 512
        elif p99 <= 1024:
            recommended_max = 1024
        elif p99 <= 2048:
            recommended_max = 2048
        elif p99 <= 4096:
            recommended_max = 4096
        else:
            recommended_max = 8192

        return {
            "count": total,
            "min": min_tok,
            "max": max_tok,
            "mean": mean_tok,
            "median": p50,
            "p50": p50,
            "p90": p90,
            "p95": p95,
            "p99": p99,
            "buckets": buckets,
            "recommended_max_seq_length": recommended_max,
        }
