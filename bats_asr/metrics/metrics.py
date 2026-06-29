from __future__ import annotations

import re
import unicodedata


def clean_text(text: str) -> str:
    text = str(text)
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u200b", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def levenshtein(a: str, b: str) -> int:
    a = clean_text(a)
    b = clean_text(b)

    n, m = len(a), len(b)
    dp = list(range(m + 1))

    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i

        for j in range(1, m + 1):
            old = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = old

    return dp[m]


def cer(reference: str, hypothesis: str) -> float:
    reference = clean_text(reference)
    hypothesis = clean_text(hypothesis)
    return levenshtein(reference, hypothesis) / max(1, len(reference))


def character_accuracy(reference: str, hypothesis: str) -> float:
    return 1.0 - cer(reference, hypothesis)
