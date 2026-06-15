"""
Step 2 — Clean and normalize source texts
Corpus: Korean news articles on AI and youth employment (article_01~20.txt)
Transformations are logged to cleaned/cleaning_log.csv for reproducibility.
"""

import csv
import os
import re
import unicodedata

SRC_DIR = "/home/eunji/project/articles"
OUT_DIR = "/home/eunji/project/cleaned"
LOG_PATH = "/home/eunji/project/cleaned/cleaning_log.csv"

os.makedirs(OUT_DIR, exist_ok=True)

TRANSFORMS = [
    "unicode_normalize_NFC",
    "strip_trailing_whitespace",
    "collapse_blank_lines",
    "repair_hyphenation",
    "normalize_quotes",
    "remove_zero_width_chars",
]


def unicode_normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def strip_trailing_whitespace(text: str) -> str:
    lines = [l.rstrip() for l in text.splitlines()]
    return "\n".join(lines)


def collapse_blank_lines(text: str) -> str:
    # Collapse 3+ consecutive blank lines to 2
    return re.sub(r"\n{3,}", "\n\n", text)


def repair_hyphenation(text: str) -> str:
    # Repair word-break hyphens across line ends (common in OCR; rare here but logged)
    return re.sub(r"-\n(\S)", r"\1", text)


def normalize_quotes(text: str) -> str:
    # Normalize curly quotes to straight quotes for consistency
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    return text


def remove_zero_width(text: str) -> str:
    # Remove zero-width spaces and BOM artifacts
    return re.sub(r"[​‌‍﻿]", "", text)


PIPELINE = [
    ("unicode_normalize_NFC",    unicode_normalize),
    ("strip_trailing_whitespace", strip_trailing_whitespace),
    ("collapse_blank_lines",     collapse_blank_lines),
    ("repair_hyphenation",       repair_hyphenation),
    ("normalize_quotes",         normalize_quotes),
    ("remove_zero_width_chars",  remove_zero_width),
]


def clean_file(filename: str) -> dict:
    path = os.path.join(SRC_DIR, filename)
    with open(path, encoding="utf-8") as f:
        original = f.read()

    text = original
    changes = []
    for name, fn in PIPELINE:
        result = fn(text)
        diff = len(original) - len(result) if name == "remove_zero_width_chars" else 0
        if result != text:
            changes.append(name)
        text = result

    # Count stats
    original_lines  = original.count("\n") + 1
    cleaned_lines   = text.count("\n") + 1
    original_chars  = len(original)
    cleaned_chars   = len(text)

    out_path = os.path.join(OUT_DIR, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)

    return {
        "filename":         filename,
        "original_lines":   original_lines,
        "cleaned_lines":    cleaned_lines,
        "original_chars":   original_chars,
        "cleaned_chars":    cleaned_chars,
        "chars_removed":    original_chars - cleaned_chars,
        "transforms_applied": "|".join(changes) if changes else "none",
        "status":           "changed" if changes else "already_clean",
    }


def main():
    files = sorted(
        f for f in os.listdir(SRC_DIR) if f.endswith(".txt")
    )
    log_rows = []
    for fname in files:
        row = clean_file(fname)
        log_rows.append(row)
        status_mark = "✎" if row["status"] == "changed" else "✓"
        print(f"  {status_mark} {fname:25s}  lines {row['original_lines']:>4}→{row['cleaned_lines']:>4}"
              f"  chars {row['original_chars']:>6}→{row['cleaned_chars']:>6}"
              f"  [{row['transforms_applied']}]")

    with open(LOG_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)

    changed = sum(1 for r in log_rows if r["status"] == "changed")
    print(f"\n  총 {len(log_rows)}개 파일 처리 — {changed}개 변환, {len(log_rows)-changed}개 원본 유지")
    print(f"  → {LOG_PATH}")
    print(f"  → 정제 파일: {OUT_DIR}/")


if __name__ == "__main__":
    main()
