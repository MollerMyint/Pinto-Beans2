"""
clean_cpp_markdown.py
---------------------
Strip obvious boilerplate from Cal Poly Pomona website markdown files.

What it removes:
- repeated top navigation / utility lines before and inside content
- footer blocks starting at markers like "Ripped green paper" or "Copyright"
- social-only / logo-only / search/menu image lines
- low-value "Follow Us" sections
- known footer-only standalone links

Usage:
    python clean_cpp_markdown.py
    python clean_cpp_markdown.py --input-dir ./raw --output-dir ./cleaned
    python clean_cpp_markdown.py file1.md file2.md ..

Example:
    python clean_cpp_markdown.py raw_pages/_class__about.md
"""

from __future__ import annotations

import re
import sys
import argparse
from pathlib import Path
from typing import Iterable


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Tightened patterns so we do not accidentally remove real content lines like
# "Apply online via Cal State Apply".
UTILITY_LINE_PATTERNS = [
    r"^\[apply\]\([^)]+\)$",
    r"^\[visit\]\([^)]+\)$",
    r"^\[info\]\([^)]+\)$",
    r"^\[give\]\([^)]+\)$",
    r"^\[mycpp\]\([^)]+\)$",
    r"^apply$",
    r"^visit$",
    r"^info$",
    r"^give$",
    r"^mycpp$",
    r"^search$",
    r"^menu$",
    r"^skip to content.*$",
    r"^commencement$",
    r"^today's hours:.*$",
    r"^events calendar$",
    r"^reserve a study room$",
    r"^current services$",
]

FOOTER_START_PATTERNS = [
    r"!\[Ripped green paper.*\]",
    r"^Copyright .*$",
    r"^A campus of$",
    r"^\[Feedback\].*$",
    r"^\[Privacy\].*$",
    r"^\[Accessibility\].*$",
    r"^\[Document Readers\].*$",
]

# Used to distinguish low-value footer links from real content links.
FOOTER_URL_TOKENS = [
    "maps.cpp.edu",
    "cpp.edu/apply",
    "cpp.edu/contact",
    "cpp.edu/website-feedback",
    "policystat.com",
    "cpp.edu/accessibility",
    "cpp.edu/file-viewers",
    "cpp.edu/outreach/tours",
]

NOISE_HEADING_PATTERNS = {
    "follow us:",
}

SOCIAL_LINE_PATTERNS = [
    r".*instagram.*",
    r".*linkedin.*",
    r".*youtube.*",
    r".*facebook.*",
    r"^\[x.*",
    r".*tiktok.*",
    r".*snapchat.*",
    r".*vimeo.*",
    r".*threads.*",
]

MENU_BLOCK_STARTS = {"menu"}


# ---------------------------------------------------------------------------
# Line helpers
# ---------------------------------------------------------------------------

def normalize(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def is_heading(line: str) -> bool:
    return bool(re.match(r"^#{1,6}\s+\S", line.strip()))


def heading_text(line: str) -> str:
    return re.sub(r"^#{1,6}\s+", "", line).strip().lower()


def is_list_line(line: str) -> bool:
    s = normalize(line).lower()
    return bool(s) and (
        s.startswith(("* ", "- ", "+ "))
        or s == "|"
        or s.count("|") >= 2
    )


def is_social_or_utility(line: str) -> bool:
    s = normalize(line)
    lower = s.lower()

    for pat in UTILITY_LINE_PATTERNS:
        if re.match(pat, lower):
            return True

    for pat in SOCIAL_LINE_PATTERNS:
        if re.match(pat, lower):
            return True

    if s.startswith("![") and "logo" in lower:
        return True

    if s.startswith(("![Open search", "![Close", "![Search]")):
        return True

    return False


def is_footer_start(line: str) -> bool:
    s = normalize(line)
    return any(re.match(pat, s, flags=re.IGNORECASE) for pat in FOOTER_START_PATTERNS)


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def find_content_start(lines: list[str]) -> int:
    """
    Skip everything before the first likely real content heading.

    Prefer a top-level H1 if available, otherwise allow H2 as a fallback in
    case some pages were scraped inconsistently.
    """
    for i, line in enumerate(lines):
        if re.match(r"^#\s+\S", line.strip()):
            return i

    for i, line in enumerate(lines):
        if re.match(r"^##\s+\S", line.strip()):
            return i

    return 0


def remove_footer(lines: list[str]) -> list[str]:
    for i, line in enumerate(lines):
        if is_footer_start(line):
            return lines[:i]
    return lines


def remove_noise_sections(lines: list[str]) -> list[str]:
    """
    Remove:
    - utility/social lines anywhere
    - "Menu" blocks followed by navigation lists
    - low-value headings like "Follow Us:" and their following content
    - plain-text "Follow Us:" blocks even if not marked as headings
    """
    cleaned: list[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]
        lower = normalize(line).lower()

        # Drop utility / social lines anywhere
        if is_social_or_utility(line):
            i += 1
            continue

        # Drop plain-text Follow Us block
        if lower == "follow us:":
            i += 1
            while i < len(lines):
                if is_heading(lines[i]):
                    break
                nxt = normalize(lines[i]).lower()
                if not nxt or is_social_or_utility(lines[i]) or re.match(r"^!\[.*\]\(.*\)$", normalize(lines[i])):
                    i += 1
                    continue
                # If prose resumes, stop skipping
                break
            continue

        # Drop "Menu" heading + following list-heavy block
        if lower in MENU_BLOCK_STARTS:
            i += 1
            while i < len(lines):
                nxt = normalize(lines[i])
                if not nxt:
                    i += 1
                    continue
                if is_heading(lines[i]):
                    break
                if is_list_line(lines[i]) or is_social_or_utility(lines[i]):
                    i += 1
                    continue
                # If actual prose resumes, stop skipping
                break
            continue

        # Drop low-value headings like "Follow Us:" + content until next heading
        if is_heading(line) and heading_text(line) in NOISE_HEADING_PATTERNS:
            i += 1
            while i < len(lines) and not is_heading(lines[i]):
                i += 1
            continue

        cleaned.append(line)
        i += 1

    return cleaned


def strip_redundant_link_lines(lines: list[str]) -> list[str]:
    """
    Remove lines that are purely:
    - a single image
    - a known footer-only link

    Preserve real content links in prose or lists.
    """
    cleaned: list[str] = []

    for line in lines:
        s = normalize(line)

        if is_heading(line) or not s:
            cleaned.append(line)
            continue

        # Pure image-only line: ![alt](url)
        if re.match(r"^!\[.*\]\(.*\)$", s):
            continue

        # Single standalone link: [text](url)
        # Only remove if it looks like a known footer link.
        if re.match(r"^\[[^\]]+\]\([^)]+\)$", s):
            lower = s.lower()
            if any(token in lower for token in FOOTER_URL_TOKENS):
                continue

        cleaned.append(line)

    return cleaned


def collapse_blank_lines(lines: Iterable[str]) -> list[str]:
    result: list[str] = []
    prev_blank = False

    for line in lines:
        blank = normalize(line) == ""
        if blank and prev_blank:
            continue
        result.append("" if blank else line.rstrip())
        prev_blank = blank

    while result and not normalize(result[0]):
        result.pop(0)
    while result and not normalize(result[-1]):
        result.pop()

    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def clean_cpp_markdown(md_text: str) -> str:
    lines = md_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    # Preserve source line if present before stripping nav/header content
    source_prefix: list[str] = []
    for line in lines[:5]:
        s = normalize(line)
        if s.startswith("**Source:**"):
            source_prefix = [line, "---", ""]
            break

    lines = lines[find_content_start(lines):]
    lines = remove_footer(lines)
    lines = remove_noise_sections(lines)
    lines = strip_redundant_link_lines(lines)
    lines = collapse_blank_lines(lines)

    return "\n".join(source_prefix + lines).strip() + "\n"


def process_file(src: Path, dst: Path) -> None:
    raw = src.read_text(encoding="utf-8")
    cleaned = clean_cpp_markdown(raw)

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(cleaned, encoding="utf-8")

    orig = len(raw.splitlines())
    out = len(cleaned.splitlines())
    removed = orig - out
    pct = (removed / orig * 100) if orig else 0.0

    print(f"  {src.name:<60}  {orig:>4} -> {out:>4} lines  ({removed:>4} removed, {pct:>3.0f}%)")

    if out < 3:
        print(f"  WARNING: cleaned output for {src.name} is very short; inspect manually.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Strip boilerplate from CPP website .md files")
    parser.add_argument("files", nargs="*", help="Specific .md files to process")
    parser.add_argument("--input-dir", default=".", help="Source directory (default: current directory)")
    parser.add_argument("--output-dir", default="cleaned", help="Output directory (default: ./cleaned)")
    args = parser.parse_args()

    if args.files:
        sources = [Path(f) for f in args.files]
    else:
        sources = sorted(Path(args.input_dir).glob("*.md"))

    if not sources:
        print("No .md files found.")
        sys.exit(1)

    out_dir = Path(args.output_dir)
    print(f"Cleaning {len(sources)} file(s) -> {out_dir}/\n")

    for src in sources:
        process_file(src, out_dir / src.name)

    print("\nDone.")


if __name__ == "__main__":
    main()