# utils/chunker.py
import re

def split_on_headings_and_windows(lines, max_words=350, min_words=20):
    """
    lines: list of file lines (strings)
    Returns list of chunks with metadata:
    {"chunk": str, "start_line": int, "end_line": int}
    Strategy:
    - Split on Markdown headings (lines that start with '#') to preserve logical sections.
    - For each section, further window it by max_words.
    """
    text = "".join(lines)
    # Split into sections at headings but keep headings with the section
    parts = re.split(r'(?m)(?=^#{1,6}\s)', text)
    chunks = []
    for part in parts:
        words = part.split()
        if len(words) < min_words:
            # skip tiny noisy parts
            continue
        # compute approximate line indices by searching for the substring
        start_idx = text.find(part)
        if start_idx == -1:
            continue
        # find start_line by counting newlines before start_idx
        start_line = text[:start_idx].count("\n")
        # create windows
        for i in range(0, len(words), max_words):
            sub = " ".join(words[i:i + max_words])
            if len(sub.split()) < min_words:
                continue
            # estimate end_line by counting newlines within the substring
            offset = text[start_idx:start_idx + len(sub)].count("\n")
            end_line = start_line + offset
            chunks.append({
                "chunk": sub,
                "start_line": start_line + 1,   # 1-indexed
                "end_line": end_line + 1
            })
            # advance start_idx for next window (approximate)
            # find the next window position to recompute start_line for accuracy
            # we do rough approach; it's fine for metadata
            start_idx += len(sub)
            start_line = end_line
    return chunks
