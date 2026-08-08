#!/usr/bin/env python3
"""Strip biscotti scaffolding from a HEIR-emitted MLIR module.

Removes `func.func @main` and any `func.func private @*_dummy` — the
wrapper that calls the identity dummy and returns. Leaves the real
hoisted compute (e.g. `@mm_clone_0_0`) untouched.

Usage:
  strip_scaffold.py input.mlir -o output.mlir
"""

import argparse
import re
from pathlib import Path


def strip(text: str) -> str:
  """Drop func ops matching main or *_dummy from a flat module { ... }."""
  header_re = re.compile(
      r"\bfunc\.func\b(?:\s+private)?\s+@([A-Za-z0-9_]+)\s*\("
  )

  keep = []
  i = 0
  while i < len(text):
    m = header_re.search(text, i)
    if not m:
      keep.append(text[i:])
      break
    name = m.group(1)
    drop = (name == "main") or name.endswith("_dummy")
    if not drop:
      keep.append(text[i : m.end()])
      i = m.end()
      continue

    # Skip past the arg-list parens.
    j = m.end()
    depth_parens = 1
    while j < len(text) and depth_parens > 0:
      if text[j] == "(":
        depth_parens += 1
      elif text[j] == ")":
        depth_parens -= 1
      j += 1
    # Scan past the (optional) return-type annotation to the body brace.
    while j < len(text) and text[j] not in "{;":
      j += 1
    if j >= len(text) or text[j] == ";":
      # Declaration-only — no body to consume.
      i = j + 1
      continue
    # Match the body's braces.
    depth = 1
    j += 1
    while j < len(text) and depth > 0:
      if text[j] == "{":
        depth += 1
      elif text[j] == "}":
        depth -= 1
      j += 1
    # Consume a trailing newline for cleanliness.
    if j < len(text) and text[j] == "\n":
      j += 1
    keep.append(text[i : m.start()])
    i = j
  return "".join(keep)


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("input", type=Path)
  ap.add_argument("-o", "--output", type=Path, required=True)
  args = ap.parse_args()
  args.output.write_text(strip(args.input.read_text()))


if __name__ == "__main__":
  main()
