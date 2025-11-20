#!/usr/bin/env python3
from pathlib import Path
import sys

SEP_FMT = "\n\n======= {name} =======\n\n"


def concat(list_file: Path, out):
    if not list_file.exists():
        out.write(f"Error: Cannot read file '{list_file}'\n")
        return 1

    try:
        # utf-8-sig handles potential BOMs (common on Windows-edited files)
        lines = list_file.read_text(encoding="utf-8-sig").splitlines()
    # --- FIX: Catch specific errors ---
    except (OSError, UnicodeDecodeError) as e:
        out.write(f"Error: Cannot read file '{list_file}': {e}\n")
        return 1

    for raw in lines:
        name = raw.strip()

        # Skip empty lines and comment lines (after stripping)
        if not name or name.startswith("#"):
            continue

        out.write(SEP_FMT.format(name=name))
        p = Path(name)
        if p.exists() and p.is_file():
            try:
                out.write(p.read_text(encoding="utf-8"))
            # --- FIX: Catch specific errors ---
            except (OSError, UnicodeDecodeError) as e:
                out.write(f"Error: Cannot read file '{name}': {e}\n")
        else:
            out.write(f"Error: Cannot read file '{name}'\n")
        out.write("\n")

    # Ensure at least one empty line at the end
    out.write("\n")
    return 0


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    if len(argv) != 1 or argv[0] in {"-h", "--help"}:
        sys.stderr.write("Usage: python tools/concat_files.py <filelist>\n")
        return 2

    list_file = Path(argv[0])
    return concat(list_file, sys.stdout)


if __name__ == "__main__":
    sys.exit(main())
