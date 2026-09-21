"""Allow `python -m datajev`."""

from __future__ import annotations

from datajev.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
