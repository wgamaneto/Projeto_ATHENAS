#!/usr/bin/env python3
"""Simple dashboard script to analyze backend logs."""

import re
from pathlib import Path

LOG_FILE = Path("logs/backend.log")


def parse_logs(log_file: Path):
    """Parse log file for response times and tokens.

    Returns a tuple (times, tokens) where times is a list of floats in seconds
    and tokens is a list of integers.
    """
    times = []
    tokens = []
    time_pattern = re.compile(r"Tempo de resposta: ([0-9.]+)s")
    token_pattern = re.compile(r"Tokens usados: (\d+)")

    if not log_file.exists():
        return times, tokens

    with log_file.open() as fh:
        for line in fh:
            match_time = time_pattern.search(line)
            if match_time:
                times.append(float(match_time.group(1)))
            match_tokens = token_pattern.search(line)
            if match_tokens:
                tokens.append(int(match_tokens.group(1)))
    return times, tokens


def main() -> None:
    times, tokens = parse_logs(LOG_FILE)
    if not times and not tokens:
        print(f"Nenhum dado disponível em {LOG_FILE}.")
        return

    avg_time = sum(times) / len(times) if times else 0.0
    total_tokens = sum(tokens)
    print(f"Tempo médio de resposta: {avg_time:.2f}s")
    print(f"Total de tokens usados: {total_tokens}")


if __name__ == "__main__":
    main()
