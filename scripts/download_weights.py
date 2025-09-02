#!/usr/bin/env python3
import os
import sys

URL = "https://drive.google.com/uc?id=1qw8YG5RieJB7qf-Pj_FfqCLHdZtTXpFi"
OUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights.dat")

def main():
    try:
        import gdown
    except Exception as e:
        print("gdown is required. Install with: pip install gdown", file=sys.stderr)
        sys.exit(1)

    print(f"Downloading weights to: {OUT}")
    gdown.download(URL, OUT, quiet=False)
    if os.path.exists(OUT):
        print("Download completed.")
    else:
        print("Download failed.", file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    main()
