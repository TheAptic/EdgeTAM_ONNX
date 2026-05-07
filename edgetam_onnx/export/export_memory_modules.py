from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--note", default="Memory/tracker export intentionally deferred")
    args = parser.parse_args()
    print(args.note)
    print("Tracker state should stay outside ONNX and be passed as explicit tensors between calls.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
