from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--note", default="Runtime runner scaffold only")
    args = parser.parse_args()
    print(args.note)
    print("Use runtime/ort_session.py + tracker_state.py to integrate split ONNX models in app loop.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
