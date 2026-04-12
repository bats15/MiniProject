"""CLI entrypoint to create dummy dataset for testing."""

import argparse

from dataset import create_dummy_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Create dummy audio dataset")
    parser.add_argument("--output", type=str, default="./data", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples")
    parser.add_argument("--duration", type=int, default=5, help="Unused compatibility flag")
    args = parser.parse_args()

    _ = args.duration  # Keep CLI compatible with README examples.
    create_dummy_dataset(args.output, num_samples=args.num_samples)


if __name__ == "__main__":
    main()
