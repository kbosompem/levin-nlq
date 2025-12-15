#!/usr/bin/env python3
"""
Test the fine-tuned Datalevin NLQ model with sample queries.
"""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_mlx_model(model_path: str, queries: list):
    """Test using MLX (for Mac)."""
    try:
        from mlx_lm import load, generate
    except ImportError:
        print("MLX not available. Install with: pip install mlx mlx-lm")
        return

    print(f"Loading model from {model_path}...")
    model, tokenizer = load(model_path)

    # Sample schema for testing
    schema = """{:user/name {:db/valueType :db.type/string}
 :user/email {:db/valueType :db.type/string}
 :user/age {:db/valueType :db.type/long}
 :user/active {:db/valueType :db.type/boolean}}"""

    print("\n" + "=" * 60)
    print("Testing Datalevin NLQ Model")
    print("=" * 60)

    for query in queries:
        prompt = f"<|user|>Schema: {schema}\n\n{query}<|assistant|>"
        print(f"\n📝 Query: {query}")

        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=200,
            verbose=False
        )

        # Extract just the generated part (after the prompt)
        generated = response[len(prompt):].strip() if response.startswith(prompt) else response
        print(f"🔍 Generated: {generated}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test Datalevin NLQ model")
    parser.add_argument(
        "--model",
        default=str(Path(__file__).parent.parent / "models" / "datalevin-fused"),
        help="Path to model directory"
    )
    args = parser.parse_args()

    test_queries = [
        "Find all users",
        "Find users named John",
        "Find active users older than 30",
        "Count all users",
        "Average user age",
        "Find users with gmail addresses",
        "Find users without email",
        "Get user names and emails",
    ]

    test_mlx_model(args.model, test_queries)


if __name__ == "__main__":
    main()
