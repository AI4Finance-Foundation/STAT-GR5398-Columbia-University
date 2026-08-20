#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import sys

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
sys.path.insert(0, SRC_DIR)

from modules.common_utils import load_config 
from modules.llm_gateway import call_llm, load_llm_settings


TEST_PROMPT = "Please reply with one short sentence confirming API connectivity."


def run_single_provider_test(config, provider: str, model: str = None):
    print(f"\n=== Testing provider={provider} model={model or '[config default]'} ===")
    settings = load_llm_settings(config, provider=provider, model=model)
    response_text = call_llm(
        settings=settings,
        instructions="You are a concise assistant.",
        prompt=TEST_PROMPT,
        max_output_tokens=5000,
        temperature=0.0,
    )
    print(f"Model: {settings.model}")
    print(f"Response: {response_text}")


def parse_args():
    parser = argparse.ArgumentParser(description="Live API smoke test for OpenAI/Claude keys used in workflow.")
    parser.add_argument("--config-file", default=None, help="Path to config.ini.")
    parser.add_argument("--openai-model", default=None, help="Override OpenAI model for this test.")
    parser.add_argument("--claude-model", default=None, help="Override Claude model for this test.")
    parser.add_argument("--skip-openai", action="store_true", help="Skip OpenAI test.")
    parser.add_argument("--skip-claude", action="store_true", help="Skip Claude test.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config_file)

    if not args.skip_openai:
        try:
            run_single_provider_test(config, provider="openai", model=args.openai_model)
        except Exception as e:
            print(f"OpenAI test failed: {e}")

    if not args.skip_claude:
        try:
            run_single_provider_test(config, provider="claude", model=args.claude_model)
        except Exception as e:
            print(f"Claude test failed: {e}")


if __name__ == "__main__":
    main()
