#!/usr/bin/env python3
"""
Quick exploration: What does ollama.list() return?
Run: python tmp/explore_ollama_list.py
"""

import ollama
import json

try:
    client = ollama.Client()
    result = client.list()

    print("=== RAW OUTPUT ===")
    print(type(result))
    print(json.dumps(result, indent=2, default=str))

    print("\n=== MODELS ===")
    if hasattr(result, "models") or isinstance(result, dict) and "models" in result:
        models = result.get("models", []) if isinstance(result, dict) else result.models
        for m in models:
            print(f"  - {m}")

except Exception as e:
    print(f"ERROR: {e}")
    print(f"Type: {type(e)}")
