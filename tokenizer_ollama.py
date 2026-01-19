"""
Ollama Tokenizer API Test
Tests tokenize and detokenize endpoints with llama3 1b model
"""

import requests
import json
import time

class OllamaTokenizer:
    def __init__(self, model="llama3.2:1b", base_url="http://127.0.0.1:11434"):
        self.model = model
        self.base_url = base_url
        self.tokenize_url = f"{base_url}/api/tokenize"
        self.detokenize_url = f"{base_url}/api/detokenize"

    def tokenize(self, text):
        """Tokenize text using Ollama API"""
        payload = {
            "model": self.model,
            "content": text
        }

        try:
            response = requests.post(
                self.tokenize_url,
                headers={"Content-Type": "application/json"},
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error during tokenization: {e}")
            return None

    def detokenize(self, tokens):
        """Detokenize tokens using Ollama API"""
        payload = {
            "model": self.model,
            "tokens": tokens
        }

        try:
            response = requests.post(
                self.detokenize_url,
                headers={"Content-Type": "application/json"},
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error during detokenization: {e}")
            return None

    def benchmark_tokenize(self, text, runs=5):
        """Benchmark tokenization performance"""
        times = []

        print(f"\nBenchmarking tokenization ({runs} runs)...")
        for i in range(runs):
            start = time.time()
            result = self.tokenize(text)
            elapsed = time.time() - start
            times.append(elapsed)
            print(f"Run {i+1}: {elapsed:.4f}s")

        avg_time = sum(times) / len(times)
        print(f"Average: {avg_time:.4f}s")
        return times


def test_basic():
    """Test basic tokenize/detokenize functionality"""
    print("=" * 60)
    print("Testing Ollama Tokenizer API with llama3:1b")
    print("=" * 60)

    tokenizer = OllamaTokenizer(model="llama3:1b")

    # Test tokenization
    test_text = "hello"
    print(f"\n1. Tokenizing: '{test_text}'")
    result = tokenizer.tokenize(test_text)

    if result:
        print(f"Result: {json.dumps(result, indent=2)}")

        # Extract tokens if available
        if "tokens" in result:
            tokens = result["tokens"]
            print(f"\nTokens: {tokens}")

            # Test detokenization
            print(f"\n2. Detokenizing: {tokens}")
            detoken_result = tokenizer.detokenize(tokens)

            if detoken_result:
                print(f"Result: {json.dumps(detoken_result, indent=2)}")
        else:
            print("Warning: 'tokens' key not found in response")
    else:
        print("Failed to tokenize. Is Ollama server running?")
        print("\nTo start Ollama server with debug logs:")
        print("  OLLAMA_DEBUG=1 OLLAMA_TOKENIZER_DEBUG=1 ollama serve")
        return

    # Test with longer text
    print("\n" + "=" * 60)
    long_text = "The quick brown fox jumps over the lazy dog"
    print(f"\n3. Tokenizing longer text: '{long_text}'")
    result = tokenizer.tokenize(long_text)

    if result:
        print(f"Result: {json.dumps(result, indent=2)}")
        if "tokens" in result:
            print(f"Token count: {len(result['tokens'])}")

    # Benchmark
    print("\n" + "=" * 60)
    tokenizer.benchmark_tokenize("hello", runs=5)


def check_ollama_version():
    """Check Ollama version"""
    try:
        response = requests.get("http://127.0.0.1:11434/api/version", timeout=2)
        if response.status_code == 200:
            version_data = response.json()
            print(f"Ollama version: {version_data}")
            return version_data
    except:
        pass
    return None


def test_with_model(model_name):
    """Test with a specific model"""
    print(f"\nTesting with model: {model_name}")
    tokenizer = OllamaTokenizer(model=model_name)

    result = tokenizer.tokenize("hello")
    if result:
        print(f"Success: {json.dumps(result, indent=2)}")
    else:
        print(f"Failed with model: {model_name}")


if __name__ == "__main__":
    # Check if server is running
    try:
        response = requests.get("http://127.0.0.1:11434/api/tags", timeout=2)
        print("Ollama server is running!")

        # Check version
        check_ollama_version()

        print(f"\nAvailable models:")
        models_data = response.json()
        if "models" in models_data:
            for model in models_data["models"]:
                print(f"  - {model.get('name', 'unknown')}")
        print()
    except requests.exceptions.RequestException:
        print("=" * 60)
        print("ERROR: Ollama server is not running!")
        print("=" * 60)
        print("\nPlease start the Ollama server first:")
        print("  ollama serve")
        print("\nOr with debug logging:")
        print("  OLLAMA_DEBUG=1 OLLAMA_TOKENIZER_DEBUG=1 ollama serve")
        print("\nMake sure you have llama3.2:1b installed:")
        print("  ollama pull llama3.2:1b")
        print("=" * 60)
        exit(1)

    # Check if tokenize endpoint is available
    print("\nChecking if tokenize API is available...")
    test_response = requests.post(
        "http://127.0.0.1:11434/api/tokenize",
        json={"model": "llama3.2:1b", "content": "test"},
        timeout=5
    )

    if test_response.status_code == 404:
        print("\n" + "=" * 60)
        print("TOKENIZE API NOT AVAILABLE")
        print("=" * 60)
        print(f"\nYour Ollama version (0.13.5) doesn't support tokenize/detokenize.")
        print("\nTo get these endpoints, you need to:")
        print("  1. Update Ollama to the latest version:")
        print("     https://ollama.com/download")
        print("\n  2. Or build from source with tokenizer support:")
        print("     https://github.com/ollama/ollama")
        print("\nCURL test commands (will work once API is available):")
        print("\n  # Tokenize")
        print("  curl -s http://127.0.0.1:11434/api/tokenize \\")
        print("    -H 'Content-Type: application/json' \\")
        print("    -d '{\"model\":\"llama3.2:1b\",\"content\":\"hello\"}'")
        print("\n  # Detokenize")
        print("  curl -s http://127.0.0.1:11434/api/detokenize \\")
        print("    -H 'Content-Type: application/json' \\")
        print("    -d '{\"model\":\"llama3.2:1b\",\"tokens\":[2050]}'")
        print("\n" + "=" * 60)
        exit(0)

    # Run tests
    print("\nTokenize API is available! Running tests...\n")
    test_basic()

    # You can also test with other models
    # test_with_model("mistral:latest")
