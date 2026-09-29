import argparse
import hashlib
import json
import runpy
from pathlib import Path

from tokenizers import Tokenizer


def prepare(model_directory, sglang_directory, destination):
    tokenizer = Tokenizer.from_file(str(model_directory / "tokenizer.json"))
    encoder_path = (
        sglang_directory / "python/sglang/srt/entrypoints/openai/encoding_dsv41.py"
    )
    encode_messages = runpy.run_path(str(encoder_path))["encode_messages"]
    marker = "__SWA_REPLAY_PADDING__"
    padding = tokenizer.encode(
        " This is reference material for a cache performance experiment. "
        "The answer is the single letter stated at the beginning of the document. "
        * 2048,
        add_special_tokens=False,
    ).ids
    requests = []
    for request_index, answer in enumerate("ABCDEFGH"):
        prompt = encode_messages(
            [
                {
                    "role": "user",
                    "content": (
                        f"Document {request_index}: remember the answer letter {answer}. "
                        "Ignore the reference padding that follows.\n"
                        f"{marker}\n"
                        "What is the answer letter at the start of this document? "
                        "Reply with that uppercase letter only, without explanation."
                    ),
                }
            ],
            thinking_mode="chat",
        )
        prefix, suffix = prompt.split(marker)
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False).ids
        suffix_ids = tokenizer.encode(suffix, add_special_tokens=False).ids
        padding_length = 8192 - len(prefix_ids) - len(suffix_ids)
        assert 0 < padding_length <= len(padding)
        token_ids = prefix_ids + padding[:padding_length] + suffix_ids
        assert len(token_ids) == 8192
        requests.append(
            {
                "name": f"document_{request_index}",
                "cache_salt": f"swa-replay-document-{request_index}",
                "expected_answer": answer,
                "prompt_tokens": len(token_ids),
                "input_ids_sha256": hashlib.sha256(
                    json.dumps(token_ids, separators=(",", ":")).encode()
                ).hexdigest(),
                "input_ids": token_ids,
            }
        )
    dataset = {
        "batch_size": 8,
        "input_tokens_per_request": 8192,
        "output_tokens_per_request": 2,
        "thinking_mode": "chat",
        "model_directory": str(model_directory),
        "requests": requests,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n")
    print(f"Saved eight independent 8192-token requests to {destination}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-directory", type=Path, default=Path("/root/models/DeepSeek-V4.1-Flash")
    )
    parser.add_argument("--sglang-directory", type=Path, default=Path("/root/sglang"))
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "requests.json"
    )
    arguments = parser.parse_args()
    prepare(arguments.model_directory, arguments.sglang_directory, arguments.output)


if __name__ == "__main__":
    main()
