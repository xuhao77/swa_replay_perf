import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def capture(command):
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    sglang_directory = Path(os.environ.get("SGLANG_DIRECTORY", "/root/sglang"))
    model_directory = Path(
        os.environ.get("MODEL_DIRECTORY", "/root/models/DeepSeek-V4.1-Flash")
    )
    flexkv_directory = Path(os.environ.get("FLEXKV_DIRECTORY", "/root/FlexKV"))
    metadata = {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "sglang_directory": str(sglang_directory),
        "model_directory": str(model_directory),
        "git_head": capture(["git", "-C", str(sglang_directory), "rev-parse", "HEAD"]),
        "git_status": capture(
            ["git", "-C", str(sglang_directory), "status", "--short"]
        ),
        "flexkv_git_head": capture(
            ["git", "-C", str(flexkv_directory), "rev-parse", "HEAD"]
        ),
        "flexkv_git_status": capture(
            ["git", "-C", str(flexkv_directory), "status", "--short"]
        ),
        "gpu": capture(["nvidia-smi", "-q"]),
        "topology": capture(["nvidia-smi", "topo", "-m"]),
        "cpu": capture(["lscpu"]),
        "memory": capture(["free", "-h"]),
        "packages": {
            distribution.metadata["Name"]: distribution.version
            for distribution in importlib.metadata.distributions()
            if distribution.metadata["Name"]
        },
        "source_sha256": {},
    }
    source_paths = [
        sglang_directory / "python/sglang/srt/model_executor/encoder_swa_replay.py",
        sglang_directory / "python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py",
        sglang_directory / "python/sglang/srt/mem_cache/dsv41_request_window.py",
        sglang_directory / "python/sglang/srt/layers/attention/deepseek_v4_backend.py",
        sglang_directory / "python/sglang/srt/managers/schedule_batch.py",
        sglang_directory / "python/sglang/srt/managers/schedule_policy.py",
        sglang_directory / "python/sglang/srt/arg_groups/deepseek_v4_hook.py",
        sglang_directory / "python/sglang/kernels/ops/attention/dsv4/kv_layout.py",
        model_directory / "config.json",
        model_directory / "tokenizer.json",
        sglang_directory / "python/sglang/srt/mem_cache/unified_radix_cache.py",
        sglang_directory
        / "python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py",
        sglang_directory
        / "python/sglang/srt/mem_cache/storage/flexkv/flexkv_hybrid_radix_cache.py",
        flexkv_directory / "flexkv/integration/sglang/connector.py",
    ]
    source_paths.extend(Path(__file__).parent.glob("*.py"))
    source_paths.extend(Path(__file__).parent.glob("*.sh"))
    source_paths.append(Path(__file__).with_name("flexkv_host.yaml"))
    distribution = importlib.metadata.distribution("sglang-kernel")
    for relative_path in distribution.files or []:
        if "flashmla_ops" in str(relative_path) or str(relative_path).endswith(
            "flash_mla.py"
        ):
            source_paths.append(Path(distribution.locate_file(relative_path)))
    flashinfer_version = importlib.metadata.version("flashinfer-python")
    source_paths.append(
        Path.home()
        / ".cache/sglang/.cache/flashinfer"
        / flashinfer_version
        / "100a/cached_ops/fused_moe_trtllm_sm100/fused_moe_trtllm_sm100.so"
    )
    for path in source_paths:
        if path.is_file():
            with path.open("rb") as source:
                metadata["source_sha256"][str(path)] = hashlib.file_digest(
                    source, "sha256"
                ).hexdigest()
    (arguments.output / "environment.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    shutil.copyfile(
        model_directory / "config.json", arguments.output / "model_config.json"
    )
    (arguments.output / "packages.txt").write_text(
        "\n".join(
            f"{name}=={version}"
            for name, version in sorted(metadata["packages"].items())
        )
        + "\n"
    )
    print(f"Saved environment and source fingerprints to {arguments.output}")


if __name__ == "__main__":
    main()
