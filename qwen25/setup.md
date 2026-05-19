---
description: Build or check the Qwen2.5 C++ OpenVINO runner (from safetensors)
allowed-tools: Bash, Read
---

Build or validate the Qwen2.5 C++ runner. Parse $ARGUMENTS for subcommand: build, check, or info.

### build

```bash
cd /home/yxu28/projects/run_qwen25
./build_and_run.sh
```

### check

Verify these exist and report:
- Binary: `/home/yxu28/projects/run_qwen25/build/run_qwen25`
- Script: `/home/yxu28/projects/run_qwen25/hf_to_openvino.py`
- Library: `/home/yxu28/projects/openvino.pipeline.mx/build/cp310-cp310-linux_x86_64/src/cpp/libcomposable_pipeline.so`
- Models: `/mnt/yxu28/models/.cache/huggingface/hub/models--Qwen--Qwen2.5*`

### info

Project: `/home/yxu28/projects/run_qwen25/`
- `modeling_qwen25.hpp/cpp` — C++ Qwen2.5 model (GQA, KV-cache, RoPE, SwiGLU, RMSNorm)
- `run_qwen25.cpp` — C++ runner (loads config.json + safetensors, compiles, generates)
- `hf_to_openvino.py` — Python pipeline (download, tokenize, run)
- `CMakeLists.txt` — links against composable_pipeline + OpenVINO runtime
- Performance: CPU ~83 tok/s, GPU ~40-53 tok/s (0.5B model)

$ARGUMENTS
