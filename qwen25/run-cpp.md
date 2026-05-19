---
description: Run Qwen2.5 C++ binary directly with raw token IDs (no Python/tokenizer)
allowed-tools: Bash, Read
---

Run the Qwen2.5 C++ binary with raw token IDs. Parse $ARGUMENTS for model path, device, and token IDs.

```bash
export LD_LIBRARY_PATH="/home/yxu28/projects/openvino.pipeline.mx/build/cp310-cp310-linux_x86_64/build_ov/install/runtime/lib/intel64:/home/yxu28/projects/openvino.pipeline.mx/build/cp310-cp310-linux_x86_64/src/cpp:$LD_LIBRARY_PATH"

/home/yxu28/projects/run_qwen25/build/run_qwen25 MODEL_DIR --device DEVICE TOKEN_IDS...
```

Default model dir: `/mnt/yxu28/models/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987`

Qwen2.5 special tokens: 151643=`<|im_start|>`, 151644=`<|im_end|>`, 151645=EOS

Example:
```bash
/home/yxu28/projects/run_qwen25/build/run_qwen25 /mnt/yxu28/models/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987 --device GPU 151643 108386 198
```

$ARGUMENTS
