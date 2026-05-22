from transformers import pipeline

pipe = pipeline("image-text-to-text", model="Qwen/Qwen3.6-35B-A3B")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "/home/yxu28/projects/openvino.pipeline.mx/tests/test_data/cat_120_100.png"},
            {"type": "text", "text": "What is in this image?"}
        ]
    },
]
result = pipe(text=messages)
print(result)