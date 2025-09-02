# DeepSeek-Coder-V2 Quantization Guide

This guide provides comprehensive information on how to quantize DeepSeek-Coder-V2 models for efficient inference with various frameworks including vLLM, SGLang, and other popular inference engines.

## Table of Contents
1. [Overview](#overview)
2. [Supported Quantization Methods](#supported-quantization-methods)
3. [Quantization with vLLM](#quantization-with-vllm)
4. [Quantization with SGLang](#quantization-with-sglang)
5. [Quantization with llama.cpp](#quantization-with-llamacpp)
6. [Quantization with AutoGPTQ](#quantization-with-autogptq)
7. [Performance Comparison](#performance-comparison)
8. [Troubleshooting](#troubleshooting)

## Overview

DeepSeek-Coder-V2 models can be quantized to reduce memory usage and improve inference speed while maintaining high code generation quality. The models support various quantization methods depending on your inference framework and requirements.

## Supported Quantization Methods

### Model Variants
- **DeepSeek-Coder-V2-Lite-Base** (16B total, 2.4B active parameters)
- **DeepSeek-Coder-V2-Lite-Instruct** (16B total, 2.4B active parameters)
- **DeepSeek-Coder-V2-Base** (236B total, 21B active parameters)
- **DeepSeek-Coder-V2-Instruct** (236B total, 21B active parameters)

### Quantization Levels
- **FP16/BF16**: Full precision (recommended for maximum quality)
- **INT8**: 8-bit quantization (good balance of speed and quality)
- **INT4**: 4-bit quantization (maximum compression, some quality loss)
- **FP8**: 8-bit floating point (supported by SGLang)

## Quantization with vLLM

### Prerequisites
```bash
# Install vLLM with DeepSeek-Coder-V2 support
pip install vllm
# Note: You need to merge PR #4650 for full support
# https://github.com/vllm-project/vllm/pull/4650
```

### Basic Quantization Example
```python
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Load model with quantization
model_name = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Quantize to INT8
llm = LLM(
    model=model_name,
    tensor_parallel_size=1,
    max_model_len=8192,
    trust_remote_code=True,
    enforce_eager=True,
    quantization="int8"  # or "int4", "fp16", "bf16"
)

# Use the model
sampling_params = SamplingParams(
    temperature=0.3, 
    max_tokens=256, 
    stop_token_ids=[tokenizer.eos_token_id]
)

messages = [{"role": "user", "content": "Write a Python function to calculate fibonacci numbers."}]
prompt_token_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
outputs = llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)
print(outputs[0].outputs[0].text)
```

### Advanced vLLM Quantization Options
```python
# INT4 quantization with specific settings
llm = LLM(
    model=model_name,
    tensor_parallel_size=2,
    max_model_len=8192,
    trust_remote_code=True,
    enforce_eager=True,
    quantization="int4",
    gpu_memory_utilization=0.9,
    max_num_batched_tokens=4096
)
```

## Quantization with SGLang

SGLang provides excellent support for DeepSeek-Coder-V2 with built-in quantization optimizations.

### FP8 Quantization (Recommended)
```bash
# Install SGLang
pip install sglang

# Launch server with FP8 quantization
python3 -m sglang.launch_server \
    --model neuralmagic/DeepSeek-Coder-V2-Instruct-FP8 \
    --tp 8 \
    --trust-remote-code \
    --kv-cache-dtype fp8_e5m2
```

### BF16 with Torch Compile
```bash
# BF16 with compilation for better performance
python3 -m sglang.launch_server \
    --model deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct \
    --trust-remote-code \
    --enable-torch-compile
```

### Python Client Example
```python
import openai

client = openai.Client(
    base_url="http://127.0.0.1:30000/v1", 
    api_key="EMPTY"
)

response = client.chat.completions.create(
    model="default",
    messages=[
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Write a quicksort implementation in Python."},
    ],
    temperature=0,
    max_tokens=512,
)
print(response.choices[0].message.content)
```

## Quantization with llama.cpp

For CPU inference or maximum compatibility, you can use llama.cpp with GGUF format.

### Convert to GGUF Format
```bash
# Install llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

# Convert model to GGUF format
python3 convert.py deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct \
    --outfile deepseek-coder-v2-lite-instruct.gguf \
    --outtype q4_0
```

### Quantization Levels Available
- `q4_0`: 4-bit quantization (recommended)
- `q4_1`: 4-bit quantization with improved accuracy
- `q5_0`: 5-bit quantization
- `q5_1`: 5-bit quantization with improved accuracy
- `q8_0`: 8-bit quantization

### Inference with llama.cpp
```bash
# Run inference with quantized model
./main -m deepseek-coder-v2-lite-instruct.gguf \
    -n 512 \
    -p "Write a Python function to sort a list:" \
    --repeat_penalty 1.1
```

## Quantization with AutoGPTQ

For GPU inference with GPTQ quantization.

### Installation
```bash
pip install auto-gptq
pip install transformers
```

### Quantization Example
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

# Load model and tokenizer
model_name = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Configure quantization
quantize_config = BaseQuantizeConfig(
    bits=4,  # 4-bit quantization
    group_size=128,
    desc_act=False,
)

# Quantize the model
model = AutoGPTQForCausalLM.from_pretrained(
    model_name,
    quantize_config=quantize_config,
    trust_remote_code=True,
    device_map="auto"
)

# Save quantized model
model.save_quantized("deepseek-coder-v2-lite-instruct-gptq")

# Load and use quantized model
quantized_model = AutoGPTQForCausalLM.from_quantized(
    "deepseek-coder-v2-lite-instruct-gptq",
    device_map="auto",
    trust_remote_code=True
)
```

## Performance Comparison

| Quantization Method | Memory Usage | Speed | Quality | Framework |
|-------------------|-------------|-------|--------|-----------|
| FP16/BF16 | 100% | 100% | 100% | All |
| INT8 | ~50% | ~120% | ~95% | vLLM, SGLang |
| INT4 | ~25% | ~150% | ~90% | vLLM, llama.cpp |
| FP8 | ~50% | ~130% | ~98% | SGLang |
| GPTQ-4bit | ~25% | ~140% | ~88% | AutoGPTQ |

*Note: Performance metrics are approximate and may vary based on hardware and specific use cases.*

## Memory Requirements

### GPU Memory (per GPU)
- **FP16/BF16**: 
  - Lite models: ~32GB
  - Full models: ~84GB
- **INT8**: 
  - Lite models: ~16GB
  - Full models: ~42GB
- **INT4**: 
  - Lite models: ~8GB
  - Full models: ~21GB

### CPU Memory
- **GGUF q4_0**: 
  - Lite models: ~8GB
  - Full models: ~21GB

## Troubleshooting

### Common Issues

1. **"Model not found" error**
   ```bash
   # Ensure you're using the correct model name
   model_name = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
   ```

2. **Out of memory errors**
   ```python
   # Reduce tensor parallelism or use quantization
   llm = LLM(
       model=model_name,
       tensor_parallel_size=1,  # Reduce from 8 to 1
       quantization="int8"
   )
   ```

3. **vLLM compatibility issues**
   ```bash
   # Check if you have the latest vLLM with DeepSeek support
   pip install --upgrade vllm
   # Or use the PR branch: https://github.com/vllm-project/vllm/pull/4650
   ```

4. **Tokenization errors**
   ```python
   # Always use trust_remote_code=True
   tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
   ```

### Performance Optimization Tips

1. **Use tensor parallelism for large models**
   ```python
   tensor_parallel_size=8  # For 236B models
   ```

2. **Enable torch.compile with SGLang**
   ```bash
   --enable-torch-compile
   ```

3. **Optimize batch size**
   ```python
   max_num_batched_tokens=4096  # Adjust based on your GPU memory
   ```

4. **Use appropriate quantization for your use case**
   - **Maximum quality**: FP16/BF16
   - **Balanced**: INT8/FP8
   - **Maximum speed**: INT4

## Additional Resources

- [vLLM Documentation](https://docs.vllm.ai/)
- [SGLang Documentation](https://docs.sgl.ai/)
- [llama.cpp Documentation](https://github.com/ggerganov/llama.cpp)
- [AutoGPTQ Documentation](https://github.com/PanQiWei/AutoGPTQ)
- [DeepSeek-Coder-V2 Paper](https://arxiv.org/pdf/2406.11931)

## Support

If you encounter issues with quantization, please:
1. Check the troubleshooting section above
2. Search existing issues: https://github.com/deepseek-ai/DeepSeek-Coder-V2/issues
3. Create a new issue with detailed error information
4. Contact the team at service@deepseek.com
