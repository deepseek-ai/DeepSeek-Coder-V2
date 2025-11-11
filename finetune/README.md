## How to Fine-tune DeepSeek-Coder-V2 

We provide script `finetune_deepseekcoder.py` for users to finetune our models on downstream tasks.

The script supports the training with [DeepSpeed](https://github.com/microsoft/DeepSpeed). You need install required packages by:

```bash
pip install flash-attn --no-build-isolation
pip install -r requirements.txt
```

Please follow [Sample Dataset Format](https://huggingface.co/datasets/nickrosh/Evol-Instruct-Code-80k-v1) to prepare your training data.

You can download the sample dataset from [HuggingFace](https://huggingface.co/datasets/nickrosh/Evol-Instruct-Code-80k-v1) by:

```bash
wget https://huggingface.co/datasets/nickrosh/Evol-Instruct-Code-80k-v1/resolve/main/EvolInstruct-Code-80k.json
```

Each line is a json-serialized string with two required fields `instruction` and `output`.

After data preparation, you can use the sample shell script to finetune `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct`. Remember to specify `DATA_PATH`, `OUTPUT_PATH`.
And please choose appropriate hyper-parameters(e.g., `learning_rate`, `per_device_train_batch_size`) according to your scenario. For devices supported by flash_attention, you can refer [here](https://github.com/Dao-AILab/flash-attention).
For this configuration, zero_stage needs to be set to 3.

```bash
DATA_PATH="<your_data_path>"
OUTPUT_PATH="<your_output_path>"
MODEL_PATH="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"

deepspeed finetune_deepseekcoder.py \
    --model_name_or_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_PATH \
    --num_train_epochs 3 \
    --model_max_length 1024 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 100 \
    --learning_rate 2e-5 \
    --warmup_steps 10 \
    --logging_steps 1 \
    --lr_scheduler_type "cosine" \
    --gradient_checkpointing True \
    --report_to "tensorboard" \
    --deepspeed configs/ds_config_zero3.json \
    --bf16 True \
    --use_lora False
```

You can also finetune the model with 4/8-bits qlora, feel free to try it. For this configuration, it is possible to run on a single A100 80G GPU, and adjustments can be made according to your resources.

```bash
DATA_PATH="<your_data_path>"
OUTPUT_PATH="<your_output_path>"
MODEL_PATH="<your_model_path>"

deepspeed finetune.py \
    --model_name_or_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_PATH \
    --num_train_epochs 3 \
    --model_max_length 1024 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 100 \
    --learning_rate 2e-5 \
    --warmup_steps 10 \
    --logging_steps 1 \
    --lr_scheduler_type "cosine" \
    --gradient_checkpointing True \
    --report_to "tensorboard" \
    --deepspeed configs/ds_config_zero2_no_offload.json \
    --bf16 True \
    --use_lora True \
    --bits 4 \
    --max_grad_norm 0.3 \
    --double_quant \
    --lora_r 64 \
    --lora_alpha 16 \
    --quant_type nf4 \
```
