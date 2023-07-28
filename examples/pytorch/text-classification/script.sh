# # LoRA
# python run_glue_lora.py \
#   --model_name_or_path roberta-large --fp16 \
#   --task_name mnli --label_names labels \
#   --do_train --per_device_train_batch_size 32 \
#   --do_eval --per_device_eval_batch_size 32 \
#   --logging_strategy steps --logging_steps 0.05 \
#   --evaluation_strategy steps --eval_steps 0.25 --save_strategy steps --save_steps 0.25 \
#   --optim adamw_torch --learning_rate 1e-3 --weight_decay 0.01 --warmup_ratio 0.05 \
#   --num_train_epochs 2 \
#   --max_seq_length 128 \
#   --output_dir ~/tmp/nli/mnli_LoRA --overwrite_output_dir \
#   --report_to wandb --run_name mnli_LoRA

# # LoRA (from paper)
# python run_glue_lora.py \
#   --model_name_or_path roberta-large --fp16 \
#   --task_name mnli \
#   --do_train --per_device_train_batch_size 4 \
#   --do_eval --per_device_eval_batch_size 4 \
#   --logging_strategy steps --logging_steps 0.01 \
#   --evaluation_strategy steps --eval_steps 0.05 --save_strategy steps --save_steps 0.05 \
#   --optim adamw_torch --learning_rate 3e-4 --weight_decay 0.01 --warmup_ratio 0.06 \
#   --num_train_epochs 10 \
#   --max_seq_length 128 \
#   --output_dir ~/tmp/nli/mnli_LoRA_paper --overwrite_output_dir \
#   --report_to wandb --run_name mnli_LoRA_paper


# # LoRA (new hyperparams)
# python run_glue_lora.py \
#   --model_name_or_path roberta-large --fp16 \
#   --task_name mnli --label_names labels \
#   --do_train --per_device_train_batch_size 32 \
#   --do_eval --per_device_eval_batch_size 32 \
#   --logging_strategy steps --logging_steps 0.025 \
#   --evaluation_strategy steps --eval_steps 0.125 --save_strategy steps --save_steps 0.125 \
#   --optim adamw_torch --learning_rate 5e-4 --weight_decay 0.01 --warmup_ratio 0.025 \
#   --num_train_epochs 4 \
#   --max_seq_length 128 \
#   --output_dir ~/tmp/nli/mnli_LoRA_2 --overwrite_output_dir \
#   --report_to wandb --run_name mnli_LoRA_2


# EVAL
python run_glue_lora.py \
  --model_name_or_path /home/chikara/tmp/nli/mnli_LoRA --fp16_full_eval \
  --task_name mnli --label_names labels \
  --do_eval --per_device_eval_batch_size 16 \
  --output_dir ~/tmp/nli/eval/full_fp16