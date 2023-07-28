# # LoRA
# python run_qa_lora.py \
#   --model_name_or_path roberta-large --fp16 \
#   --dataset_name squad_v2 --version_2_with_negative \
#   --do_train --per_device_train_batch_size 16 \
#   --do_eval --per_device_eval_batch_size 16 \
#   --logging_strategy steps --logging_steps 0.05 \
#   --evaluation_strategy steps --eval_steps 0.25 --save_strategy steps --save_steps 0.25 \
#   --optim adamw_torch --learning_rate 1e-3 --weight_decay 0.01 --warmup_ratio 0.05 \
#   --num_train_epochs 2 \
#   --max_seq_length 384 \
#   --doc_stride 128 \
#   --output_dir ~/tmp/qa/squadv2_LoRA --overwrite_output_dir \
#   --report_to wandb --run_name squadv2_LoRA

# # LoRA (new hyperparams)
# python run_qa_lora.py \
#   --model_name_or_path roberta-large --fp16 \
#   --dataset_name squad_v2 --version_2_with_negative \
#   --do_train --per_device_train_batch_size 16 \
#   --do_eval --per_device_eval_batch_size 16 \
#   --logging_strategy steps --logging_steps 0.025 \
#   --evaluation_strategy steps --eval_steps 0.125 --save_strategy steps --save_steps 0.125 \
#   --optim adamw_torch --learning_rate 5e-4 --weight_decay 0.01 --warmup_ratio 0.025 \
#   --num_train_epochs 4 \
#   --max_seq_length 384 \
#   --doc_stride 128 \
#   --output_dir ~/tmp/qa/squadv2_LoRA_2 --overwrite_output_dir \
#   --report_to wandb --run_name squadv2_LoRA_2


# # LoRA (new hyperparams)
# python run_qa_lora.py \
#   --model_name_or_path roberta-large --fp16 \
#   --dataset_name squad_v2 --version_2_with_negative \
#   --do_train --per_device_train_batch_size 4 \
#   --do_eval --per_device_eval_batch_size 4 \
#   --logging_strategy steps --logging_steps 0.05 \
#   --evaluation_strategy steps --eval_steps 0.25 --save_strategy steps --save_steps 0.25 \
#   --optim adamw_torch --learning_rate 3e-4 --weight_decay 0.01 --warmup_ratio 0.05 \
#   --num_train_epochs 2 \
#   --max_seq_length 384 \
#   --doc_stride 128 \
#   --output_dir ~/tmp/qa/squadv2_LoRA_3 --overwrite_output_dir \
#   --report_to wandb --run_name squadv2_LoRA_3

# # LoRA (new hyperparams)
# python run_qa_lora.py \
#   --model_name_or_path roberta-large --fp16 \
#   --dataset_name squad_v2 --version_2_with_negative \
#   --do_train --per_device_train_batch_size 16 \
#   --do_eval --per_device_eval_batch_size 16 \
#   --logging_strategy steps --logging_steps 0.05 \
#   --evaluation_strategy steps --eval_steps 0.25 --save_strategy steps --save_steps 0.25 \
#   --optim adamw_torch --learning_rate 1e-3 --weight_decay 0 --warmup_ratio 0.05 \
#   --num_train_epochs 2 \
#   --max_seq_length 384 \
#   --doc_stride 128 \
#   --output_dir ~/tmp/qa/squadv2_LoRA_4 --overwrite_output_dir \
#   --report_to wandb --run_name squadv2_LoRA_4

# EVAL
python run_qa_lora.py \
  --model_name_or_path /home/chikara/tmp/qa/squadv2_LoRA --fp16_full_eval \
  --dataset_name squad_v2 --version_2_with_negative \
  --do_eval --per_device_eval_batch_size 16 \
  --output_dir ~/tmp/qa/test