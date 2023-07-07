# # Full fine-tuning
# python run_qa.py \
#   --model_name_or_path roberta-base \
#   --dataset_name squad_v2 --version_2_with_negative \
#   --do_train --per_device_train_batch_size 32 \
#   --do_eval --per_device_eval_batch_size 32 --evaluation_strategy epoch \
#   --logging_strategy steps --logging_steps 0.05 --save_strategy epoch \
#   --optim adamw_torch --learning_rate 5e-5 --warmup_ratio 0.05 \
#   --num_train_epochs 2 \
#   --max_seq_length 384 \
#   --doc_stride 128 \
#   --output_dir ~/tmp/qa/base_squad_v2 --overwrite_output_dir \
#   --report_to wandb --run_name base_qa

# # LoRA
# python run_qa_lora.py \
#   --model_name_or_path roberta-base \
#   --dataset_name squad_v2 --version_2_with_negative \
#   --do_train --per_device_train_batch_size 32 \
#   --do_eval --per_device_eval_batch_size 32 --evaluation_strategy epoch \
#   --logging_strategy steps --logging_steps 0.05 --save_strategy epoch \
#   --optim adamw_torch --learning_rate 1e-3 --warmup_ratio 0.05 \
#   --num_train_epochs 2 \
#   --max_seq_length 384 \
#   --doc_stride 128 \
#   --output_dir ~/tmp/qa/base_lora_squad_v2 --overwrite_output_dir \
#   --report_to wandb --run_name base_qa_lora

# # Full fine-tuning
# python run_qa.py \
#   --model_name_or_path roberta-large \
#   --dataset_name squad_v2 --version_2_with_negative \
#   --do_train --per_device_train_batch_size 8 \
#   --do_eval --per_device_eval_batch_size 8 --evaluation_strategy epoch \
#   --logging_strategy steps --logging_steps 0.05 --save_strategy epoch \
#   --optim adamw_torch --learning_rate 5e-5 --warmup_ratio 0.05 \
#   --num_train_epochs 2 \
#   --max_seq_length 384 \
#   --doc_stride 128 \
#   --output_dir ~/tmp/qa/large_squad_v2 --overwrite_output_dir \
#   --report_to wandb --run_name large_qa

# # LoRA
# python run_qa_lora.py \
#   --model_name_or_path roberta-large \
#   --dataset_name squad_v2 --version_2_with_negative \
#   --do_train --per_device_train_batch_size 32 \
#   --do_eval --per_device_eval_batch_size 32 --evaluation_strategy epoch \
#   --logging_strategy steps --logging_steps 0.05 --save_strategy epoch \
#   --optim adamw_torch --learning_rate 1e-3 --warmup_ratio 0.05 \
#   --num_train_epochs 2 \
#   --max_seq_length 384 \
#   --doc_stride 128 \
#   --output_dir ~/tmp/qa/large_lora_squad_v2 --overwrite_output_dir \
#   --report_to wandb --run_name large_qa_lora

# # Full fine-tuning with RoBERTa's paper hyperparams
# python run_qa.py \
#   --model_name_or_path roberta-large \
#   --dataset_name squad_v2 --version_2_with_negative \
#   --do_train --per_device_train_batch_size 8 --gradient_accumulation_steps 6 \
#   --do_eval --per_device_eval_batch_size 8 --evaluation_strategy epoch \
#   --logging_strategy steps --logging_steps 0.05 --save_strategy epoch \
#   --optim adamw_torch --learning_rate 1.5e-5 --weight_decay 0.01 --warmup_ratio 0.06 \
#   --num_train_epochs 2 \
#   --max_seq_length 384 \
#   --doc_stride 128 \
#   --output_dir ~/tmp/qa/large_fairseq_squad_v2 --overwrite_output_dir \
#   --report_to wandb --run_name large_fairseq_qa

# # Full fine-tuning with RoBERTa's paper hyperparams (modified)
# python run_qa.py \
#   --model_name_or_path roberta-large \
#   --dataset_name squad_v2 --version_2_with_negative \
#   --do_train --per_device_train_batch_size 8 \
#   --do_eval --per_device_eval_batch_size 8 --evaluation_strategy epoch \
#   --logging_strategy steps --logging_steps 0.05 --save_strategy epoch \
#   --optim adamw_torch --learning_rate 1e-5 --weight_decay 0.01 --warmup_ratio 0.025 \
#   --num_train_epochs 4 \
#   --max_seq_length 384 \
#   --doc_stride 128 \
#   --output_dir ~/tmp/qa/large_best_squad_v2 --overwrite_output_dir \
#   --report_to wandb --run_name large_best_qa

# # LoRA with RoBERTa's paper hyperparams (modified)
# python run_qa_lora.py \
#   --model_name_or_path roberta-large \
#   --dataset_name squad_v2 --version_2_with_negative \
#   --do_train --per_device_train_batch_size 32 \
#   --do_eval --per_device_eval_batch_size 32 --evaluation_strategy epoch \
#   --logging_strategy steps --logging_steps 0.05 --save_strategy epoch \
#   --optim adamw_torch --learning_rate 1e-3 --weight_decay 0.01 --warmup_ratio 0.025 \
#   --num_train_epochs 4 \
#   --max_seq_length 384 \
#   --doc_stride 128 \
#   --output_dir ~/tmp/qa/large_lora_best_squad_v2 --overwrite_output_dir \
#   --report_to wandb --run_name large_best_qa_lora

# LoRA with RoBERTa's paper hyperparams (modified)
python run_qa_lora.py \
  --model_name_or_path roberta-large \
  --dataset_name squad_v2 --version_2_with_negative \
  --do_train --per_device_train_batch_size 4 \
  --do_eval --per_device_eval_batch_size 4 --evaluation_strategy epoch \
  --logging_strategy steps --logging_steps 0.05 --save_strategy epoch \
  --optim adamw_torch --learning_rate 3e-4 --weight_decay 0 --warmup_ratio 0.06 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir ~/tmp/qa/large_lora_best_2_squad_v2 --overwrite_output_dir \
  --report_to wandb --run_name large_best_2_qa_lora

# LoRA with RoBERTa's paper hyperparams (modified)
python run_qa_lora.py \
  --model_name_or_path roberta-large \
  --dataset_name squad_v2 --version_2_with_negative \
  --do_train --per_device_train_batch_size 48 \
  --do_eval --per_device_eval_batch_size 48 --evaluation_strategy epoch \
  --logging_strategy steps --logging_steps 0.05 --save_strategy epoch \
  --optim adamw_torch --learning_rate 1e-4 --weight_decay 0 --warmup_ratio 0.06 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir ~/tmp/qa/large_lora_no_decay_squad_v2 --overwrite_output_dir \
  --report_to wandb --run_name large_lora_no_decay

# LoRA with RoBERTa's paper hyperparams (modified)
python run_qa_lora.py \
  --model_name_or_path roberta-large \
  --dataset_name squad_v2 --version_2_with_negative \
  --do_train --per_device_train_batch_size 48 \
  --do_eval --per_device_eval_batch_size 48 --evaluation_strategy epoch \
  --logging_strategy steps --logging_steps 0.05 --save_strategy epoch \
  --optim adamw_torch --learning_rate 1e-4 --weight_decay 0.01 --warmup_ratio 0.06 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir ~/tmp/qa/large_lora_decay_squad_v2 --overwrite_output_dir \
  --report_to wandb --run_name large_lora_decay

# LoRA with RoBERTa's paper hyperparams (modified)
python run_qa_lora.py \
  --model_name_or_path roberta-large \
  --dataset_name squad_v2 --version_2_with_negative \
  --do_train --per_device_train_batch_size 48 \
  --do_eval --per_device_eval_batch_size 48 --evaluation_strategy epoch \
  --logging_strategy steps --logging_steps 0.05 --save_strategy epoch \
  --optim adamw_torch --learning_rate 1e-4 --weight_decay 0 --warmup_ratio 0.06 \
  --num_train_epochs 5 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir ~/tmp/qa/large_lora_long_squad_v2 --overwrite_output_dir \
  --report_to wandb --run_name large_lora_long

