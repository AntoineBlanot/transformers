# LoRA
python run_qa_lora.py \
  --model_name_or_path roberta-large --fp16 \
  --dataset_name squad_v2 --version_2_with_negative \
  --do_train --per_device_train_batch_size 16 \
  --do_eval --per_device_eval_batch_size 16 \
  --logging_strategy steps --logging_steps 0.05 \
  --evaluation_strategy steps --eval_steps 0.25 --save_strategy steps --save_steps 0.25 \
  --optim adamw_torch --learning_rate 1e-3 --weight_decay 0.01 --warmup_ratio 0.05 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir ~/tmp/qa/new/LoRA --overwrite_output_dir \
  --report_to wandb --run_name LoRA
