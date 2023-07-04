# Full fine-tuning
accelerate launch run_qa_no_trainer.py \
  --model_name_or_path roberta-base \
  --dataset_name squad_v2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --checkpointing_steps epoch \
  --output_dir ~/tmp/base_squad_v2 \
  --version_2_with_negative \
  --with_tracking --report_to wandb

# LoRa
accelerate launch run_qa_lora.py \
  --model_name_or_path roberta-base \
  --dataset_name squad_v2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --checkpointing_steps epoch \
  --output_dir ~/tmp/base_lora_squad_v2 \
  --version_2_with_negative \
  --with_tracking --report_to wandb
