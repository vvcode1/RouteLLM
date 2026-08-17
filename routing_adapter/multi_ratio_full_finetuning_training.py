import os
import pandas as pd
import torch
import transformers
import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
    get_linear_schedule_with_warmup
)
from datetime import datetime
import shutil

def load_dataset_by_ratio(normal_count, anomaly_count, base_dataset_path):
    # Construct dataset filename based on ratio
    dataset_filename = f"lora_{normal_count}n+{anomaly_count}ab.json"
    dataset_path = os.path.join(base_dataset_path, dataset_filename)

    # Alternative naming convention
    if not os.path.exists(dataset_path):
        dataset_filename = f"lora_{anomaly_count}ab+{normal_count}n.json"
        dataset_path = os.path.join(base_dataset_path, dataset_filename)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    print(f"Loading dataset: {dataset_filename}")
    df = pd.read_json(dataset_path)
    ds = Dataset.from_pandas(df)

    print(f"Dataset loaded: {len(ds)} samples")
    print(f"Ratio: {normal_count} normal : {anomaly_count} anomaly")

    return ds, dataset_filename

def create_tokenizer(model_path):
    """Initialize and configure tokenizer."""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def create_processing_function(tokenizer, max_length=512):
    """Create data processing function for tokenization."""
    def process_func(example):
        instruction = tokenizer(
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\nYou are an Internet expert, good at judging the routing anomaly.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{example['instruction'] + example['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            add_special_tokens=False
        )
        response = tokenizer(f"{example['output']}<|eot_id|>", add_special_tokens=False)

        input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
        attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
        labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]

        # Truncate if too long
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            labels = labels[:max_length]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    return process_func

def setup_model_for_full_finetuning(model_path):
    """Load model for full fine-tuning."""
    print("Loading model for full fine-tuning...")
    print("WARNING: This will load the full model into memory and update all parameters.")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",  # Use flash attention for efficiency
    )

    # Enable gradient computation for all parameters
    for param in model.parameters():
        param.requires_grad = True

    print(f"Model loaded with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
    print(f"Model dtype: {model.dtype}")

    return model

def calculate_memory_requirements(model):
    """Calculate approximate memory requirements for training."""
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    model_memory = param_count * 2 / (1024**3)  # GB
    gradient_memory = param_count * 2 / (1024**3)  # GB
    optimizer_memory = param_count * 8 / (1024**3)  # GB

    total_memory = model_memory + gradient_memory + optimizer_memory

    print(f"\nMemory Requirements Estimate:")
    print(f"- Model parameters: {model_memory:.2f} GB")
    print(f"- Gradients: {gradient_memory:.2f} GB")
    print(f"- Optimizer states: {optimizer_memory:.2f} GB")
    print(f"- Minimum GPU memory needed: {total_memory:.2f} GB")
    print(f"- Recommended GPU memory: {total_memory * 1.5:.2f} GB (including activations and buffers)")

    return total_memory

def train_model_for_ratio(normal_count, anomaly_count, config):
    """Train model for specific normal:anomaly ratio using full fine-tuning."""

    ratio_name = f"{normal_count}n_{anomaly_count}ab"
    print(f"\n{'='*70}")
    print(f"Full Fine-tuning for Ratio: {normal_count} Normal : {anomaly_count} Anomaly")
    print(f"{'='*70}")

    try:
        # Load dataset for this ratio
        ds, dataset_filename = load_dataset_by_ratio(
            normal_count,
            anomaly_count,
            config['base_dataset_path']
        )

        # Create tokenizer
        tokenizer = create_tokenizer(config['model_path'])

        # Process dataset
        print("Tokenizing dataset...")
        process_func = create_processing_function(tokenizer, config['max_length'])
        tokenized_ds = ds.map(process_func, remove_columns=ds.column_names)

        # Setup model for full fine-tuning
        model = setup_model_for_full_finetuning(config['model_path'])

        # Calculate memory requirements
        memory_needed = calculate_memory_requirements(model)

        # Check available GPU memory
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"Available GPU memory: {gpu_memory:.2f} GB")
            if memory_needed > gpu_memory * 0.8:
                print("WARNING: Estimated memory usage exceeds 80% of available GPU memory.")
                print("Consider using gradient checkpointing or reducing batch size.")

        # Configure output directory for this ratio
        output_dir = os.path.join(config['base_output_dir'], f"full_ft_ratio_{ratio_name}")

        # Training arguments optimized for full fine-tuning
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=config['batch_size'],
            per_device_eval_batch_size=config['batch_size'],
            gradient_accumulation_steps=config['gradient_accumulation_steps'],
            logging_steps=config['logging_steps'],
            num_train_epochs=config['num_epochs'],
            save_steps=config['save_steps'],
            eval_steps=config.get('eval_steps', config['save_steps']),
            learning_rate=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0.01),
            lr_scheduler_type=config.get('lr_scheduler_type', 'cosine'),
            warmup_steps=config.get('warmup_steps', 100),
            save_total_limit=config.get('save_total_limit', 3),
            load_best_model_at_end=False,
            metric_for_best_model="loss",
            greater_is_better=False,
            save_strategy="steps",
            evaluation_strategy="no",  # Set to "steps" if you have validation data
            logging_strategy="steps",
            gradient_checkpointing=config.get('gradient_checkpointing', True),
            dataloader_pin_memory=False,  # Can cause memory issues with large models
            remove_unused_columns=True,
            label_names=["labels"],
            logging_dir=f"{output_dir}/logs",
            report_to="tensorboard" if config.get('use_tensorboard', True) else None,
            run_name=f"full_ft_{ratio_name}",
            dataloader_num_workers=0,
            fp16=False,  # Use bfloat16 instead
            bf16=True,
            tf32=True if torch.cuda.is_available() else False,
        )

        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_ds,
            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
            tokenizer=tokenizer,
        )

        # Start training
        print(f"Starting full fine-tuning for ratio {ratio_name}...")
        start_time = datetime.now()

        # Clear cache before training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        train_result = trainer.train()
        end_time = datetime.now()

        # Log results
        training_time = end_time - start_time
        print(f"Full fine-tuning completed for ratio {ratio_name}!")
        print(f"Training time: {training_time}")
        print(f"Final training loss: {train_result.training_loss:.6f}")

        # Save model and tokenizer
        print(f"Saving model to {output_dir}")
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)

        # Save model configuration
        model_config = {
            'model_type': 'full_finetuning',
            'base_model': config['model_path'],
            'ratio': f"{normal_count}:{anomaly_count}",
            'dataset_filename': dataset_filename,
            'total_samples': len(ds),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'training_args': training_args.to_dict(),
        }

        config_path = os.path.join(output_dir, 'model_config.json')
        with open(config_path, 'w') as f:
            json.dump(model_config, f, indent=2, default=str)

        # Save training summary
        summary = {
            'model_type': 'full_finetuning',
            'ratio': f"{normal_count}:{anomaly_count}",
            'dataset_filename': dataset_filename,
            'total_samples': len(ds),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'training_loss': train_result.training_loss,
            'training_time': str(training_time),
            'output_dir': output_dir,
            'timestamp': datetime.now().isoformat(),
            'memory_estimate_gb': memory_needed,
        }

        summary_path = os.path.join(output_dir, 'training_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"Training summary saved to {summary_path}")
        return summary

    except Exception as e:
        error_msg = f"Error in full fine-tuning for ratio {ratio_name}: {str(e)}"
        print(f"ERROR: {error_msg}")
        return {'ratio': f"{normal_count}:{anomaly_count}", 'error': error_msg, 'model_type': 'full_finetuning'}

def main():
    """Main training function for full fine-tuning with multiple ratios."""

    print(f"Transformers version: {transformers.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name} ({props.total_memory // 1024**3} GB)")

    # Configuration for full fine-tuning
    config = {
        'model_path': os.environ.get(
            'ROUTE_LLM_BASE_MODEL',
            'meta-llama/Meta-Llama-3.1-8B-Instruct',
        ),
        'base_dataset_path': '../models',
        'base_output_dir': './output/full_finetuning',
        'max_length': 512,  # Slightly longer for full fine-tuning
        'batch_size': 1,    # Keep small due to memory constraints
        'gradient_accumulation_steps': 8,  # Increase to maintain effective batch size
        'logging_steps': 10,
        'num_epochs': 1,    # Usually fewer epochs needed for full fine-tuning
        'save_steps': 200,  # Save more frequently
        'learning_rate': 5e-6,  # Lower learning rate for full fine-tuning
        'weight_decay': 0.01,
        'warmup_steps': 100,
        'gradient_checkpointing': True,  # Essential for memory efficiency
        'use_tensorboard': True,
        'save_total_limit': 2,  # Limit saved checkpoints to save disk space
    }


    ratio_configs = [
        (1000, 1000),  # 1:1 ratio (smaller dataset for testing)
        (3000, 3000),  # 1:1 ratio
    ]

    # Create base output directory
    os.makedirs(config['base_output_dir'], exist_ok=True)

    print(f"\n{'='*70}")
    print("STARTING FULL FINE-TUNING")
    print(f"{'='*70}")
    print("WARNING: Full fine-tuning will update ALL model parameters.")
    print("This requires significantly more GPU memory and training time than LoRA.")
    print(f"Starting full fine-tuning for {len(ratio_configs)} configurations:")

    for i, (normal, anomaly) in enumerate(ratio_configs, 1):
        print(f"{i}. Normal: {normal}, Anomaly: {anomaly} (ratio {normal}:{anomaly})")

    # Track results for all ratios
    all_results = []

    # Train model for each ratio
    for i, (normal_count, anomaly_count) in enumerate(ratio_configs, 1):
        print(f"\n{'#'*70}")
        print(f"CONFIGURATION {i}/{len(ratio_configs)} - FULL FINE-TUNING")
        print(f"{'#'*70}")

        result = train_model_for_ratio(normal_count, anomaly_count, config)
        all_results.append(result)

        # Aggressive memory cleanup between trainings
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    # Save overall results summary
    results_df = pd.DataFrame(all_results)
    results_path = os.path.join(config['base_output_dir'], 'full_finetuning_summary.csv')
    results_df.to_csv(results_path, index=False)

    print(f"\n{'='*70}")
    print("FULL FINE-TUNING COMPLETED FOR ALL RATIOS")
    print(f"{'='*70}")
    print(f"Results summary saved to: {results_path}")
    print("\nSummary:")
    if 'training_loss' in results_df.columns:
        print(results_df[['ratio', 'total_samples', 'training_loss']].to_string(index=False))
    else:
        print(results_df[['ratio', 'total_samples']].to_string(index=False))

    successful_trainings = len([r for r in all_results if 'error' not in r])
    print(f"\nSuccessful trainings: {successful_trainings}/{len(ratio_configs)}")

if __name__ == "__main__":
    main()
