import os
import pandas as pd
import torch
import transformers
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
    GenerationConfig
)
from peft import LoraConfig, TaskType, get_peft_model
from datetime import datetime

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

def create_processing_function(tokenizer, max_length=384):
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

def setup_lora_model(model_path, lora_config=None):
    """Load model and apply LoRA configuration."""
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    model.enable_input_require_grads()

    if lora_config is None:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1
        )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model

def train_model_for_ratio(normal_count, anomaly_count, config):
    """Train model for specific normal:anomaly ratio."""

    ratio_name = f"{normal_count}n_{anomaly_count}ab"
    print(f"\n{'='*60}")
    print(f"Training Model for Ratio: {normal_count} Normal : {anomaly_count} Anomaly")
    print(f"{'='*60}")

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

        # Setup model with LoRA
        model = setup_lora_model(config['model_path'], config.get('lora_config'))

        # Configure output directory for this ratio
        output_dir = os.path.join(config['base_output_dir'], f"ratio_{ratio_name}")

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=config['batch_size'],
            gradient_accumulation_steps=config['gradient_accumulation_steps'],
            logging_steps=config['logging_steps'],
            num_train_epochs=config['num_epochs'],
            save_steps=config['save_steps'],
            learning_rate=config['learning_rate'],
            save_on_each_node=True,
            gradient_checkpointing=False,
            logging_dir=f"{output_dir}/logs",
            report_to="tensorboard" if config.get('use_tensorboard', True) else None,
            run_name=f"lora_training_{ratio_name}",
        )

        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_ds,
            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        )

        # Start training
        print(f"Starting training for ratio {ratio_name}...")
        start_time = datetime.now()
        train_result = trainer.train()
        end_time = datetime.now()

        # Log results
        training_time = end_time - start_time
        print(f"Training completed for ratio {ratio_name}!")
        print(f"Training time: {training_time}")
        print(f"Final training loss: {train_result.training_loss:.6f}")

        # Save model and tokenizer
        print(f"Saving model to {output_dir}")
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)

        # Save training summary
        summary = {
            'ratio': f"{normal_count}:{anomaly_count}",
            'dataset_filename': dataset_filename,
            'total_samples': len(ds),
            'training_loss': train_result.training_loss,
            'training_time': str(training_time),
            'output_dir': output_dir,
            'timestamp': datetime.now().isoformat()
        }

        summary_path = os.path.join(output_dir, 'training_summary.json')
        pd.Series(summary).to_json(summary_path, indent=2)

        print(f"Training summary saved to {summary_path}")
        return summary

    except Exception as e:
        error_msg = f"Error training model for ratio {ratio_name}: {str(e)}"
        print(f"ERROR: {error_msg}")
        return {'ratio': f"{normal_count}:{anomaly_count}", 'error': error_msg}

def main():
    """Main training function for multiple ratios."""

    print(f"Transformers version: {transformers.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # Configuration
    config = {
        'model_path': os.environ.get(
            'ROUTE_LLM_BASE_MODEL',
            'meta-llama/Meta-Llama-3.1-8B-Instruct',
        ),
        'base_dataset_path': '../models',
        'base_output_dir': './output/multi_ratio_lora',
        'max_length': 384,
        'batch_size': 1,
        'gradient_accumulation_steps': 2,
        'logging_steps': 10,
        'num_epochs': 1,
        'save_steps': 500,
        'learning_rate': 5e-5,
        'use_tensorboard': True,
    }

    # Define different normal:anomaly ratios to test
    ratio_configs = [
        (3000, 3000),  # 1:1 ratio
        # 10:1 ratio
        # 100:1 ratio
        # 1000:1 ratio
         # 10000:1 ratio
    ]

    # Create base output directory
    os.makedirs(config['base_output_dir'], exist_ok=True)

    print(f"\nStarting multi-ratio LoRA training for {len(ratio_configs)} configurations:")
    for i, (normal, anomaly) in enumerate(ratio_configs, 1):
        print(f"{i}. Normal: {normal}, Anomaly: {anomaly} (ratio {normal}:{anomaly})")

    # Track results for all ratios
    all_results = []

    # Train model for each ratio
    for i, (normal_count, anomaly_count) in enumerate(ratio_configs, 1):
        print(f"\n{'#'*70}")
        print(f"CONFIGURATION {i}/{len(ratio_configs)}")
        print(f"{'#'*70}")

        result = train_model_for_ratio(normal_count, anomaly_count, config)
        all_results.append(result)

        # Clean up GPU memory between trainings
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save overall results summary
    results_df = pd.DataFrame(all_results)
    results_path = os.path.join(config['base_output_dir'], 'all_ratios_summary.csv')
    results_df.to_csv(results_path, index=False)

    print(f"\n{'='*70}")
    print("TRAINING COMPLETED FOR ALL RATIOS")
    print(f"{'='*70}")
    print(f"Results summary saved to: {results_path}")
    print("\nSummary:")
    print(results_df[['ratio', 'total_samples', 'training_loss']].to_string(index=False))

    successful_trainings = len([r for r in all_results if 'error' not in r])
    print(f"\nSuccessful trainings: {successful_trainings}/{len(ratio_configs)}")

if __name__ == "__main__":
    main()
