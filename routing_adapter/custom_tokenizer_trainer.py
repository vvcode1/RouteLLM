import os
import json
import sentencepiece as spm
from transformers import AutoTokenizer
from tqdm import tqdm
import argparse
from typing import List, Tuple

class CustomTokenizerTrainer:
    def __init__(self, model_name: str, data_path: str, output_dir: str = "./tokenizer_output"):
        self.model_name = model_name
        self.data_path = data_path
        self.output_dir = output_dir
        self.spm_training_file = os.path.join(output_dir, "spm_training_data.txt")
        self.tokenizer_prefix = os.path.join(output_dir, "custom_tokenizer")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

    def extract_training_data(self) -> None:

        print(f"Extracting training data from: {self.data_path}")

        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        dataset = []

        try:
            # Try JSONL format first
            with open(self.data_path, "r", encoding="utf-8") as fin:
                for line_num, line in enumerate(fin, 1):
                    if line.strip():  # Skip empty lines
                        try:
                            data = json.loads(line.strip())
                            # Extract instruction and output fields
                            if "instruction" in data:
                                dataset.append(data["instruction"])
                            if "input" in data and data["input"].strip():
                                dataset.append(data["input"])
                            if "output" in data:
                                dataset.append(data["output"])
                        except json.JSONDecodeError as e:
                            print(f"Warning: Skipping malformed JSON on line {line_num}: {e}")
                            continue
        except Exception:
            # If JSONL fails, try regular JSON format
            try:
                with open(self.data_path, "r", encoding="utf-8") as fin:
                    data = json.load(fin)
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict):
                                if "instruction" in item:
                                    dataset.append(item["instruction"])
                                if "input" in item and item["input"].strip():
                                    dataset.append(item["input"])
                                if "output" in item:
                                    dataset.append(item["output"])
            except Exception as e:
                raise ValueError(f"Could not parse data file as JSONL or JSON: {e}")

        # Write training data for SentencePiece
        print(f"Writing {len(dataset)} text samples to {self.spm_training_file}")
        with open(self.spm_training_file, "w", encoding="utf-8") as fout:
            for text in dataset:
                if text and text.strip():  # Skip empty texts
                    fout.write(text.strip() + "\n")

        print(f"Training data extraction completed: {len(dataset)} samples")

    def train_custom_tokenizer(self, vocab_size: int = 32000, model_type: str = "bpe",
                             character_coverage: float = 1.0,
                             user_defined_symbols: List[str] = None) -> None:

        print("Training custom SentencePiece tokenizer...")

        if user_defined_symbols is None:
            user_defined_symbols = ['<|begin_of_text|>', '<|end_of_text|>',
                                   '<|start_header_id|>', '<|end_header_id|>',
                                   '<|eot_id|>', '<|reserved_special_token|>']

        if not os.path.exists(self.spm_training_file):
            raise FileNotFoundError(f"Training data file not found: {self.spm_training_file}. Run extract_training_data() first.")

        training_args = {
            "input": self.spm_training_file,
            "model_prefix": self.tokenizer_prefix,
            "vocab_size": vocab_size,
            "model_type": model_type,
            "character_coverage": character_coverage,
            "user_defined_symbols": user_defined_symbols,
            "pad_id": 0,
            "unk_id": 1,
            "bos_id": 2,
            "eos_id": 3,
            "split_by_whitespace": True,
            "split_digits": True,
            "normalization_rule_name": "nmt_nfkc_cf",
        }

        print(f"Training parameters: {training_args}")

        spm.SentencePieceTrainer.Train(**training_args)

        print(f"Custom tokenizer training completed!")
        print(f"Model saved as: {self.tokenizer_prefix}.model")
        print(f"Vocabulary saved as: {self.tokenizer_prefix}.vocab")

    def compare_tokenizers(self, test_data_path: str = None, sample_size: int = 1000) -> Tuple[float, float]:
        print("Comparing tokenizers...")

        # Load reference tokenizer
        print(f"Loading reference tokenizer from: {self.model_name}")
        reference_tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        print(f"Reference tokenizer special tokens: {reference_tokenizer.all_special_tokens}")
        print(f"Reference tokenizer vocab size: {reference_tokenizer.vocab_size}")

        # Load custom tokenizer
        custom_model_path = f"{self.tokenizer_prefix}.model"
        if not os.path.exists(custom_model_path):
            raise FileNotFoundError(f"Custom tokenizer model not found: {custom_model_path}. Train tokenizer first.")

        sp_model = spm.SentencePieceProcessor()
        sp_model.Load(custom_model_path)
        print(f"Custom tokenizer vocab size: {sp_model.vocab_size()}")

        # Prepare test data
        test_file = test_data_path if test_data_path else self.spm_training_file

        if not os.path.exists(test_file):
            raise FileNotFoundError(f"Test data file not found: {test_file}")

        with open(test_file, "r", encoding="utf-8") as fin:
            test_texts = fin.readlines()

        # Limit sample size
        test_texts = test_texts[:sample_size]

        print(f"Testing on {len(test_texts)} samples...")

        # Compare tokenization
        reference_total_tokens = 0
        custom_total_tokens = 0
        valid_samples = 0

        for text in tqdm(test_texts, desc="Comparing tokenizers"):
            text = text.strip()
            if len(text) < 10:  # Skip very short texts
                continue

            valid_samples += 1

            # Reference tokenizer
            reference_tokens = reference_tokenizer.tokenize(text)
            reference_total_tokens += len(reference_tokens)

            # Custom tokenizer
            custom_tokens = sp_model.EncodeAsPieces(text)
            custom_total_tokens += len(custom_tokens)

        # Calculate averages
        reference_avg = reference_total_tokens / valid_samples if valid_samples > 0 else 0
        custom_avg = custom_total_tokens / valid_samples if valid_samples > 0 else 0

        # Display results
        print("\nTokenization Comparison Results:")
        print(f"{'='*50}")
        print(f"Valid test samples: {valid_samples}")
        print(f"Reference tokenizer average tokens per text: {reference_avg:.2f}")
        print(f"Custom tokenizer average tokens per text: {custom_avg:.2f}")

        if reference_avg > 0:
            efficiency_gain = ((reference_avg - custom_avg) / reference_avg) * 100
            print(f"Efficiency gain: {efficiency_gain:.2f}% {'(better)' if efficiency_gain > 0 else '(worse)'}")

        # Save comparison results
        results = {
            "reference_model": self.model_name,
            "custom_tokenizer": custom_model_path,
            "test_samples": valid_samples,
            "reference_avg_tokens": reference_avg,
            "custom_avg_tokens": custom_avg,
            "efficiency_gain_percent": efficiency_gain if reference_avg > 0 else 0
        }

        results_path = os.path.join(self.output_dir, "comparison_results.json")
        with open(results_path, "w", encoding="utf-8") as fout:
            json.dump(results, fout, indent=2)

        print(f"Comparison results saved to: {results_path}")

        return reference_avg, custom_avg

def main():
    """Main function with command line argument support."""
    parser = argparse.ArgumentParser(description="Train and compare custom tokenizers")
    parser.add_argument("--model_name", type=str,
                       default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                       help="Path to reference model for comparison")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to training data (JSON/JSONL format)")
    parser.add_argument("--output_dir", type=str, default="./tokenizer_output",
                       help="Directory to save outputs")
    parser.add_argument("--vocab_size", type=int, default=32000,
                       help="Vocabulary size for custom tokenizer")
    parser.add_argument("--model_type", type=str, default="bpe",
                       choices=["bpe", "unigram", "word", "char"],
                       help="SentencePiece model type")
    parser.add_argument("--character_coverage", type=float, default=1.0,
                       help="Character coverage for tokenizer")
    parser.add_argument("--sample_size", type=int, default=1000,
                       help="Number of samples for comparison")
    parser.add_argument("--skip_training", action="store_true",
                       help="Skip data extraction and training, only compare")
    parser.add_argument("--test_data_path", type=str, default=None,
                       help="Separate test data path for comparison")

    args = parser.parse_args()

    # Initialize trainer
    trainer = CustomTokenizerTrainer(
        model_name=args.model_name,
        data_path=args.data_path,
        output_dir=args.output_dir
    )

    try:
        if not args.skip_training:
            # Extract training data
            trainer.extract_training_data()

            # Train custom tokenizer
            trainer.train_custom_tokenizer(
                vocab_size=args.vocab_size,
                model_type=args.model_type,
                character_coverage=args.character_coverage
            )

        # Compare tokenizers
        trainer.compare_tokenizers(
            test_data_path=args.test_data_path,
            sample_size=args.sample_size
        )

        print("\nTokenizer training and comparison completed successfully!")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
