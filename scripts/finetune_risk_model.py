"""
Fine-tune a small LLM for credit risk classification, then apply ternary quantization.

Trains the model to take borrower history notes and output structured JSON:
  {"Risk_Flag": "Low|Medium|High", "Audit_Memo": "..."}

Usage:
    # Step 1: Generate training data (if not done already)
    python scripts/generate_risk_data.py --count 1000

    # Step 2: Fine-tune + quantize
    python scripts/finetune_risk_model.py \
        --model microsoft/Phi-3-mini-4k-instruct \
        --data data/risk_training.jsonl \
        --epochs 3 \
        --output checkpoints/risk-model-ternary

Supported base models:
    - microsoft/Phi-3-mini-4k-instruct  (3.8B, recommended)
    - google/gemma-2b-it                (2B)
    - Qwen/Qwen2-1.5B-Instruct         (1.5B, fast)
    - TinyLlama/TinyLlama-1.1B-Chat-v1.0 (1.1B, smallest)
"""

import argparse
import json
import os
import sys
import time
import logging
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── System prompt matching the lead ranking pipeline ────────────────

SYSTEM_PROMPT = (
    "You are an independent credit risk auditor. Review the borrower notes below. "
    "Summarize the borrower's 'Execution Risk' in one sentence and assign a "
    "'Risk Flag' (Low, Medium, High). "
    'Return response in JSON format with keys: "Risk_Flag" and "Audit_Memo".'
)


# ── Dataset ─────────────────────────────────────────────────────────

class RiskDataset(Dataset):
    """JSONL dataset of (borrower_notes -> risk JSON) pairs."""

    def __init__(self, path: str, tokenizer, max_length: int = 512):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line.strip())
                self.examples.append(obj)

        logger.info(f"Loaded {len(self.examples)} training examples from {path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]

        # Build prompt matching the pipeline's exact format
        user_msg = f"Borrower Notes:\n{ex['input']}"
        target = json.dumps(ex["output"], ensure_ascii=False)

        # Format as instruction-following conversation
        prompt = (
            f"<|system|>\n{SYSTEM_PROMPT}\n"
            f"<|user|>\n{user_msg}\n"
            f"<|assistant|>\n{target}"
        )

        encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Labels = input_ids (causal LM), mask padding with -100
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ── Training loop ──────────────────────────────────────────────────

def train_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    steps = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        steps += 1

        if steps % 10 == 0:
            avg = total_loss / steps
            logger.info(f"  Epoch {epoch} | Step {steps} | Loss: {avg:.4f}")

    return total_loss / max(steps, 1)


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    steps = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            total_loss += outputs.loss.item()
            steps += 1

    return total_loss / max(steps, 1)


# ── Main ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fine-tune LLM for credit risk")
    parser.add_argument("--model", default="Qwen/Qwen2-1.5B-Instruct",
                        help="HuggingFace model ID")
    parser.add_argument("--data", default="data/risk_training.jsonl",
                        help="Training data JSONL")
    parser.add_argument("--output", default="checkpoints/risk-model-ternary",
                        help="Output directory for ternary model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=0.05,
                        help="Ternary quantization threshold")
    parser.add_argument("--skip-finetune", action="store_true",
                        help="Skip fine-tuning, only apply ternary quantization")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Load model + tokenizer ──
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,  # Full precision for training
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {param_count:,} ({param_count * 4 / 1024**3:.2f} GB FP32)")

    if not args.skip_finetune:
        # ── Load dataset ──
        if not os.path.exists(args.data):
            logger.error(f"Training data not found: {args.data}")
            logger.error("Run: python scripts/generate_risk_data.py --count 1000")
            sys.exit(1)

        dataset = RiskDataset(args.data, tokenizer, args.max_length)

        # Split train/val
        val_size = int(len(dataset) * args.val_split)
        train_size = len(dataset) - val_size
        train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size)

        logger.info(f"Training: {train_size} examples, Validation: {val_size} examples")

        # ── Train ──
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        best_val_loss = float("inf")
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
            val_loss = evaluate(model, val_loader, device)
            elapsed = time.time() - t0

            logger.info(
                f"Epoch {epoch}/{args.epochs} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Time: {elapsed:.0f}s"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Save best checkpoint before quantization
                best_path = os.path.join(args.output, "best-ft")
                os.makedirs(best_path, exist_ok=True)
                model.save_pretrained(best_path)
                tokenizer.save_pretrained(best_path)
                logger.info(f"Saved best model (val_loss={val_loss:.4f})")

    # ── Apply ternary quantization ──
    from models.ternary_llm import convert_llm_to_ternary, TernaryLLM

    logger.info(f"Applying ternary quantization (threshold={args.threshold})")
    model = model.to("cpu")  # Quantize on CPU
    model = convert_llm_to_ternary(model, threshold=args.threshold)

    stats = TernaryLLM.get_stats(model)
    logger.info(f"Ternary stats:")
    logger.info(f"  Ternary layers: {stats['ternary_linear']}")
    logger.info(f"  Preserved layers: {stats['preserved_linear']}")
    logger.info(f"  FP16 size: {stats['fp16_size_mb']:.1f} MB")
    logger.info(f"  Ternary size: {stats['ternary_size_mb']:.1f} MB")
    logger.info(f"  Compression: {stats['compression_vs_fp16']:.1f}x vs FP16")

    # Save ternary model
    os.makedirs(args.output, exist_ok=True)
    ternary_path = os.path.join(args.output, "ternary_model.pt")
    TernaryLLM.save_ternary(model, ternary_path)

    # Also save tokenizer alongside
    tokenizer.save_pretrained(args.output)

    # Save metadata
    meta = {
        "base_model": args.model,
        "threshold": args.threshold,
        "epochs": args.epochs if not args.skip_finetune else 0,
        "task": "credit_risk_classification",
        "output_format": {"Risk_Flag": "Low|Medium|High", "Audit_Memo": "string"},
        **stats,
    }
    with open(os.path.join(args.output, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"\nDone! Ternary model saved to {args.output}/")
    logger.info(f"  Model: {ternary_path}")
    logger.info(f"  Size: {stats['ternary_size_mb']:.1f} MB (was {stats['fp16_size_mb']:.1f} MB FP16)")


if __name__ == "__main__":
    main()
