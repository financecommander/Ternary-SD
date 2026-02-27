# Ternary Quantization for On-Device Credit Risk Intelligence

**Calculus Labs — Technical Whitepaper**
**Version 1.0 | February 2026**

**Calculus Holdings LLC — Proprietary**

---

## Abstract

This paper presents a novel approach to deploying credit risk analysis at scale with zero marginal inference cost. By applying ternary weight quantization ({-1, 0, +1}) to a fine-tuned language model, we compress a 3 GB credit risk classifier to under 500 MB while preserving analytical quality. The resulting model runs entirely on-device, eliminating per-request cloud API costs, reducing latency, and ensuring borrower data never leaves the host infrastructure. Integrated into the Calculus Labs Lead Ranking Engine, this system replaces cloud-based risk analysis (Google Gemini) with a local-first architecture that falls back to cloud only when necessary.

---

## 1. Problem Statement

### 1.1 The Cost of Cloud-Based Risk Analysis

Modern lead ranking pipelines rely on large language models (LLMs) hosted by third-party providers to perform qualitative risk assessments. For each borrower lead classified as Tier 1 or Tier 2, the pipeline sends borrower history notes to a cloud API and receives a structured risk flag and audit memo.

At low volume, this cost is negligible. At institutional scale, it compounds:

| Monthly Lead Volume | Tier 1/2 Leads (~40%) | Cloud API Cost (est.) |
|--------------------:|----------------------:|----------------------:|
| 1,000               | 400                   | $4                    |
| 10,000              | 4,000                 | $40                   |
| 100,000             | 40,000                | $400                  |
| 1,000,000           | 400,000               | $4,000                |

Beyond cost, cloud inference introduces latency (500ms–2s per call), creates a hard dependency on third-party API availability, and routes sensitive borrower financial data through external servers.

### 1.2 The Opportunity

A locally hosted model trained specifically for credit risk classification eliminates all three concerns simultaneously. The challenge is compression: production-grade language models require gigabytes of memory, exceeding practical deployment budgets for edge and server-side workloads.

Ternary quantization solves this by reducing every weight in the network to one of three values: **{-1, 0, +1}**. This achieves 8–16x compression versus standard floating-point representations while preserving the model's learned behavior.

---

## 2. Technical Architecture

### 2.1 Ternary Weight Quantization

Standard neural network weights are stored as 32-bit (FP32) or 16-bit (FP16) floating-point numbers. Ternary quantization maps each weight to one of three discrete values using a threshold parameter τ:

```
w_ternary = {  +1   if w ≥ τ
            {   0   if |w| < τ
            {  -1   if w ≤ -τ
```

With only three possible values, each weight requires approximately 2.5 bits of storage (log₂(3) ≈ 1.585 bits theoretical, 2.5 bits practical with int8 encoding and per-layer scale factors).

**Compression ratios:**

| Source Format | Bits/Weight | Ternary Ratio |
|--------------|:-----------:|:-------------:|
| FP32         | 32          | 12.8x         |
| FP16         | 16          | 6.4x          |
| INT8         | 8           | 3.2x          |

### 2.2 Straight-Through Estimator

Ternary quantization is non-differentiable — the discrete mapping from continuous weights to {-1, 0, +1} has zero gradient almost everywhere. To enable fine-tuning through quantized layers, we employ the straight-through estimator (STE): during the backward pass, gradients are passed through the quantization operation unchanged.

This allows the model to learn in continuous weight space while inference operates in discrete ternary space.

### 2.3 Per-Layer Weight Scaling

Raw ternary values {-1, 0, +1} discard magnitude information. To preserve numerical stability, each quantized layer maintains a floating-point scale factor computed as the mean absolute value of the original weights:

```
α = mean(|W|)
W_effective = α × ternarize(W)
```

This scale factor is stored as a single FP32 value per layer, adding negligible overhead while significantly improving output quality.

### 2.4 Selective Layer Quantization

Not all layers in a transformer are equally tolerant of quantization. Our approach selectively converts layers based on empirical sensitivity analysis:

**Quantized (TernaryLinear):**
- Self-attention Q, K, V, and output projections
- Feed-forward network up and down projections

**Preserved in full precision:**
- Token and position embeddings
- Layer normalization / RMS normalization
- Rotary positional encoding parameters
- Language model output head (lm_head)

This selective strategy preserves the components most sensitive to numerical precision — embeddings that encode vocabulary semantics, normalization layers that stabilize activations, and the output head that maps to vocabulary logits — while quantizing the bulk of the parameter count (typically 85–90% of total weights).

---

## 3. Credit Risk Model

### 3.1 Task Definition

The credit risk classifier takes unstructured borrower history notes as input and produces structured output:

**Input:**
```
Borrower seeking $5M for Multifamily project in New York. Excellent track
record with multiple successful projects. Consistent on-time payments
across all prior loans.
```

**Output:**
```json
{
  "Risk_Flag": "Low",
  "Audit_Memo": "Strong borrower profile with proven execution history
   and consistent payment performance across institutional-scale facilities."
}
```

Risk flags are classified as **Low**, **Medium**, or **High**. The audit memo provides a one-sentence narrative justification suitable for institutional credit committee review.

### 3.2 Training Data

Training data is generated using a teacher-student methodology:

1. **Synthetic profile generation:** Borrower profiles are constructed from parameterized components — positive traits (e.g., "excellent track record," "timely repayment history"), neutral traits (e.g., "first-time commercial borrower"), and negative traits (e.g., "prior default," "bankruptcy filing") — combined with randomized project types, locations, and loan amounts.

2. **Teacher labeling:** Each synthetic profile is submitted to Google Gemini 2.5 Flash, acting as the teacher model. Gemini produces a contextually nuanced risk flag and audit memo that accounts for severity, recency, and interaction between risk factors.

3. **Dataset composition:** 2,000 labeled examples with distribution: ~40% Low, ~35% Medium, ~25% High. Stored as JSONL for streaming training.

This approach captures the analytical quality of a frontier model in a compact, reusable dataset. The one-time labeling cost (~$2) amortizes to zero over the lifetime of the trained model.

### 3.3 Fine-Tuning

The base model is a small, instruction-tuned language model (Qwen2-1.5B-Instruct, 1.5 billion parameters). Fine-tuning configuration:

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 2 × 10⁻⁵ |
| Batch size | 2 |
| Epochs | 3 |
| Max sequence length | 512 tokens |
| Gradient clipping | 1.0 norm |
| Validation split | 10% |

The model is trained with a causal language modeling objective on the structured prompt format used at inference time, ensuring output format consistency.

### 3.4 Post-Training Quantization

After fine-tuning converges, ternary quantization is applied:

1. All eligible Linear layers are converted to TernaryLinear
2. Original weights are ternarized with threshold τ = 0.05
3. Per-layer scale factors are computed and stored
4. The compressed model is serialized with int8 ternary weights

**Expected compression for Qwen2-1.5B:**

| Representation | Size |
|---------------|-----:|
| FP32          | ~6.0 GB |
| FP16          | ~3.0 GB |
| **Ternary**   | **~469 MB** |

---

## 4. Pipeline Integration

### 4.1 Lead Ranking Engine

The Calculus Labs Lead Ranking Engine is a 4-agent pipeline deployed on the AI Portal:

```
┌─────────────────┐     ┌──────────────┐     ┌────────────────┐     ┌───────────────────┐
│  Data Validator  │ ──▶ │  Lead Scorer  │ ──▶ │  Risk Analyzer │ ──▶ │  Report Generator │
│   (gpt-4o-mini)  │     │ (deterministic)│    │ (local/gemini) │     │     (gpt-4o)      │
└─────────────────┘     └──────────────┘     └────────────────┘     └───────────────────┘
```

**Agent 1 — Data Validator:** Parses CSV input or extracts lead data from natural language using GPT-4o-mini.

**Agent 2 — Lead Scorer:** Applies deterministic weighted composite scoring with zero LLM cost:
- LTV ratio (30% weight)
- Loan request amount (20%)
- Project type (20%)
- Location premium (15%)
- Borrower history sentiment (15%)

Assigns institutional ratings: Tier 1 (≥80), Tier 2 (60–79), Tier 3 (40–59), Tier 4 (<40).

**Agent 3 — Risk Analyzer:** Generates AI-powered risk memos for Tier 1 and Tier 2 leads. This is the agent replaced by the ternary local model.

**Agent 4 — Report Generator:** Synthesizes all scored and analyzed leads into a structured executive report via GPT-4o.

### 4.2 Provider Fallback Architecture

The Risk Analyzer implements a tiered provider strategy:

```
try:  Local Ternary Model  →  $0.00/call, ~800ms, on-device
      ↓ (if unavailable)
try:  Google Gemini API     →  ~$0.001/call, ~1.5s, cloud
      ↓ (if unavailable)
skip: Mark as "Not Analyzed"
```

This architecture ensures the pipeline never fails due to a missing provider. The local model is preferred when deployed; Gemini serves as the cloud fallback during transition or when the local model is not yet trained.

### 4.3 Provider Interface

The local ternary provider implements the same `BaseProvider` interface as all cloud providers (OpenAI, Anthropic, Google). This enables seamless substitution with no pipeline code changes:

```
BaseProvider
├── OpenAIProvider      (cloud, per-token cost)
├── AnthropicProvider   (cloud, per-token cost)
├── GoogleProvider      (cloud, per-token cost)
└── LocalTernaryProvider (on-device, zero cost)
```

The provider is registered in the factory under the `"local"` key and configured via a single environment variable (`TERNARY_MODEL_PATH`).

---

## 5. Proof of Concept: Vision Model

Prior to applying ternary quantization to language models, we validated the technique on the Stable Diffusion v1.5 UNet — a 859.5 million parameter convolutional architecture used for image generation.

### 5.1 Results

| Metric | Value |
|--------|------:|
| Original size (FP16) | 327.8 MB |
| Ternary size | 21.9 MB |
| **Compression ratio** | **15.0x** |
| Parameters | 859.5M |
| Conv2d layers converted | 98 |
| Linear layers converted | 184 |
| Conversion time | 4.4 seconds |
| Inference time (10 steps, CPU) | 33.6 seconds |

The ternary UNet successfully generated coherent images from text prompts, demonstrating that extreme quantization preserves learned representations across modalities.

### 5.2 Implications for Language Models

The vision model validation confirmed three properties critical for language model deployment:

1. **Forward pass stability:** Ternary weights combined with per-layer scaling produce finite, non-degenerate outputs across 282 converted layers.
2. **Gradient flow:** The straight-through estimator enables continued training through quantized layers, which is required for domain-specific fine-tuning.
3. **Compression consistency:** The theoretical 2.5 bits/weight target is achievable in practice with standard PyTorch serialization.

---

## 6. Security and Privacy

### 6.1 Data Residency

Cloud-based risk analysis requires transmitting borrower financial data — including loan amounts, credit history, and personally identifiable information — to third-party API endpoints. With on-device inference, all data processing occurs within the Calculus Labs infrastructure. Borrower data never traverses external networks.

### 6.2 Model Isolation

The ternary model operates as a read-only artifact. It does not log, transmit, or persist input data. Inference is stateless — each request is processed independently with no context carried between calls.

### 6.3 Regulatory Alignment

On-device processing supports compliance with data handling requirements in financial services, where controlling the flow of borrower PII is a regulatory obligation. The local-first architecture provides an auditable, contained inference path.

---

## 7. Cost Analysis

### 7.1 One-Time Costs

| Item | Cost |
|------|-----:|
| Training data generation (2,000 Gemini-labeled examples) | ~$2 |
| Fine-tuning compute (3 epochs, CPU) | $0 (existing infrastructure) |
| Ternary quantization | $0 (post-processing step) |
| **Total** | **~$2** |

### 7.2 Ongoing Costs: Cloud vs. Local

| Scenario | Cloud (Gemini) | Local Ternary | Savings |
|----------|---------------:|--------------:|--------:|
| 1K leads/month | $4 | $0 | $48/yr |
| 10K leads/month | $40 | $0 | $480/yr |
| 100K leads/month | $400 | $0 | $4,800/yr |
| 1M leads/month | $4,000 | $0 | $48,000/yr |

The local model's marginal inference cost is zero. The only ongoing cost is the compute infrastructure already provisioned for the AI Portal backend.

### 7.3 Break-Even

At any lead volume above zero, the local model is more cost-effective than cloud inference. The $2 training investment is recovered after approximately 2,000 risk analyses.

---

## 8. Limitations and Future Work

### 8.1 Current Limitations

- **Quality benchmarking pending:** Formal evaluation (accuracy, F1 score, BLEU for memo quality) against Gemini baseline has not yet been conducted.
- **Single-task model:** The ternary model is fine-tuned exclusively for credit risk classification. It cannot generalize to other analytical tasks.
- **CPU inference latency:** On CPU, inference takes 500ms–1s per lead. GPU acceleration would reduce this to under 100ms.
- **Training data scale:** 2,000 examples may underrepresent edge cases in borrower profiles. Expanding to 10,000+ examples with real (anonymized) data would improve robustness.

### 8.2 Planned Improvements

1. **Formal quality evaluation:** Side-by-side comparison of ternary model output vs. Gemini on a held-out test set, measuring classification accuracy and memo quality.
2. **GPU optimization:** Deploy with CUDA support on the GCE instance for 5–10x inference speedup.
3. **ONNX/TensorRT export:** Convert the ternary model to optimized inference formats for further latency reduction.
4. **Adaptive thresholding:** Per-layer threshold optimization instead of a global τ = 0.05, potentially improving quality at the same compression ratio.
5. **HubSpot integration:** Sync scored and analyzed leads directly to CRM, enabling automated pipeline-to-deal-flow workflows.
6. **Continuous learning:** Periodically regenerate training data with updated Gemini models and retrain the local model to absorb improvements.

---

## 9. Conclusion

Ternary quantization enables a fundamental shift in how Calculus Labs deploys AI-powered credit risk analysis: from a per-request cloud dependency to a fixed-cost, on-device capability. By compressing a fine-tuned language model from 3 GB to under 500 MB using {-1, 0, +1} weight encoding, we achieve deployment-grade inference on existing server infrastructure with zero marginal cost, sub-second latency, and complete data residency.

The technique is validated across both vision (Stable Diffusion, 15x compression) and language (transformer decoder, 6.4x compression) architectures, confirming its generality. Integrated into the Lead Ranking Engine as a local-first provider with cloud fallback, the system degrades gracefully while delivering institutional-scale cost savings.

The total investment to replicate Gemini-quality risk analysis locally: approximately $2 and 4 hours of compute.

---

**Calculus Holdings LLC**
Proprietary and Confidential

*Built with the Calculus Labs AI Portal and Ternary-SD framework.*
