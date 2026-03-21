# Parameter Golf Research Log

## Current SOTA: 1.1428 BPB (thwu1, 2026-03-20)

### Baseline Recipe (what we're starting from)
- 10 unique layers, dim=512, 8 heads, 4 KV heads (GQA), 3× MLP (1536 hidden), ReLU²
- SmearGate + BigramHash(10240, XOR hash, 128-dim projected to 512)
- Mixed int5 (MLP) / int6 (attention) post-training quantization + zstd-22
- SWA over last 40% of warmdown, every 50 steps
- 3% magnitude pruning before quantization
- Muon optimizer: WD=0.04, momentum warmup 0.92→0.99 over 1500 steps, matrix_lr=0.02
- AdamW for embeddings/scalars: WD=0.04
- Orthogonal init, zero-init for output projections (scaled by 1/√(2L))
- Tied FP16 embeddings, last-layer keys (blocks.8.attn.c_k) kept FP16
- U-Net skip connections (encoder/decoder halves)
- Sliding window eval: stride=64, seq_len=2048
- Training: seq_len=2048, batch=786K tokens, grad_clip=0.3, 600s wallclock cap

---

## Ideas to Explore (ordered by expected impact)

### Tier 1: High confidence, moderate effort

#### Idea 1: Trigram Hash Embedding
**Hypothesis**: BigramHash captures (prev, curr) pairs. Extending to (prev_prev, prev, curr) captures higher-order local patterns — common 3-word phrases, syntax, and local dependencies that bigrams miss.

**Implementation**: Add `TrigramHashEmbedding` class alongside BigramHash. Use 8192 buckets, 64-dim, projected to 512. Hash: `xor(hash1*t[i], hash2*t[i-1], hash3*t[i-2]) % (buckets-1)`. Positions 0-1 use a sentinel index. Parameter cost: 8192×64 + 64×512 = ~557K params → ~200KB after int6+zstd.

**Expected gain**: 0.002-0.005 BPB
**Status**: [ ] Not started

---

#### Idea 2: Per-Layer Quantization Bitwidth Search
**Hypothesis**: Not all layers are equally sensitive to quantization error. Some layers (especially early/late) can tolerate int4 while others need int6. Reallocating bits per-layer can reduce total model size, letting us fit an 11th layer or wider MLP.

**Implementation**: After training, quantize each layer individually at int4/int5/int6/int8 and measure BPB delta. Build a DP/greedy allocation: minimize total BPB cost subject to 16MB artifact budget.

**Expected gain**: 0.003-0.008 BPB (via fitting more capacity)
**Status**: [ ] Not started

---

#### Idea 3: Entropy-Coded Weights (ANS/Huffman)
**Hypothesis**: Quantized int5/int6 values are peaked near zero. Uniform-width coding wastes bits on common values. ANS coding with a tuned frequency table can beat zstd-22 by 5-15%.

**Implementation**: After quantization, compute per-layer histograms of quantized values. Build an ANS frequency table. Encode weights with rANS. Ship the frequency table + encoded blob. Decoder is ~50 lines.

**Expected gain**: 0.5-1.5MB saved → reinvest in model capacity
**Status**: [ ] Not started

---

#### Idea 4: Gated Linear Attention (GLA) Hybrid
**Hypothesis**: Replace 2-3 middle layers with GLA (O(n) in seq_len). This enables training/eval at 4096+ seq_len within the same wallclock, and longer context = better BPB.

**Implementation**: Implement GLA layer (data-dependent decay gate + linear attention). Replace blocks 3-5 with GLA. Increase train_seq_len to 4096.

**Expected gain**: 0.005-0.015 BPB (from longer context)
**Risk**: GLA quality may be worse per-layer; net effect uncertain
**Status**: [ ] Not started

---

### Tier 2: Creative, higher risk/reward

#### Idea 5: Test-Time Training (TTT) with LoRA
**Hypothesis**: The rules allow test-time training on already-evaluated tokens. Training tiny LoRA adapters (rank-1) on each sliding window's past context before predicting future tokens gives the model local adaptation for free. The LoRA TTT submission hit 1.1928 with a much weaker base model — combining with current SOTA base should be very strong.

**Implementation**: At eval time, for each sliding window: (1) freeze base weights, (2) initialize rank-1 LoRA on Q/V projections, (3) do 3-5 gradient steps on the "context" portion of the window, (4) predict "future" tokens with adapted model. LoRA params don't count against 16MB.

**Expected gain**: 0.01-0.03 BPB
**Risk**: Eval time budget (10 min on 8×H100); may need careful batching
**Status**: [ ] Not started

---

#### Idea 6: Mixture-of-Experts MLP
**Hypothesis**: Replace 3× MLP with top-1 MoE over 4 experts (each 1.5×). Same parameter count but each token routes to a specialized expert. MoE consistently improves loss at fixed parameter count.

**Implementation**: MoE layer with learned router (512→4 linear), top-1 selection, 4 experts each 1.5× width. Load balancing loss (0.01 weight). Replace 2-3 MLP layers.

**Expected gain**: 0.005-0.01 BPB
**Risk**: Load balance, quantization friendliness, torch.compile compat
**Status**: [ ] Not started

---

#### Idea 7: Depth-Recurrence Hybrid (revised)
**Hypothesis**: 8 unique layers + 1 shared layer iterated 4× in the middle = effective depth 12 with storage cost of 9 blocks. More effective depth than current 10-layer SOTA.

**Implementation**: Modify GPT to have prelude(4) + shared_block×4 + coda(4). With int5/int6, 9 stored blocks fits in budget. Add per-iteration learned gate (tiny).

**Expected gain**: 0.003-0.008 BPB
**Risk**: Shared block doesn't specialize enough; torch.compile issues
**Status**: [ ] Not started

---

#### Idea 8: Progressive QAT (int6 STE)
**Hypothesis**: Current SOTA uses post-training quantization only. QAT (#3 submission, 1.1502) eliminates the quantization gap. Combining QAT with the full SOTA recipe could close the gap between pre-quant and post-quant BPB.

**Implementation**: Train at full precision for first 70% of steps, then enable int6 STE fake-quantization for last 30%. Use the same fake_quantize_ste as the original script but with 6-bit (not 4-bit).

**Expected gain**: 0.002-0.005 BPB
**Status**: [ ] Not started

---

### Tier 3: Moonshots

#### Idea 9: Multi-Token Prediction Head
Auxiliary head predicting 2 tokens ahead forces richer representations. May improve main prediction quality.

#### Idea 10: Custom Tokenizer for BPB
Retrain a 2048-vocab tokenizer to improve tokens-per-byte ratio. The BPB metric directly benefits from this.

#### Idea 11: Neural Weight Compression
Train a tiny autoencoder to compress/decompress weight matrices, beating general-purpose codecs.

---

## Experiment Log

| Run | Changes | val_bpb | Delta | Compressed Size | Notes |
|-----|---------|---------|-------|-----------------|-------|
| baseline | SOTA recipe reproduction | 1.1438 | - | 15.79MB | 6470 steps, 600s, SWA 24 ckpts |
| #2 trigram+QAT(70%) | Trigram(4096,32) + QAT at 70% wallclock | 1.1630 | +0.0192 | TBD | QAT recompile cost ~130s (22% budget). FAIL |
| #3 trigram+QAT(start) | Trigram(4096,32) + QAT from step 0 | TBD | TBD | TBD | Running... |
