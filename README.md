# candle-plamo3

100% Pure Rust (`candle-core` / `candle-nn` / `candle-transformers` / `tokenizers`) implementation of the **PLaMo 3 (`plamo3`)** GGUF model loader and tokenizer, with zero C/C++ bindings (no `llama.cpp` dependency).

Tested with `pfnet/plamo-3-translate-0.8b-GGUF` (`Q4_K_M`).

## Features

- **Pure Rust GGUF Loader (`Plamo3ModelWeights`)**:
  - Fused QKV projection (`attn_qkv.weight`) and SwiGLU MLP (`ffn_up.weight`)
  - Per-head QK-Norm (`attn_q_norm.weight`, `attn_k_norm.weight`)
  - Pre-Norm and Post-Norm (`post_attention_norm`, `post_ffw_norm`) with folded RMSNorm weights
  - Layer-wise RoPE switching between SWA (`freq_base_swa`) and Full Attention YaRN (`sliding_window_pattern`)
  - CPU flash attention & interleaved KV cache support
- **Pure Rust Tokenizer (`Plamo3Tokenizer`)**:
  - Built directly from GGUF embedded `tokenizer.ggml.*` metadata (no external `tokenizer.json` needed)
  - Unigram model + PLaMo 3 boundary split rules (` {2,}`, `(.)\1{3,}`) + `<0xXX>` byte fallback decoding
- **PFML3 & External Dialect Config (`pfml3`)**:
  - Helpers for `JA -> EN` and `EN -> JA` (`polite` / `plain`)
  - `DialectConfig` JSON loader supporting `<|plamo:tag|>context`, `<|plamo:tag|>glossary`, and few-shot style transfer

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
candle-plamo3 = { git = "https://github.com/<your-account>/candle-plamo3" }
```

Basic translation example:

```rust
use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_plamo3::{pfml3, Plamo3ModelWeights, Plamo3Tokenizer};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;
    let mut file = std::fs::File::open(Path::new("plamo-3-translate-0.8b.gguf"))?;
    let content = gguf_file::Content::read(&mut file)?;
    let tokenizer = Plamo3Tokenizer::from_gguf(&content)?;
    let mut model = Plamo3ModelWeights::from_gguf(content, &mut file, &device)?;

    let prompt = pfml3::prompt_ja_to_en("今日はいい天気ですね。");
    let prompt_tokens = tokenizer.encode(&prompt, true)?;
    let prompt_len = prompt_tokens.len();

    let mut logits_processor = LogitsProcessor::from_sampling(2026, Sampling::ArgMax);
    let mut generated = Vec::new();
    let mut next_token = 0u32;

    for step in 0..128 {
        let (input, pos) = if step == 0 {
            (Tensor::new(prompt_tokens.as_slice(), &device)?.unsqueeze(0)?, 0)
        } else {
            (Tensor::new(&[next_token], &device)?.unsqueeze(0)?, prompt_len + step - 1)
        };
        let logits = model.forward(&input, pos)?.squeeze(0)?;
        next_token = logits_processor.sample(&logits)?;

        if next_token == tokenizer.eos_token_id
            || next_token == tokenizer.eot_token_id
            || next_token == tokenizer.eom_token_id
        {
            break;
        }
        generated.push(next_token);
        if tokenizer.decode(&generated, false)?.contains("<|plamo:tag|>") {
            break;
        }
    }

    let text = tokenizer.decode(&generated, true)?;
    println!("{}", text.trim());
    Ok(())
}
```

## Notes / Limitations

- **Sliding Window Attention (SWA)**: Currently uses standard causal attention up to `4096` positions. For sequences within `window_size` (`<= 2048` tokens), this produces mathematically identical attention to SWA. Explicit sliding-window masking for >2048 tokens is planned for future work.
- **Model License**: Model weights (e.g., `pfnet/plamo-3-translate-0.8b-GGUF`) are subject to the PLaMo Community License. This crate's source code is dual-licensed under MIT / Apache-2.0.
