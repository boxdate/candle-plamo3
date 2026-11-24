//! Pure Rust PLaMo 3 (`plamo3`) GGUF loader and tokenizer for `candle`.
//!
//! Provides:
//! - [`Plamo3ModelWeights`]: Quantized GGUF model loader and forward pass (`QMatMul`, `RmsNorm`, QK-Norm, Post-Norm, YaRN/SWA RoPE).
//! - [`Plamo3Tokenizer`]: Pure Rust tokenizer built directly from GGUF embedded `tokenizer.ggml.*` metadata.
//! - [`pfml3`]: Prompt builder utilities for PLaMo 3 Translate (`JA <-> EN`, glossary, context, few-shot, and external JSON dialect profiles).

pub mod model;
pub mod tokenizer;

pub use model::Plamo3ModelWeights;
pub use tokenizer::{pfml3, Plamo3Tokenizer};
