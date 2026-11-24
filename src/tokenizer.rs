use candle_core::quantized::gguf_file;
use candle_core::Result;
use tokenizers::models::unigram::Unigram;
use tokenizers::pre_tokenizers::sequence::Sequence;
use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
use tokenizers::tokenizer::{AddedToken, SplitDelimiterBehavior, Tokenizer};

/// GGUF メタデータ内の `tokenizer.ggml.*` から PLaMo 3 トークナイザーを構築する。
/// 100% Pure Rust (`tokenizers` クレートのみ) で Aho-Corasick / Unigram 語彙と
/// Pre-tokenizer 境界ルール・Byte Fallback (`<0xXX>`) デコードを再現する。
pub struct Plamo3Tokenizer {
    inner: Tokenizer,
    tokens: Vec<String>,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub eot_token_id: u32,
    pub eom_token_id: u32,
}

impl Plamo3Tokenizer {
    pub fn from_gguf(ct: &gguf_file::Content) -> Result<Self> {
        let md = &ct.metadata;

        let tokens_val = md
            .get("tokenizer.ggml.tokens")
            .ok_or_else(|| candle_core::Error::Msg("missing tokenizer.ggml.tokens".into()))?
            .to_vec()?;
        let scores_val = md
            .get("tokenizer.ggml.scores")
            .ok_or_else(|| candle_core::Error::Msg("missing tokenizer.ggml.scores".into()))?
            .to_vec()?;
        let types_val = md
            .get("tokenizer.ggml.token_type")
            .ok_or_else(|| candle_core::Error::Msg("missing tokenizer.ggml.token_type".into()))?
            .to_vec()?;

        let bos_token_id = md
            .get("tokenizer.ggml.bos_token_id")
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(1);
        let eos_token_id = md
            .get("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(1);
        let eom_token_id = md
            .get("tokenizer.ggml.eom_token_id")
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(2);
        let eot_token_id = md
            .get("tokenizer.ggml.eot_token_id")
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(16);
        let unk_token_id = md
            .get("tokenizer.ggml.unknown_token_id")
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(0) as usize;

        let mut tokens = Vec::with_capacity(tokens_val.len());
        let mut vocab = Vec::with_capacity(tokens_val.len());
        let mut special_tokens = Vec::new();

        for i in 0..tokens_val.len() {
            let tok_str = tokens_val[i].to_string()?.clone();
            let score = scores_val[i].to_f32()? as f64;
            let ttype = types_val[i].to_i32()?;

            if tok_str.starts_with("<|plamo:") && tok_str.ends_with("|>") {
                special_tokens.push(AddedToken::from(tok_str.clone(), true));
            }

            // Byte token in GGUF has score -1e7; ensure Unigram model keeps byte fallback score reasonable
            let adj_score = if ttype == 6 && score <= -1e6 {
                -100.0
            } else {
                score
            };

            tokens.push(tok_str.clone());
            vocab.push((tok_str, adj_score));
        }

        let unigram = Unigram::from(vocab, Some(unk_token_id), true).map_err(|e| {
            candle_core::Error::Msg(format!("Failed to build Unigram tokenizer: {e}"))
        })?;

        let mut tokenizer = Tokenizer::new(unigram);

        // Pre-tokenizer matching PLaMo 3 boundary rules:
        // 1. Consecutive spaces >= 2 (` {2,}`)
        // 2. Repeated characters >= 4 (`(.)\1{3,}`) via Oniguruma regex
        let split_spaces = Split::new(
            SplitPattern::Regex(" {2,}".to_string()),
            SplitDelimiterBehavior::Isolated,
            false,
        )
        .map_err(|e| candle_core::Error::Msg(format!("Split regex error: {e}")))?;

        let split_repeats = Split::new(
            SplitPattern::Regex("(.)\\1{3,}".to_string()),
            SplitDelimiterBehavior::Isolated,
            false,
        )
        .map_err(|e| candle_core::Error::Msg(format!("Split repeat regex error: {e}")))?;

        tokenizer.with_pre_tokenizer(Some(Sequence::new(vec![
            split_spaces.into(),
            split_repeats.into(),
        ])));

        tokenizer.add_special_tokens(&special_tokens);

        Ok(Self {
            inner: tokenizer,
            tokens,
            bos_token_id,
            eos_token_id,
            eot_token_id,
            eom_token_id,
        })
    }

    /// Encode text into token IDs. If `add_bos` is true, prepends `<|plamo:bos|>` (id=1).
    pub fn encode(&self, text: &str, add_bos: bool) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| candle_core::Error::Msg(format!("Tokenize error: {e}")))?;
        let mut ids = Vec::with_capacity(encoding.get_ids().len() + usize::from(add_bos));
        if add_bos {
            ids.push(self.bos_token_id);
        }
        ids.extend_from_slice(encoding.get_ids());
        Ok(ids)
    }

    /// Decode token IDs into a UTF-8 string, matching `Plamo3Tokenizer.convert_tokens_to_string`.
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        let mut bytes = Vec::new();
        for &id in ids {
            let Some(tok) = self.tokens.get(id as usize) else {
                continue;
            };
            if skip_special_tokens && tok.starts_with("<|plamo:") && tok.ends_with("|>") {
                continue;
            }
            if tok.len() == 6 && tok.starts_with("<0x") && tok.ends_with('>') {
                if let Ok(b) = u8::from_str_radix(&tok[3..5], 16) {
                    bytes.push(b);
                    continue;
                }
            }
            bytes.extend_from_slice(tok.as_bytes());
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.tokens.get(id as usize).map(|s| s.as_str())
    }

    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }
}

/// PLaMo 3 Translate 固有の Raw PFML3 プロンプト構築関数
pub mod pfml3 {
    #[cfg(feature = "dialect-config")]
    use serde::{Deserialize, Serialize};
    #[cfg(feature = "dialect-config")]
    use std::path::Path;

    /// `<|plamo:tag|>glossary` に渡す用語ペア
    #[derive(Debug, Clone)]
    #[cfg_attr(feature = "dialect-config", derive(Serialize, Deserialize))]
    pub struct GlossaryEntry {
        pub source: String,
        pub target: String,
    }

    /// `<|plamo:tag|>input` / `<|plamo:tag|>output` に渡す対訳お手本（Few-shot）ペア
    #[derive(Debug, Clone)]
    #[cfg_attr(feature = "dialect-config", derive(Serialize, Deserialize))]
    pub struct FewShotEntry {
        pub en: String,
        pub ja: String,
    }

    /// 外部 JSON ファイルから読み込み可能な方言・話し方プロファイル。
    #[derive(Debug, Clone)]
    #[cfg_attr(feature = "dialect-config", derive(Serialize, Deserialize))]
    pub struct DialectConfig {
        /// モード表示名
        pub name: String,
        /// `"polite"` / `"plain"` / `None`
        pub form: Option<String>,
        /// `<|plamo:tag|>context` に渡す文脈・トーン指示（任意）
        pub context: Option<String>,
        /// `<|plamo:tag|>glossary` に渡す用語集リスト
        #[cfg_attr(feature = "dialect-config", serde(default))]
        pub glossary: Vec<GlossaryEntry>,
        /// スタイル転写を安定させるための Few-shot 対訳リスト
        #[cfg_attr(feature = "dialect-config", serde(default))]
        pub few_shots: Vec<FewShotEntry>,
    }

    impl DialectConfig {
        /// JSON ファイルから `DialectConfig` を読み込む
        #[cfg(feature = "dialect-config")]
        pub fn from_json_file<P: AsRef<Path>>(
            path: P,
        ) -> std::result::Result<Self, Box<dyn std::error::Error>> {
            let content = std::fs::read_to_string(path.as_ref())?;
            let cfg: Self = serde_json::from_str(&content)?;
            Ok(cfg)
        }

        /// デフォルトの標準語（です・ます調）設定を返す
        pub fn standard_polite() -> Self {
            Self {
                name: "標準語（です・ます調）".to_string(),
                form: Some("polite".to_string()),
                context: None,
                glossary: Vec::new(),
                few_shots: Vec::new(),
            }
        }
    }

    /// ① 日本語 → 英語（JA to EN）のプロンプト
    pub fn prompt_ja_to_en(ja_text: &str) -> String {
        format!(
            "<|plamo:tag|>dataset<|plamo:msg|>translation\
             <|plamo:tag|>input<|plamo:key|>lang<|plamo:val|>ja<|plamo:msg|>{ja_text}\
             <|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>en<|plamo:msg|>"
        )
    }

    /// ② 英語 → 日本語・敬体（EN to JA polite）のプロンプト
    #[allow(dead_code)]
    pub fn prompt_en_to_ja_polite(en_text: &str) -> String {
        prompt_en_to_ja_with_config(en_text, &DialectConfig::standard_polite())
    }

    /// ③ 英語 → 日本語（外部 JSON から読み込んだ `DialectConfig` による Glossary / Context / Few-shot 対応版）
    /// PLaMo 3 公式 Jinja テンプレートの PFML3 構造に完全準拠：
    /// `<|plamo:tag|>dataset<|plamo:msg|>translation`
    ///   `[<|plamo:tag|>context<|plamo:msg|>{context}]`
    ///   `[<|plamo:tag|>input...<|plamo:tag|>output... (few_shots)]`
    ///   `[<|plamo:tag|>glossary<|plamo:index|>{src}<|plamo:item|>{tgt}...]`
    ///   `<|plamo:tag|>input<|plamo:key|>lang<|plamo:val|>en<|plamo:msg|>{en_text}`
    ///   `<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>ja[<|plamo:key|>form<|plamo:val|>{form}]<|plamo:msg|>`
    pub fn prompt_en_to_ja_with_config(en_text: &str, config: &DialectConfig) -> String {
        let mut s = String::from("<|plamo:tag|>dataset<|plamo:msg|>translation");

        if let Some(ctx) = &config.context {
            if !ctx.trim().is_empty() {
                s.push_str("<|plamo:tag|>context<|plamo:msg|>");
                s.push_str(ctx);
            }
        }

        for shot in &config.few_shots {
            s.push_str("<|plamo:tag|>input<|plamo:key|>lang<|plamo:val|>en<|plamo:msg|>");
            s.push_str(&shot.en);
            s.push_str("<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>ja<|plamo:msg|>");
            s.push_str(&shot.ja);
        }

        if !config.glossary.is_empty() {
            s.push_str("<|plamo:tag|>glossary");
            for entry in &config.glossary {
                s.push_str("<|plamo:index|>");
                s.push_str(&entry.source);
                s.push_str("<|plamo:item|>");
                s.push_str(&entry.target);
            }
        }

        s.push_str("<|plamo:tag|>input<|plamo:key|>lang<|plamo:val|>en<|plamo:msg|>");
        s.push_str(en_text);
        s.push_str("<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>ja");
        if let Some(form) = &config.form {
            if !form.is_empty() {
                s.push_str("<|plamo:key|>form<|plamo:val|>");
                s.push_str(form);
            }
        }
        s.push_str("<|plamo:msg|>");
        s
    }
}

#[cfg(test)]
mod tests {
    use super::pfml3::*;

    #[test]
    fn test_prompt_ja_to_en() {
        let p = prompt_ja_to_en("こんにちは");
        assert_eq!(
            p,
            "<|plamo:tag|>dataset<|plamo:msg|>translation\
             <|plamo:tag|>input<|plamo:key|>lang<|plamo:val|>ja<|plamo:msg|>こんにちは\
             <|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>en<|plamo:msg|>"
        );
    }

    #[test]
    fn test_prompt_en_to_ja_polite() {
        let p = prompt_en_to_ja_polite("Hello");
        assert_eq!(
            p,
            "<|plamo:tag|>dataset<|plamo:msg|>translation\
             <|plamo:tag|>input<|plamo:key|>lang<|plamo:val|>en<|plamo:msg|>Hello\
             <|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>ja<|plamo:key|>form<|plamo:val|>polite<|plamo:msg|>"
        );
    }

    #[test]
    fn test_prompt_en_to_ja_with_dialect_config() {
        let cfg = DialectConfig {
            name: "test".to_string(),
            form: None,
            context: Some("方言で".to_string()),
            glossary: vec![GlossaryEntry {
                source: "Bullet".to_string(),
                target: "Bullet".to_string(),
            }],
            few_shots: vec![FewShotEntry {
                en: "Hi".to_string(),
                ja: "やあ".to_string(),
            }],
        };
        let p = prompt_en_to_ja_with_config("Good morning", &cfg);
        assert!(p.contains("<|plamo:tag|>context<|plamo:msg|>方言で"));
        assert!(p.contains("<|plamo:tag|>glossary<|plamo:index|>Bullet<|plamo:item|>Bullet"));
        assert!(p.contains("<|plamo:tag|>input<|plamo:key|>lang<|plamo:val|>en<|plamo:msg|>Hi"));
        assert!(p.ends_with("<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>ja<|plamo:msg|>"));
    }
}
