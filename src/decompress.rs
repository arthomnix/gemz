use std::error::Error;
use std::fs::read;
use std::io::{Read, Seek};
use std::sync::Arc;
use anyhow::bail;
use bitvec::macros::internal::funty::Integral;
use bitvec::prelude::*;
use candle_core::{Device, Tensor};
use candle_core::quantized::gguf_file::Content;
use tokenizers::Tokenizer;

pub fn decompress(input: impl AsRef<[u8]>, tokenizer: impl AsRef<[u8]>, model: &mut (impl Seek + Read)) -> anyhow::Result<String> {
    let tokenizer = Tokenizer::from_bytes(tokenizer)
        .map_err(Into::<Arc<dyn Error + Send + Sync>>::into)?;
    let token_bits = (tokenizer.get_vocab_size(false) as f32).log2().ceil() as usize;

    // Load model
    let device = Device::Cpu;
    let ct = Content::read(model)?;
    let model = crate::model::quantized_gemma3::ModelWeights::from_gguf(ct, model, &device)?;
    let mut kv_cache = None;

    // Load BOS token
    let bos = tokenizer.token_to_id("<bos>").expect("Tokenizer does not have BOS token!");
    let input_tensor = Tensor::new(&[bos], &device)?.unsqueeze(0)?;
    model.forward(&input_tensor, &mut kv_cache, 0)?;

    // Load input to bit slice
    let input_bv = BitSlice::<u8, Lsb0>::from_slice(input.as_ref());
    let mut reader_pos = 0;

    // Validate file header
    let magic = read_bits::<u32>(input_bv, &mut reader_pos, 32);
    if magic != 0x47454D5A.to_be()  { // "GEMZ"
        bail!("Invalid magic {magic:x} - is this a gmz file?");
    }
    let version = read_bits::<u8>(input_bv, &mut reader_pos, 8);
    if version != 0 {
        bail!("Invalid file version {version} - this implementation only understands version 0");
    }
    let model_id = read_bits::<u32>(input_bv, &mut reader_pos, 32);
    if model_id != 0 {
        bail!("Invalid model ID {model_id} - this implementation only supports ggml-org/gemma-3-270m-gguf:q4_k_m (ID 0)");
    }

    // Get first token and load into model
    let mut tokens = vec![];
    let is_idx = input_bv[reader_pos];
    reader_pos += 1;
    if is_idx {
        bail!("Invalid gmz file - first token must not be an index");
    }
    let first_tok = read_bits::<u32>(input_bv, &mut reader_pos, token_bits);
    tokens.push(first_tok);

    while reader_pos < input_bv.len() {
        let is_idx = input_bv[reader_pos];
        reader_pos += 1;

        let last_tok = *tokens.last().unwrap();
        let input_tensor = Tensor::new(&[last_tok], &device)?.unsqueeze(0)?;
        let logits = model.forward(&input_tensor, &mut kv_cache, tokens.len())?.squeeze(0)?;

        if is_idx {
            let idx = read_bits::<usize>(input_bv, &mut reader_pos, 3);
            let logits_vec = logits.to_vec1::<f32>()?;

            let mut logits_vec = logits_vec.into_iter().enumerate().map(|(i, prob)| {
                (i as u32, prob)
            }).collect::<Vec<_>>();
            logits_vec.sort_by(|(_, p1), (_, p2)| p2.total_cmp(p1));
            tokens.push(logits_vec[idx].0)
        } else {
            if input_bv.len() - reader_pos < token_bits { break; }

            let tok = read_bits::<u32>(input_bv, &mut reader_pos, token_bits);
            tokens.push(tok);
        }
    }

    let out_str = tokenizer.decode(&tokens, true)
        .map_err(Into::<Arc<dyn Error + Send + Sync>>::into)?;

    Ok(out_str)
}

fn read_bits<T: Integral>(input_bv: &BitSlice<u8>, reader_pos: &mut usize, num_bits: usize) -> T {
    let result = input_bv[*reader_pos..*reader_pos + num_bits].load_le::<T>();
    *reader_pos += num_bits;
    result
}