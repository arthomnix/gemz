use std::error::Error;
use std::io::{Read, Seek, Write};
use std::sync::Arc;
use bitvec::prelude::*;
use candle_core::{Device, Tensor};
use candle_core::quantized::gguf_file::Content;
use tokenizers::Tokenizer;

pub fn compress(input: &str, tokenizer: impl AsRef<[u8]>, model: &mut (impl Seek + Read)) -> anyhow::Result<Vec<u8>> {
    // Tokenize input text
    let tokenizer = Tokenizer::from_bytes(tokenizer)
        .map_err(Into::<Arc<dyn Error + Send + Sync>>::into)?;
    let input_tokens = tokenizer.encode_fast(input, false)
        .map_err(Into::<Arc<dyn Error + Send + Sync>>::into)?;
    let mut tokens = input_tokens.get_ids().to_vec();

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

    // Perform compression

    let mut output = bitvec![u8, Lsb0;];
    output.extend("GEMZ".as_bits::<Lsb0>()); // magic number
    output.extend(0u8.view_bits::<Lsb0>()); // version
    output.extend(0u32.to_le_bytes().view_bits::<Lsb0>()); // model ID (0 = ggml-org/gemma-3-270m-gguf:q4_k_m)

    // First token
    output.push(false);
    output.extend(&tokens[0].to_le_bytes().view_bits::<Lsb0>()[..token_bits]);
    print!("{}", tokenizer.id_to_token(tokens[0]).unwrap());
    std::io::stdout().flush();

    for index in 0..input_tokens.len() - 1 {
        let context = &tokens[index..=index];
        let input_tensor = Tensor::new(context, &device)?.unsqueeze(0)?;

        let logits = model.forward(&input_tensor, &mut kv_cache, index + 1)?.squeeze(0)?;
        let logits_vec = logits.to_vec1::<f32>()?;
        let mut logits_vec = logits_vec.into_iter().enumerate().map(|(i, prob)| {
            (i as u32, prob)
        }).collect::<Vec<_>>();
        logits_vec.sort_by(|(_, p1), (_, p2)| p2.total_cmp(p1));

        let top_8 = &logits_vec[..8];
        if let Some(index) = top_8.iter().position(|(tok, _)| *tok == tokens[index + 1]) {
            output.push(true);
            output.extend(&index.view_bits::<Lsb0>()[..3]);
            print!("<{index}>");
        } else {
            let kv_cache_checkpoint = kv_cache.clone();

            let mut top_8_embeddings = vec![];
            for (tok, _) in top_8 {
                let input_tensor = Tensor::new(&[*tok], &device)?.unsqueeze(0)?;
                let x = model.forward_inner(&input_tensor, &mut kv_cache, index + 2)?
                    .into_iter()
                    .map(|x| x.squeeze(0).unwrap().squeeze(0).unwrap())
                    .collect::<Vec<_>>();
                top_8_embeddings.push(x);
                kv_cache = kv_cache_checkpoint.clone();
            }

            let real_tok = tokens[index + 1];
            let input_tensor = Tensor::new(&[real_tok], &device)?.unsqueeze(0)?;
            let real_embedding = model.forward_inner(&input_tensor, &mut kv_cache, index + 2)?
                .into_iter()
                .map(|x| x.squeeze(0).unwrap().squeeze(0).unwrap())
                .collect::<Vec<_>>();
            kv_cache = kv_cache_checkpoint.clone();

            let mut min_distance = f32::INFINITY;
            let mut min_index = 0;
            for (i, x) in top_8_embeddings.iter_mut().enumerate() {
                let dist = x.iter().zip(real_embedding.iter()).map(|(t1, t2)| {
                    (t1 - t2).unwrap().sqr().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap()
                }).sum::<f32>();

                if dist < min_distance {
                    min_distance = dist;
                    min_index = i;
                }
            }

            if min_distance < 100000000.0 {
                let tok = top_8[min_index].0;
                tokens[index + 1] = tok;

                output.push(true);
                output.extend(&min_index.view_bits::<Lsb0>()[..3]);
                print!("[{min_index}]");
            } else {
                output.push(false);
                output.extend(&tokens[index + 1].to_le_bytes().view_bits::<Lsb0>()[..token_bits]);
                print!("{}", tokenizer.id_to_token(tokens[index + 1]).unwrap());
            }
        }
        std::io::stdout().flush();
    }

    println!();

    output.set_uninitialized(false);
    Ok(output.into_vec())
}