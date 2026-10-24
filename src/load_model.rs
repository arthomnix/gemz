use std::error::Error;
use std::fs;
use std::sync::Arc;
use candle_core::Device;
use candle_core::quantized::gguf_file::Content;
use tokenizers::Tokenizer;
use crate::model::quantized_gemma3::ModelWeights;

pub fn load_tokenizer(model_id: u32) -> anyhow::Result<Tokenizer> {
    let path = dirs::data_local_dir().expect("No data directory found!")
        .join(format!("gemz/models/{model_id}/tokenizer.json"));

    Ok(Tokenizer::from_file(path).map_err(Into::<Arc<dyn Error + Send + Sync>>::into)?)
}

pub fn load_model(model_id: u32, device: &Device) -> anyhow::Result<ModelWeights> {
    let path = dirs::data_local_dir().expect("No data directory found!")
        .join(format!("gemz/models/{model_id}/model.gguf"));
    let mut reader = fs::File::open(path)?;
    let ct = Content::read(&mut reader)?;
    let model = ModelWeights::from_gguf(ct, &mut reader, &device)?;
    Ok(model)
}