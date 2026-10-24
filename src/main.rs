use std::fs;
use std::io::{Seek, SeekFrom};

mod model;
mod compress;
mod decompress;

fn main() -> anyhow::Result<()> {
    let tokenizer_path = "/home/arthomnix/code/gemma3-270m/pt/tokenizer.json";
    let model_path = "/home/arthomnix/code/gemma3-270m/gemma-3-270m-q4_k_m.gguf";
    let input_path = "test_input.txt";

    let tokenizer = fs::read(tokenizer_path)?;
    let mut model = fs::File::open(model_path)?;

    let text = fs::read_to_string(input_path)?;

    let out = compress::compress(&text, &tokenizer, &mut model)?;
    fs::write("test.gmz", &out)?;

    println!("Compression ratio: {}", text.len() as f32 / out.len() as f32);

    model.seek(SeekFrom::Start(0))?;
    let decompressed = decompress::decompress(&out, &tokenizer, &mut model)?;
    println!("{decompressed}");
    Ok(())
}
