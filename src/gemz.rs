use std::fs;
use std::path::PathBuf;
use gemz::compress;
use clap::Parser;

#[derive(Parser, Debug)]
struct Args {
    #[arg()]
    input_path: PathBuf,

    #[arg(short, long = "output")]
    output_path: Option<PathBuf>,

    #[arg(short, long = "tokenizer")]
    tokenizer_path: PathBuf,

    #[arg(short, long = "model")]
    model_path: PathBuf,

    #[arg(short = 'z', long, default_value_t = 0.0)]
    lossy_threshold: f32,

    #[arg(short = 'l', long, default_value_t = 6)]
    lossy_evaluation_layer: usize,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let tokenizer = fs::read(&args.tokenizer_path)?;
    let mut model = fs::File::open(&args.model_path)?;

    let text = fs::read_to_string(&args.input_path)?;

    let output_path = match args.output_path {
        Some(p) => p,
        None => (args.input_path.into_string().expect("input path contains invalid characters") + ".gmz").into(),
    };

    let out = compress::compress(&text, &tokenizer, &mut model, args.lossy_threshold, args.lossy_evaluation_layer)?;
    fs::write(output_path, &out)?;

    println!("Compression successful. Compression ratio: {}", text.len() as f32 / out.len() as f32);
    Ok(())
}
