use std::fs;
use std::path::PathBuf;
use gemz::decompress;
use clap::Parser;

#[derive(Parser, Debug)]
struct Args {
    #[arg()]
    input_path: PathBuf,

    #[arg(short, long = "output")]
    output_path: Option<PathBuf>,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();


    let text = fs::read(&args.input_path)?;

    let output_path = match args.output_path {
        Some(p) => p,
        None => {
            let s = args.input_path.into_string().expect("input path contains invalid characters");
            let s_trimmed = s.trim_end_matches(".gmz");
            if s_trimmed == &s {
                (s + ".txt").into()
            } else {
                s_trimmed.into()
            }
        }
    };

    let out = decompress::decompress(&text)?;
    fs::write(output_path, out.as_bytes())?;
    
    Ok(())
}
