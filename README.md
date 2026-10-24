# GemZ: Efficient Text Compression with Small Language Models

This is an implementation of a text compression algorithm using small transformer-based language models
(currently, the only supported model is Google Gemma 3 270M in Q4_K_M quantisation).
This algorithm achieves compression ratios far higher than those achieved by traditional compression
algorithms (zip, gzip, etc), and also enables the possibility of lossy compression of text.

## The Idea

An LLM (or small language model) is, at its core, a statistical model designed to predict the next token in a stream
of text. If we feed the first token of the text to be compressed to the LLM, we can compare its output to the real
next token. If the real token appears in the top 8 most likely tokens predicted by the LLM, we can just store a flag
saying where the token appeared in this prediction order, instead of storing the whole token. On the other end, an
identical language model can be used to recover the original tokens and decompress the stream.

### Lossy compression

To achieve lossy compression, we check if any of the top 8 tokens predicted by the model are similar enough to the real token
that they can be substituted. To achieve this, we run each of the tokens through a certain number of layers of the model again, and
also do this with the real token. After running these layers, we get a vector in the model's embedding space representing
each of these tokens after being put in some context by the first few layers of the model. We then calculate the distance
between each of these vectors and the corresponding vector for the real token, and if one of these distances is below a
threshold, we pretend as if the model predicted this token correctly. This threshold, and the number of layers of the
model to use, can be controlled with command-line paramaters.

## Installation

Install with `cargo install --git https://github.com/arthomnix/gemz`.

This program also requires model files to be manually downloaded. The model files must be placed in `<DATA_LOCAL_DIR>/gemz/models/0/`.
`<DATA_LOCAL_DIR>` is usually `~/.local/share` on Linux systems.

The model files required are:
* Gemma 3 270M Q4_K_0 in GGUF format - download from https://huggingface.co/gguf-org/gemma-3-270m-gguf/tree/main, and save as `model.gguf` in the correct folder
* Gemma 3 tokenizer - download `tokenizer.json` from https://huggingface.co/google/gemma-3-270m/tree/main, and place it in the correct folder

## Usage

* Losslessly compress a text file:
    ```shell
    gemz file.txt
    ```
* Lossily compress a text file:
    ```shell
    gemz file.txt --lossy-threshold 0.1 --lossy-evaluation-layer 2
    ```
    (the above two parameters can be tuned to control the lossiness of the compression)

* Decompress a compressed file:
    ```shell
    ungemz file.txt.gmz 
    ```

By default, `gemz` will output to a file with `.gmz` appended to the name, and `ungemz` will output to a file with `.gmz` removed from the name.
You can change the output file with the `-o` flag.