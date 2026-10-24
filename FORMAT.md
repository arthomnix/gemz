# .gmz file format, version 0.0

* Little endian, LSB first

## Header
* Magic number [0x47 0x45 0x4D 0x5A] ("GEMZ")
* 8-bit format version (currently 0)
* 32-bit Model ID
    * Currently defined values:
      * 0 = gguf-org/gemma-3-270m-gguf q4_k_m

## Compression stream
* The stream contains a combination of full token IDs and predicted token indexes
* Full token IDs consist of a zero bit followed by the token ID in the model's embedding
  * Stored using the minimum number of bits required for the model's embedding (for Gemma 3, this is 18 bits)
* Predicted tokens consist of a one bit followed by a three bit index
  * The index represents where the model ranks this token in order of likelihood (e.g. if the model predicts the next token is the most likely, this is stored as an index of zero)
