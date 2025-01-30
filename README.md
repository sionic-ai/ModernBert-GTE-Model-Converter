# ModernBert-GTE Model Converter

A tool for manual conversion of ModernBert-GTE models with preserved trainable variables and direct control over model outputs.

## Features

- Manual conversion of ModernBert-GTE models with preserved model architecture
- Resolution for ONNX conversion failures
- Solution for protobuf 2GB file size limitation
- Preservation of trainable variable information in the model structure

## Project Structure

```
├── ModernGTETFModel.py            # Model architecture definition
├── ModernGTETFWeightConverter.py  # Weight conversion and saving implementation
└── model_conversion_validator.py  # Conversion validation code
```

## Prerequisites

- Python 3.11
- Virtual environment
- Git LFS
- transformers==4.47.1
- keras==3.7.0
- tensorflow==2.16.2

## Installation

1. Clone the repository:
```bash
git clone https://github.com/sionic-ai/ModernBert-GTE-Model-Converter
cd ./ModernBert-GTE-Model-Converter
```

2. Set up Python virtual environment:
```bash
uv venv
source ./venv/bin/activate
uv sync
#uv add -r requirements.txt
```

## Usage

1. Convert the model weights:
```bash
uv run ModernGTETFWeightConverter.py
```

2. Validate the converted model:
```bash
uv run model_conversion_validator.py
```

## Model Architecture

The converter works with XLMRobertaModel architecture with the following specifications:

- Hidden size: 768
- Attention heads: 12
- Hidden layers: 22
- Maximum position embeddings: 8192
- Intermediate size: 1152
- Vocabulary size: 50368

## Input/Output Example

```
Encoder Layer 22:
PyTorch shape: torch.Size([2, 842, 768])
    dims: [batch_size=2, seq_len=842, hidden_dim=768]
TensorFlow shape: (2, 842, 768)
    dims: [batch_size=2, seq_len=842, hidden_dim=768]
  -> MSE: 32.437252
  -> CLS Token Cosine Similarity: 1.000000
[TensorFlow] Final Embeddings Shape: (2, 768)

=== 3) PT vs. TF 최종 임베딩 비교 ===
[[ 0.22076873 -1.3506409  -2.0690584  ... -0.38223675  1.0725555
  -0.5355745 ]
 [ 0.12727773 -1.3734885  -2.333182   ... -0.3719053   1.4427259
  -0.78435   ]]
[[ 0.2207656  -1.350643   -2.0690598  ... -0.38224083  1.0725522
  -0.5355765 ]
 [ 0.12727615 -1.3734777  -2.3331754  ... -0.37190926  1.442732
  -0.7843542 ]]
===== Queries =====
[0] 이 모델은 무엇을 하는 모델인가요?
[1] 이 모델은 무엇을 하는 모델인가요?이 모델은 무엇을 ...

===== PyTorch Embeddings (shape) =====
(2, 768)
===== TF Embeddings (shape) =====
(2, 768)

===== Pairwise Cosine Similarity (PT vs TF) =====
Query 0 Cosine Similarity: 1.0000
Query 1 Cosine Similarity: 1.0000

===== MSE (PT vs TF) =====
MSE: 0.000000

===== Sample Differences (first query, first 5 dims) =====
[ 3.1292439e-06  2.1457672e-06  1.4305115e-06 -1.4305115e-06
  2.6226044e-06]

```

## Contributing

Feel free to open issues and pull requests for any improvements or bug fixes.

## License

MIT License

```
Copyright (c) 2025 [Sionic AI Inc.]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

Note: This project is specifically designed for ModernBert-GTE model conversion and may require specific hardware configurations and dependencies. Please ensure your environment meets the requirements before proceeding with the conversion.
