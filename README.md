# AirLLM Llama-3.1-8B Test

A simple application to run Meta's Llama-3.1-8B model using AirLLM, optimized for both CPU and GPU environments.

## Description

This project provides a straightforward way to interact with the Llama-3.1-8B large language model using AirLLM, which helps optimize memory usage and performance. The implementation includes features like:

- Automatic device detection (CPU/GPU)
- Dynamic prompt length handling
- Fallback mechanisms for generation errors
- Debug mode configuration

## Prerequisites

- Python 3.8+
- Hugging Face account with access to Meta's Llama-3.1-8B model
- API token from Hugging Face (set as environment variable)

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/airllm-test.git
   cd airllm-test
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   Note: The requirements file includes different PyTorch installation options. By default, it uses CUDA. Uncomment the appropriate line based on your system.

3. Create a `.env` file in the project root with your Hugging Face token:

   ```plaintext
   HF_TOKEN=your_hugging_face_token_here
   DEVICE=auto  # Options: auto, cpu, cuda
   ```

## Usage

Run the application with:

```bash
python main.py
```

The program will:

1. Authenticate with Hugging Face
2. Load the Llama-3.1-8B model
3. Prompt you to enter a question or instruction
4. Generate and display the model's response

## Configuration Options

You can configure the application behavior through environment variables in your `.env` file:

- `HF_TOKEN`: Your Hugging Face API token (required)
- `DEVICE`: Preferred device (`cpu`, `cuda`, or leave empty for auto-detection)
- `AIRLLM_DEBUG`: Set to `1` to enable debug mode
- `AIRLLM_PREFETCHING`: Set to `1` to enable prefetching

## Troubleshooting

If you encounter generation errors:

- The application will automatically attempt a simplified generation configuration
- Check that your Hugging Face token has access to the Llama-3.1-8B model
- Ensure you're using compatible versions of AirLLM and transformers

## Acknowledgments

- [Meta AI](https://ai.meta.com/) for creating the Llama models
- [AirLLM](https://github.com/lyogavin/Anima/tree/main/air_llm) for the optimization library
