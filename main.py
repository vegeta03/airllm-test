# Fix for Llama-3.1-8B with AirLLM on CPU

import torch
import os
import transformers
import sys
from huggingface_hub import login

# Login to Hugging Face to access the model
login(token=os.getenv("HF_TOKEN"))

# Configure AirLLM for better compatibility
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"  # Disable warnings
os.environ["AIRLLM_DEBUG"] = "1"  # Enable debug mode
os.environ["AIRLLM_PREFETCHING"] = "1"  # Enable prefetching

# Import after setting environment variables
from airllm import AutoModel

# Show system information
print(f"Python version: {sys.version}")
print(f"Transformers version: {transformers.__version__}")

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Smaller context size to reduce memory usage
MAX_LENGTH = 64

# Model configuration
model_id = "meta-llama/Llama-3.1-8B"

# Initialize model with minimal parameters
model = AutoModel.from_pretrained(
    model_id,
    device="cpu"  # Explicitly use CPU
)

# Explicitly set padding token
model.tokenizer.pad_token = model.tokenizer.eos_token

print("\nPreparing to run Llama-3.1-8B with AirLLM...")

# Input text for generation
input_text = [
    'Explain quantum computing in simple terms.'
]

# Tokenize input with proper settings
print("Tokenizing input...")
input_tokens = model.tokenizer(
    input_text,
    return_tensors="pt",
    truncation=True,
    max_length=MAX_LENGTH,
    padding="max_length"
)

print("\nStarting text generation...")

# Try generation with error handling
try:
    # First attempt with standard settings
    generation_output = model.generate(
        input_tokens['input_ids'],
        max_new_tokens=20,
        do_sample=False  # Deterministic generation is more stable
    )
    
    # Decode output
    if hasattr(generation_output, 'sequences'):
        # If return_dict_in_generate was True
        output = model.tokenizer.decode(generation_output.sequences[0])
    else:
        # Standard output format
        output = model.tokenizer.decode(generation_output[0])
        
    print("\nGeneration successful! Output:")
    print(output)
    
except Exception as e:
    print(f"\nGeneration error: {type(e).__name__}: {e}")
    print("\nTrying with simplified configuration...")
    
    try:
        # Simplified generation settings
        generation_output = model.generate(
            input_tokens['input_ids'],
            max_length=input_tokens['input_ids'].shape[1] + 10  # Input length + 10 new tokens
        )
        
        output = model.tokenizer.decode(generation_output[0])
        print("\nSimplified generation successful! Output:")
        print(output)
        
    except Exception as e2:
        print(f"\nSimplified generation failed: {type(e2).__name__}: {e2}")
        print("\nThe error is likely due to compatibility issues between AirLLM and Llama-3.1-8B.")
        print("You may need to update AirLLM or use an older version of transformers.")

