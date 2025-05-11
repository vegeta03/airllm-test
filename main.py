# Fix for Llama-3.1-8B with AirLLM on CPU

import dotenv
import torch
import os
import transformers
import sys
from huggingface_hub import login

# Login to Hugging Face to access the model
dotenv.load_dotenv(override=True)
login(token=os.environ.get("HF_TOKEN"))

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

# Get input from the user
print("\nEnter your question or prompt (press Enter when done):")
user_input = input("\n> ")

# Function to calculate optimal MAX_LENGTH based on input size
def calculate_max_length(text, model_tokenizer):
    # Use the model's own tokenizer for accurate token counting
    try:
        # Get token count using the model's tokenizer
        token_count = len(model_tokenizer.encode(text))
        
        # Set reasonable bounds (min 64, max 4096)
        min_length = 64
        max_length = 4096
        
        # Set MAX_LENGTH to at least double the token count to allow room for generation
        # but cap it within reasonable bounds
        suggested_length = max(min_length, min(token_count * 2, max_length))
        
        print(f"Input token count (using Llama tokenizer): {token_count}")
        print(f"Setting MAX_LENGTH to: {suggested_length}")
        return suggested_length
    except Exception as e:
        print(f"Error calculating token length: {e}")
        print("Defaulting to MAX_LENGTH = 512")
        return 512

# Calculate MAX_LENGTH dynamically based on input
MAX_LENGTH = calculate_max_length(user_input, model.tokenizer) + 5

# Create a list with the user input
input_text = [user_input]

# Tokenize input with proper settings
print("\nTokenizing input...")
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
        max_new_tokens=2048,  # Maximum output token length for Llama-3.1-8B
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
            max_length=input_tokens['input_ids'].shape[1] + 2048  # Account for input tokens + maximum output length
        )
        
        output = model.tokenizer.decode(generation_output[0])
        print("\nSimplified generation successful! Output:")
        print(output)
        
    except Exception as e2:
        print(f"\nSimplified generation failed: {type(e2).__name__}: {e2}")
        print("\nThe error is likely due to compatibility issues between AirLLM and Llama-3.1-8B.")
        print("You may need to update AirLLM or use an older version of transformers.")

