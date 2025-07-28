import numpy as np
import torch
import os
from diffusers import StableDiffusionPipeline
from dotenv import load_dotenv

load_dotenv()

def load_model():
    """
    Loads the Stable Diffusion model from local .safetensors files and returns a callable pipeline for inference.
    """
    model_path = "./sd3_medium_incl_clips_t5xxlfp16.safetensors"  # Path to the local model file
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading Stable Diffusion model from: {model_path}")
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        local_files_only=True  # Ensures the model is loaded locally
    )
    pipeline = pipeline.to(device)

    print("Model loaded successfully.")
    return pipeline


def load_data():
    pass




def main():
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Train model on CSD data")
    parser.add_argument("--data_dir", type=str, default='./data', help="Directory containing training data")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs for training")
    args = parser.parse_args()

    print("Loading model...")
    model = load_model()

    print("Model is ready for inference.")
    # Example inference
    prompt = "A futuristic cityscape at sunset"
    image = model(prompt).images[0]
    image.save("output.png")
    print("Inference completed. Image saved as output.png.")


if __name__ == '__main__':
    main()