#!/usr/bin/env python3
"""
Batch process emotion classification using multi-image input for Llama 3.2 11B Vision.
Passes 10 frames at once to get a single unified classification per video row.
"""

import argparse
import torch
import pandas as pd
import os
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, MllamaForConditionalGeneration

MODEL_PATH = "/data/open-weight-llms/models/llama-3.2-11b-vision"
VALID_EMOTIONS = [
    "Neutral", "Happy", "Contempt", 
    "Disgust", "Surprised", "Sad", "Angry"
]

# Strict prompt for multi-frame analysis
PROMPT = (
    f"These images are frames from a single video.  "
    f"classify the exact emotion being displayed. "
    f"Choose exactly one from this list: {', '.join(VALID_EMOTIONS)}. "
    f"Output ONLY the emotion name, and nothing else."
)

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-image Emotion detection with Llama 3.2")
    parser.add_argument("--csv", type=str, default="index_sampled_k10_m5.csv")
    parser.add_argument("--output", type=str, default="results_llama_multi_frame.csv")
    parser.add_argument("--base-dir", type=str, default="/home/videep/research/cvpr-edu-affect-2026")
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"GPUs visible to PyTorch: {torch.cuda.device_count()}")

    # 1. Load Data
    df = pd.read_csv(args.csv)
    frame_cols = [f"frame_{i:02d}" for i in range(1, 11)]
    
    if 'multi_frame_pred' not in df.columns:
        df['multi_frame_pred'] = None

    # 2. Load Model & Processor
    print(f"Loading Llama 3.2 Vision from: {MODEL_PATH}")
    # use_fast=False is safer for the vision tower in this architecture
    processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=False)
    
    model = MllamaForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0", 
    )

    print("Starting multi-image inference...")
    
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing videos"):
        images = []
        for col in frame_cols:
            image_rel_path = row.get(col)
            if pd.notna(image_rel_path):
                full_path = os.path.join(args.base_dir, str(image_rel_path))
                try:
                    images.append(Image.open(full_path).convert("RGB"))
                except:
                    continue

        if not images:
            continue

        try:
           
         
            content = [{"type": "image"} for _ in range(len(images))]
            content.append({"type": "text", "text": PROMPT})
            
            messages = [{"role": "user", "content": content}]
            
            # Apply template
            input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
            
            # Process with add_special_tokens=False to prevent the double-token gibberish bug
            inputs = processor(
                images=images, 
                text=input_text, 
                add_special_tokens=False, 
                return_tensors="pt"
            ).to(model.device)

            # Generate
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs, 
                    max_new_tokens=15, 
                    do_sample=False,  # Prevents NaN/Inf CUDA asserts
                    temperature=None,
                    top_p=None
                )

          
            prompt_length = inputs["input_ids"].shape[-1]
            generated_ids = output_ids[0][prompt_length:]
            response = processor.decode(generated_ids, skip_special_tokens=True).strip()
            
            df.at[index, 'multi_frame_pred'] = response
                
        except Exception as e:
            df.at[index, 'multi_frame_pred'] = f"ERROR: {str(e)}"
        
        # Incremental save
        df.to_csv(args.output, index=False)

    print(f"\nProcessing complete! Results saved to: {args.output}")

if __name__ == "__main__":
    main()