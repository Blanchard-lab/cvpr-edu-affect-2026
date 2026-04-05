#!/usr/bin/env python3
"""
Batch process emotion classification using multi-image input for Qwen 2.5 VL 7B.
Passes 10 frames at once to get a single unified classification per video row.
"""

import argparse
import torch
import pandas as pd
import os
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

MODEL_PATH = ""
VALID_EMOTIONS = [
    "Neutral", "Happy", "Contempt", 
    "Disgust", "Surprised", "Sad", "Angry"
]


PROMPT = (
    f"These images are frames from a single video. "
    f"classify the exact emotion being displayed. "
    f"Choose exactly one from this list: {', '.join(VALID_EMOTIONS)}. "
    f"Output ONLY the emotion name, and nothing else."
)

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-image Emotion detection with Qwen 2.5 VL")
    parser.add_argument("--csv", type=str, default="index_sampled_k10_m5.csv")
    parser.add_argument("--output", type=str, default="results_qwen_multi_frame.csv")
    parser.add_argument("--base-dir", type=str, default="/home/videep/research/cvpr-edu-affect-2026")
    return parser.parse_args()

def main():
    args = parse_args()
    
    df = pd.read_csv(args.csv)
    frame_cols = [f"frame_{i:02d}" for i in range(1, 11)]
    
    if 'multi_frame_pred' not in df.columns:
        df['multi_frame_pred'] = None

    print(f"Loading Qwen 2.5 VL model from: {MODEL_PATH}")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
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
            
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            inputs = processor(
                text=[text],
                images=images, # Pass the list of 10 images here
                padding=True,
                return_tensors="pt"
            ).to(model.device)

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs, 
                    max_new_tokens=10, 
                    do_sample=False
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0].strip()
            
            df.at[index, 'multi_frame_pred'] = response
                
        except Exception as e:
            df.at[index, 'multi_frame_pred'] = f"ERROR: {str(e)}"
        
        # Save results incrementally
        df.to_csv(args.output, index=False)

    print(f"\nSaved multi-frame results to: {args.output}")

if __name__ == "__main__":
    main()