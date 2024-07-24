import re
import textwrap
import argparse
from pathlib import Path
import tqdm
import jsonlines
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import os
from datasets import load_dataset
import pandas as pd

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def generate_sample(model, tokenizer, question):
    prompt = (
        f"'Describe the input SMILES molecule.\n{question}\n"
    )

    #messages = [{"role": "user", "content": prompt}]
    #inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(DEVICE)

    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False).to(DEVICE)
    response = model.generate(**inputs, 
                         #do_sample=True,
                         #temperature=0.0, 
                         #top_p=None,
                         #num_beams=3,
                         #no_repeat_ngram_size=3,
                         eos_token_id=tokenizer.eos_token_id,  # End of sequence token
                         pad_token_id=tokenizer.eos_token_id,  # Pad token
                         max_new_tokens=256,
                        )
    #output = tokenizer.decode(response.squeeze()[inputs.shape[1]:], skip_special_tokens=True)
    output = tokenizer.decode(response.squeeze()[len(inputs['input_ids'][0]):], skip_special_tokens=True)
    return output



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test HF checkpoint.")
    parser.add_argument(
        "-c",
        "--checkpoint-path",
        type=Path,
        help="Checkpoint path",
    )
    parser.add_argument(
        "-f",
        "--sample-input-file",
        type=str,
        default=None,
        help="LPM24-eval for captioning",
    )
    parser.add_argument(
        "-o", "--sample-output-file", type=str, default="smi2cap_submit.txt"
    )

    args = parser.parse_args()
    print("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path, trust_remote_code=True
    )

    print("Loading model ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path,
        device_map="auto",
        trust_remote_code=True,
    ).eval()
    model.generation_config = GenerationConfig.from_pretrained(
        args.checkpoint_path, trust_remote_code=True
    )
    model.generation_config.do_sample = False  # use greedy decoding
    model.generation_config.repetition_penalty = 1.0  # disable repetition penalty

    os.makedirs(os.path.dirname(args.sample_output_file), exist_ok=True)
    lpm_eval=load_dataset("language-plus-molecules/LPM-24_eval-caption")
    lpm_eval=pd.DataFrame(lpm_eval['train'])
    lpm_eval=lpm_eval[:7314]
    with open(args.sample_output_file, 'w') as output:
        for task_id, input in enumerate(tqdm.tqdm(lpm_eval['molecule'])):
            response = generate_sample(
                model, tokenizer, input
            )
            output.write(response.replace('\n',' ')+'\n')
