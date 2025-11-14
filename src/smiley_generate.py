"""
SmileyLlama inference utilities for constraint-based generation.
Provides model loading and SMILES generation functions used by smiley_generate_constraint.
"""
from __future__ import annotations

import random
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .utils import all_valid_smiles

MODEL_NAME = "THGLab/Llama-3.1-8B-SmileyLlama-1.1"

# System message for SmileyLlama prompt template
SMILEY_SYSTEM_MSG = "You love and excel at generating SMILES strings of drug-like molecules"


def _format_smiley_prompt(user_instruction: str) -> str:
    """
    Format prompt for SmileyLlama using the expected template format.
    
    Format: ### Instruction:\n{system}\n\n### Input:\n{user}\n\n### Response:\n
    This matches the format expected by SmileyLlama model.
    """
    return f"### Instruction:\n{SMILEY_SYSTEM_MSG}\n\n### Input:\n{user_instruction}\n\n### Response:\n"


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(
    device: Optional[str] = None,
    quantize: bool = True,
    low_cpu_mem_usage: bool = True,
    max_memory: Optional[dict] = None,
) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": low_cpu_mem_usage,
        "torch_dtype": torch.float16 if device.startswith("cuda") else torch.float32,
    }
    if device.startswith("cuda"):
        model_kwargs["device_map"] = "auto"
        if max_memory:
            model_kwargs["max_memory"] = max_memory
    if quantize and device.startswith("cuda"):
        try:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        except Exception:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)
    except Exception:
        # Fallback to full precision if quantized load fails.
        model_kwargs.pop("quantization_config", None)
        model_kwargs["torch_dtype"] = torch.float16 if device.startswith("cuda") else torch.float32
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)
    if not device.startswith("cuda"):
        model.to(device)
    model.eval()
    return tokenizer, model


def _gather_smiles(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    prompt: str,  # This should be the formatted prompt with ### Instruction/### Input/### Response
    target_n: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    batch_size: int,
) -> List[str]:
    device = next(model.parameters()).device
    collected: List[str] = []
    seen = set()
    
    # Extract the response marker to find where the model's response starts
    response_marker = "### Response:\n"
    
    # Create progress bar for Smiley generation
    pbar = tqdm(
        total=target_n,
        desc="Generating SmileyLlama molecules",
        unit="mol",
        ncols=80,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )
    
    batch_count = 0
    try:
        while len(collected) < target_n:
            batch_count += 1
            prompts = [prompt] * batch_size
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            valid_count = 0
            batch_molecules = []
            for text in decoded:
                # Extract response part (after "### Response:\n")
                if response_marker in text:
                    response_start = text.find(response_marker) + len(response_marker)
                    body = text[response_start:].strip()
                else:
                    # Fallback: remove prompt prefix if response marker not found
                    body = text[len(prompt) :].strip() if text.startswith(prompt) else text.strip()
                
                smiles_list = all_valid_smiles(body)
                for smi in smiles_list:
                    if smi not in seen:
                        seen.add(smi)
                        collected.append(smi)
                        batch_molecules.append(smi)
                        valid_count += 1
                        if len(collected) >= target_n:
                            break
                if len(collected) >= target_n:
                    break
            
            # Update progress bar with batch of molecules
            if batch_molecules:
                pbar.update(len(batch_molecules))
            
            # Update progress bar with current status
            pbar.set_postfix({
                "Batch": f"{batch_count}",
                "Valid": f"{valid_count}",
                "Temp": f"{temperature:.1f}",
                "Total": f"{len(collected)}/{target_n}"
            })
    finally:
        pbar.close()
    
    return collected[:target_n]
