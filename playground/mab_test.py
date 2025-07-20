import time

import torch
from sglang.srt.entrypoints.engine import Engine

spec_algorithm = "EAGLE"  # [None, "EAGLE", "EAGLE3", "LOOKAHEAD"]

def main():
    # Sample prompts.
    prompts = [
        # "Juan asked his neighbor, Herb, how much his house was worth. Herb answered that he paid $76,000 for the house. If Juan's house is 30% less expensive than Herb's, calculate the value of the two houses combined.",
        "Mike decides he wants to replace his movie collection with digital versions.  He has 600 movies.  A third of the movies are in various series and he knows he can get those for only $6 of the cost of a normal movie by just buying the series together.  40% of the remaining movies are older movies which are $5.  How much does replacing the movies cost if a normal movie costs $10?",
    ]

    # Create a sampling params object.
    sampling_params = {
        "n": 1,
        "temperature": 0.0,
        # "repetition_penalty": 1,
        "max_new_tokens": 2048,
        # "top_k": 1,
    }
    # Set speculative args based on algorithm
    speculative_args = {}
    if spec_algorithm == "EAGLE3":
        speculative_args = {
            "speculative_algorithm": "EAGLE3",
            "speculative_draft_model_path": "/dataset/model/eagle/sglang-EAGLE3-LLaMA3.1-Instruct-8B",
            "speculative_num_steps": 8,
            "speculative_eagle_topk": 8,
            "speculative_num_draft_tokens": 64,
            "speculative_eagle_mab_algorithm": "EG",
            "speculative_eagle_mab_configs": ["8_8_48","8_4_8"], # "8_8_32", "8_8_16", 
            "speculative_mab_window_size": 100,
            "mem_fraction_static": 0.65,
        }
    elif spec_algorithm == "EAGLE":
        speculative_args = {
            "speculative_algorithm": "EAGLE",
            "speculative_draft_model_path": "/dataset/model/eagle/sglang-EAGLE-LLaMA3-Instruct-8B",
            "speculative_num_steps": 8,
            "speculative_eagle_topk": 8,
            "speculative_num_draft_tokens": 48,
            "speculative_eagle_mab_algorithm": "PREDEFINED", # "EG", "PREDEFINED", "UCB1"
            "speculative_eagle_mab_configs": ["8_8_32","8_8_16","8_8_8"], # "8_8_32", "8_8_16", 
            "speculative_mab_window_size": 100,
            "mem_fraction_static": 0.65,
        }
    else:
        raise ValueError(f"Unsupported speculative algorithm: {spec_algorithm}")

    # Create an LLM.
    llm = Engine(
        model_path="/dataset/model/llama3.1/Llama-3.1-8B-Instruct",
        dtype="float16",
        cuda_graph_max_bs=128,
        tp_size=1,
        **speculative_args,
    )

    for idx in range(50):
        torch.cuda.synchronize()
        start = time.time()
        outputs = llm.generate(prompts, sampling_params)
        torch.cuda.synchronize()
        cos = time.time() - start
        completion_tokens = 0
        verify_tokens = 0

        # Print the outputs.
        for prompt, output in zip(prompts, outputs):
            completion_tokens += output["meta_info"]["completion_tokens"]
            has_verify = "spec_verify_ct" in output["meta_info"]
            if has_verify:
                verify_tokens += output["meta_info"]["spec_verify_ct"]
            else:
                verify_tokens += output["meta_info"]["completion_tokens"]
            print(f"{output["meta_info"]}")
            print("======================" * 3)
        accept_length = completion_tokens / verify_tokens if verify_tokens > 0 else 1.0
        print(f"Run:{idx + 1}, {spec_algorithm}, Accept length: {accept_length:.3f}, TPS =: {completion_tokens/cos}\n\n")

if __name__ == "__main__":
    main()
