import argparse
import asyncio
import json
import os
import time
from types import SimpleNamespace

import numpy as np
import requests

from sglang.bench_serving import benchmark, set_global_args
from sglang.srt.server_args import ServerArgs
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    kill_process_tree,
    popen_launch_server,
)


def node0_print(msg):
    if server_args.node_rank == 0:
        print(msg)


prompts = [
    "Human: Give me a fully functional FastAPI server. Show the full, long python code without stop.\n\nAssistant:",
    "Human: Imagine you are an experienced Ethereum developer tasked with creating a smart contract for a blockchain messenger. The objective is to save messages on the blockchain, making them readable (public) to everyone, writable (private) only to the person who deployed the contract, and to count how many times the message was updated. Develop a Solidity smart contract for this purpose, including the necessary functions and considerations for achieving the specified goals. Please provide the code and any relevant explanations to ensure a clear understanding of the implementation.\n\nAssistant:",
    "Human: Write a travel blog post to Hawaii.\n\nAssistant:",
    "Human: I want you to act as an English translator, spelling corrector and improver. I will speak to you in any language and you will detect the language, translate it and answer in the corrected and improved version of my text, in English. I want you to replace my simplified A0-level words and sentences with more beautiful and elegant, upper level English words and sentences. Keep the meaning same, but make them more literary. My first sentence is 'istanbulu cok seviyom burada olmak cok guzel'. Answer in more than 5000 words.\n\nAssistant:",
    "Human: I want you to act as a storyteller. You will come up with entertaining stories that are engaging, imaginative and captivating for the audience. It can be fairy tales, educational stories or any other type of stories which has the potential to capture people's attention and imagination. Depending on the target audience, you may choose specific themes or topics for your storytelling session e.g., if it's children then you can talk about animals; If it's adults then history-based tales might engage them better etc. Answer in more than 5000 words. My first request is 'I need an interesting story on perseverance.'\n\nAssistant:",
    "Human: Solve x^2 = -1. Think step-by-step. Give me a long detailed explanation. \n\nAssistant:",
    "Human: Tell me about the president of the USA in wikipedia style.\n\nAssistant:",
    "Human: Hello? Who are you? Write code, math, and poem to explanin yourself.\n\nAssistant:",
]

prompts = [
    "user\nGiven the expression $(x-y)^3 \\div (x-y)^2 \\cdot (y-x)$, find the simplified form.\n\nPresent the answer in LaTex format: \\boxed{Your answer}\nassistant\n",
    "user\nGiven the set \\( A = \\{x \\mid x^{2} - 3x - 10 \\leq 0\\} \\) and \\( B = \\{x \\mid m + 1 \\leq x \\leq 2m - 1\\} \\), if \\( A \\cup B = A \\), determine the range of the real number \\( m \\).\n\nPresent the answer in LaTex format: \\boxed{Your answer}\nassistant\n",
    "user\nGiven $\\overrightarrow{a}=(2\\sin \\alpha,1)$, $\\overrightarrow{b}=(\\cos \\alpha,1)$, $\\alpha \\in (0, \\frac{\\pi}{4})$.\n(1) If $\\overrightarrow{a} \\parallel \\overrightarrow{b}$, find the value of $\\tan \\alpha$;\n(2) If $\\overrightarrow{a} \\cdot \\overrightarrow{b} = \\frac{9}{5}$, find the value of $\\sin(2\\alpha + \\frac{\\pi}{4})$.\n\nPresent the answer in LaTex format: \\boxed{Your answer}\nassistant\n",
    "user\nLet  $f: \\mathbb{R} \\rightarrow \\mathbb{R}$  be a function such that   $$ \\displaystyle{f(f(x)) = \\frac{x^2 - x}{2}\\cdot f(x) + 2-x,} $$   for all  $x \\in  \\mathbb{R}.$  Find all possible values of  $f(2).$ \n\nPresent the answer in LaTex format: \\boxed{Your answer}\nassistant\n",
]

# prompts = [
#     "system\n\nWhen tackling complex reasoning tasks, you have access to the following actions. Use them as needed to progress through your thought process.\n\n[ASSESS]\n\n[ADVANCE]\n\n[VERIFY]\n\n[SIMPLIFY]\n\n[SYNTHESIZE]\n\n[PIVOT]\n\n[OUTPUT]\n\nYou should strictly follow the format below:\n\n[ACTION NAME]\n\n# Your action step 1\n\n# Your action step 2\n\n# Your action step 3\n\n...\n\nNext action: [NEXT ACTION NAME]\n\n\nuser\nGiven the expression $(x-y)^3 \\div (x-y)^2 \\cdot (y-x)$, find the simplified form.\n\nPresent the answer in LaTex format: \\boxed{Your answer}\nassistant\n",
#     "system\n\nWhen tackling complex reasoning tasks, you have access to the following actions. Use them as needed to progress through your thought process.\n\n[ASSESS]\n\n[ADVANCE]\n\n[VERIFY]\n\n[SIMPLIFY]\n\n[SYNTHESIZE]\n\n[PIVOT]\n\n[OUTPUT]\n\nYou should strictly follow the format below:\n\n[ACTION NAME]\n\n# Your action step 1\n\n# Your action step 2\n\n# Your action step 3\n\n...\n\nNext action: [NEXT ACTION NAME]\n\n\nuser\nGiven the set \\( A = \\{x \\mid x^{2} - 3x - 10 \\leq 0\\} \\) and \\( B = \\{x \\mid m + 1 \\leq x \\leq 2m - 1\\} \\), if \\( A \\cup B = A \\), determine the range of the real number \\( m \\).\n\nPresent the answer in LaTex format: \\boxed{Your answer}\nassistant\n",
#     "system\n\nWhen tackling complex reasoning tasks, you have access to the following actions. Use them as needed to progress through your thought process.\n\n[ASSESS]\n\n[ADVANCE]\n\n[VERIFY]\n\n[SIMPLIFY]\n\n[SYNTHESIZE]\n\n[PIVOT]\n\n[OUTPUT]\n\nYou should strictly follow the format below:\n\n[ACTION NAME]\n\n# Your action step 1\n\n# Your action step 2\n\n# Your action step 3\n\n...\n\nNext action: [NEXT ACTION NAME]\n\n\nuser\nGiven $\\overrightarrow{a}=(2\\sin \\alpha,1)$, $\\overrightarrow{b}=(\\cos \\alpha,1)$, $\\alpha \\in (0, \\frac{\\pi}{4})$.\n(1) If $\\overrightarrow{a} \\parallel \\overrightarrow{b}$, find the value of $\\tan \\alpha$;\n(2) If $\\overrightarrow{a} \\cdot \\overrightarrow{b} = \\frac{9}{5}$, find the value of $\\sin(2\\alpha + \\frac{\\pi}{4})$.\n\nPresent the answer in LaTex format: \\boxed{Your answer}\nassistant\n",
#     "system\n\nWhen tackling complex reasoning tasks, you have access to the following actions. Use them as needed to progress through your thought process.\n\n[ASSESS]\n\n[ADVANCE]\n\n[VERIFY]\n\n[SIMPLIFY]\n\n[SYNTHESIZE]\n\n[PIVOT]\n\n[OUTPUT]\n\nYou should strictly follow the format below:\n\n[ACTION NAME]\n\n# Your action step 1\n\n# Your action step 2\n\n# Your action step 3\n\n...\n\nNext action: [NEXT ACTION NAME]\n\n\nuser\nLet  $f: \\mathbb{R} \\rightarrow \\mathbb{R}$  be a function such that   $$ \\displaystyle{f(f(x)) = \\frac{x^2 - x}{2}\\cdot f(x) + 2-x,} $$   for all  $x \\in  \\mathbb{R}.$  Find all possible values of  $f(2).$ \n\nPresent the answer in LaTex format: \\boxed{Your answer}\nassistant\n",
# ]


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False):
        return []


def send_one_batch(base_url, num_prompts, batch_size):

    system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."

    # Add system prompt before each prompt
    new_prompts = [system_prompt + p for p in prompts]
    padded_prompts = (new_prompts * ((num_prompts + len(new_prompts) - 1) // len(new_prompts)))[:num_prompts]

    # format: (prompt, input_len, output len). We set input_len as a dummy value 0.
    input_requests = [(p, 0, 512) for p in padded_prompts]
    print(input_requests)

    # We need to set some dummy values in order to call `benchmark` below.
    args = SimpleNamespace(
        disable_ignore_eos=False,
        disable_stream=False,
        return_logprob=False,
        backend="sglang",
        dataset_name="custom",
        num_prompts=None,
        sharegpt_output_len=None,
        random_input_len=None,
        random_output_len=None,
        random_range_ratio=None,
        output_file=None,
    )
    set_global_args(args)
    tokenizer = FakeTokenizer()

    # Run benchmark
    results = asyncio.run(
        benchmark(
            backend="sglang",
            api_url=f"{base_url}/generate",
            base_url=base_url,
            model_id="default",
            tokenizer=tokenizer,
            input_requests=input_requests,
            request_rate=float("inf"),
            max_concurrency=batch_size,
            disable_tqdm=False,
            lora_name=None,
            extra_request_body={},
            profile=None,
        )
    )

    print(results)
    exit()

    assert results["completed"] == len(input_requests)
    acc_length = results["accept_length"] or 1.0
    avg_output_token = results["total_output_tokens"] / results["completed"]

    server_info = requests.get(base_url + "/get_server_info").json()
    # We use 20% percentile instead of median on purpose
    step_time = np.percentile(server_info["step_time_dict"][str(batch_size)], 20)
    speed = 1 / step_time * acc_length

    return (
        round(acc_length, 3),
        round(step_time, 5),
        round(speed, 3),
        avg_output_token,
    )


def main(args, server_args):
    base_url = "http://127.0.0.1:20000"

    configs = []
    for batch_size in args.batch_size:
        for steps in args.steps:
            for topk in args.topk:
                for num_draft_tokens in args.num_draft_tokens:
                    if steps * topk + 1 < num_draft_tokens:
                        continue

                    if (steps == 0 or topk == 0 or num_draft_tokens == 0) and (steps + topk + num_draft_tokens != 0):
                        # steps == 0 and topk == 0 and num_draft_tokens == 0 is a special case for non-speculative decoding.
                        continue

                    configs.append((batch_size, steps, topk, num_draft_tokens))

    for i in range(args.start, args.end or len(configs)):
        batch_size, steps, topk, num_draft_tokens = configs[i]

        node0_print(f"Start {i=}: {batch_size=}, {steps=}, {topk=}, {num_draft_tokens=}")

        # Create an LLM.
        if steps == 0:
            other_args = []
        else:
            # Only add speculative args if they're not already in server_args
            other_args = [
                "--speculative-algorithm",
                args.speculative_alg,
                "--speculative-num-steps",
                steps,
                "--speculative-eagle-topk",
                topk,
                "--speculative-num-draft-tokens",
                num_draft_tokens,
            ]
            # Add draft model path last to avoid duplication
            if server_args.speculative_draft_model_path is not None:
                other_args.extend(["--speculative-draft-model-path", server_args.speculative_draft_model_path])

        other_args.extend(
            [
                "--cuda-graph-max-bs",
                batch_size,
                "--max-running-requests",
                batch_size,
                "--mem-fraction-static",
                server_args.mem_fraction_static,
                "--tp-size",
                server_args.tp_size,
                "--dtype",
                server_args.dtype,
            ]
        )

        if server_args.trust_remote_code:
            other_args.extend(["--trust-remote-code"])

        if server_args.enable_flashinfer_mla:
            other_args.extend(["--enable-flashinfer-mla"])

        if server_args.quantization:
            other_args.extend(["--quantization", server_args.quantization])

        process = popen_launch_server(
            args.model_path,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            env={
                "SGLANG_RECORD_STEP_TIME": "1",
                **os.environ,
            },
        )

        try:
            # Warmup
            send_one_batch(base_url, batch_size, batch_size)

            # Benchmark
            acc_length, step_time, speed, completion_tokens = send_one_batch(
                base_url, max(args.num_prompts, batch_size), batch_size
            )
        finally:
            kill_process_tree(process.pid)

        node0_print(
            f"Finish {i=}: {batch_size=}, {steps=}, {topk=}, {num_draft_tokens=}, {speed=:.2f} token/s, step_time={step_time * 1000:.2f} ms"
        )

        record = {
            "model_path": args.model_path,
            "batch_size": batch_size,
            "steps": steps,
            "topk": topk,
            "num_draft_tokens": num_draft_tokens,
            "acc_length": acc_length,
            "step_time": step_time,
            "speed": speed,
            "completion_tokens": completion_tokens,
        }

        with open(args.output, "a") as fout:
            fout.write(json.dumps(record) + "\n")

        # Wait for the server to shutdown
        time.sleep(5)


# The __main__ condition is necessary here because we use "spawn" to create subprocesses
# Spawn starts a fresh program every time, if there is no __main__, it will run into infinite loop to keep spawning processes from sgl.Engine
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    parser.add_argument("--batch-size", type=int, nargs="+", default=(1, 2, 4, 8, 16))
    parser.add_argument("--steps", type=int, nargs="+", default=(0, 1, 3, 5, 7))
    parser.add_argument("--topk", type=int, nargs="+", default=(0, 1, 2, 4, 8))
    parser.add_argument("--num_draft_tokens", type=int, nargs="+", default=(0, 2, 4, 8, 16, 32))
    parser.add_argument("--num-prompts", type=int, default=4)
    # parser.add_argument("--mem-fraction", type=float, default=0.8)
    # parser.add_argument("--dtype", type=str, default="float16")
    # parser.add_argument("--enable-torch-compile", action="store_true")
    parser.add_argument("--speculative-alg", type=str, default="EAGLE3")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--output", type=str, default="output.jsonl")
    args = parser.parse_args()
    server_args: ServerArgs = ServerArgs.from_cli_args(args)

    main(args, server_args)
