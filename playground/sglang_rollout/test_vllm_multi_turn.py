# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import json
import os
from typing import Any, Dict

import numpy as np
import ray
from omegaconf import DictConfig, OmegaConf
from openai.types.chat.chat_completion import ChatCompletion
from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              ChatCompletionResponse,
                                              ChatCompletionStreamResponse,
                                              ErrorResponse)

from tests.workers.rollout.async_rollout_utils import \
    init_async_rollout_manager
from verl.protocol import DataProto

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["VLLM_USE_V1"] = "1"
os.environ["NCCL_DEBUG"] = "WARN"
os.environ["VLLM_LOGGING_LEVEL"] = "WARN"


def init_config() -> DictConfig:
    config = OmegaConf.load("verl/trainer/config/ppo_trainer.yaml")
    model_path = "/nobackup/model/qwen3/Qwen3-0.6B"
    config.actor_rollout_ref.model.path = model_path
    config.actor_rollout_ref.rollout.mode = "async"
    config.actor_rollout_ref.rollout.chat_scheduler = "examples.ppo_trainer.naive_chat_scheduler.NaiveChatCompletionScheduler"
    config.actor_rollout_ref.rollout.prompt_length = 4096
    config.actor_rollout_ref.rollout.response_length = 4096

    # test sleep/wake_up with fsdp offload
    config.actor_rollout_ref.actor.fsdp_config.param_offload = True
    config.actor_rollout_ref.actor.fsdp_config.optimizer_offload = True

    return config


def test_vllm_multi_turn(config):
    ray.init(
        runtime_env={
            # "env_vars": {
            #     # "TOKENIZERS_PARALLELISM": "true",
            #     "NCCL_DEBUG": "WARN",
            #     "VLLM_LOGGING_LEVEL": "WARN",
            #     "VLLM_USE_V1": "1",
            # }
        }
    )

    # =========================== 1. Init rollout manager ===========================
    model_name = "/".join(config.actor_rollout_ref.model.path.split("/")[-2:])
    async_rollout_manager = init_async_rollout_manager(config)

    print(f"async_rollout_manager: {async_rollout_manager}")

    # test sleep and wake_up
    async_rollout_manager.sleep()
    
    # Demonstrate partial wake-up (experimental approach to avoid collective operation hangs)
    print("\n=== Testing partial wake-up (experimental) ===")
    async_rollout_manager.wake_up(server_indices=[0, 1, 2, 3])  # True partial wake-up

    async_chat_scheduler = async_rollout_manager.chat_scheduler

    # =========================== 2. Multi turn rollout  ===========================
    async def callback(completions: ChatCompletion, info: Dict[str, Any], exception: Exception):
        # Handle exceptions first
        if exception is not None:
            print(f"[round={info.get('round', 'unknown')}] Error: {exception}")
            return
            
        # Handle None completions
        if completions is None:
            print(f"[round={info.get('round', 'unknown')}] Error: No completion returned")
            return
            
        messages, round = info["messages"], info["round"]
        message = completions.choices[0].message
        messages.append({"role": message.role, "content": message.content})
        print(f"[round={round}] role: {message.role}, content: {message.content}")

        extra_headers = {"x-request-id": completions.id}
        if round == 0:
            messages.append({"role": "user", "content": "What is your name?"})
            await async_chat_scheduler.submit_chat_completions(
                callback=callback,
                callback_additional_info={"messages": messages, "round": 1},
                model=model_name,
                messages=messages,
                extra_headers=extra_headers,
            )
        elif round == 1:
            messages.append({"role": "user", "content": "What is your favorite color?"})
            await async_chat_scheduler.submit_chat_completions(
                callback=callback,
                callback_additional_info={"messages": messages, "round": 2},
                model=model_name,
                messages=messages,
                extra_headers=extra_headers,
            )
        else:
            print("Done!")

    messages = [{"role": "user", "content": "Let's play a role playing game. Your name is Bob, your favorite color is red."}]
    async_rollout_manager.submit_chat_completions(
        callback=callback,
        callback_additional_info={"messages": messages, "round": 0},
        model=model_name,
        messages=messages,
    )
    print(f"messages: {messages}") #messages: [{'role': 'user', 'content': "Let's play a role playing game. Your name is Bob, your favorite color is red."}, {'role': 'assistant', 'content': "<think>\nOkay, the user wants a role-playing game where Bob is playing, with his favorite color red. Let me start by building the game. Roleplaying is a fun way to interact. I need to set up Bob's character with red as his favorite.\n\nFirst, I should define Bob's traits and abilities. Since he likes red, maybe he's a red-haired artist or someone who loves the color. His strengths could include his creativity and maybe some magical abilities related to colors. I need to make sure the conversation flows naturally, avoiding any complicated plot points.\n\nI should also consider what Bob might do in his game. Maybe he finds a magical object that uses red, or he has a task that requires red. Including elements like dialogue and choices keeps the game engaging. Need to check for any inconsistencies and ensure that the character actions and dialogue are coherent.\n</think>\n\nBob stepped onto the stage with a crimson sunhat, his red hair brushing against his cheeks. The air hummed with a whisper of uncommon energy.  \n\n**Bob**: (leaning over his workbench) Hey, the mural... it’s *red*! My latest creation. Color magic? Definitely.  \n\n**Bob**: (pulling a crimson stone from his pocket) Dad told me to protect the colors... but the sun isn’t always kind.  \n\n**Bob**: (gesturing to a glowing vortex above) This is a red lantern. Five minutes left on my journey to bring it to the tavern.  \n\n**Bob**: (pauses, then sighs)... Are you sure about the zone? I heard something... protective.  \n\n**Bob**: (nodding) Maybe it’s a trick. The only way to truly see the colors is if they’re *can’t* be *forbidden*.  \n\n*(the lantern glowed, and a faint sound of whispers shifted through the air.)*"}, {'role': 'user', 'content': 'What is your name?'}, {'role': 'assistant', 'content': '<think>\nOkay, the user asked for my name after I mentioned being Bob, red, and a sunhat. I need to respond naturally. Let me start with that information and then expand on my personality. Maybe add some details about my favorite color or actions. Also, considering the previous conversation where I mentioned the mural and the red lantern, I can connect these elements. Keeping the tone friendly and interesting is key here. Let me make sure it flows well and stays true to the scenario.\n</think>\n\nYour name is Bob, and your favorite color is red!  \nI’m a passionate storyteller and artist whose work blends magic and realism. My favorite action is crafting creations that bring in the wonders of the world, like the red lantern or the mural. Have any stories come to mind? Let me know! 🌈✨'}, {'role': 'user', 'content': 'What is your favorite color?'}, {'role': 'assistant', 'content': '<think>\nOkay, the user asked, "What is your favorite color?" and I already provided "red" as the answer. Now they\'re responding again with this same question. Let me check if they\'re repeating the same thing or changing the topic.\n\nHmm, maybe they want more information. I should acknowledge their question and offer to share. It\'s good to keep the conversation open. I should mention that my favorite color is red, reinforcing the previous answer. Also, since the story involves red, maybe adding that it\'s central to the world could be a nice touch. Let me make sure the response is friendly and engaging. Alright, I\'ll say something like "Yes, red is my favorite color!" and invite them to share their own favorite. That should cover it.\n</think>\n\nMy favorite color is red, and I still love the vibrant energy it brings to my creations! 🌟 What\'s your favorite color, too?'}]
    assert len(messages) == 6
    for round, message in enumerate(messages):
        if round % 2 == 0:
            assert message["role"] == "user"
        else:
            assert message["role"] == "assistant"

    # =========================== 3. Generate sequences  ===========================
    raw_prompts = [
        [
            {
                "role": "user",
                "content": "Let's play a role playing game. Your name is Alice, your favorite color is blue.",
            }
        ],
        [{"role": "user", "content": "Let's play a role playing game. Your name is Bob, your favorite color is red."}],
    ]
    batch = DataProto(
        non_tensor_batch={
            "raw_prompt": np.array(raw_prompts),
        },
    )
    result = async_rollout_manager.generate_sequences(prompts=batch)
    seq_len = result.batch["prompts"].size(1) + result.batch["responses"].size(1)
    assert len(result) == 2
    assert result.batch["input_ids"].size(1) == seq_len
    assert result.batch["attention_mask"].size(1) == seq_len
    assert result.batch["position_ids"].size(1) == seq_len

    ray.shutdown()


async def test_vllm_streaming_response(config):
    ray.init(
        runtime_env={
            "env_vars": {
                # "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "WARN",
                "VLLM_USE_V1": "1",
            }
        }
    )

    model_name = "/".join(config.actor_rollout_ref.model.path.split("/")[-2:])
    async_rollout_manager = init_async_rollout_manager(config)
    async_llm_server = async_rollout_manager.async_llm_servers[0]

    # non-streaming request
    request = ChatCompletionRequest(
        model=model_name,
        messages=[{"role": "user", "content": "What is your name?"}],
        stream=False,
    )
    generator = async_llm_server.chat_completion_generator.remote(request)
    async for ref in generator:
        status_code, data = await ref
        print(f">>>> status_code: {status_code}, {data}")
        data = data[len("data: ") :].rstrip()
        if status_code != 200:
            response = ErrorResponse(**json.loads(data))
        else:
            response = ChatCompletionResponse(**json.loads(data))
            assert response.choices[0].message.role == "assistant"
            assert response.choices[0].message.content is not None

    # streaming request
    request = ChatCompletionRequest(
        model=model_name,
        messages=[{"role": "user", "content": "How are you?"}],
        stream=True,
    )
    generator = async_llm_server.chat_completion_generator.remote(request)
    async for ref in generator:
        status_code, data = await ref
        print(f">>>> status_code: {status_code}, {data}")
        data = data[len("data: ") :].rstrip()
        if status_code != 200:
            response = ErrorResponse(**json.loads(data))
        elif data == "[DONE]":
            break
        else:
            response = ChatCompletionStreamResponse(**json.loads(data))
            assert response.choices[0].delta.role is None or response.choices[0].delta.role == "assistant"
            assert response.choices[0].delta.content is not None

    ray.shutdown()


if __name__ == "__main__":
    config = init_config()
    test_vllm_multi_turn(config)
    # asyncio.run(test_vllm_streaming_response(config))

# torchrun --standalone --nnodes=1 --nproc_per_node=8 $(which pytest) -s playground/sglang_rollout/test_vllm_multi_turn.py

# python3 playground/sglang_rollout/test_vllm_multi_turn.py