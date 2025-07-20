<h1 style="text-align: center;">fastrl: Efficient Reinforcement Learning for LLM Based on VeRL</h1>





# Installation

```bash
git clone --recursive https://github.com/mit-han-lab/fastrl.git
conda create --name fastrl python=3.12
conda activate fastrl
pip install -e .
```

Install liger-kernel and flash_attn
```bash
pip install liger-kernel
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```


Install SGLang

```bash
cd third-party/sglang
pip install -e "python[dev]" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python
```
If your editor cannot find the sglang module, you can add the following to your `.env` file:
```bash
PYTHONPATH=${PYTHONPATH}:/YOUR_PATH/fastrl/third-party/sglang/python
```

Install Megatron

```bash
pip install megatron-core
pip install --no-build-isolation transformer_engine[pytorch]
```

<details>
<summary>Install vLLM</summary>

```bash
cd third-party/vllm
VLLM_USE_PRECOMPILED=1 pip install -e .
```
If your editor cannot find the vllm module, you can add the following to your `.env` file:
```bash
PYTHONPATH=${PYTHONPATH}:/YOUR_PATH/fastrl/third-party/vllm
```

Install flashinfer (optional)
```bash
pip install https://github.com/flashinfer-ai/flashinfer/releases/download/v0.2.3/flashinfer_python-0.2.3+cu124torch2.6-cp38-abi3-linux_x86_64.whl
```
</details>

# RL Training

Data: [Eurus-2-RL-Data](https://huggingface.co/datasets/PRIME-RL/Eurus-2-RL-Data)

```bash
bash example/train_grpo.sh
```

# SFT

```bash
bash example/run_sft.sh
```

### Skip tokenization

(DO ONLY ONCE) Add following code to the end of `_prepare_dataset` in `trl.trainer.sft_trainer` to cache tokenized dataset. Note the `cache_path` must concat ***tokenized*** after the original dataset path.
```python
print(f"Caching tokenized dataset")
print(dataset)
cache_path="/nobackup/xxx/dataset/reasoning/OpenR1-Math-220k/data/tokenized"
os.makedirs(cache_path, exist_ok=True)
dataset.save_to_disk(cache_path)
exit(0)
```

Then you can add `--use_cached_dataset True \` in `run_sft.sh` to skip tokenization.

# Evaluation


`fastrl evaluate (./fastrl/evaluate)` is deprecated (i.e., following instructions below). Try `./evaluate` instead (adopted from QwQ).

Eval one task

```bash
fastrl evaluate --model /nobackup/model/deepseek-r1/DeepSeek-R1-Distill-Qwen-7B --dataset AIME24 --split train --tp 4 --temperatures 0.0
```

Eval multiple tasks
```bash
bash example/eval.sh
```

After evaluating, you can summarize results in `eval/summary_results.csv` across tasks via
```bash
python example/parse_results.py
```
> \[!Note\]
> `livecodebench`'s `results.json` file is large (8GB), please take care to clean it.
··

# Acknowledgement

Training based on [verl](https://github.com/volcengine/verl), evaluation based on [skythrought](https://github.com/NovaSky-AI/SkyThought), [open-r1](https://github.com/huggingface/open-r1), [Qwen2.5-Math](https://github.com/QwenLM/Qwen2.5-Math).


