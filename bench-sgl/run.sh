
# llama2-7B
python bench-sgl/bench_speculative.py.py --model-path /nobackup/model/llama2/Llama-2-7b-chat-hf --speculative-draft-model-path /nobackup/model/eagle-sgl/sglang-EAGLE-llama2-chat-7B --batch-size 8 --steps 0 5 --topk 0 4 --num_draft_tokens 0 8 --enable-torch-compile --cuda-graph-max-bs 2 --dtype half

# llama3-8B
python bench-sgl/bench_speculative.py --model-path /nobackup/model/llama3/Meta-Llama-3-8B-Instruct --speculative-draft-model-path /nobackup/model/eagle-sgl/sglang-EAGLE-LLaMA3-Instruct-8B --batch-size 1 --steps 5 --topk 4 --num_draft_tokens 8 --mem-fraction 0.7 --trust-remote-code --dtype float16 --enable-torch-compile

# llama3.1-8B
python bench-sgl/bench_speculative.py --model-path /nobackup/model/llama3.1/Llama-3.1-8B-Instruct --speculative-draft-model-path /nobackup/qinghao/runs/eagle/Llama-3.1-8B-Instruct/state_100 --batch-size 1 --steps 0 5 --topk 0 4 --num_draft_tokens 0 8 --mem-fraction 0.8 --trust-remote-code --dtype float16 --enable-torch-compile


# llama3.1-8B Eagle3
python bench-sgl/bench_speculative.py --model-path /nobackup/model/llama3.1/Llama-3.1-8B-Instruct --speculative-draft-model-path /nobackup/model/eagle-sgl/sglang-EAGLE3-LLaMA3.1-Instruct-8B --batch-size 1 --steps 0 8 --topk 0 8 --num_draft_tokens 0 64 --mem-fraction 0.8 --trust-remote-code --dtype float16 --enable-torch-compile --speculative-algorithm EAGLE3

# DeepSeek-llama3.1-8B Eagle3
python bench-sgl/bench_speculative.py --model-path /nobackup/model/deepseek-r1/DeepSeek-R1-Distill-Llama-8B --speculative-draft-model-path /nobackup/model/eagle-sgl/sglang-EAGLE3-DeepSeek-R1-Distill-LLaMA-8B --batch-size 1 --steps 0 8 --topk 0 8 --num_draft_tokens 0 64 --mem-fraction 0.8 --trust-remote-code --dtype float16 --enable-torch-compile --speculative-alg EAGLE3

================================================================================================================
# Llama-3.1-8B-Instruct
python -m sglang.launch_server --model-path /nobackup/model/llama3.1/Llama-3.1-8B-Instruct --port 30000 --cuda-graph-max-bs 1

python -m sglang.launch_server --model /nobackup/model/llama3.1/Llama-3.1-8B-Instruct --speculative-algo EAGLE3 \
    --speculative-draft /nobackup/model/eagle-sgl/sglang-EAGLE3-LLaMA3.1-Instruct-8B --speculative-num-steps 8 \
    --speculative-eagle-topk 8 --speculative-num-draft-tokens 64 --speculative-token-map /nobackup/model/eagle-sgl/sglang-EAGLE-LLaMA3-Instruct-8B/freq_32768.pt \
    --cuda-graph-max-bs 1 --mem-fraction 0.7 --dtype float16 --port 30000


# DeepSeek-R1-Distill-Llama-8B 
python -m sglang.launch_server --model-path /nobackup/model/deepseek-r1/DeepSeek-R1-Distill-Llama-8B --port 30000 --cuda-graph-max-bs 1

python -m sglang.launch_server --model /nobackup/model/deepseek-r1/DeepSeek-R1-Distill-Llama-8B --speculative-algo EAGLE3 \
    --speculative-draft /nobackup/model/eagle-sgl/sglang-EAGLE3-DeepSeek-R1-Distill-LLaMA-8B --speculative-num-steps 8 \
    --speculative-eagle-topk 8 --speculative-num-draft-tokens 64 \
    --cuda-graph-max-bs 1 --mem-fraction 0.7 --dtype float16 --port 30000


# Eagle-2 Llama3
python -m sglang.launch_server --model-path /nobackup/model/llama3/Meta-Llama-3-8B-Instruct  --port 30000 --cuda-graph-max-bs 1

python -m sglang.launch_server --model /nobackup/model/llama3/Meta-Llama-3-8B-Instruct --speculative-algo EAGLE \
    --speculative-draft /nobackup/model/eagle-sgl/sglang-EAGLE-LLaMA3-Instruct-8B --speculative-num-steps 8 \
    --speculative-eagle-topk 6 --speculative-num-draft-tokens 32 \
    --cuda-graph-max-bs 1 --mem-fraction 0.7 --dtype float16 --port 30000

python -m sglang.launch_server --model /nobackup/model/llama3/Meta-Llama-3-8B-Instruct --speculative-algo EAGLE \
    --speculative-draft /nobackup/model/eagle-sgl/leptonai-EAGLE-Llama-3.1-8B-Instruct --speculative-num-steps 8 \
    --speculative-eagle-topk 6 --speculative-num-draft-tokens 32 \
    --cuda-graph-max-bs 1 --mem-fraction 0.7 --dtype float16 --port 30000



# Eagle-2 Qwen2.5
python -m sglang.launch_server --model-path /nobackup/model/qwen2.5/Qwen2.5-7B-Instruct  --port 30000 --cuda-graph-max-bs 1

python -m sglang.launch_server --model /nobackup/model/qwen2.5/Qwen2.5-7B-Instruct --speculative-algo EAGLE \
    --speculative-draft /nobackup/model/eagle-sgl/EAGLE-Qwen2.5-7B-Instruct --speculative-num-steps 8 \
    --speculative-eagle-topk 6 --speculative-num-draft-tokens 32 \
    --cuda-graph-max-bs 1 --mem-fraction 0.7 --dtype float16 --port 30000


# Run
python bench-sgl/bench_over_dataset.py --question-file /home/qinghao/workdir/fastrl/bench-sgl/bench_data/mt_bench.jsonl --model-type llama

python bench-sgl/bench_over_dataset.py --question-file /home/qinghao/workdir/fastrl/bench-sgl/bench_data/mt_bench.jsonl --model-type qwen

python bench-sgl/bench_over_dataset.py --question-file /home/qinghao/workdir/fastrl/bench-sgl/bench_data/mt_bench.jsonl --model-type deepseek

================================================================================================================
# Profile & Benchmark Long-tail
python -m sglang.launch_server --model-path /nobackup/model/qwen2.5/Qwen2.5-7B --tensor 2 --enable-metrics --log-requests-level 2 --decode-log-interval 50 --show-time-cost --disable-cuda-graph --disable-radix-cache --context-length 40000 2>&1 | tee server_log_qwen25_7B_wo_cuda_graph.txt


python -m sglang.launch_server --model-path /nobackup/model/llama3.1/Llama-3.1-8B-Instruct --tensor 2 --enable-metrics --log-requests-level 2 --decode-log-interval 50 --show-time-cost --disable-cuda-graph --disable-radix-cache --context-length 40000 2>&1 | tee server_log_llama31_8B_wo_cuda_graph.txt


SIMULATE_ACC_LEN=6 python -m sglang.launch_server --model /nobackup/model/llama3.1/Llama-3.1-8B-Instruct --tensor 2 --enable-metrics --log-requests-level 2 --decode-log-interval 50 --show-time-cost --disable-cuda-graph --disable-radix-cache --context-length 40000 --speculative-algo EAGLE \
    --speculative-draft /nobackup/model/eagle-sgl/sglang-EAGLE3-LLaMA3.1-Instruct-8B-bak --speculative-num-steps 8 \
    --speculative-eagle-topk 8 --speculative-num-draft-tokens 64 \
    --mem-fraction 0.7 --dtype float16 2>&1 | tee server_log_llama31_8B_eagle.txt


# --disable-radix-cache 

python -m sglang.bench_serving --backend sglang --dataset-name fastrl --dataset-path /nobackup/qinghao/trace/ShareGPT_V3_unfiltered_cleaned_split.json --random-input 1024 --random-output 32768 --random-range-ratio 1  --request-rate 128 --num-prompt 128


================================================================================================================
# Bandit

python -m sglang.launch_server --model /nobackup/model/llama3.1/Llama-3.1-8B-Instruct --tensor 2 --enable-metrics --log-requests-level 2 --decode-log-interval 50 --show-time-cost --disable-cuda-graph --disable-radix-cache --context-length 40000 --speculative-algo EAGLE \
    --speculative-draft /nobackup/model/eagle-sgl/sglang-EAGLE-LLaMA3-Instruct-8B \
    --speculative-num-steps 8 \
    --speculative-eagle-topk 8 --speculative-num-draft-tokens 64 \
    --speculative-eagle-mab-algorithm EG \
    --speculative-eagle-mab-configs 2_2_4,3_4_8 \
    --speculative-mab-window-size 5 \
    --mem-fraction 0.6 --dtype float16 2>&1 | tee server_log_llama31_8B_eagle.txt

python -m sglang.bench_serving --backend sglang --dataset-name random --dataset-path /nobackup/qinghao/trace/ShareGPT_V3_unfiltered_cleaned_split.json --random-input 1024 --random-output 2048 --random-range-ratio 1  --request-rate 128 --num-prompt 128