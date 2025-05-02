================================================================================================================
# Llama-3.1-8B-Instruct
================================================================================================================

python -m sglang.launch_server --model-path /nobackup/model/llama3.1/Llama-3.1-8B-Instruct --tensor 2 --enable-metrics --log-requests-level 2 --decode-log-interval 50 --show-time-cost --disable-radix-cache --context-length 40000 --max-running-requests 1


SIMULATE_ACC_LEN=6 python -m sglang.launch_server --model /nobackup/model/llama3.1/Llama-3.1-8B-Instruct --tensor 2 --enable-metrics --log-requests-level 2 --decode-log-interval 50 --show-time-cost --disable-radix-cache --context-length 40000 --speculative-algo EAGLE \
    --speculative-draft /nobackup/model/eagle-sgl/leptonai-EAGLE-Llama-3.1-8B-Instruct --speculative-num-steps 8 \
    --speculative-eagle-topk 8 --speculative-num-draft-tokens 48 \
    --mem-fraction 0.7 --dtype float16  --max-running-requests 1


python -m sglang.bench_serving --backend sglang --dataset-name random --dataset-path /nobackup/qinghao/trace/ShareGPT_V3_unfiltered_cleaned_split.json --random-input 256 --random-output 1024 --random-range-ratio 1  --request-rate 1 --num-prompt 5

python -m sglang.bench_serving --backend sglang --dataset-name fastrl --dataset-path /nobackup/qinghao/trace/ShareGPT_V3_unfiltered_cleaned_split.json --random-input 1024 --random-output 32768 --random-range-ratio 1  --request-rate 128 --num-prompt 128



================================================================================================================
# Qwen-32B
================================================================================================================

python -m sglang.launch_server --model-path /nobackup/model/qwen2.5/Qwen2.5-32B --tensor 4 --enable-metrics --log-requests-level 2 --decode-log-interval 50 --show-time-cost --disable-radix-cache --context-length 40000 2>&1 | tee server_log_qwen32B.txt

SIMULATE_ACC_LEN=6.3 python -m sglang.launch_server --model /nobackup/model/qwen2.5/Qwen2.5-32B --tensor 4 --enable-metrics --log-requests-level 2 --decode-log-interval 50 --show-time-cost --disable-radix-cache --context-length 40000 --speculative-algo EAGLE \
    --speculative-draft /nobackup/model/eagle-sgl/sglang-EAGLE-Qwen-32B-dummy --speculative-num-steps 8 \
    --speculative-eagle-topk 8 --speculative-num-draft-tokens 48 \
    --mem-fraction 0.6 --dtype float16

python -m sglang.bench_serving --backend sglang --dataset-name fastrl --dataset-path /nobackup/qinghao/trace/ShareGPT_V3_unfiltered_cleaned_split.json --random-input 1024 --random-output 32768 --random-range-ratio 1  --request-rate 128 --num-prompt 128


================================================================================================================
# Llama-70B
================================================================================================================

python -m sglang.launch_server --model-path /nobackup/model/llama3.3/Llama-3.3-70B-Instruct --tensor 8 --enable-metrics --log-requests-level 2 --decode-log-interval 50 --show-time-cost --disable-radix-cache --context-length 40000 2>&1 | tee server_log_llama70B.txt

SIMULATE_ACC_LEN=4.2 python -m sglang.launch_server --model /nobackup/model/llama3.3/Llama-3.3-70B-Instruct --tensor 8 --enable-metrics --log-requests-level 2 --decode-log-interval 50 --show-time-cost --disable-radix-cache --context-length 40000 --speculative-algo EAGLE \
    --speculative-draft /nobackup/model/eagle-sgl/EAGLE-Llama-3.3-70B-Instruct --speculative-num-steps 8 \
    --speculative-eagle-topk 8 --speculative-num-draft-tokens 48 \
    --mem-fraction 0.6 --dtype float16 2>&1 | tee server_log_llama70B_eagle.txt

python -m sglang.bench_serving --backend sglang --dataset-name fastrl --dataset-path /nobackup/qinghao/trace/ShareGPT_V3_unfiltered_cleaned_split.json --random-input 1024 --random-output 32768 --random-range-ratio 1  --request-rate 128 --num-prompt 128




