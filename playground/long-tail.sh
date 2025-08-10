# SIMULATE_ACC_LEN=6 
python -m sglang.launch_server --model /nobackup/model/llama3.1/Llama-3.1-8B-Instruct --tensor 2 --enable-metrics --log-requests-level 2 --decode-log-interval 50 --show-time-cost --context-length 40000 --speculative-algo EAGLE \
    --speculative-draft /nobackup/model/eagle-sgl/sglang-EAGLE-LLaMA3-Instruct-8B \
    --speculative-num-steps 8 \
    --speculative-eagle-topk 8 --speculative-num-draft-tokens 48 \
    --speculative-eagle-mab-algorithm "PREDEFINED" \
    --speculative-eagle-mab-configs 8_8_32,8_8_16,8_8_8 \
    --mem-fraction 0.6 --max-running-requests 48 --dtype float16 2>&1 | tee server_log_llama31_8B_eagle.txt


python -m sglang.bench_serving --backend sglang --dataset-name fastrl --dataset-path /nobackup/qinghao/trace/ShareGPT_V3_unfiltered_cleaned_split.json --random-input 1024 --random-output 32768 --random-range-ratio 1  --request-rate 128 --num-prompt 128