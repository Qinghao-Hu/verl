python -m sglang.launch_server --model /local/model/llama3.1/Llama-3.1-8B-Instruct --speculative-algo EAGLE \
    --speculative-draft /nobackup/model/eagle-sgl/sglang-EAGLE-Llama-3.1-Instruct-8B --speculative-num-steps 8 \
    --speculative-eagle-topk 48 --speculative-num-draft-tokens 48 \
    --cuda-graph-max-bs 16 --mem-fraction 0.7 --dtype float16 --port 30000

# Run
python  /home/qinghao/workdir/verl-dev/playground/archive/bench-sgl/bench_over_dataset.py  --question-file /home/qinghao/workdir/fastrl/bench-sgl/bench_data/mt_bench.jsonl --model-type llama