# 1. launch server

python3 -m sglang.launch_server --model-path /nobackup/model/deepseek-r1/DeepSeek-R1-Distill-Llama-8B --port 30000

# with data parallelism
# python3 -m sglang_router.launch_server --model-path /nobackup/model/deepseek-r1/DeepSeek-R1-Distill-Llama-8B --port 30000 --dp-size 4

# with speculative decoding
python -m sglang.launch_server --model /nobackup/model/deepseek-r1/DeepSeek-R1-Distill-Llama-8B --speculative-algo EAGLE3 \
    --speculative-draft /nobackup/model/eagle-sgl/sglang-EAGLE3-DeepSeek-R1-Distill-LLaMA-8B --speculative-num-steps 8 \
    --speculative-eagle-topk 8 --speculative-num-draft-tokens 64 \
    --cuda-graph-max-bs 1 --mem-fraction 0.7 --dtype float16 --port 30000

========================================


# 2. run benchmark
python3 bench-sgl/reasoning_eval/bench_sglang.py --parallel 256 --port 30000 --data-path /home/qinghao/workdir/fastrl/bench-sgl/reasoning_eval/aime_2024_problems.parquet --question-key Problem --answer-key Answer --num-tries 2


# python3 bench-sgl/reasoning_eval/bench_sglang.py --parallel 256 --port 30000 --data-path /home/qinghao/workdir/fastrl/bench-sgl/reasoning_eval/aime_2024_problems.parquet --question-key Problem --answer-key Answer --num-tries 2
# Generating train split: 30 examples [00:00, 2167.26 examples/s]
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 60/60 [12:11<00:00, 12.19s/it]
# Overall Accuracy: 0.48333333333333334
# Mean Standard Error of Accuracy across questions: 0.21666666666666667
# Output throughput: 1254.0063939741933 token/s