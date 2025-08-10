export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS=1

BATCH_SIZES=(256)
TP_SIZES=(2 4 8)
# INPUT_LENS=(512 1024 2048) # 4096 8192 16384)

INPUT_LENS=(1024) 
# (1024)
# export CUDA_VISIBLE_DEVICES=4,5,6,7

run_name="qwen2.5-32b"
# model="/nobackup/model/llama2/Llama-2-7b-chat-hf-bak/"
model="/nobackup/model/qwen2.5/Qwen2.5-32B-Instruct"

for global_bs in "${BATCH_SIZES[@]}"; do
    for tp_size in "${TP_SIZES[@]}"; do
        for input_len in "${INPUT_LENS[@]}"; do
            
            bs=$((global_bs / (8 / tp_size)))

            python -m sglang.bench_one_batch \
                --model-path $model \
                --batch-size $bs --input-len $input_len --output-len 64 \
                --max-running-requests $bs \
                --mem-fraction-static 0.7 \
                --tp-size $tp_size --run-name $run_name #--disable-cuda-graph # --max-prefill-tokens 1101344 --max-total-tokens 1101344 # 2>&1 | tee "$log_file"
        done
    done
done