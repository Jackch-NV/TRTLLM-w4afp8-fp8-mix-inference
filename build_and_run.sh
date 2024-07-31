MODEL_PATH=$1
FP8_MODULES=$2
###Step 1. w4a8 awq quantization
python3 ../quantization/quantize.py --model_dir $MODEL_PATH                                    --output_dir ./tllm_checkpoint_llama_awq                                    --dtype float16                                    --qformat w4a8_awq    --calib_size 256 --kv_cache_dtype fp8 --fp8_modules $FP8_MODULES

###Step 2. fp8 quantization
python3 ../quantization/quantize.py --model_dir $MODEL_PATH  --output_dir ./tllm_checkpoint_llama_fp8 --dtype float16 --qformat fp8 --kv_cache_dtype fp8

###Step 3. combine two quantized checkpoint
python3 combine_fp8_into_w4a8_awq.py tllm_checkpoint_llama_fp8/rank0.safetensors tllm_checkpoint_llama_awq/rank0.safetensors tllm_checkpoint_llama_awq/config.json o.safetensors

mv o.safetensors tllm_checkpoint_llama_awq/rank0.safetensors


###Step 4. build engine
trtllm-build --checkpoint_dir ./tllm_checkpoint_llama_awq             --output_dir llama_awq             --gemm_plugin float16             --max_batch_size 8             --max_input_len 2048             --max_seq_len 4096

###Step 5. download test data
mkdir data; wget https://people.eecs.berkeley.edu/~hendrycks/data.tar -O data/mmlu.tar
tar -xf data/mmlu.tar -C data && mv data/data data/mmlu

###Step 6. test with mmlu
python3 ../mmlu.py --hf_model_dir $MODEL_PATH --engine_dir "llama_awq" --test_trt_llm 
