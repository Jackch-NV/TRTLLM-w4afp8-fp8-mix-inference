下载源码
```
git clone https://github.com/NVIDIA/TensorRT-LLM -b v0.11.0
```
拷贝修改的文件到指定目录
```
cd TensorRT-LLM
cp ./w4afp8_fp8_example/build_and_run.sh examples/llama/
cp ./w4afp8_fp8_example/combine_fp8_into_w4a8_awq.py examples/llama/
cp ./w4afp8_fp8_example/examples/quantization/quantize.py examples/quantization/ 
cp ./w4afp8_fp8_example/tensorrt_llm/builder.py tensorrt_llm/builder.py
cp ./w4afp8_fp8_example/tensorrt_llm/models/modeling_utils.py tensorrt_llm/models/modeling_utils.py 
cp ./w4afp8_fp8_example/tensorrt_llm/quantization/quantize.py tensorrt_llm/quantization/quantize.py 
cp ./w4afp8_fp8_example/tensorrt_llm/quantization/quantize_by_modelopt.py tensorrt_llm/quantization/quantize_by_modelopt.py
```
编译安装
```
python3 ./scripts/build_wheel.py --trt_root /usr/local/tensorrt --cuda_architectures "90" -c
pip install ./build/tensorrt_llm*.whl
```

运行测试
```
cd examples/llama/
```

我们可以考虑让以下5个gemm 设置为fp8 精度, 以下是gemm和它对应的fp8_modules_list(用逗号分隔，不要有空格）

qkv gemm : "*q_proj*,*k_proj*,*v_proj*"

attention o_proj : "*o_proj*"

mlp up_proj : "*up*"

mlp gate : "*gate*"

mlp down_proj : "*down*"

我们也可以增加"*layers.xxx*"来控制只把某些层的某些gemm设置为fp8，比如只把第22层和23层的gate gemm 设置为fp8，则fp8_modules_list = "*layers.22.*gate*,*layers.23.*gate*"

运行示例
```
bash -x build_and_run.sh $LLAMA_PATH "*q_proj*,*k_proj*,*k_proj*"
bash -x build_and_run.sh $LLAMA_PATH "*q_proj*,*k_proj*,*k_proj*,*up*" 
bash -x build_and_run.sh $LLAMA_PATH "*q_proj*,*k_proj*,*k_proj*,*down*" 
bash -x build_and_run.sh $LLAMA_PATH "*q_proj*,*k_proj*,*k_proj*,*gate*" 
bash -x build_and_run.sh $LLAMA_PATH "*q_proj*,*k_proj*,*k_proj*,*o_proj*"
bash -x build_and_run.sh $LLAMA_PATH "*layers.1.*" 
```




