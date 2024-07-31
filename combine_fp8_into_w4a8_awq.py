import torch
import sys
from safetensors.torch import load_file
import json

import re
def check_exclude_modules(exclude_modules, name):
    if name in exclude_modules:
        return True
    if len(exclude_modules) == 0:
        return False
    for exclude_module in exclude_modules:
        regex_pattern = exclude_module.replace('.', '\.').replace('*', '.*')
        pattern = re.compile(regex_pattern)
        if pattern.search(name):
            return True

fp8_file_path = sys.argv[1]
w4a8_awq_file_path = sys.argv[2]
w4a8_awq_config = sys.argv[3]
output_file_path = sys.argv[4]

fp8_data = load_file(fp8_file_path)
w4a8_awq_data = load_file(w4a8_awq_file_path)

output_data = {}

with open(w4a8_awq_config, 'r') as file:
    w4a8_config = json.load(file)
fp8_modules = w4a8_config["quantization"]["fp8_modules"]
print(fp8_modules)

for int8_name in fp8_data:
    if check_exclude_modules(fp8_modules, int8_name):
        print("fp8", int8_name)
        output_data[int8_name] = fp8_data[int8_name]

for w4a8_name in w4a8_awq_data:
    if not check_exclude_modules(fp8_modules, w4a8_name):
        print("w4a8_awq", w4a8_name)
        output_data[w4a8_name] = w4a8_awq_data[w4a8_name]

from safetensors.torch import save_file
save_file(output_data, output_file_path)
