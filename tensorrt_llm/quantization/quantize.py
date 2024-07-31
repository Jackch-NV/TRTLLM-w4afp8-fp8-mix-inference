import fnmatch

from .._utils import get_init_params
from ..layers import (MLP, Attention, ColumnLinear, Embedding, GatedMLP,
                      LayerNorm, RmsNorm, RowLinear)
from ..models.modeling_utils import QuantConfig
from ..parameter import Parameter
from .layers import (FP8Linear, FP8RowLinear, Int8SmoothQuantLinear,
                     Int8SmoothQuantRowLinear, SmoothQuantAttention,
                     SmoothQuantGatedMLP, SmoothQuantLayerNorm, SmoothQuantMLP,
                     SmoothQuantRmsNorm, WeightOnlyGroupwiseQuantColumnLinear,
                     WeightOnlyGroupwiseQuantRowLinear,
                     WeightOnlyQuantColumnLinear, WeightOnlyQuantEmbedding,
                     WeightOnlyQuantRowLinear)
from .mode import W8A8_SQ_PLUGIN_LIST, QuantAlgo

### M
from tensorrt_llm.quantization import QuantMode
import re
def check_exclude_modules(exclude_modules, name):
    if name in exclude_modules:
        return True
    for exclude_module in exclude_modules:
        if len(exclude_module) == 0:
            continue
        regex_pattern = exclude_module.replace('.', '\.').replace('*', '.*')
        pattern = re.compile(regex_pattern)
        if pattern.search(name):
            return True


def quantize_layers(
    model,
    quant_config: QuantConfig,
    quant_map,
    preprocess_init_params=None,
):
    exclude_modules = quant_config.exclude_modules or [
        '*lm_head',
        '*router',
        '*vocab_embedding',
        '*position_embedding',
        '*block_embedding',
    ]
    fp8_modules = quant_config.fp8_modules
    for name, module, parent in model.named_modules_with_parent():
        module_name = name.rsplit('.', 1)[-1]
        is_excluded = False
        for exclude_module in exclude_modules:
            if fnmatch.fnmatchcase(name, exclude_module):
                is_excluded = True
                break
        if not is_excluded:
            quant_cls = None
            if check_exclude_modules(fp8_modules, name):
                quant_map_fp8 = {
                    ColumnLinear: FP8Linear,
                    RowLinear: FP8RowLinear,
                }
                for cls in quant_map_fp8:
                    if isinstance(module, cls):
                        quant_cls = quant_map_fp8[cls]
                        break
            else:
                for cls in quant_map:
                    if isinstance(module, cls):
                        quant_cls = quant_map[cls]
                        break

            if quant_cls is None:
                continue

            init_params = get_init_params(module, quant_cls)
            if "bias" in init_params:
                init_params["bias"] = init_params["bias"] is not None
            if isinstance(module, ColumnLinear):
                init_params[
                    "out_features"] = module.out_features * module.tp_size
            elif isinstance(module, RowLinear):
                init_params["in_features"] = module.in_features * module.tp_size
            #M 
            if (not check_exclude_modules(fp8_modules, name)) and (preprocess_init_params is not None):
                preprocess_init_params(init_params, name, module)
            quant_layer = quant_cls(**init_params)
            if parent is not None:
                setattr(parent, module_name, quant_layer)
            else:
                model = quant_layer

    setattr(model, 'quant_mode', quant_config.quant_mode)
    return model


def weight_only_quantize(model, quant_config: QuantConfig):
    assert quant_config.quant_mode.is_weight_only()

    quant_map = {
        ColumnLinear: WeightOnlyQuantColumnLinear,
        RowLinear: WeightOnlyQuantRowLinear,
        Embedding: WeightOnlyQuantEmbedding,
    }

    def preprocess_init_params(init_params, name, module):
        init_params["quant_mode"] = quant_config.quant_mode
        if isinstance(module, ColumnLinear):
            module_name = name.rsplit('.', 1)[-1]
            init_params["transb"] = module_name == "lm_head"

    model = quantize_layers(
        model,
        quant_config,
        quant_map,
        preprocess_init_params,
    )
    return model


def weight_only_groupwise_quantize(model, quant_config: QuantConfig):
    assert quant_config.quant_mode.is_weight_only()

    quant_map = {
        ColumnLinear: WeightOnlyGroupwiseQuantColumnLinear,
        RowLinear: WeightOnlyGroupwiseQuantRowLinear,
    }

    def preprocess_init_params(init_params, name, module):
        init_params["group_size"] = quant_config.group_size
        init_params["pre_quant_scale"] = quant_config.pre_quant_scale
        init_params["zero"] = quant_config.has_zero_point
        init_params[
            "use_w4a8_awq"] = quant_config.quant_algo == QuantAlgo.W4A8_AWQ

    model = quantize_layers(
        model,
        quant_config,
        quant_map,
        preprocess_init_params,
    )
    return model


def smooth_quantize_ootb(
    model,
    quant_config: QuantConfig,
):
    quant_map = {
        ColumnLinear: Int8SmoothQuantLinear,
        RowLinear: Int8SmoothQuantRowLinear,
    }

    model = quantize_layers(
        model,
        quant_config,
        quant_map,
    )
    return model


def smooth_quantize_plugin(model, quant_mode):
    quant_map = {
        RmsNorm: SmoothQuantRmsNorm,
        LayerNorm: SmoothQuantLayerNorm,
        GatedMLP: SmoothQuantGatedMLP,
        MLP: SmoothQuantMLP,
        Attention: SmoothQuantAttention,
    }
    for name, layer, parent in model.named_modules_with_parent():
        layer_name = name.rsplit('.', 1)[-1]
        if layer_name in ['ln_f', 'ln_embed']:
            continue

        quant_cls = None
        for cls in quant_map:
            if isinstance(layer, cls):
                quant_cls = quant_map[cls]
                break

        if quant_cls is None:
            continue

        init_params = get_init_params(layer, quant_cls)
        init_params["quant_mode"] = quant_mode
        if isinstance(layer, Attention):
            init_params[
                "num_attention_heads"] = layer.num_attention_heads * layer.tp_size
        quant_layer = quant_cls(**init_params)
        if parent is not None:
            setattr(parent, layer_name, quant_layer)
        else:
            model = quant_layer

    setattr(model, 'quant_mode', quant_mode)
    return model


def smooth_quantize(model, quant_config: QuantConfig):
    assert quant_config.quant_mode.has_act_and_weight_quant()
    if quant_config.quant_algo in W8A8_SQ_PLUGIN_LIST:
        return smooth_quantize_plugin(model, quant_config.quant_mode)
    else:
        return smooth_quantize_ootb(model, quant_config)


def fp8_quantize(model, quant_config: QuantConfig):
    assert quant_config.quant_mode.has_fp8_qdq()

    quant_map = {
        ColumnLinear: FP8Linear,
        RowLinear: FP8RowLinear,
    }

    model = quantize_layers(
        model,
        quant_config,
        quant_map,
    )
    return model


def kv_cache_quantize(model, quant_config: QuantConfig):
    assert quant_config.quant_mode.has_kv_cache_quant()
    for name, module in model.named_modules():
        if isinstance(module, (Attention, SmoothQuantAttention)):
            module.kv_cache_scaling_factor = Parameter(shape=(1, ),
                                                       dtype='float32')


def quantize(model, quant_config: QuantConfig):
    quant_mode = quant_config.quant_mode

    if quant_mode.has_fp8_qdq():
        model = fp8_quantize(model, quant_config)
    elif quant_mode.has_act_and_weight_quant():
        model = smooth_quantize(model, quant_config)
    elif quant_mode.is_weight_only():
        if quant_mode.has_per_group_scaling():
            model = weight_only_groupwise_quantize(model, quant_config)
        else:
            model = weight_only_quantize(model, quant_config)

    if quant_mode.has_kv_cache_quant():
        model = kv_cache_quantize(model, quant_config)

    return model
