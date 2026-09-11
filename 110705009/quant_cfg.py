from hqq.core.quantize import BaseQuantizeConfig

def get_quant_config_deit(model):
    """
    微調 DeiT-S 模型的量化配置，平衡準確率和模型大小
    - 目標：恢復 90% 左右的準確率
    - 保持模型大小在 18-20 MiB 範圍內
    """
    quant_config = {}
    
    # 定義量化配置
    q8_config = BaseQuantizeConfig(nbits=8, group_size=32)
    q6_config = BaseQuantizeConfig(nbits=6, group_size=32)
    q5_config = BaseQuantizeConfig(nbits=5, group_size=64)
    q4_config = BaseQuantizeConfig(nbits=4, group_size=64)
    q3_config = BaseQuantizeConfig(nbits=3, group_size=128)
    
    # 輸入和輸出層維持高精度
    quant_config['patch_embed.proj'] = q8_config
    quant_config['head'] = q8_config
    
    # 根據模型結構，blocks.0 到 blocks.11
    n_blocks = 12
    
    # 比上一版本提高一些關鍵層的精度
    for i in range(n_blocks):
        # 前1個和最後1個 block 使用高精度
        if i < 1 or i >= n_blocks - 1:
            # 注意力機制層
            quant_config[f'blocks.{i}.attn.qkv'] = q8_config  # 提高至 8 位
            quant_config[f'blocks.{i}.attn.proj'] = q6_config
            # MLP 層
            quant_config[f'blocks.{i}.mlp.fc1'] = q6_config
            quant_config[f'blocks.{i}.mlp.fc2'] = q5_config
        else:
            # 中間層提高精度
            quant_config[f'blocks.{i}.attn.qkv'] = q6_config  # 提高至 6 位
            quant_config[f'blocks.{i}.attn.proj'] = q4_config
            # MLP 層
            quant_config[f'blocks.{i}.mlp.fc1'] = q4_config  # 提高至 5 位
            quant_config[f'blocks.{i}.mlp.fc2'] = q4_config
    
    # 歸一化層
    for i in range(n_blocks):
        quant_config[f'blocks.{i}.norm1'] = q6_config
        quant_config[f'blocks.{i}.norm2'] = q6_config
    
    quant_config['norm_pre'] = q6_config
    quant_config['norm'] = q6_config
    
    return quant_config

def get_quant_config_slm(model):
    """
    為 Llama 3.2-1B-Instruct 模型創建優化量化配置
    - 優先保持PPL在14左右，同時儘量保持加速比接近1.3
    - 僅使用 GemLite 支持的位元數：1, 2, 4, 8, 16
    """
    quant_config = {}
    
    # 只使用 GemLite 支持的位元數
    q8_config = BaseQuantizeConfig(nbits=8, group_size=128)
    q4_config = BaseQuantizeConfig(nbits=4, group_size=64)
    q2_config = BaseQuantizeConfig(nbits=2, group_size=64)
    
    # 對於 MLP 層，使用確保大小兼容的組大小
    mlp_q8_config = BaseQuantizeConfig(nbits=8, group_size=32)
    mlp_q4_config = BaseQuantizeConfig(nbits=4, group_size=32)
    mlp_q2_config = BaseQuantizeConfig(nbits=2, group_size=32)
    
    # 對於特別大的層 (lm_head)
    lm_head_config = BaseQuantizeConfig(nbits=8, group_size=256)
    
    # 定義層數
    n_layers = 16
    
    # 大幅增加高精度層的數量
    for i in range(n_layers):
        # 首8層和尾4層使用高精度
        if i < 8 or i >= n_layers - 4:
            # 注意力層全部使用8位
            quant_config[f'model.layers.{i}.self_attn.q_proj'] = q8_config
            quant_config[f'model.layers.{i}.self_attn.k_proj'] = q8_config
            quant_config[f'model.layers.{i}.self_attn.v_proj'] = q4_config
            quant_config[f'model.layers.{i}.self_attn.o_proj'] = q4_config
            
            # MLP層對於首尾層也使用8位
            if i < 3 or i >= n_layers - 2:
                quant_config[f'model.layers.{i}.mlp.gate_proj'] = mlp_q8_config
                quant_config[f'model.layers.{i}.mlp.up_proj'] = mlp_q8_config
                quant_config[f'model.layers.{i}.mlp.down_proj'] = mlp_q8_config
            elif i == 5:
                quant_config[f'model.layers.{i}.mlp.gate_proj'] = mlp_q2_config  # 從4位降至2位
                quant_config[f'model.layers.{i}.mlp.up_proj'] = mlp_q2_config    # 從4位降至2位
                quant_config[f'model.layers.{i}.mlp.down_proj'] = mlp_q2_config
                
            else:
                # 其他層使用4位以保持一定速度
                quant_config[f'model.layers.{i}.mlp.gate_proj'] = mlp_q4_config
                quant_config[f'model.layers.{i}.mlp.up_proj'] = mlp_q4_config
                quant_config[f'model.layers.{i}.mlp.down_proj'] = mlp_q4_config
        # 中間少數幾層可以使用低精度
        else:
            # 即使在中間層也使用較高精度的注意力層
            quant_config[f'model.layers.{i}.self_attn.q_proj'] = q8_config
            quant_config[f'model.layers.{i}.self_attn.k_proj'] = q8_config
            quant_config[f'model.layers.{i}.self_attn.v_proj'] = q4_config
            quant_config[f'model.layers.{i}.self_attn.o_proj'] = q4_config
            
            # 這些中間層的MLP可以用2位
            quant_config[f'model.layers.{i}.mlp.gate_proj'] = mlp_q2_config
            quant_config[f'model.layers.{i}.mlp.up_proj'] = mlp_q2_config
            quant_config[f'model.layers.{i}.mlp.down_proj'] = mlp_q2_config
    
    # 嵌入層和頭部層使用最高精度
    quant_config['model.embed_tokens'] = q8_config
    quant_config['lm_head'] = lm_head_config
    
    # 所有歸一化層使用8位
    #for i in range(n_layers):
    #    quant_config[f'model.layers.{i}.input_layernorm'] = q8_config
    #    quant_config[f'model.layers.{i}.post_attention_layernorm'] = q8_config
    for i in range(n_layers):
        if i < 5 or i >= n_layers - 3:  # 最關鍵的首尾層保持高精度
            quant_config[f'model.layers.{i}.input_layernorm'] = q8_config
            quant_config[f'model.layers.{i}.post_attention_layernorm'] = q8_config
        else:  # 其他層降至4位
            quant_config[f'model.layers.{i}.input_layernorm'] = q4_config
            quant_config[f'model.layers.{i}.post_attention_layernorm'] = q4_config
    
    quant_config['model.norm'] = q8_config
    for i in range(8, 12):
        quant_config[f'model.layers.{i}.mlp.gate_proj'] = mlp_q4_config
        quant_config[f'model.layers.{i}.mlp.up_proj'] = mlp_q4_config
        quant_config[f'model.layers.{i}.mlp.down_proj'] = mlp_q4_config
    
    return quant_config