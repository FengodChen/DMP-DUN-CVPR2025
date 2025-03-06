from .DMP_DUN_plus_4step import cfg_gen as cfg_gen_baseline

def cfg_gen(img_channels, cs_ratio, name):
    cfg = cfg_gen_baseline(img_channels, cs_ratio, name,)
    cfg["base_config"]["net_kwargs"]["time_step"] = 50
    return cfg
