from src.networks.decoders import Decoders

def get_model(cfg):
    c_dim = cfg['model']['c_dim']  # feature dimensions
    truncation = cfg['model']['truncation']
    learnable_beta = cfg['rendering']['learnable_beta']
    input_ch = cfg['model']['input_ch']
    input_ch_pos = cfg['model']['input_ch_pos']
    #decoder = Decoders(c_dim=c_dim, truncation=truncation, learnable_beta=learnable_beta)
    decoder = Decoders(cfg, input_ch=input_ch, input_ch_pos=input_ch_pos)
    return decoder

