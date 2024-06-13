def get_network(network_name):
    network_name = network_name.lower()
    # Original GR-ConvNet
    if network_name == 'grconvnet':
        from .grconvnet import GenerativeResnet
        return GenerativeResnet
    # Configurable GR-ConvNet with multiple dropouts
    elif network_name == 'grconvnet2':
        from .grconvnet2 import GenerativeResnet
        return GenerativeResnet
    # Configurable GR-ConvNet with dropout at the end
    elif network_name == 'grconvnet3':
        from .grconvnet3 import GenerativeResnet
        return GenerativeResnet
    # Inverted GR-ConvNet
    elif network_name == 'grconvnet4':
        from .grconvnet4 import GenerativeResnet
        return GenerativeResnet
    elif network_name == 'ragt':
        from .ragt.ragt import RAGT
        return RAGT
    elif network_name == 'ggcnn':
        from .ggcnn import GGCNN
        return GGCNN
    elif network_name == 'lgrconvnet3':
        from .lgrconvnet3 import GenerativeResnet
        return GenerativeResnet
    elif network_name == 'lggcnn':
        from .lggcnn import LGGCNN
        return LGGCNN
    elif network_name == 'lragt':
        from .ragt.ragt import LRAGT
        return LRAGT
    elif network_name == 'clipfusion':
        from .clipfusion import CLIPFusion
        return CLIPFusion
    elif network_name == 'lgdm':
        from .lgdm.network import LGDM
        return LGDM
    else:
        raise NotImplementedError('Network {} is not implemented'.format(network_name))
