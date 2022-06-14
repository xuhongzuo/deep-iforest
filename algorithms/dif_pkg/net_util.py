from algorithms.dif_pkg import net_torch


def choose_net(network_name):
    if network_name == 'layer2':
        return net_torch.Net2
    elif network_name == 'layer3-skip3':
        return net_torch.Net3S3
    elif network_name == 'layer4':
        return net_torch.Net4
    elif network_name == 'layer5-skip3':
        return net_torch.Net5S3
    elif network_name == 'layer6':
        return net_torch.Net6
    elif network_name == 'layer6-skip3':
        return net_torch.Net6S3
    elif network_name == 'layer7-skip3':
        return net_torch.Net7S3
    elif network_name == 'layer8-skip3':
        return net_torch.Net8S3
    elif network_name == 'layer4-skip1':
        return net_torch.Net4S1
    elif network_name == 'layer4-skip2':
        return net_torch.Net4S2
    elif network_name == 'layer4-skip3':
        return net_torch.Net4S3
    elif network_name == 'gru':
        return net_torch.GRUNet
    elif network_name == 'lstm':
        return net_torch.LSTMNet
    elif network_name == 'gin':
        from algorithms.dif_pkg import net_graph
        return net_graph.GinEncoderGraph
    else:
        raise NotImplementedError("")
