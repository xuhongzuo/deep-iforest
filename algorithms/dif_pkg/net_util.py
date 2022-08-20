from algorithms.dif_pkg import net_torch


def choose_net(network_name):
    if network_name.startswith('layer'):
        return net_torch.MLPnet
    elif network_name == 'gru':
        return net_torch.GRUNet
    elif network_name == 'lstm':
        return net_torch.LSTMNet
    elif network_name == 'gin':
        from algorithms.dif_pkg import net_graph
        return net_graph.GinEncoderGraph
    else:
        raise NotImplementedError("")
