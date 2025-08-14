

def create_layers(layer_type, layer_args, num_layers):
    if layer_type is None:
        layers_list = [None] * num_layers
        args_list = [None] * num_layers
        return layers_list, args_list

    if isinstance(layer_type, list):
        assert len(layer_type) == num_layers
        layers_list = layer_type
        if isinstance(layer_args, list):
            assert len(layer_args) == num_layers
            args_list = layer_args
        else:
            args_list = [layer_args for _ in range(num_layers)]
    else:
        layers_list = [layer_type for _ in range(num_layers)]
        args_list = [layer_args for _ in range(num_layers)]
    return layers_list, args_list


def miniblock(input_size, output_size, layer_type, layer_args, dropout_layer, dropout_args, norm_layer, norm_args, activation, act_args):
    # Main layer
    if layer_args is None:
        layers = [layer_type(input_size, output_size)]
    elif isinstance(layer_args, tuple):
        layers = [layer_type(input_size, output_size, *layer_args)]
    elif isinstance(layer_args, dict):
        layers = [layer_type(input_size, output_size, **layer_args)]
    else:
        raise ValueError(f"layer_args must be None, tuple or dict, got {type(layer_args)}")

    def create_layer_with_args(layer, args):
        if layer is None:
            return None
        if args is None:
            return layer()
        elif isinstance(args, tuple):
            return layer(*args)
        elif isinstance(args, dict):
            return layer(**args)
        else:
            return layer(args)

    # Dropout
    if dropout_layer is not None:
        drop = create_layer_with_args(dropout_layer, dropout_args)
        if drop is not None:
            layers.append(drop)
    # Normalization
    if norm_layer is not None:
        norm = create_layer_with_args(norm_layer, norm_args)
        if norm is not None:
            layers.append(norm)
    # Activation
    if activation is not None:
        act = create_layer_with_args(activation, act_args)
        if act is not None:
            layers.append(act)
    return layers