from torch import nn, Tensor
from itertools import tee
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, remove_self_loops
import torch.nn.functional as F


class FeatureSteeredConvolution(MessagePassing):
    """Implementation of feature steered convolutions.

    References
    ----------
    .. [1] Verma, Nitika, Edmond Boyer, and Jakob Verbeek.
       "Feastnet: Feature-steered graph convolutions for 3d shape analysis."
       Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_heads: int,
        ensure_trans_invar: bool = True,
        bias: bool = True,
        with_self_loops: bool = True,
    ):
        super(FeatureSteeredConvolution ,self).__init__(aggr="mean")
        # num_heads = num_heads[0] #if I dont do this, i have (2,) instead of 2

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.with_self_loops = with_self_loops

        self.linear = nn.Linear(
            in_features=in_channels,
            out_features=out_channels * num_heads,
            bias=False,
        )
        self.u = nn.Linear(
            in_features=in_channels,
            out_features=num_heads,
            bias=False,
        )
        self.c = nn.Parameter(Tensor(num_heads))

        if not ensure_trans_invar:
            self.v = nn.Linear(
                in_features=in_channels,
                out_features=num_heads,
                bias=False,
            )
        else:
            self.register_parameter("v", None)

        if bias:
            self.bias = nn.Parameter(Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialization of tuneable network parameters."""
        nn.init.uniform_(self.linear.weight)
        nn.init.uniform_(self.u.weight)
        nn.init.normal_(self.c, mean=0.0, std=0.1)
        if self.bias is not None:
            nn.init.normal_(self.bias, mean=0.0, std=0.1)
        if self.v is not None:
            nn.init.uniform_(self.v.weight)

    def forward(self, x, edge_index):
        """Forward pass through a feature steered convolution layer.

        Parameters
        ----------
        x: torch.tensor [|V|, in_features]
            Input feature matrix, where each row describes
            the input feature descriptor of a node in the graph.
        edge_index: torch.tensor [2, E]
            Edge matrix capturing the graph's
            edge structure, where each row describes an edge
            between two nodes in the graph.
        Returns
        -------
        torch.tensor [|V|, out_features]
            Output feature matrix, where each row corresponds
            to the updated feature descriptor of a node in the graph.
        """
        if self.with_self_loops:
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index=edge_index, num_nodes=x.shape[0])

        out = self.propagate(edge_index, x=x)
        return out if self.bias is None else out + self.bias

    def _compute_attention_weights(self, x_i, x_j):
        """Computation of attention weights.

        Parameters
        ----------
        x_i: torch.tensor [|E|, in_feature]
            Matrix of feature embeddings for all central nodes,
            collecting neighboring information to update its embedding.
        x_j: torch.tensor [|E|, in_features]
            Matrix of feature embeddings for all neighboring nodes
            passing their messages to the central node along
            their respective edge.
        Returns
        -------
        torch.tensor [|E|, M]
            Matrix of attention scores, where each row captures
            the attention weights of transformed node in the graph.
        """
        if x_j.shape[-1] != self.in_channels:
            raise ValueError(
                f"Expected input features with {self.in_channels} channels."
                f" Instead received features with {x_j.shape[-1]} channels."
            )
        if self.v is None:
            attention_logits = self.u(x_i - x_j) + self.c
        else:
            attention_logits = self.u(x_i) + self.b(x_j) + self.c
        return F.softmax(attention_logits, dim=1)

    def message(self, x_i, x_j):
        """Message computation for all nodes in the graph.

        Parameters
        ----------
        x_i: torch.tensor [|E|, in_feature]
            Matrix of feature embeddings for all central nodes,
            collecting neighboring information to update its embedding.
        x_j: torch.tensor [|E|, in_features]
            Matrix of feature embeddings for all neighboring nodes
            passing their messages to the central node along
            their respective edge.
        Returns
        -------
        torch.tensor [|E|, out_features]
            Matrix of updated feature embeddings for
            all nodes in the graph.
        """
        attention_weights = self._compute_attention_weights(x_i, x_j)
        x_j = self.linear(x_j).view(-1, self.num_heads, self.out_channels)
        return (attention_weights.view(-1, self.num_heads, 1) * x_j).sum(dim=1)
    
class GraphFeatureEncoder(nn.Module):
    """Graph neural network consisting of stacked graph convolutions."""
    def __init__(
        self,
        in_features,
        conv_channels,
        num_heads,
        apply_batch_norm: int = True,
        ensure_trans_invar: bool = True,
        bias: bool = True,
        with_self_loops: bool = True,
    ):
        super(GraphFeatureEncoder,self).__init__()
        # conv_channels = conv_channels[0] #if I dont do this, i have ([16,32,128,254],) instead of [16,32,128,254]

        conv_params = dict(
            num_heads=num_heads,
            ensure_trans_invar=ensure_trans_invar,
            bias=bias,
            with_self_loops=with_self_loops,
        )
        self.apply_batch_norm = apply_batch_norm

        *first_conv_channels, final_conv_channel = conv_channels

        conv_layers = get_conv_layers(
            channels= [in_features] + conv_channels,
            conv=FeatureSteeredConvolution,
            conv_params=conv_params,
        )
        self.conv_layers = nn.ModuleList(conv_layers)

        self.batch_layers = [None for _ in first_conv_channels]
        if apply_batch_norm:
            self.batch_layers = nn.ModuleList(
                [nn.BatchNorm1d(channel) for channel in first_conv_channels]
            )

    def forward(self, x, edge_index):
        *first_conv_layers, final_conv_layer = self.conv_layers
        for conv_layer, batch_layer in zip(first_conv_layers, self.batch_layers):
            x = conv_layer(x, edge_index)
            x = F.relu(x)
            if batch_layer is not None:
                x = batch_layer(x)
        return final_conv_layer(x, edge_index)


class MeshSeg(nn.Module):
    """Mesh segmentation network."""
    def __init__(
        self,
        in_features,
        encoder_features,
        conv_channels,
        encoder_channels,
        decoder_channels,
        num_classes,
        num_heads,
        apply_batch_norm=True,
    ):
        super(MeshSeg,self).__init__()
        self.input_encoder = get_mlp_layers(
            channels=[in_features] + encoder_channels,
            activation=nn.ReLU,
        )
        self.gnn = GraphFeatureEncoder(
            in_features=encoder_features,
            conv_channels=conv_channels,
            num_heads=num_heads,
            apply_batch_norm=apply_batch_norm,
        )
        *_, final_conv_channel = conv_channels

        self.final_projection = get_mlp_layers(
            [final_conv_channel] + decoder_channels + [num_classes],
            activation=nn.ReLU,
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.input_encoder(x)
        x = self.gnn(x, edge_index)
        return self.final_projection(x)
    
class MeshSegbis(nn.Module):
    """Mesh segmentation network."""
    def __init__(
        self,
        in_features,
        encoder_features,
        conv_channels,
        encoder_channels,
        decoder_channels,
        num_classes,
        num_heads,
        gnn,
        apply_batch_norm=True,
    ):
        super(MeshSeg,self).__init__()
        self.input_encoder = get_mlp_layers(
            channels=list([in_features]) + list(encoder_channels),
            activation=nn.ReLU,
        )
        self.gnn = gnn
        *_, final_conv_channel = conv_channels

        self.final_projection = get_mlp_layers(
            [final_conv_channel] + decoder_channels + [num_classes],
            activation=nn.ReLU,
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.input_encoder(x)
        x = self.gnn(x, edge_index)
        return self.final_projection(x)



def pairwise(iterable):
    """Iterate over all pairs of consecutive items in a list.
    Notes
    -----
        [s0, s1, s2, s3, ...] -> (s0,s1), (s1,s2), (s2, s3), ...
    """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def get_mlp_layers(channels: list, activation, output_activation=nn.Identity):
    """Define basic multilayered perceptron network."""
    layers = []
    *intermediate_layer_definitions, final_layer_definition = pairwise(channels)

    for in_ch, out_ch in intermediate_layer_definitions:
        intermediate_layer = nn.Linear(in_ch, out_ch)
        layers += [intermediate_layer, activation()]

    layers += [nn.Linear(*final_layer_definition), output_activation()]
    return nn.Sequential(*layers)

def get_conv_layers(channels: list, conv: MessagePassing, conv_params: dict):
    """Define convolution layers with specified in and out channels.

    Parameters
    ----------
    channels: list
        List of integers specifying the size of the convolution channels.
    conv: MessagePassing
        Convolution layer.
    conv_params: dict
        Dictionary specifying convolution parameters.

    Returns
    -------
    list
        List of convolutions with the specified channels.
    """

    conv_layers = [
        conv(in_ch, out_ch, **conv_params) for in_ch, out_ch in pairwise(channels)
    ]
    return conv_layers