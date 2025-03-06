import numpy as np
JOINT_COUNT = 33

USE_ATTENTION = False
def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def normalize_digraph(A, dim=0):
    # A is a 2D square array
    Dl = np.sum(A, dim)
    h, w = A.shape
    Dn = np.zeros((w, w))

    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)

    AD = np.dot(A, Dn)
    return AD


class Graph:
    """The Graph to model the skeletons.

    Args:
        layout (str): must be one of the following candidates: 'mediapipe'. Default: 'mediapipe'.
        mode (str): must be one of the following candidates: 'stgcn_spatial', 'spatial'. Default: 'spatial'.
        max_hop (int): the maximal distance between two connected nodes.
            Default: 1
    """

    def __init__(
        self,
        layout="mediapipe",
        mode="spatial",
        max_hop=1,
        nx_node=1,
        num_filter=3,
        init_std=0.02,
        init_off=0.04,
    ):

        self.max_hop = max_hop
        self.layout = layout
        self.mode = mode
        self.num_filter = num_filter
        self.init_std = init_std
        self.init_off = init_off
        self.nx_node = nx_node

        assert (
            nx_node == 1 or mode == "random"
        ), "nx_node can be > 1 only if mode is 'random'"
        assert layout in ["mediapipe"]

        self.get_layout(layout)
        #  self.hop_dis = get_hop_distance(self.num_node, self.inward, max_hop)

        assert hasattr(self, mode), f"Do Not Exist This Mode: {mode}"
        self.A = getattr(self, mode)()

    def __str__(self):
        return self.A

    def get_layout(self, layout):
        if layout == "mediapipe":
            self.num_node = 33
            self.inward = [
                # Torso
                (12, 11),
                (24, 12),
                (24, 23),
                (23, 11),  # shoulders to hips
                # Right arm
                (14, 12),
                (16, 14),
                (18, 16),
                (20, 18),
                (20, 22),  # shoulder to fingertip
                # Left arm
                (13, 11),
                (15, 13),
                (17, 15),
                (19, 17),
                (21, 19),  # shoulder to fingertip
                # Right leg
                (26, 24),
                (28, 26),
                (30, 28),
                (32, 30),  # hip to foot
                # Left leg
                (25, 23),
                (27, 25),
                (29, 27),
                (31, 29),  # hip to foot
                # Face
                (1, 0),
                (2, 1),
                (3, 2),
                (7, 3),  # right eye
                (4, 0),
                (5, 4),
                (6, 5),
                (8, 6),  # left eye
                (10, 9),  # mouth
                # Add connections to nose (0) from shoulders
                # (0, 11), (0, 12)  # do we include this connection dear friend?
            ]
            self.center = 11  # Left shoulder as the center
        else:
            raise ValueError(f"Do Not Exist This Layout: {layout}")
        self.self_link = [(i, i) for i in range(self.num_node)]
        self.outward = [(j, i) for (i, j) in self.inward]
        self.neighbor = self.inward + self.outward

    def spatial(self):
        Iden = edge2mat(self.self_link, self.num_node)
        In = normalize_digraph(edge2mat(self.inward, self.num_node))
        Out = normalize_digraph(edge2mat(self.outward, self.num_node))
        A = np.stack((Iden, In, Out))
        return A

import torch
import torch.nn as nn
import torch.nn.functional as F


class AGCN(nn.Module):
    def __init__(
        self, in_channels, out_channels, adaptive="init", conv_pos="pre", with_res=False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.graph = Graph()
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.num_subsets = A.size(0)
        # print("NUM_SUBSET:", self.num_subsets)
        self.adaptive = adaptive
        self.conv_pos = conv_pos
        self.with_res = with_res

        self.A = nn.Parameter(A.clone())

        if self.conv_pos == "pre":
            self.conv = nn.Conv2d(in_channels, out_channels * self.num_subsets, 1)
        elif self.conv_pos == "post":
            self.conv = nn.Conv2d(self.num_subsets * in_channels, out_channels, 1)

        if self.with_res:
            if in_channels != out_channels:
                self.residual = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels),
                )
            else:
                self.residual = lambda x: x

        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        n, c, t, v = x.shape
        res = self.residual(x) if self.with_res else 0

        A = self.A

        if self.conv_pos == "pre":
            x = self.conv(x)
            x = x.view(n, self.num_subsets, -1, t, v)
            x = torch.einsum("nkctv,kvw->nctw", (x, A)).contiguous()
        elif self.conv_pos == "post":
            x = torch.einsum("nctv,kvw->nkctw", (x, A)).contiguous()
            x = x.view(n, -1, t, v)
            x = self.conv(x)

        return F.relu(self.batch_norm(x) + res)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        if hasattr(m, "weight"):
            # He initialization for convolutional layers
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias, 0)

    elif classname.find("BatchNorm") != -1:
        if hasattr(m, "weight") and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            m.bias.data.fill_(0)

class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1),
        )

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class MTCN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dilations=[1, 2, 3, 4],
        residual=True,
        residual_kernel_size=1,
    ):

        super().__init__()
        assert (
            out_channels % (len(dilations) + 2) == 0
        ), "# out channels should be multiples of # branches"

        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size] * len(dilations)
        # Temporal Convolution branches
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
                    nn.BatchNorm2d(branch_channels),
                    nn.ReLU(inplace=True),
                    TemporalConv(
                        branch_channels,
                        branch_channels,
                        kernel_size=ks,
                        stride=stride,
                        dilation=dilation,
                    ),
                )
                for ks, dilation in zip(kernel_size, dilations)
            ]
        )

        # Additional Max & 1x1 branch
        self.branches.append(
            nn.Sequential(
                nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
                nn.BatchNorm2d(branch_channels),
            )
        )

        self.branches.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0,
                    stride=(stride, 1),
                ),
                nn.BatchNorm2d(branch_channels),
            )
        )

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(
                in_channels,
                out_channels,
                kernel_size=residual_kernel_size,
                stride=stride,
            )

        # initialize
        self.apply(weights_init)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out = self.bn(out)
        out += res
        return out




class STSAM(nn.Module):
    def __init__(self, in_channels):
        super(STSAM, self).__init__()

        # 1x1 convolutions for Q, K, V
        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # 1x1 convolutions for scaling attention maps
        self.Ws = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.Wt = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        N, C, T, V = x.size()

        # Generate Q, K, V
        Q = self.query_conv(x)
        K = self.key_conv(x)
        V = self.value_conv(x)

        # Spatial attention
        Qs = torch.mean(Q, dim=2, keepdim=True)  # (N, C, 1, V)
        Ks = torch.mean(K, dim=2, keepdim=True)  # (N, C, 1, V)
        Vs = torch.mean(V, dim=2, keepdim=True)  # (N, C, 1, V)

        # Temporal attention
        Qt = torch.mean(Q, dim=3, keepdim=True)  # (N, C, T, 1)
        Kt = torch.mean(K, dim=3, keepdim=True)  # (N, C, T, 1)
        Vt = torch.mean(V, dim=3, keepdim=True)  # (N, C, T, 1)

        # Compute attention maps
        Ms = torch.matmul(Qs.transpose(2, 3), Ks) / torch.sqrt(torch.tensor(C, dtype=torch.float))  # Spatial attention
        Ms = torch.softmax(Ms, dim=-1)
        Ms = torch.matmul(Ms, Vs.transpose(2, 3)).transpose(2, 3)

        Mt = torch.matmul(Qt.transpose(2, 3), Kt) / torch.sqrt(torch.tensor(C, dtype=torch.float))  # Temporal attention
        Mt = torch.softmax(Mt, dim=-1)
        Mt = torch.matmul(Mt, Vt.transpose(2, 3)).transpose(2, 3)

        # Scale attention maps
        Ms1 = torch.sigmoid(self.Ws(Ms))  # (N, 1, 1, V)
        Mt1 = torch.sigmoid(self.Wt(Mt))  # (N, 1, T, 1)

        # Apply attention with residual connections
        out = (x + x * Ms1) + (x * Mt1)

        return out, Ms1, Mt1

import torch
import torch.nn as nn
import torch.nn.functional as F


class STSAE_GCN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, residual=True, stride=1):
        super(STSAE_GCN_Block, self).__init__()

        # Graph Convolution
        self.agcn = AGCN(in_channels, out_channels)

        # Replace separate attention modules with STSAM
        if USE_ATTENTION:
          self.stsam = STSAM(out_channels)  # Spatio-Temporal Self-Attention Module

        # Multi-Scale Temporal Convolution Network
        self.mtcn = MTCN(out_channels, out_channels)

        # Residual Connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(
                in_channels, out_channels, kernel_size=1, stride=stride
            )

        # Activation
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)

        # Store Attention Maps
        self.Ms = None  # Spatial Attention Map
        self.Mt = None  # Temporal Attention Map

    def forward(self, x):
        res = self.residual(x)

        # Apply Graph Convolution
        x = self.agcn(x)

        # Apply STSAM and store attention maps
        if USE_ATTENTION:
          x, self.Ms, self.Mt = self.stsam(x)

        # Apply Multi-Scale Temporal Convolution
        x = self.bn(self.mtcn(x)) + res

        return self.relu(x)

    def get_attention(self):
        """
        Returns the stored attention maps.
        """
        return {"Ms": self.Ms, "Mt": self.Mt}


class STSAE_GCN(nn.Module):
    def __init__(
        self, in_channels, hidden_channels, num_classes, num_frames, num_blocks=9
    ):
        super(STSAE_GCN, self).__init__()
        self.num_blocks = num_blocks
        num_nodes = 33
        self.batch_norm = nn.BatchNorm1d(in_channels * num_nodes)
        self.blocks = nn.ModuleList(
            [
                STSAE_GCN_Block(
                    in_channels if i == 0 else hidden_channels,
                    hidden_channels,
                    residual=False if i == 0 else True,
                )
                for i in range(num_blocks)
            ]
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling
        self.batch_norm_out = nn.BatchNorm1d(hidden_channels)
        self.fc = nn.Linear(
            hidden_channels, num_classes
        )  # Only hidden_channels remain after pooling

        # Initialize weights
        self.apply(weights_init)

    def forward(self, x):
        # x shape: (batch_size, in_channels, num_frames, num_nodes)

        # print("SHAPE X:", x.shape)
        N, C, T, V = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(N, V * C, T)
        x = self.batch_norm(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1)


        for block in self.blocks:
            # print("X SHAPE:", x.shape)
            # print(len(self.blocks))
            x = block(x)
            # print("X SHAPE:", x.shape)

        # Apply Global Average Pooling
        x = (
            self.avg_pool(x).squeeze(-1).squeeze(-1)
        )  # Reduce to (batch_size, hidden_channels)

        x = self.batch_norm_out(x)
        # Fully connected layer
        x = self.fc(x)
        return x

    def count_parameters(self):
        total_params = 0
        param_str = ""
        for name, parameter in self.named_parameters():
            if parameter.requires_grad:
                params = parameter.numel()
                param_str += f"{name}: {params}\n"
                total_params += params
        param_str += f"Total Trainable Params: {total_params}"
        print(param_str)
        return param_str, total_params
