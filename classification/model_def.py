import torch
import numpy
import os
import torch.nn as nn

class AGCN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AGCN, self).__init__()
        # Initialize adjacency matrix for BlazePose (33 joints)
        self.num_nodes = 33
        # Define the natural connections in BlazePose skeleton
        self.edges = [
            # Torso
            (11, 12), (12, 24), (24, 23), (23, 11),  # shoulders to hips
            # Right arm
            (12, 14), (14, 16), (16, 18), (18, 20), (20, 22),  # shoulder to fingertip
            # Left arm
            (11, 13), (13, 15), (15, 17), (17, 19), (19, 21),  # shoulder to fingertip
            # Right leg
            (24, 26), (26, 28), (28, 30), (30, 32),  # hip to foot
            # Left leg
            (23, 25), (25, 27), (27, 29), (29, 31),  # hip to foot
            # Face
            (0, 1), (1, 2), (2, 3), (3, 7),  # right eye
            (0, 4), (4, 5), (5, 6), (6, 8),  # left eye
            (9, 10),  # mouth
            # Add connections to nose (0) from shoulders
            # (0, 11), (0, 12)  # do we include this connection dear friend?
        ]
        # Create adjacency matrix
        A = numpy.zeros((self.num_nodes, self.num_nodes))
        for i, j in self.edges:
            A[i, j] = 1
            A[j, i] = 1  # Undirected graph
        # Convert to tensor and make it a parameter
        self.A = nn.Parameter(torch.from_numpy(A.astype(numpy.float32)))
        # Create identity matrix
        self.identity = nn.Parameter(torch.eye(self.num_nodes), requires_grad=False)
        # 1x1 convolution for feature transformation
        self.W = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        # Compute degree matrix
        D = torch.sum(self.A, dim=1)
        D = torch.diag(torch.pow(D, -0.5))
        # D_r = torch.diag(torch.pow(D, 0.5))
        # Normalized adjacency matrix
        A_norm = torch.matmul(torch.matmul(D, self.A + self.identity), D)
        # Reshape input for matrix multiplication
        # N, C, T, V = x.size()
        x_reshape = x.permute(0, 2, 3, 1).contiguous()  # N, T, V, C
        # Apply GCN operation
        x_gc = torch.matmul(A_norm, x_reshape)  # N, T, V, C
        # Reshape back
        x_gc = x_gc.permute(0, 3, 1, 2).contiguous()  # N, C, T, V
        # Apply 1x1 convolution
        out = self.W(x_gc)
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
        out = (x + x * Ms1) + (x + x * Mt1)
        return out


class MTCN(nn.Module):
    def __init__(self, in_channels, hidden_channels=None):
        super(MTCN, self).__init__()
        # If hidden_channels not specified, make it divisible by 6
        if hidden_channels is None:
            hidden_channels = in_channels - (in_channels % 6)
        assert hidden_channels % 6 == 0, "var: hidden_channels should always be multple of 6 because 6 branches"
        self.branch_channels = hidden_channels // 6
        # Initial 1x1 conv to reduce channels
        self.init_conv = nn.Conv2d(
            in_channels,
            hidden_channels,
            kernel_size=1
        )
        # Branch 1: 1x1 Conv
        self.branch1 = nn.Conv2d(
            hidden_channels,
            self.branch_channels,
            kernel_size=1
        )
        # Branch 2: Max Pooling followed by 1x1 Conv to adjust channels
        self.branch2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1, 3), padding=(0, 1), stride=1),
            nn.Conv2d(hidden_channels, self.branch_channels, kernel_size=1)
        )
        # Branches 3-6: 1D Conv with different dilations
        self.branches = nn.ModuleList([
            nn.Conv2d(
                hidden_channels,
                self.branch_channels,
                kernel_size=(1, 3),
                padding=(0, dilation),
                dilation=(1, dilation)
            ) for dilation in range(1, 5)
        ])
        # Final 1x1 conv to restore original channel count
        self.final_conv = nn.Conv2d(hidden_channels, in_channels, kernel_size=1)
    def forward(self, x):
        # x shape: (batch_size, C, V, T)
        # Initial channel reduction
        x = self.init_conv(x)
        # Process each branch
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        # Process dilated convolution branches
        branch_outputs = [branch1, branch2]
        for branch in self.branches:
            branch_outputs.append(branch(x))
        # Concatenate all branch outputs
        x = torch.cat(branch_outputs, dim=1)
        # Final 1x1 conv
        x = self.final_conv(x)
        return x


class STSAE_GCN_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(STSAE_GCN_Block, self).__init__()
        self.agcn = AGCN(in_channels, out_channels)
        self.stsam = STSAM(out_channels)
        self.mtcn = MTCN(out_channels, 48)
    def forward(self, x):
        x = self.agcn(x)
        x = self.stsam(x)
        x = self.mtcn(x)
        return x

class STSAE_GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, num_frames, num_blocks=9):
        super(STSAE_GCN, self).__init__()
        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList([
            STSAE_GCN_Block(in_channels if i == 0 else hidden_channels,
                            hidden_channels)
            for i in range(num_blocks)
        ])
        num_nodes = 33
        self.fc = nn.Linear(hidden_channels * num_nodes * num_frames, num_classes)
    def forward(self, x):
        # x shape: (batch_size, in_channels, num_frames, num_nodes)
        for block in self.blocks:
            x = block(x)
        # Global average pooling
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    def count_parameters(self):
        total_params = 0
        for name, parameter in self.named_parameters():
            if parameter.requires_grad:
                params = parameter.numel()
                print(f"{name}: {params}")
                total_params += params
        print(f"Total Trainable Params: {total_params}")

