import torch
import torch.nn as nn
import torch.nn.functional as F
import util


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,wv->ncwl', (x, A))
        return x.contiguous()


class d_nconv(nn.Module):
    def __init__(self):
        super(d_nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncwl,nvw->ncvl', (x, A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)
    def forward(self, x):
        return self.mlp(x)


class linear_(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear_, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 2), dilation=2, padding=(0, 0), stride=(1, 1),
                                   bias=True)
    def forward(self, x):
        return self.mlp(x)

"""
class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2
        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h
"""


class dgcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, order=2):
        super(dgcn, self).__init__()
        self.d_nconv = d_nconv()
        c_in = (order * 3 + 1) * c_in
        # self.mlp = linear_(c_in, c_out)
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.d_nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.d_nconv(x1, a)
                out.append(x2)
                x1 = x2
        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


# class hgcn(nn.Module):
#     def __init__(self, c_in, c_out, dropout, order=2):
#         super(hgcn, self).__init__()
#         self.nconv = nconv()
#         c_in = (order + 1) * c_in
#         self.mlp = linear(c_in, c_out)
#         self.dropout = dropout
#         self.order = order

#     def forward(self, x, G):
#         out = [x]
#         support = [G]
#         for a in support:
#             x1 = self.nconv(x, a)
#             out.append(x1)
#             for k in range(2, self.order + 1):
#                 x2 = self.nconv(x1, a)
#                 out.append(x2)
#                 x1 = x2
#         h = torch.cat(out, dim=1)
#         h = self.mlp(h)
#         h = F.dropout(h, self.dropout, training=self.training)
#         return h


# class hgcn_edge_At(nn.Module):
#     def __init__(self, c_in, c_out, dropout, order=1):
#         super(hgcn_edge_At, self).__init__()
#         self.nconv = nconv()
#         c_in = (order + 1) * c_in
#         self.mlp = linear(c_in, c_out)
#         self.dropout = dropout
#         self.order = order

#     def forward(self, x, G):
#         out = [x]
#         support = [G]
#         for a in support:
#             x1 = self.nconv(x, a)
#             out.append(x1)
#             for k in range(2, self.order + 1):
#                 x2 = self.nconv(x1, a)
#                 out.append(x2)
#                 x1 = x2
#         h = torch.cat(out, dim=1)
#         h = self.mlp(h)
#         h = F.dropout(h, self.dropout, training=self.training)
#         return h
    
class gcn_glu(nn.Module):
    def __init__(self,c_in,c_out):
        super(gcn_glu,self).__init__()
        self.nconv = nconv()
        self.mlp = linear(c_in,2*c_out)
        self.c_out = c_out
    def forward(self, x, A):
        # (3N, B, C)
        x = x.unsqueeze(3) # (3N, B, C, 1)
        x = x.permute(1, 2, 0, 3) # (3N, B, C, 1)->(B, C, 3N, 1)   
        ax = self.nconv(x, A)
        # print(ax.shape) # torch.Size([8, 24, 621, 1])
        axw = self.mlp(ax) # (B, 2C', 3N, 1)
        axw_1,axw_2 = torch.split(axw, [self.c_out, self.c_out], dim=1)
        # print(axw_1.shape, axw_2.shape) # torch.Size([8, 24, 621, 1]) torch.Size([8, 24, 621, 1])
        axw_new = axw_1 * torch.sigmoid(axw_2) # (B, C', 3N, 1)
        axw_new = axw_new.squeeze(3) # (B, C', 3N)
        axw_new = axw_new.permute(2, 0, 1) # (3N, B, C')
        # print(axw_new.shape) # torch.Size([621, 8, 24])
        return axw_new

class stsgcm(nn.Module):
    def __init__(self, num_nodes, num_of_features, output_features_num):
        super(stsgcm,self).__init__()
        c_in = num_of_features
        c_out = output_features_num
        gcn_num = 3
        self.gcn_glu = nn.ModuleList()
        for _ in range(gcn_num):
            self.gcn_glu.append(gcn_glu(c_in,c_out))
            c_in = c_out
        self.num_nodes = num_nodes
        self.gcn_num = gcn_num
    def forward(self, x, A):
        # (3N, B, C)
        need_concat = []
        # print(x.shape, A.shape) # torch.Size([621, 8, 24]) torch.Size([621, 621])
        for i in range(self.gcn_num):   
            x = self.gcn_glu[i](x, A)
            need_concat.append(x)
            # print(x.shape) # orch.Size([621, 8, 24])
        # (3N, B, C')
        # print(x.shape, self.num_nodes)
        need_concat = [i[(self.num_nodes):(2*self.num_nodes),:,:].unsqueeze(0) for i in need_concat] # (1, N, B, C')
        # print(need_concat[0].shape)
        outputs = torch.stack(need_concat,dim=0) # (3, N, B, C')
        # print(outputs.shape)
        outputs = torch.max(outputs, dim=0).values # (1, N, B, C')
        # print(outputs.shape) # torch.Size([1, 207, 8, 24])
        return outputs
    
class stsgcl(nn.Module):
    def __init__(self, c_in, c_out, input_length, num_of_vertices):
        super(stsgcl,self).__init__()
        # print(c_in, c_out, dropout, input_length, num_of_vertices)
        #        24     1     0.3      11->9->7            207
        self.T = input_length
        self.num_of_vertices = num_of_vertices
        self.input_features_num = c_in
        output_features_num = c_out
        self.stsgcm = nn.ModuleList()
        self.mlp = linear(c_in, c_out)
        for _ in range(self.T-2):
            self.stsgcm.append(stsgcm(num_of_vertices, self.input_features_num, output_features_num))
        # position_embedding
        self.temporal_emb = torch.nn.init.xavier_normal_(torch.empty(1, self.T, 1, self.input_features_num), gain=0.0003).cuda()
        self.spatial_emb = torch.nn.init.xavier_normal_(torch.empty(1, 1, self.num_of_vertices, self.input_features_num), gain=0.0003).cuda()

    def forward(self, x, A):
        # (B, T, N, C)
        # position_embedding
        # x的shape是[8, 24, 207, 11], temporal_emb的是[1, 24, 1, 24], 因为self.input_features_num = c_in是24
        # torch.Size([8, 11, 207, 24]) torch.Size([621, 621]) torch.Size([1, 10, 1, 24]) torch.Size([1, 1, 207, 24])
        # print(self.T, x.shape, A.shape, self.temporal_emb.shape, self.spatial_emb.shape)
        x = x+self.temporal_emb
        x = x+self.spatial_emb
        data = x
        need_concat = []
        for i in range(self.T-2):
            # print(data.shape) # torch.Size([8, 11, 207, 24])
            t = data[:,i:i+3,:,:]
            # print(t.shape)    # torch.Size([8, 3, 207, 24])
            # print(self.num_of_vertices, self.input_features_num) # 207 24
            t = t.reshape([-1, 3 * self.num_of_vertices, self.input_features_num])
            # print(t.shape)    # torch.Size([8, 621, 24])
            t = t.permute(1, 0, 2)
            # print(t.shape)    # [621, 8, 24]
            t = self.stsgcm[i](t, A)
            # print(t.shape)    # [1, 207, 8, 24]
            t = t.permute(2, 0, 1, 3).squeeze(1)
            # print(t.shape)    # [8, 207, 24]
            need_concat.append(t)
        outputs = torch.stack(need_concat, dim=1)  # (B, T - 2, N, C')
        # print(outputs.shape) 
        return outputs


# class linear_1(nn.Module):
#     def __init__(self, c_in, c_out):
#         super(linear_1, self).__init__()
#         self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 15), padding=(0, 0), stride=(1, 1), bias=True)
#     def forward(self, x):
#         return self.mlp(x)
    
class hgcn_edge_At(nn.Module):
    def __init__(self, c_in, c_out, dropout, order=1):
        super(hgcn_edge_At, self).__init__()
        self.nconv = nconv()
        c_in = (order + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, G):
        out = [x]
        support = [G]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2
        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

"""
class dhgcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, order=2):
        super(dhgcn, self).__init__()
        self.d_nconv = d_nconv()
        c_in = (order + 1) * c_in
        self.mlp = linear_(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, G):
        out = [x]
        support = [G]
        for a in support:
            x1 = self.d_nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.d_nconv(x1, a)
                out.append(x2)
                x1 = x2
        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h
"""


class spatial_attention(nn.Module):
    def __init__(self, in_channels, num_of_timesteps, num_of_edge, num_of_vertices):
        super(spatial_attention, self).__init__()
        self.W1 = nn.Parameter(torch.randn(num_of_timesteps).cuda(), requires_grad=True).cuda()
        self.W2 = nn.Parameter(torch.randn(num_of_timesteps).cuda(), requires_grad=True).cuda()
        self.W3 = nn.Parameter(torch.randn(in_channels, int(in_channels / 2)).cuda(), requires_grad=True).cuda()
        self.W4 = nn.Parameter(torch.randn(in_channels, int(in_channels / 2)).cuda(), requires_grad=True).cuda()
        self.out_conv = nn.Conv2d(in_channels=in_channels,
                                  out_channels=in_channels,
                                  kernel_size=(1, 1))

    def forward(self, x, idx, idy):
        # print(x.shape, self.W1.shape) # torch.Size([8, 207, 24, 12]) torch.Size([12])
        lhs = torch.matmul(torch.matmul(x, self.W1), self.W3)
        rhs = torch.matmul(torch.matmul(x, self.W2), self.W4)
        sum = torch.cat([lhs[:, idx, :], rhs[:, idy, :]], dim=2)
        sum = torch.unsqueeze(sum, dim=3).transpose(1, 2)
        S = self.out_conv(sum)
        S = torch.squeeze(S).transpose(1, 2)
        return S

class output_layer(nn.Module):
    def __init__(self, residual_channels, input_length):
        super(output_layer,self).__init__()
        nhid = residual_channels
        # print(nhid, input_length)
        self.fully_1 = torch.nn.Conv2d(input_length * nhid, 128, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)
        self.fully_2 = torch.nn.Conv2d(128, 1, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, data):
        # (B, T, N, C)
        _, time_num, node_num, feature_num = data.size()
        data = data.permute(0, 2, 1, 3) # (B, T, N, C)->(B, N, T, C)
        # print(data.shape) # torch.Size([8, 207, 4, 24])
        data = data.reshape([-1, node_num, time_num*feature_num, 1]) # (B, N, T, C)->(B, N, T*C, 1)
        # print(data.shape) # torch.Size([8, 207, 96, 1])
        data = data.permute(0, 2, 1, 3) # (B, N, T*C, 1)->(B, T*C, N, 1)
        # print(data.shape) # torch.Size([8, 96, 207, 1])
        data = self.fully_1(data) # (B, 128, N, 1)
        data = torch.relu(data)
        data = self.fully_2(data) # (B, 1, N, 1)
        data = data.squeeze(dim=3) # (B, 1, N)
        # print(data.shape) # torch.Size([8, 1, 207])
        return data # (B, 1, N)
    
class ddstgcn(nn.Module):
    def __init__(self, batch_size, H_a, H_b, G0, G1, indices, G0_all, G1_all, H_T_new, lwjl, num_nodes,
                 dropout=0.3, supports=None, supports_=None, in_dim=2, out_dim=12, residual_channels=40, dilation_channels=40,
                 skip_channels=320, end_channels=640, kernel_size=2, blocks=4, layers=1):
        super(ddstgcn, self).__init__()
        self.batch_size = batch_size
        self.H_a = H_a
        self.H_b = H_b
        self.G0 = G0
        self.G1 = G1
        # self.H_T_new = H_T_new
        self.lwjl = lwjl
        self.indices = indices
        self.G0_all = G0_all
        self.G1_all = G1_all

        self.edge_node_vec1 = nn.Parameter(torch.rand(self.H_a.size(1), 10).cuda(), requires_grad=True).cuda()
        self.edge_node_vec2 = nn.Parameter(torch.rand(10, self.H_a.size(0)).cuda(), requires_grad=True).cuda()

        self.node_edge_vec1 = nn.Parameter(torch.rand(self.H_a.size(0), 10).cuda(), requires_grad=True).cuda()
        self.node_edge_vec2 = nn.Parameter(torch.rand(10, self.H_a.size(1)).cuda(), requires_grad=True).cuda()

        self.hgcn_w_vec_edge_At_forward = nn.Parameter(torch.rand(self.H_a.size(1)).cuda(), requires_grad=True).cuda()
        self.hgcn_w_vec_edge_At_backward = nn.Parameter(torch.rand(self.H_a.size(1)).cuda(), requires_grad=True).cuda()

        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.dgconv = nn.ModuleList()
        self.stsgcl = nn.ModuleList()
        # self.filter_convs_h = nn.ModuleList()
        # self.gate_convs_h = nn.ModuleList()
        self.SAt_forward = nn.ModuleList()
        self.SAt_backward = nn.ModuleList()
        self.hgconv_edge_At_forward = nn.ModuleList()
        self.hgconv_edge_At_backward = nn.ModuleList()
        # self.gconv_dgcn_w = nn.ModuleList()
        # self.dhgconv = nn.ModuleList()
        self.bn_g = nn.ModuleList()
        # self.bn_hg = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim, out_channels=residual_channels, kernel_size=(1, 1))
        self.supports = supports
        self.supports_ = supports_
        self.num_nodes = 3*num_nodes
        # receptive_field = 1
        receptive_field = 0
        self.supports_len = 0
        self.supports_len += len(supports)
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).cuda(), requires_grad=True).cuda()
        self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).cuda(), requires_grad=True).cuda()
        self.nodevec1_ = nn.Parameter(torch.randn(self.num_nodes, 10).cuda(), requires_grad=True).cuda()
        self.nodevec2_ = nn.Parameter(torch.randn(10, self.num_nodes).cuda(), requires_grad=True).cuda()
        self.supports_len += 1

        input_length = 12

        for b in range(blocks):
            additional_scope = kernel_size
            new_dilation = 2
            for i in range(layers):
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size), dilation=new_dilation))
                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                # self.filter_convs_h.append(nn.Conv2d(in_channels=1 + residual_channels * 2,
                #                                      out_channels=dilation_channels,
                #                                      kernel_size=(1, kernel_size), dilation=new_dilation))
                # self.gate_convs_h.append(nn.Conv2d(in_channels=1 + residual_channels * 2,
                #                                    out_channels=dilation_channels,
                #                                    kernel_size=(1, kernel_size), dilation=new_dilation))
                # self.SAt_forward.append(spatial_attention(residual_channels, int(13 - receptive_field + 1),
                #                                           self.indices.size(1), num_nodes))
                # self.SAt_backward.append(spatial_attention(residual_channels, int(13 - receptive_field + 1),
                #                                            self.indices.size(1), num_nodes))
                self.SAt_forward.append(spatial_attention(residual_channels, int(12 - receptive_field),
                                                          self.indices.size(1), num_nodes))
                self.SAt_backward.append(spatial_attention(residual_channels, int(12 - receptive_field),
                                                           self.indices.size(1), num_nodes))
                # receptive_field += (additional_scope * 2)
                receptive_field += additional_scope
                # self.dgconv.append(dgcn(dilation_channels, int(residual_channels / 2), dropout))
                #  ----- No Hypergraph Convolution, residual channels needn't reduce by half
                self.dgconv.append(dgcn(dilation_channels, residual_channels, dropout))
                self.stsgcl.append(stsgcl(residual_channels, dilation_channels, input_length, num_nodes))
                input_length -= 2
                #                                  24        1     0.3      12            1722
                self.hgconv_edge_At_forward.append(hgcn_edge_At(residual_channels, 1, dropout))
                self.hgconv_edge_At_backward.append(hgcn_edge_At(residual_channels, 1, dropout))
                # self.gconv_dgcn_w.append(gcn(residual_channels, 1, dropout, support_len=2, order=1))
                # self.dhgconv.append(dhgcn(dilation_channels, int(residual_channels / 2), dropout))
                # self.bn_g.append(nn.BatchNorm2d(int(residual_channels / 2)))
                #  ----- No Hypergraph Convolution, residual channels needn't reduce by half
                self.bn_g.append(nn.BatchNorm2d(residual_channels))
                # self.bn_hg.append(nn.BatchNorm2d(int(residual_channels / 2)))

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.receptive_field = receptive_field
        self.bn_start = nn.BatchNorm2d(in_dim, affine=False)  
        self.new_supports_w = [
            torch.zeros([num_nodes, num_nodes]).repeat([self.batch_size, 1, 1]).cuda()]
        self.new_supports_w = self.new_supports_w + [
            torch.zeros([num_nodes, num_nodes]).repeat([self.batch_size, 1, 1]).cuda()]
        self.new_supports_w = self.new_supports_w + [
            torch.zeros([num_nodes, num_nodes]).repeat([self.batch_size, 1, 1]).cuda()]
        # self.new_supports_w_ = [
        #     torch.zeros([self.num_nodes, self.num_nodes]).repeat([self.batch_size, 1, 1]).cuda()]
        # self.new_supports_w_ = self.new_supports_w_ + [
        #     torch.zeros([self.num_nodes, self.num_nodes]).repeat([self.batch_size, 1, 1]).cuda()]
        # self.new_supports_w_ = self.new_supports_w_ + [
        #     torch.zeros([self.num_nodes, self.num_nodes]).repeat([self.batch_size, 1, 1]).cuda()]
        # # print(self.new_supports_w[0].shape) # torch.Size([8, 621, 621])

        self.predict_length = 12
        self.mask = nn.Parameter(torch.rand(3*num_nodes, 3*num_nodes).cuda(), requires_grad=True).cuda()
        self.output_layer = nn.ModuleList()
        for _ in range(self.predict_length):
            self.output_layer.append(output_layer(residual_channels, input_length))
        # print(in_dim, residual_channels) # 1 24
        self.start_conv= nn.Conv2d(in_dim, residual_channels, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)
        # self.start_conv = nn.Conv2d(in_channels=in_dim, out_channels=residual_channels, kernel_size=(1, 1))

    def forward(self, input):
        # print("2、", input.shape)
        in_len = input.size(3)
        # print(in_len, self.receptive_field)
        # if in_len < self.receptive_field:
        #     x = nn.functional.pad(input, (self.receptive_field - in_len, 0, 0, 0))
        # else:
        #     x = input
        x = input

        # print(x.shape)  # [8, 1, 207, 12]
        # x = self.bn_start(x) # 数据的归一化处理，这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定
        # print(x.shape)  # [8, 1, 207, 12]
        x = self.start_conv(x)
        # print(x.shape)  # [8, 24, 207, 12]
        x = torch.relu(x) # 激活函数，全称为修正线性单元。它的主要作用是将输入值限制在一个非负的范围内，
                          # 即当输入值小于0时，输出值为0;当输入值大于等于0时，输出值等于输入值本身
        adj = self.mask * self.supports_
        # print(x.shape) # [8, 24, 207, 12]
        
        skip = 0
        adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        adp_new = adp.repeat([self.batch_size, 1, 1])
        new_supports = self.supports + [adp]
        # print(new_supports[0].shape)
        # adp_ = F.softmax(F.relu(torch.mm(self.nodevec1_, self.nodevec2_)), dim=1)
        # adp_new_ = adp_.repeat([self.batch_size, 1, 1])
        # # print(self.supports_[0].shape)
        # new_supports_ = self.supports_ + [adp_]
        # print(new_supports_[0].shape)
        # edge_node_H = (self.H_T_new * (torch.mm(self.edge_node_vec1, self.edge_node_vec2)))
        # self.H_a_ = (self.H_a * (torch.mm(self.node_edge_vec1, self.node_edge_vec2)))
        # self.H_b_ = (self.H_b * (torch.mm(self.node_edge_vec1, self.node_edge_vec2)))
        G0G1_edge_At_forward = self.G0_all @ (torch.diag_embed(self.hgcn_w_vec_edge_At_forward)) @ self.G1_all
        G0G1_edge_At_backward = self.G0_all @ (torch.diag_embed(self.hgcn_w_vec_edge_At_backward)) @ self.G1_all
        self.new_supports_w[2] = adp_new.cuda()
        # self.new_supports_w_[2] = adp_new_.cuda()
        # forward_medium = torch.eye(self.num_nodes).repeat([self.batch_size, 1, 1]).cuda()
        # backward_medium = torch.eye(self.num_nodes).repeat([self.batch_size, 1, 1]).cuda()
        forward_medium = torch.eye(int(self.num_nodes/3)).repeat([self.batch_size, 1, 1]).cuda()
        backward_medium = torch.eye(int(self.num_nodes/3)).repeat([self.batch_size, 1, 1]).cuda()
        forward_medium_ = torch.eye(self.num_nodes).repeat([self.batch_size, 1, 1]).cuda()
        backward_medium_ = torch.eye(self.num_nodes).repeat([self.batch_size, 1, 1]).cuda()

        for i in range(self.blocks * self.layers):
            # ----------  Graph to Hypergraph Dual Transformation start  ------------ #
            # edge_feature = util.feature_node_to_edge(x, self.H_a_, self.H_b_, operation="concat")
            # edge_feature = torch.cat([edge_feature, self.lwjl.repeat(1, 1, 1, edge_feature.size(3))], dim=1)
            # ----------   Graph to Hypergraph Dual Transformation end   ------------ #
            # ----------  Hypergraph Gate-TCN start  ------------ #
            # filter_h = self.filter_convs_h[i](edge_feature)
            # filter_h = torch.tanh(filter_h)
            # gate_h = self.gate_convs_h[i](edge_feature)
            # gate_h = torch.sigmoid(gate_h)
            # x_h = filter_h * gate_h
            # ----------   Hypergraph Gate-TCN end   ------------ #

            # ----------  Forward edge Extaction start  ------------ #
            # print(x.shape, x.transpose(1, 2).shape, self.indices[0].shape, self.indices[1].shape) # torch.Size([8, 24, 207, 13]) torch.Size([8, 207, 24, 13]) torch.Size([1722]) torch.Size([1722])
            batch_edge_forward = self.SAt_forward[i](x.transpose(1, 2), self.indices[0], self.indices[1])
            batch_edge_forward = torch.unsqueeze(batch_edge_forward, dim=3)
            batch_edge_forward = batch_edge_forward.transpose(1, 2)
            # ----------   Forward edge Extaction end   ------------ #
            # ----------  Forward edge HGCN start  ------------ #
            # print(batch_edge_forward.shape, G0G1_edge_At_forward.shape) # torch.Size([8, 24, 1722, 1]) torch.Size([1722, 1722])
            batch_edge_forward = self.hgconv_edge_At_forward[i](batch_edge_forward, G0G1_edge_At_forward)
            batch_edge_forward = torch.squeeze(batch_edge_forward)
            # ----------   Forward edge HGCN end   ------------ #
            # print(batch_edge_forward.shape) # torch.Size([8, 1722])
            # print(self.indices[0].shape, self.indices[1].shape) # torch.Size([1722]) torch.Size([1722])
            forward_medium[:, self.indices[0], self.indices[1]] = torch.sigmoid((batch_edge_forward))
            self.new_supports_w[0] = forward_medium
            # print(self.new_supports_w[0].shape) # torch.Size([8, 207, 207])
            forward_medium_[:, self.indices[0], self.indices[1]] = torch.sigmoid((batch_edge_forward))
            # self.new_supports_w_[0] = forward_medium_
            # print(self.new_supports_w_[0].shape) # torch.Size([8, 621, 621])
            # ----------  Backward edge Extaction start  ------------ #
            batch_edge_backward = self.SAt_backward[i](x.transpose(1, 2), self.indices[0], self.indices[1])
            batch_edge_backward = torch.unsqueeze(batch_edge_backward, dim=3)
            batch_edge_backward = batch_edge_backward.transpose(1, 2)
            # ----------   Backward edge Extaction end   ------------ #
            # ----------  Backward edge HGCN start  ------------ #
            batch_edge_backward = self.hgconv_edge_At_backward[i](batch_edge_backward, G0G1_edge_At_backward)
            batch_edge_backward = torch.squeeze(batch_edge_backward)
            # ----------   Backward edge HGCN end   ------------ #
            backward_medium[:, self.indices[0], self.indices[1]] = torch.sigmoid((batch_edge_backward))
            self.new_supports_w[1] = backward_medium.transpose(1, 2)
            backward_medium_[:, self.indices[0], self.indices[1]] = torch.sigmoid((batch_edge_backward))
            # self.new_supports_w_[1] = backward_medium_.transpose(1, 2)
            # print(self.new_supports_w[0].shape, self.new_supports_w[1].shape, new_supports[0].shape, new_supports[1].shape) # torch.Size([8, 207, 207]) torch.Size([207, 207])
            self.new_supports_w[0] = self.new_supports_w[0] * new_supports[0]
            self.new_supports_w[1] = self.new_supports_w[1] * new_supports[1]
            # self.new_supports_w_[0] = self.new_supports_w_[0] * new_supports_[0]
            # self.new_supports_w_[1] = self.new_supports_w_[1] * new_supports_[1]
            # ----------  Traffic Gate-TCN start  ------------ #
            residual = x
            # print(residual.shape) # [8, 24, 207, 13]
            # filter = self.filter_convs[i](residual)
            filter = residual
            # print(filter.shape) # [8, 24, 207, 11]
            filter = torch.tanh(filter)
            # gate = self.gate_convs[i](residual)
            gate = residual
            gate = torch.sigmoid(gate)
            # print(filter.shape, gate.shape) # [8, 24, 207, 11] [8, 24, 207, 11]
            x = filter * gate
            # ----------   Traffic Gate-TCN end   ------------ #
            # ----------    DGCN start    ------------ #
            # print(x.shape, self.new_supports_w[0].shape) # torch.Size([8, 24, 207, 11]) torch.Size([8, 207, 207])
            x = self.dgconv[i](x, self.new_supports_w)
            # print(x.shape, self.new_supports_w_[0].shape) # torch.Size([8, 24, 207, 11]) torch.Size([8, 621, 621])
            # new_supports_w_ = torch.cat(self.new_supports_w_, dim=0)
            # new_supports_w_ = torch.max(new_supports_w_, dim=0).values
            # print(x.shape, self.supports_.shape)
            
            # x = x.permute(0, 3, 2, 1)
            # x = self.stsgcl[i](x, self.supports_)
            # # print(x.shape) # torch.Size([8, 9, 207, 24])
            # x = x.permute(0, 3, 2, 1)

            # print(x.shape) # torch.Size([8, 24, 207, 10])
            x = x.permute(0, 3, 2, 1) # （B,C,N,T）
            # x = self.input_layer(x)
            # x = torch.relu(x)
            # x = x.permute(0, 3, 2, 1)  # （B,T,N,C'）
            # print(data.shape) # [8, 12, 207, 24]
            # print(self.mask.shape, self.A.shape) # torch.Size([621, 621]) torch.Size([621, 621])
            # for i in range(self.layers):
                # print(data.shape, adj.shape) # torch.Size([8, 12, 207, 24]) torch.Size([621, 621])
            x = self.stsgcl[i](x, adj)
            # # (B, 4, N, C')
            # need_concat = []
            # print(x.shape)
            # for i in range(self.predict_length):
            #     output = self.output_layer[i](data) # (B, 1, N)
            #     need_concat.append(output.squeeze(1))
            # x = torch.stack(need_concat, dim=1)  # (B, 12, N)
            # x = x.unsqueeze(3) # (B, 12, N, 1)

            x = x.permute(0, 3, 2, 1) # （B,C,N,T）
            # print(x.shape) # torch.Size([8, 24, 207, 9])
            x = self.bn_g[i](x)
            # print(x.shape) # torch.Size([8, 24, 207, 9])
            # ----------     DGCN end     ------------ #

            # dhgcn_w_input = residual
            # ---------- Pooling start ------------ #
            # dhgcn_w_input = dhgcn_w_input.transpose(1, 2)
            # dhgcn_w_input = torch.mean(dhgcn_w_input, 3)
            # dhgcn_w_input = dhgcn_w_input.transpose(0, 2)
            # dhgcn_w_input = torch.unsqueeze(dhgcn_w_input, dim=0)
            # ----------  Pooling end  ------------ #
            # ----------   GCN start   ------------ #
            # dhgcn_w_input = self.gconv_dgcn_w[i](dhgcn_w_input, self.supports)
            # ----------    GCN end    ------------ #
            # ----------  DHGCN start  ------------ #
            # dhgcn_w_input = torch.squeeze(dhgcn_w_input)
            # dhgcn_w_input = dhgcn_w_input.transpose(0, 1)
            # dhgcn_w_input = self.G0 @ (torch.diag_embed(dhgcn_w_input)) @ self.G1
            # x_h = self.dhgconv[i](x_h, dhgcn_w_input)
            # ----------   DHGCN end   ------------ #
            # ----------  Hypergraph to Graph Dual Transformation start  ------------ #
            # x_h = self.bn_hg[i](x_h)
            # ----------   Hypergraph to Graph Dual Transformation end   ------------ #

            # ---------- No Hypergraph Convolution, needn't fusion
            # x = util.fusion_edge_node(x, x_h, edge_node_H)
            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)
            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :, -s.size(3):]
            except:
                skip = 0
            skip = s + skip 
            # print(x.shape)

        x = x.permute(0, 3, 2, 1)
        need_concat = []
        for i in range(self.predict_length):
            output = self.output_layer[i](x) # (B, 1, N)
            need_concat.append(output.squeeze(1))
        x = torch.stack(need_concat, dim=1)  # (B, 12, N)
        x = x.unsqueeze(3) # (B, 12, N, 1)

        # print("4、", x.shape)
        # x = F.leaky_relu(skip)
        # x = F.leaky_relu(self.end_conv_1(x))
        # x = self.end_conv_2(x)
        # print("5、", x.shape)

        return x
