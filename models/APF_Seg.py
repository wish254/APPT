import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from pointnet2_ops import pointnet2_utils
from pointnet2_ops.pointnet2_utils import furthest_point_sample
from models.model_utils import PointNetFeaturePropagation, index_points, square_distance
from models.point_nn_seg import EncNP, DecNP
from models.point_pn import EncP
from knn_cuda import KNN
import timm
from models.transformer import *
import torch.nn.functional as F

class CosineClassifier(nn.Module):
    def __init__(self, feat_dim, num_classes, scale=30):
        super(CosineClassifier, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.scale = scale
        self.weight = nn.Parameter(torch.empty(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        x = F.normalize(x, dim=-1)
        weight = F.normalize(self.weight, dim=0)
        return F.linear(x, weight) * self.scale


def fps(data, number):
    '''
        data B N C
        number int
    '''
    fps_idx = furthest_point_sample(data[:, :, :3].contiguous(), number)
    fps_data = torch.gather(
        data, 1, fps_idx.unsqueeze(-1).long().expand(-1, -1, data.shape[-1]))
    return fps_data

class DGCNN_Propagation(nn.Module):
    def __init__(self, k=16):
        super().__init__()
        '''
        K has to be 16
        '''
        # print('using group version 2')
        self.k = k
        self.knn = KNN(k=k, transpose_mode=False)

        self.layer1 = nn.Sequential(nn.Conv2d(1536, 512, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 512),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

        self.layer2 = nn.Sequential(nn.Conv2d(1024, 768, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 768),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

    @staticmethod
    def fps_downsample(coor, x, num_group):
        xyz = coor.transpose(1, 2).contiguous() # b, n, 3
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, num_group)

        combined_x = torch.cat([coor, x], dim=1)

        new_combined_x = (
            pointnet2_utils.gather_operation(
                combined_x, fps_idx
            )
        )

        new_coor = new_combined_x[:, :3]
        new_x = new_combined_x[:, 3:]

        return new_coor, new_x

    def get_graph_feature(self, coor_q, x_q, coor_k, x_k):

        # coor: bs, 3, np, x: bs, c, np

        k = self.k
        batch_size = x_k.size(0)
        num_points_k = x_k.size(2)
        num_points_q = x_q.size(2)

        with torch.no_grad():
            _, idx = self.knn(coor_k, coor_q)  # bs k np
            assert idx.shape[1] == k
            idx_base = torch.arange(0, batch_size, device=x_q.device).view(-1, 1, 1) * num_points_k
            idx = idx + idx_base
            idx = idx.view(-1)
        num_dims = x_k.size(1)
        x_k = x_k.transpose(2, 1).contiguous()
        feature = x_k.view(batch_size * num_points_k, -1)[idx, :]
        feature = feature.view(batch_size, k, num_points_q, num_dims).permute(0, 3, 2, 1).contiguous()
        x_q = x_q.view(batch_size, num_dims, num_points_q, 1).expand(-1, -1, -1, k)
        feature = torch.cat((feature - x_q, x_q), dim=1)
        return feature

    def forward(self, coor, f, coor_q, f_q):
        """ coor, f : B 3 G ; B C G
            coor_q, f_q : B 3 N; B 3 N
        """
        # dgcnn upsample
        f_q = self.get_graph_feature(coor_q, f_q, coor, f)
        f_q = self.layer1(f_q)
        f_q = f_q.max(dim=-1, keepdim=False)[0]

        f_q = self.get_graph_feature(coor_q, f_q, coor_q, f_q)
        f_q = self.layer2(f_q)
        f_q = f_q.max(dim=-1, keepdim=False)[0]

        return f_q


class ConvBNReLU1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True, activation='relu'):
        super(ConvBNReLU1D, self).__init__()
        self.act = nn.ReLU()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
            self.act
        )

    def forward(self, x):
        return self.net(x)


class ConvBNReLURes1D(nn.Module):
    def __init__(self, channel, kernel_size=1, groups=1, res_expansion=1.0, bias=True, activation='relu'):
        super(ConvBNReLURes1D, self).__init__()
        self.act = nn.ReLU()
        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels=channel, out_channels=int(channel * res_expansion),
                      kernel_size=kernel_size, groups=groups, bias=bias),
            nn.BatchNorm1d(int(channel * res_expansion)),
            self.act
        )
        if groups > 1:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                          kernel_size=kernel_size, groups=groups, bias=bias),
                nn.BatchNorm1d(channel),
                self.act,
                nn.Conv1d(in_channels=channel, out_channels=channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel),
            )
        else:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel)
            )

    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)

class PosExtraction(nn.Module):
    def __init__(self, channels, blocks=1, groups=1, res_expansion=1, bias=True, activation='relu'):
        """
        input[b,d,g]; output[b,d,g]
        :param channels:
        :param blocks:
        """
        super(PosExtraction, self).__init__()
        operation = []
        for _ in range(blocks):
            operation.append(
                ConvBNReLURes1D(channels, groups=groups, res_expansion=res_expansion, bias=bias, activation=activation)
            )
        self.operation = nn.Sequential(*operation)

    def forward(self, x):  # [b, d, g]
        return self.operation(x)
class PointNet_FeaturePropagation(nn.Module):
    def __init__(self, in_channel, out_channel, blocks=1, groups=1, res_expansion=1.0, bias=True, activation='relu'):
        super(PointNet_FeaturePropagation, self).__init__()
        self.fuse = ConvBNReLU1D(in_channel, out_channel, 1, bias=bias)
        self.extraction = PosExtraction(out_channel, blocks, groups=groups,
                                        res_expansion=res_expansion, bias=bias, activation=activation)


    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, N, 3]
            xyz2: sampled input points position data, [B, S, 3]
            points1: input points data, [B, D', N]
            points2: input points data, [B, D'', S]
        Return:
            new_points: upsampled points data, [B, D''', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        new_points = self.fuse(new_points)
        new_points = self.extraction(new_points)
        return new_points
class APF(nn.Module):
    def __init__(self,
                 num_classes=15,
                 embed_dim=768,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 norm_layer=nn.LayerNorm,
                 act_layer=nn.GELU,
                 depth=12,
                 drop_path_rate=0.,
                 cls_dim=50,
                 de_blocks=[2, 2, 2, 2, 2],
                 res_expansion=1.0,
                 de_dims=[512, 256, 128, 128, 128],
                 en_dims=[24, 48, 96, 192, 384, 768],
                 dims=[1920, 704, 352, 176, 152, 44],
                 gmp_dim=64
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.dropout = nn.Dropout(0.1)
        self.embed_dim = self.num_features = embed_dim
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 65, embed_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.cls_dim = cls_dim
        self.label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                        nn.BatchNorm1d(64),
                                        nn.LeakyReLU(0.2))
        self.prompt_pos = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.head = nn.Sequential(
            nn.Conv1d(5888, 128, 1),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Conv1d(128, self.cls_dim, 1)
        )
        self.relu = nn.ReLU()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.reduce_dim = 512
        self.enc_norm = norm_layer(embed_dim)
        self.conv_block = nn.Conv1d(2, 1, kernel_size=1)
        self.propagation_0 = PointNetFeaturePropagation(in_channel=787,
                                                        mlp=[self.embed_dim * 4, self.embed_dim])
        self.propagation_1 = PointNetFeaturePropagation(in_channel=771,
                                                        mlp=[self.embed_dim * 4, self.embed_dim])
        self.propagation_2 = PointNetFeaturePropagation(in_channel=771,
                                                        mlp=[self.embed_dim * 4, self.embed_dim])
        self.propagation_3 = PointNetFeaturePropagation(in_channel=1152,
                                                        mlp=[self.reduce_dim * 4, self.reduce_dim])
        self.propagation_4 = PointNetFeaturePropagation(in_channel=704,
                                                        mlp=[self.reduce_dim * 4, self.reduce_dim])
        self.propagation_5 = PointNetFeaturePropagation(in_channel=608,
                                                        mlp=[self.reduce_dim * 4, self.reduce_dim])
        self.propagation_6 = PointNetFeaturePropagation(in_channel=560,
                                                        mlp=[self.reduce_dim * 4, self.reduce_dim])
        self.propagation_7 = PointNetFeaturePropagation(in_channel=536,
                                                         mlp=[self.reduce_dim * 4, self.reduce_dim])
        self.dgcnn_pro_1 = DGCNN_Propagation(k=4)
        self.dgcnn_pro_2 = DGCNN_Propagation(k=4)
        self.EncP = EncP(in_channels=7, input_points=2048, num_stages=5, embed_dim=24, k_neighbors=128, alpha=100, beta=1000,
                         LGA_block=[2, 1, 1, 1, 1], dim_expansion=[2, 2, 2, 2, 2], type='scan')
        self.decode_list = nn.ModuleList()
        en_dims.reverse()
        de_dims.insert(0, en_dims[0])
        assert len(en_dims) == len(de_dims) == len(de_blocks) + 1
        for i in range(len(en_dims) - 1):
            self.decode_list.append(
                PointNet_FeaturePropagation(dims[i], de_dims[i + 1],
                                           blocks=de_blocks[i], groups=1, res_expansion=res_expansion,
                                           bias=True, activation="relu")
            )
        self.cls_map = nn.Sequential(
            ConvBNReLU1D(16, cls_dim, bias=True, activation="relu"),
            ConvBNReLU1D(cls_dim, cls_dim, bias=True, activation="relu")
        )
        # global max pooling mapping
        self.gmp_map_list = nn.ModuleList()
        for en_dim in en_dims:
            self.gmp_map_list.append(ConvBNReLU1D(en_dim, gmp_dim, bias=True, activation="relu"))
        self.gmp_map_end = ConvBNReLU1D(gmp_dim * len(en_dims), gmp_dim, bias=True, activation="relu")

    def freeze(self):
        for param in self.parameters():
            param.requires_grad_(False)

        for name, param in self.named_parameters():
            if 'head' in name or 'enc_norm' in name or 'EncP' in name or 'prompt' in name or 'conv_block' in name\
                    or 'propagation' in name:
                param.requires_grad = True
                # print(name, param.shape)

    def forward_features(self, x, xyz, cls_label):
        B = x.shape[0]
        N = x.shape[2]
        xyz_list, x_list, x, prompt = self.EncP(xyz, x)
        l4_feature = self.propagation_3(xyz_list[4], xyz_list[5], x_list[4], x_list[5])
        l3_feature = self.propagation_4(xyz_list[3], xyz_list[4], x_list[3], l4_feature)
        l2_feature = self.propagation_5(xyz_list[2], xyz_list[3], x_list[2], l3_feature)
        l1_feature = self.propagation_6(xyz_list[1], xyz_list[2], x_list[1], l2_feature)
        l0_feature = self.propagation_7(xyz_list[0], xyz_list[1], x_list[0], l1_feature)
        G = x.shape[1]
        x_mean = x.mean(dim=1, keepdim=True)
        x_hat = torch.stack((x, x - x_mean), dim=2)
        [bs, num_token, dim_e] = x.shape
        x_hat = x_hat.reshape(bs * num_token, -1, dim_e)
        x_hat = self.conv_block(x_hat)
        x_hat = x_hat.reshape(bs, num_token, -1)
        x = x + x_hat
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.pos_embed
        feature_list = []
        for idx, blk in enumerate(self.blocks):
            x = torch.cat((x[:, :idx + 1, :], prompt, x[:, -G:, :]), dim=1)
            x = blk(x)
            prompt = (x.max(1)[0] + x.mean(1)).unsqueeze(1)
            if idx in [3, 7, 11]:
                feature_list.append(x)
        x = torch.cat((x[:, :12, :], prompt, x[:, -G:, :]), dim=1)
        feature_list = [self.enc_norm(x).transpose(-1, -2).contiguous() for x in feature_list]
        prompt_feature = torch.cat((feature_list[0][:, :, :1], feature_list[1][:, :, :1], feature_list[2][:, :, :1]),
                                   dim=1).repeat(1, 1, N)
        prompt_feature_2 = torch.cat(
            (feature_list[0][:, :, 1:2], feature_list[1][:, :, 1:2], feature_list[2][:, :, 1:2]), dim=1).repeat(1, 1, N)
        cls_label_one_hot = cls_label.view(B, 16, 1).repeat(1, 1, N)
        center_level_0 = xyz.transpose(-1, -2).contiguous()
        f_level_0 = torch.cat([cls_label_one_hot, center_level_0], 1)
        # f_level_0 = center_level_0
        center_level_1 = xyz_list[2].contiguous()
        f_level_1 = center_level_1
        center_level_2 = xyz_list[3].contiguous()
        f_level_2 = center_level_2
        center_level_3 = xyz_list[-1].contiguous()
        f_level_3 = feature_list[2][:, :, -G:]
        # f_level_0 = torch.cat((f_level_0, l0_feature), dim=1)
        f_level_2 = self.propagation_2(center_level_2, center_level_3, f_level_2, feature_list[1][:, :, -G:])
        # (16, 3, 256) (16, 3, 64) (16, 3, 256) (16, 768, 64) ==> (16, 768, 256)
        f_level_1 = self.propagation_1(center_level_1, center_level_3, f_level_1, feature_list[0][:, :, -G:])
        # (16, 3, 512) (16, 3, 64) (16, 3, 512) (16, 768, 64) ==> (16, 768, 512)
        # bottom up
        f_level_2 = self.dgcnn_pro_2(center_level_3, f_level_3, center_level_2, f_level_2)
        # (16, 3, 64) (16, 3, 64) (16, 3, 256) (16, 768, 256) ==> (16, 768, 256)
        f_level_1 = self.dgcnn_pro_1(center_level_2, f_level_2, center_level_1, f_level_1)
        # (16, 3, 256) (16, 768, 256) (16, 3, 512) (16, 768, 512) ==> (16, 768, 512)
        f_level_0 = self.propagation_0(center_level_0, center_level_1, f_level_0, f_level_1)
        # (16, 3, 2048) (16, 3, 512) (16, 19, 2048) (16, 768, 512) ==> (16, 768, 2048)
        x = torch.cat((l0_feature, f_level_0, prompt_feature, prompt_feature_2), dim=1)
        x = self.head(x)
        # x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x

    def forward(self, x, xyz, cls_label):
        x = self.forward_features(x, xyz, cls_label)
        return x

def create_point_former():
    vit = timm. create_model("vit_base_patch16_224_in21k", pretrained=True)
    checkpoint = torch.load(r'/root/wish/Point-NN-main/ViT-L-14.pt')
    checkpoint = checkpoint.state_dict()
    # for k, v in checkpoint.items():
    #     print(k, v.shape)
    clip_text = {}
    for k, v in checkpoint.items():
        if 'visual' not in k:
            new_k = k.replace("transformer.resblocks", "blocks")
            new_k = new_k.replace("ln_", "norm")
            new_k = new_k.replace("c_fc", "fc1")
            new_k = new_k.replace("c_proj", "fc2")
            new_k = new_k.replace("proj_", "qkv.")
            new_k = new_k.replace("in_", "")
            new_k = new_k.replace("out_", "")
            new_k = new_k.replace("layer_", "")
            clip_text[new_k] = v
    base_state_dict = vit.state_dict()
    block4 = {}
    block7 = {}
    for k, v in base_state_dict.items():
        if 'blocks.7' in k:
            new_k = k.replace("blocks.7", "blocks7")
            block7[new_k] = v
        if 'blocks.4' in k:
            new_k = k.replace("blocks.4", "blocks4")
            block4[new_k] = v
    # for k, v in block4.items():
    #     print(k, v.shape)
    merge_dict = {**base_state_dict, **block7, **block4}
    # for k, v in merge_dict.items():
    #     print(k, v.shape)
    # print("===========================================================================================")
    del base_state_dict['pos_embed']
    del base_state_dict['head.weight']
    del base_state_dict['head.bias']
    del merge_dict['pos_embed']
    del merge_dict['head.weight']
    del merge_dict['head.bias']
    model = APF()
    # for k, v in model.state_dict().items():
    #     print(k, v.shape)
    model.load_state_dict(base_state_dict, False)

    model.freeze()
    return model


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()

    return new_y


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss

if __name__ == '__main__':
    xyz = torch.randn(2, 2048, 3).cuda()
    x = torch.randn(2, 7, 2048).cuda()
    label = torch.randint(0, 16, (2, 1)).to('cuda')
    # print(label.shape)
    model = create_point_former().to('cuda')
    for name, param in model.named_parameters():
        print(name)
    print("======= hot ========")
    for name, param in model.named_parameters():
        if param.requires_grad is True:
            print(f"{name} : {param.shape}")
    seg_pred = model(x, xyz, to_categorical(label, 16))
    print(to_categorical(label, 16).shape)
    print(seg_pred.shape)


