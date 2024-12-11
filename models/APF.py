import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import timm
from models.point_pn import Point_PN_scan, Point_PN_mn40
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
                 drop_path_rate=0.,):
        super().__init__()
        self.encoder_large = Point_PN_scan()
        self.num_classes = num_classes
        self.dropout = nn.Dropout(0.1)
        self.embed_dim = self.num_features = embed_dim
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 65, embed_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.head = nn.Sequential(
            nn.Linear(self.embed_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(256, self.num_classes)
        )
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.enc_norm = norm_layer(self.embed_dim)
        self.conv_block = nn.Conv1d(2, 1, kernel_size=1)
    def freeze(self):
        for param in self.parameters():
            param.requires_grad_(False)

        for name, param in self.named_parameters():
            if 'head' in name or 'enc_norm' in name or 'encoder' in name or 'conv_block' in name\
                    or 'pos_embed' in name:
                param.requires_grad = True
                print(name, param.shape)

    def forward_features(self, x, xyz, extra):
        B = x.shape[0]
        x, prompt = self.encoder_large(x, xyz)
        new_x = x.max(1)[0] + x.mean(1)
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
        for idx, blk in enumerate(self.blocks):
            x = torch.cat((x[:, :idx + 1, :], prompt, x[:, -G:, :]), dim=1)
            x = blk(x)
        x = torch.cat((x[:, :12, :], prompt, x[:, -G:, :]), dim=1)
        x = self.enc_norm(x)
        x = x.mean(dim=1)[0] + x.max(dim=1)[0]
        return x, new_x

    def forward(self, x, xyz, extra=True):
        x, new_x = self.forward_features(x, xyz, extra)
        x = self.dropout(x)
        x = self.head(x)
        return x

def create_point_former(num_classes):
    vit = timm. create_model("vit_base_patch16_224_in21k", pretrained=True)
    checkpoint = torch.load(r'/root/wish/Point-NN-main/ViT-L-14.pt')
    # checkpoint = torch.load(r'/root/wish/Point-NN-main/pytorch_model_deit.bin')
    checkpoint = checkpoint.state_dict()
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
    del base_state_dict['pos_embed']
    del base_state_dict['head.weight']
    del base_state_dict['head.bias']
    model = APF(num_classes=num_classes)
    model.load_state_dict(base_state_dict, False)
    model.freeze()
    return model

if __name__ == '__main__':

    x = torch.rand(32, 1024, 4).to('cuda')
    model = create_point_former(num_classes=15).to('cuda')
    x = model(x.permute(0, 2, 1), x[:, :, :3], extra=True)
    tuning_num_params = 0
    num_params = 0
    a = 0
    for p in model.parameters():
        num_params += p.numel()
        if p.requires_grad:
            tuning_num_params += p.numel()
        else:
            a += p.numel()
    print("===============================================")
    print("model parameters: " + str(num_params))
    print("model tuning parameters: " + str(tuning_num_params))
    print("model not tuning parameters: " + str(a))
    print("tuning rate: " + str(tuning_num_params / num_params))
    print("===============================================")
    for name, param in model.named_parameters():
        if param.requires_grad is not True:
            print(f"{name},{param.shape}")
    print(x.shape)



