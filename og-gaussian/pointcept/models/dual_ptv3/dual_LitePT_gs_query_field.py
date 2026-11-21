"""
Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""


import torch, torch.nn.functional as F, torch.nn as nn
from timm.layers import DropPath
from types import SimpleNamespace
from itertools import accumulate

try:
    import flash_attn
except ImportError:
    flash_attn = None

from .LitePT import LitePT, MLP
from .cgr import QueryField_log, QueryField_log_normal
from pointcept.models.builder import MODELS
from pointcept.models.utils.structure import Point
from pointcept.models.modules import PointModule, PointSequential
from pointcept.models.utils.diffusion import timestep_embedding
import copy


def quat_to_ortho6d(quaternions):
    quaternions = F.normalize(quaternions, p=2, dim=-1, eps=1e-8)
    
    w, x, y, z = quaternions.unbind(dim=-1)
    
    two_xy = 2 * x * y
    two_xz = 2 * x * z
    two_xw = 2 * x * w
    
    two_yz = 2 * y * z
    two_yw = 2 * y * w
    
    two_zw = 2 * z * w
    
    two_xx = 2 * x * x
    two_yy = 2 * y * y
    two_zz = 2 * z * z
    
    r1_x = 1 - two_yy - two_zz
    r1_y = two_xy + two_zw
    r1_z = two_xz - two_yw
    
    r2_x = two_xy - two_zw
    r2_y = 1 - two_xx - two_zz
    r2_z = two_yz + two_xw

    r6d = torch.stack([r1_x, r1_y, r1_z, r2_x, r2_y, r2_z], dim=-1)
    
    return r6d




class PointNetHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels//2),
            nn.LayerNorm(in_channels//2),
            nn.ReLU(inplace=True),

            # nn.Linear(in_channels//2, in_channels//4),
            # nn.LayerNorm(in_channels//4),
            # nn.ReLU(inplace=True),

            nn.Linear(in_channels//2, out_channels)
        )

    def forward(self, x):  # x: (N, C)
        return self.mlp(x)


class RelationQueryAdapter(nn.Module):
    def __init__(
        self,
        channels: int,
        rel_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.0,
        alpha_init: float = 0.0,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(rel_dim),
            nn.Linear(rel_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, channels * 3),
        )
        self.alpha_delta = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
        self.alpha_scale = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
        self.alpha_shift = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))

    def forward(self, q: torch.Tensor, rel_feat: torch.Tensor) -> torch.Tensor:
        if rel_feat is None:
            return q
        if rel_feat.shape[0] != q.shape[0]:
            raise ValueError(
                f"RelationQueryAdapter expects rel_feat.shape[0] == q.shape[0], "
                f"but got {rel_feat.shape[0]} vs {q.shape[0]}."
            )

        cond = self.net(rel_feat.to(q.dtype))
        delta, scale, shift = cond.chunk(3, dim=-1)

        delta = torch.tanh(delta)
        scale = torch.tanh(scale)
        shift = torch.tanh(shift)

        q = q * (1.0 + torch.tanh(self.alpha_scale) * scale)
        q = q + torch.tanh(self.alpha_shift) * shift
        q = q + torch.tanh(self.alpha_delta) * delta
        return q


class CrossFlashAttention(nn.Module):
    def __init__(
        self,
        channels,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        rel_dim=None,
        rel_hidden_dim=128,
        rel_dropout=0.0,
        rel_alpha_init=0.0,
        use_rel_out=True,
    ):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        self.attn_drop = attn_drop

        self.q_proj = nn.Linear(channels, channels, bias=qkv_bias)
        self.k_proj = nn.Linear(channels, channels, bias=qkv_bias)
        self.v_proj = nn.Linear(channels, channels, bias=qkv_bias)
        self.out_proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rel_adapter = None
        self.rel_out = None
        self.rel_out_alpha = None
        if rel_dim is not None and rel_dim > 0:
            self.rel_adapter = RelationQueryAdapter(
                channels=channels,
                rel_dim=rel_dim,
                hidden_dim=rel_hidden_dim,
                dropout=rel_dropout,
                alpha_init=rel_alpha_init,
            )
            if use_rel_out:
                self.rel_out = nn.Sequential(
                    nn.LayerNorm(rel_dim),
                    nn.Linear(rel_dim, rel_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(rel_dropout),
                    nn.Linear(rel_hidden_dim, channels),
                )
                self.rel_out_alpha = nn.Parameter(
                    torch.tensor(rel_alpha_init, dtype=torch.float32)
                )

    def forward(self, pcd: SimpleNamespace):
        H, C = self.num_heads, self.channels

        q_in = pcd.q
        rel_feat = getattr(pcd, "rel_feat", None)
        if self.rel_adapter is not None and rel_feat is not None:
            q_in = self.rel_adapter(q_in, rel_feat)

        q_proj = self.q_proj(q_in).reshape(-1, H, C // H)     # (Nq, H, Dh)
        k_proj = self.k_proj(pcd.kv).reshape(-1, H, C // H)   # (Nk, H, Dh)
        v_proj = self.v_proj(pcd.kv).reshape(-1, H, C // H)   # (Nk, H, Dh)

        q_seq_lens = pcd.q_offset[1:] - pcd.q_offset[:-1]
        max_seqlen_q = q_seq_lens.max().item()
        kv_seq_lens = pcd.kv_offset[1:] - pcd.kv_offset[:-1]
        max_seqlen_kv = kv_seq_lens.max().item()

        kv_proj = torch.stack([k_proj, v_proj], dim=1)  # (Nk, 2, H, Dh)
        out = flash_attn.flash_attn_varlen_kvpacked_func(
            q=q_proj.half(),
            kv=kv_proj.half(),
            cu_seqlens_q=pcd.q_offset,
            cu_seqlens_k=pcd.kv_offset,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_kv,
            dropout_p=self.attn_drop if self.training else 0.0,
            softmax_scale=self.scale,
        ).reshape(-1, C)
        out = out.to(q_proj.dtype)

        out = self.out_proj(out)
        if self.rel_out is not None and rel_feat is not None:
            out = out + torch.tanh(self.rel_out_alpha) * self.rel_out(rel_feat.to(out.dtype))
        out = self.proj_drop(out)

        pcd.q = out
        return pcd


class CTDecoderBlockRelQ(PointModule):
    def __init__(
        self,
        channels,
        attn: CrossFlashAttention,
        mlp_ratio=4.0,
        proj_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
    ):
        super().__init__()
        self.channels = channels
        self.norm1 = norm_layer(channels)
        self.attn = attn
        self.norm2 = norm_layer(channels)
        self.mlp = MLP(
            in_channels=channels,
            hidden_channels=int(channels * mlp_ratio),
            out_channels=channels,
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, pcd):
        shortcut = pcd.q
        pcd.q = self.norm1(pcd.q)
        attn_out = self.attn(pcd).q
        pcd.q = shortcut + self.drop_path(attn_out)
        pcd.q = pcd.q + self.drop_path(self.mlp(self.norm2(pcd.q)))
        return pcd


class BackboneTransformerSedOed_QueryField(PointModule):
    def __init__(
        self,
        in_skill_channels,
        use_skill=True,
        out_channels=3,
        width=512,
        depths=12,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        rel_dim=4,
        rel_hidden_dim=64,
        rel_depth=2,
        rel_alpha_init=0.0,
        rel_use_out=True,
    ):
        super().__init__()
        self.width = width
        self.use_skill = use_skill
        self.rel_depth = rel_depth

        self.coord_mlp_scene = nn.Sequential(
            nn.Linear(3, width),
            nn.ReLU(),
            nn.Linear(width, width),
        )
        self.rot_mlp_scene = nn.Sequential(
            nn.Linear(6, width),
            nn.ReLU(),
            nn.Linear(width, width),
        )

        self.coord_mlp_obj = nn.Sequential(
            nn.Linear(3, width),
            nn.ReLU(),
            nn.Linear(width, width),
        )
        self.rot_mlp_obj = nn.Sequential(
            nn.Linear(6, width),
            nn.ReLU(),
            nn.Linear(width, width),
        )

        self.coord_mlp_noisy = nn.Sequential(
            nn.Linear(3, width),
            nn.ReLU(),
            nn.Linear(width, width),
        )
        self.rot_mlp_noisy = nn.Sequential(
            nn.Linear(6, width),
            nn.ReLU(),
            nn.Linear(width, width),
        )

        self.input_fusion = nn.Linear(width * 2, width)

        self.skill_proj = nn.Linear(in_skill_channels, width)
        self.time_embed = MLP(
            in_channels=width,
            hidden_channels=int(width * mlp_ratio),
            out_channels=width,
            drop=proj_drop,
        )

        self.ln_kv = norm_layer(width)
        self.ln_q = norm_layer(width)
        self.ln_post = norm_layer(width)

        block_drop_path = [x.item() for x in torch.linspace(0, drop_path, depths)]
        self.kv_prefix_len = 2 if use_skill else 1

        self.blocks = PointSequential()
        for d in range(depths):
            use_relation = d < rel_depth
            attn = CrossFlashAttention(
                channels=width,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                rel_dim=rel_dim if use_relation else None,
                rel_hidden_dim=rel_hidden_dim,
                rel_dropout=proj_drop,
                rel_alpha_init=rel_alpha_init,
                use_rel_out=rel_use_out,
            )

            self.blocks.add(
                CTDecoderBlockRelQ(
                    channels=width,
                    attn=attn,
                    mlp_ratio=mlp_ratio,
                    proj_drop=proj_drop,
                    drop_path=block_drop_path[d],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                ),
                name=f"block{d}",
            )

        self.output_proj = PointNetHead(
            in_channels=width,
            out_channels=out_channels,
        )

    def forward(self, sce_point: Point, obj_point: Point, skill_emb, t, feat):
        B = obj_point.offset.shape[0]
        device = sce_point.offset.device

        if self.use_skill:
            x_skill = self.skill_proj(skill_emb.reshape(B, -1))

        t_embed = self.time_embed(timestep_embedding(t, self.width))

        sce_coord_raw = sce_point.coord
        sce_coord_emb = self.coord_mlp_scene(sce_coord_raw)
        sce_rot_emb = self.rot_mlp_scene(quat_to_ortho6d(sce_point.gs_raw[:, 3:7]))
        sce_gs_emb = self.input_fusion(torch.cat([sce_coord_emb, sce_rot_emb], dim=-1))
        x_sce_pcd = sce_point.feat + sce_gs_emb

        obj_coord_raw = obj_point.gs_raw[:, :3]
        obj_coord_emb = self.coord_mlp_obj(obj_coord_raw)
        obj_rot_emb = self.rot_mlp_obj(quat_to_ortho6d(obj_point.gs_raw[:, 3:7]))
        obj_emb = self.input_fusion(torch.cat([obj_coord_emb, obj_rot_emb], dim=-1))

        noisy_coord_emb = self.coord_mlp_noisy(obj_point.goal[:, :3])
        noisy_rot_emb = self.rot_mlp_noisy(obj_point.goal[:, 3:9])
        noisy_emb = self.input_fusion(torch.cat([noisy_coord_emb, noisy_rot_emb], dim=-1))

        x_obj_pcd = obj_point.feat + obj_emb + noisy_emb

        zero = torch.tensor([0], device=device, dtype=sce_point.offset.dtype)
        sce_point_offset = torch.cat((zero, sce_point.offset), dim=0)
        obj_point_offset = torch.cat((zero, obj_point.offset), dim=0)

        kvs = []
        qs = []
        kv_lens = []
        q_lens = []

        for b in range(B):
            t_tok = t_embed[b:b + 1]
            sce_tok = x_sce_pcd[sce_point_offset[b]:sce_point_offset[b + 1]]

            if self.use_skill:
                skill_tok = x_skill[b:b + 1]
                kv = torch.cat([t_tok, skill_tok, sce_tok], dim=0)
            else:
                kv = torch.cat([t_tok, sce_tok], dim=0)

            q = x_obj_pcd[obj_point_offset[b]:obj_point_offset[b + 1]]

            kvs.append(kv)
            qs.append(q)
            kv_lens.append(kv.shape[0])
            q_lens.append(q.shape[0])

        kvs = torch.cat(kvs, dim=0)
        qs = torch.cat(qs, dim=0)

        kv_offset = torch.tensor([0] + list(accumulate(kv_lens)), device=device, dtype=torch.int32)
        q_offset = torch.tensor([0] + list(accumulate(q_lens)), device=device, dtype=torch.int32)

        kvs = self.ln_kv(kvs)
        qs = self.ln_q(qs)

        if feat is not None:
            if feat.shape[0] != qs.shape[0]:
                raise ValueError(
                    f"QueryField feature count must equal number of object queries. "
                    f"Got feat.shape[0]={feat.shape[0]} vs q.shape[0]={qs.shape[0]}."
                )
            feat = feat.to(qs.dtype)

        pcd = SimpleNamespace(
            kv=kvs,
            q=qs,
            kv_offset=kv_offset,
            q_offset=q_offset,
            kv_prefix_len=self.kv_prefix_len,
            obj_offset=obj_point.offset,
            sce_offset=sce_point.offset,
            rel_feat=feat,
        )

        q = self.blocks(pcd).q
        q = self.ln_post(q)
        q = self.output_proj(q)
        return q


@MODELS.register_module("SedOed_LitePT_QueryField")
class SedOed_LitePT_QueryField(PointModule):
    def __init__(
        self,
        scene_encoder_params: dict,
        object_net_params: dict,
        backbone_params: dict,
        query_field_params: dict,
        qf_normal=False
    ):
        super().__init__()
        self.scene_net = LitePT(**scene_encoder_params)
        self.object_net = LitePT(**object_net_params)
        if qf_normal:
            self.relation_extractor = QueryField_log_normal(**query_field_params)
        else:
            self.relation_extractor = QueryField_log(**query_field_params)

        backbone_params = copy.deepcopy(backbone_params)
        backbone_rel_dim = backbone_params.get("rel_dim", self.relation_extractor.out_dim)
        if backbone_rel_dim != self.relation_extractor.out_dim:
            print(
                f"[SedOed_LitePT_QueryField] Override backbone rel_dim from "
                f"{backbone_rel_dim} to {self.relation_extractor.out_dim}."
            )
        backbone_params["rel_dim"] = self.relation_extractor.out_dim
        self.backbone = BackboneTransformerSedOed_QueryField(**backbone_params)

        sce_n_parameters = sum(p.numel() for p in self.scene_net.parameters() if p.requires_grad)
        obj_n_parameters = sum(p.numel() for p in self.object_net.parameters() if p.requires_grad)
        bac_n_parameters = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        print(f"Num params (scene_net): {sce_n_parameters}")
        print(f"Num params (object_net): {obj_n_parameters}")
        print(f"Num params (backbone): {bac_n_parameters}")

    def forward(
        self,
        sce_point: Point,
        obj_point: Point,
        skill_emb,
        t,
        sce_point_emb=None,
        obj_point_emb=None,
    ):
        if not self.training and sce_point_emb is not None:
            sce_point_ = sce_point_emb
        else:
            sce_point_ = copy.deepcopy(sce_point)
            sce_point_ = self.scene_net(sce_point_)

        if not self.training and obj_point_emb is not None:
            obj_point_ = obj_point_emb
            obj_point_.goal = obj_point.goal
        else:
            obj_point_ = copy.deepcopy(obj_point)
            obj_point_ = self.object_net(obj_point_)

        query_xyz = (
            obj_point.goal[:, :3]
            if hasattr(obj_point, "goal") and obj_point.goal is not None
            else obj_point_.gs_raw[:, :3]
        )

        feat = self.relation_extractor(
            obj_point_.gs_raw,
            sce_point_.gs_raw,
            obj_point_.offset,
            sce_point_.offset,
            query_xyz=query_xyz,
        )
        

        pred = self.backbone(sce_point_, obj_point_, skill_emb, t, feat)
        result = dict(pred=pred, sce_point_emb=sce_point_, obj_point_emb=obj_point_)
        return result


if __name__ == "__main__":
    pt1 = LitePT(
        in_channels=6,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2),
        enc_depths=(2, 2, 2),
        enc_channels=(256, 256, 384),
        enc_num_head=(4, 4, 4),
        enc_patch_size=(1024, 1024, 1024),
    )
    print(sum(p.numel() for p in pt1.parameters() if p.requires_grad))
