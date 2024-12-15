from typing import Sequence, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm
from models.SwinUNETR import SwinUNETR
from models.TextAttend import TextAttend
from models.OSTR import QueryRefine
class ZePT(nn.Module):
    def __init__(self, img_size, in_channels, out_channels, backbone = 'swinunetr'):
        # encoding: rand_embedding or word_embedding
        super().__init__()
        self.backbone_name = backbone
        if backbone == 'swinunetr':
            self.backbone = SwinUNETR(img_size=img_size,
                        in_channels=in_channels,
                        out_channels=out_channels,
                        feature_size=48,
                        drop_rate=0.0,
                        attn_drop_rate=0.0,
                        dropout_path_rate=0.0,
                        use_checkpoint=False,
                        )
        else:
            raise Exception('{} backbone is not implemented in curretn version'.format(backbone))

        self.organ_numbers = 25
        self.tumor_numbers = 7

        self.organ_embedding = nn.Embedding(self.organ_numbers, 768)
        self.tumor_embedding = nn.Embedding(self.tumor_numbers, 768)
        
        
        
        in_dims = [768, 768, 384, 192]
        dims = [768, 384, 192, 96]
        TextAttendLayers = []
        for i in range(len(dims)):
            layer = TextAttend(dim=dims[i], 
                    out_dim=dims[i], 
                    num_heads=8, 
                    norm_layer=nn.LayerNorm, 
                    in_features=in_dims[i], 
                    mlp_ratio=4, 
                    hard=True, 
                    gumbel=True, 
                    sum_assign=False, 
                    assign_eps=1., 
                    gumbel_tau=1.)
            TextAttendLayers.append(layer)
        self.text_attend_layers = nn.ModuleList(TextAttendLayers)
        
        
        self.whole_controller = nn.Linear(96, 48)
        self.whole_organ_out_norm_layer = nn.LayerNorm(48)
        
        
        self.controller = nn.Linear(768, 48)
        self.out_norm_layer = nn.LayerNorm(48)
        
        proj_controllers = []
        decoder_norm_layers = []
        for i in range(len(dims)):
             proj_controllers.append(nn.Linear(768, dims[i]))
             decoder_norm_layers.append(nn.LayerNorm(dims[i]))
             
        self.proj_controllers = nn.ModuleList(proj_controllers)
        self.decoder_norm_layers = nn.ModuleList(decoder_norm_layers)
        
        self.final_proj_controller = nn.Linear(96, 48)
        self.final_decoder_norm_layer = nn.LayerNorm(48)
        
        self.num_heads = 8
        self.queryrefine = QueryRefine(in_dims=dims, hidden_dim=768, nheads=8, dec_layers=len(dims), pre_norm=False)
        
        self.cl = nn.Linear(768, 768)
        self.cl_norm_layer = nn.LayerNorm(768)
        
        self.temp = nn.Parameter(0.07*torch.ones([]))
    def load_params(self, model_dict):
        if self.backbone_name == 'swinunetr':
            store_dict = self.backbone.state_dict()
            for key in model_dict.keys():
                if 'out' not in key:
                    store_dict[key] = model_dict[key]

            self.backbone.load_state_dict(store_dict)
            print('Use pretrained weights')
        else:
            raise Exception('{} backbone is not implemented in curretn version'.format(self.backbone_name))

    
    # def min_max_normalize_batch(self, anomaly_scores):
    #     min_val = anomaly_scores.min(dim=4, keepdim=True)[0].min(dim=3, keepdim=True)[0].min(dim=2, keepdim=True)[0]
    #     max_val = anomaly_scores.max(dim=4, keepdim=True)[0].max(dim=3, keepdim=True)[0].max(dim=2, keepdim=True)[0]

    #     # Apply min-max normalization
    #     normalized_scores = (anomaly_scores - min_val) / (max_val - min_val)
    #     return normalized_scores
    
    # def forward_prediction_heads(self, organ_emb, decoder_out):
    #     N = organ_emb.shape[1]
    #     anomaly_score = torch.einsum("bnc,bcdhw->bncdhw", organ_emb, decoder_out)
    #     anomaly_map = -(anomaly_score.max(2)[0])
    #     norm_anomaly_map =  self.min_max_normalize_batch(anomaly_map)
    #     map_for_mask = norm_anomaly_map.detach().flatten(2, 4)
    #     bool_masks = []
    #     for i in range(N):
    #         each_map = map_for_mask[:, i, :]
    #         bool_map = (each_map > 0.5).bool()
    #         bool_masks.append(bool_map)
    #     bool_masks = torch.stack(bool_masks, dim=1)
    #     anomaly_mask = torch.any(bool_masks, dim=1)
    #     attn_mask = ~anomaly_mask.bool()

    #     out_mask = attn_mask.unsqueeze(1).unsqueeze(1).repeat(1, self.num_heads, self.tumor_numbers, 1).flatten(0, 1)
        
    #     assert len(out_mask.shape) == 3
    #     assert out_mask.dtype == torch.bool, "attn_mask should be of type torch.bool"
        
    #     return anomaly_map, out_mask

    # def forward(self, x_in):
    #     B = x_in.shape[0]
    #     dec4, out, feats, decoder_outs = self.backbone(x_in)
        
    #     organ_embedding = self.organ_embedding.weight
    #     organ_enconding = organ_embedding.unsqueeze(0).repeat(B, 1, 1)
        
    #     tumor_embedding = self.tumor_embedding.weight
    #     tumor_enconding = tumor_embedding.unsqueeze(0).repeat(B, 1, 1)        

    #     organ_encondings = []
    #     for i in range(len(self.text_attend_layers)):
    #         organ_enconding, _ = self.text_attend_layers[i](feats[i], organ_enconding)
    #         organ_encondings.append(organ_enconding)

    #     whole_weight = self.out_norm_layer(self.whole_controller(organ_encondings[-1]))
        
    #     B, C, D, H, W = out.size()
    #     whole_organ_logits = out.flatten(start_dim=2, end_dim=4).transpose(1, 2) @ whole_weight.transpose(1, 2)
    #     whole_organ_logits_out = whole_organ_logits.transpose(1, 2).reshape(B, self.organ_numbers, D, H, W)
    #     attn_masks = []
    #     for i in range(len(organ_encondings)):
    #         weight = organ_encondings[i]#self.proj_controllers[i](organ_enconding)
    #         _, attn_mask = self.forward_prediction_heads(weight, decoder_outs[i])
    #         attn_masks.append(attn_mask)
            
    #     final_weight = self.final_proj_controller(organ_enconding)
    #     final_weight = self.final_decoder_norm_layer(final_weight)
    #     anomaly_map, _ = self.forward_prediction_heads(final_weight, out)
        
    #     tumor_anomaly_map = torch.stack([
    #         anomaly_map[:, 1, :, :, :]+anomaly_map[:, 2, :, :, :],
    #         anomaly_map[:, 5, :, :, :],
    #         anomaly_map[:, 10, :, :, :],
    #         anomaly_map[:, 14, :, :, :],
    #         anomaly_map[:, 15, :, :, :]+anomaly_map[:, 16, :, :, :],
    #         anomaly_map[:, 17, :, :, :],
    #         anomaly_map[:, 1, :, :, :]+anomaly_map[:, 2, :, :, :],
    #     ], dim=1)    
        
    #     organ_logits_out = whole_organ_logits_out
    #     organ_logits_out[:,1,:,:,:] -= tumor_anomaly_map[:,0,:,:,:]
    #     organ_logits_out[:,2,:,:,:] -= tumor_anomaly_map[:,0,:,:,:]
        
    #     organ_logits_out[:,5,:,:,:] -= tumor_anomaly_map[:,1,:,:,:]
    #     organ_logits_out[:,10,:,:,:] -= tumor_anomaly_map[:,2,:,:,:]
    #     organ_logits_out[:,14,:,:,:] -= tumor_anomaly_map[:,3,:,:,:]
        
    #     organ_logits_out[:,15,:,:,:] -= tumor_anomaly_map[:,4,:,:,:]
    #     organ_logits_out[:,16,:,:,:] -= tumor_anomaly_map[:,4,:,:,:]
        
    #     organ_logits_out[:,17,:,:,:] -= tumor_anomaly_map[:,5,:,:,:]
        
    #     organ_logits_out[:,1,:,:,:] -= tumor_anomaly_map[:,6,:,:,:]
    #     organ_logits_out[:,2,:,:,:] -= tumor_anomaly_map[:,6,:,:,:]
        
    #     all_out = self.queryrefine(tumor_enconding, feats, attn_masks, organ_encondings[0])
        
    #     tumor_encodings = all_out[:, self.organ_numbers:]
    #     tumor_weight = self.out_norm_layer(self.controller(tumor_encodings))

        
    #     N = all_out.shape[1]
        
    #     tumor_logits = out.flatten(start_dim=2, end_dim=4).transpose(1, 2) @ tumor_weight.transpose(1, 2)
    #     tumor_logits_out = tumor_logits.transpose(1, 2).reshape(B, self.tumor_numbers, D, H, W)
    #     tumor_logits_out += tumor_anomaly_map
    #     # tumor_logits_out = tumor_anomaly_map
        
        
    #     logits_out = torch.cat([organ_logits_out, tumor_logits_out], dim=1)
        
    #     return logits_out

    def create_targets(self, N):
        sim_targets = torch.zeros(N, N).float()
        sim_targets.fill_diagonal_(1)
        
        sim_targets_q = sim_targets
        sim_targets_q[self.organ_numbers:, self.organ_numbers:] = 1.0
        sim_targets_q[self.organ_numbers:, self.organ_numbers:] = 1.0
        sim_targets_q[1, 25] = 1.0
        
        map_pair = [[1,25], [2,25], [5,26], [10,27], [14,28], [14,26], [15,29], [16,29], [17,30], [1,31], [2,31], [25, 31]]
        
        for pair in map_pair:
            sim_targets_q[pair[0], pair[1]] = 1.0
            sim_targets_q[pair[1], pair[0]] = 1.0
        
        return sim_targets, sim_targets_q
        
    def min_max_normalize_batch(self, anomaly_scores):
        min_val = anomaly_scores.min(dim=3, keepdim=True)[0].min(dim=2, keepdim=True)[0].min(dim=1, keepdim=True)[0]
        max_val = anomaly_scores.max(dim=3, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]

        # Apply min-max normalization
        normalized_scores = (anomaly_scores - min_val) / (max_val - min_val)
        return normalized_scores
    
    def forward_prediction_heads(self, organ_emb, decoder_out):
        N = organ_emb.shape[1]
        anomaly_score = torch.einsum("bnc,bcdhw->bndhw", organ_emb, decoder_out)    
        anomaly_map = -(anomaly_score.max(1)[0])
        return_anomaly_map = anomaly_map.unsqueeze(1).repeat(1, self.tumor_numbers, 1, 1, 1)
        norm_anomaly_map =  self.min_max_normalize_batch(anomaly_map)
        map_for_mask = norm_anomaly_map.detach().flatten(1, 3)
        
        attn_mask = (map_for_mask < 0.5).bool()
        
        out_mask = attn_mask.unsqueeze(1).unsqueeze(1).repeat(1, self.num_heads, self.tumor_numbers, 1).flatten(0, 1)
        
        assert len(out_mask.shape) == 3
        assert out_mask.dtype == torch.bool, "attn_mask should be of type torch.bool"
        
        return return_anomaly_map, out_mask
    
    def forward(self, x_in):
        B = x_in.shape[0]
        dec4, out, feats, decoder_outs = self.backbone(x_in)
        
        organ_embedding = self.organ_embedding.weight
        organ_enconding = organ_embedding.unsqueeze(0).repeat(B, 1, 1)
        
        tumor_embedding = self.tumor_embedding.weight
        tumor_enconding = tumor_embedding.unsqueeze(0).repeat(B, 1, 1)        

        organ_encondings = []
        for i in range(len(self.text_attend_layers)):
            organ_enconding, _ = self.text_attend_layers[i](feats[i], organ_enconding)
            organ_encondings.append(organ_enconding)

        whole_weight = self.out_norm_layer(self.whole_controller(organ_encondings[-1]))
        
        B, C, D, H, W = out.size()
        whole_organ_logits = out.flatten(start_dim=2, end_dim=4).transpose(1, 2) @ whole_weight.transpose(1, 2)
        whole_organ_logits_out = whole_organ_logits.transpose(1, 2).reshape(B, self.organ_numbers, D, H, W)
        attn_masks = []
        for i in range(len(organ_encondings)):
            weight = organ_encondings[i]#self.proj_controllers[i](organ_enconding)
            #_, attn_mask = self.forward_prediction_heads(decoder_outs[i])
            _, attn_mask = self.forward_prediction_heads(weight, decoder_outs[i])
            attn_masks.append(attn_mask)
        final_weight = self.final_proj_controller(organ_enconding)
        final_weight = self.final_decoder_norm_layer(final_weight)
    #     anomaly_map, _ = self.forward_prediction_heads(final_weight, out)
        tumor_anomaly_map, _ = self.forward_prediction_heads(final_weight, out)
        
        organ_logits_out = whole_organ_logits_out
        organ_logits_out[:,1,:,:,:] -= tumor_anomaly_map[:,0,:,:,:]
        organ_logits_out[:,2,:,:,:] -= tumor_anomaly_map[:,0,:,:,:]
        
        organ_logits_out[:,5,:,:,:] -= tumor_anomaly_map[:,1,:,:,:]
        organ_logits_out[:,10,:,:,:] -= tumor_anomaly_map[:,2,:,:,:]
        organ_logits_out[:,14,:,:,:] -= tumor_anomaly_map[:,3,:,:,:]
        
        organ_logits_out[:,15,:,:,:] -= tumor_anomaly_map[:,4,:,:,:]
        organ_logits_out[:,16,:,:,:] -= tumor_anomaly_map[:,4,:,:,:]
        
        organ_logits_out[:,17,:,:,:] -= tumor_anomaly_map[:,5,:,:,:]
        
        organ_logits_out[:,1,:,:,:] -= tumor_anomaly_map[:,6,:,:,:]
        organ_logits_out[:,2,:,:,:] -= tumor_anomaly_map[:,6,:,:,:]
        
        all_out = self.queryrefine(tumor_enconding, feats, attn_masks, organ_encondings[0])
        
        tumor_encodings = all_out[:, self.organ_numbers:]
        tumor_weight = self.out_norm_layer(self.controller(tumor_encodings))

        
        N = all_out.shape[1]
        
        tumor_logits = out.flatten(start_dim=2, end_dim=4).transpose(1, 2) @ tumor_weight.transpose(1, 2)
        tumor_logits_out = tumor_logits.transpose(1, 2).reshape(B, self.tumor_numbers, D, H, W)
        
        tumor_logits_out += tumor_anomaly_map
        
        logits_out = torch.cat([organ_logits_out, tumor_logits_out], dim=1)
        
        # weight = self.cl_norm_layer(self.cl(all_out))
        
        
        # sim_targets, sim_targets_q = self.create_targets(N)
        # sim_targets = sim_targets.to(out.device).unsqueeze(0).repeat(B, 1, 1)
        # sim_targets_q = sim_targets_q.to(out.device).unsqueeze(0).repeat(B, 1, 1)
        
        
        # sim_q2k = weight @ know_emb.transpose(1,2) / self.temp
        
        # loss_q2k = -torch.sum(F.log_softmax(sim_q2k, dim=2)*sim_targets,dim=2).mean()
        
        # sim_q2q = weight @ weight.transpose(1,2) / self.temp
        # loss_q2q = -torch.sum(F.log_softmax(sim_q2q, dim=2)*sim_targets_q,dim=2).mean()
        
        # loss_itc = loss_q2k+loss_q2q
        
        return logits_out#, loss_itc    
    # def forward_prediction_heads(self, anomaly_score):
    #     anomaly_map = -(anomaly_score.max(1)[0])
    #     return_anomaly_map = anomaly_map.unsqueeze(1).repeat(1, self.tumor_numbers, 1, 1, 1)
    #     norm_anomaly_map =  self.min_max_normalize_batch(anomaly_map)
    #     map_for_mask = norm_anomaly_map.detach().flatten(1, 3)
        
    #     attn_mask = (map_for_mask < 0.5).bool()
        
    #     out_mask = attn_mask.unsqueeze(1).unsqueeze(1).repeat(1, self.num_heads, self.tumor_numbers, 1).flatten(0, 1)
        
    #     assert len(out_mask.shape) == 3
    #     assert out_mask.dtype == torch.bool, "attn_mask should be of type torch.bool"
        
    #     return return_anomaly_map, out_mask
    
    # def forward(self, x_in):
    #     B = x_in.shape[0]
    #     dec4, out, feats, decoder_outs = self.backbone(x_in)
        
    #     organ_embedding = self.organ_embedding.weight
    #     organ_enconding = organ_embedding.unsqueeze(0).repeat(B, 1, 1)
        
    #     tumor_embedding = self.tumor_embedding.weight
    #     tumor_enconding = tumor_embedding.unsqueeze(0).repeat(B, 1, 1)        

    #     organ_encondings = []
    #     for i in range(len(self.text_attend_layers)):
    #         organ_enconding, _ = self.text_attend_layers[i](feats[i], organ_enconding)
    #         organ_encondings.append(organ_enconding)

    #     whole_weight = self.out_norm_layer(self.whole_controller(organ_encondings[-1]))
        
    #     B, C, D, H, W = out.size()
    #     whole_organ_logits = out.flatten(start_dim=2, end_dim=4).transpose(1, 2) @ whole_weight.transpose(1, 2)
    #     whole_organ_logits_out = whole_organ_logits.transpose(1, 2).reshape(B, self.organ_numbers, D, H, W)
    #     attn_masks = []
    #     for i in range(len(organ_encondings)):
    #         weight = organ_encondings[i]#self.proj_controllers[i](organ_enconding)
    #         _, attn_mask = self.forward_prediction_heads(decoder_outs[i])
    #         attn_masks.append(attn_mask)
        
    #     tumor_anomaly_map, _ = self.forward_prediction_heads(out)
        
    #     organ_logits_out = whole_organ_logits_out
    #     organ_logits_out[:,1,:,:,:] -= tumor_anomaly_map[:,0,:,:,:]
    #     organ_logits_out[:,2,:,:,:] -= tumor_anomaly_map[:,0,:,:,:]
        
    #     organ_logits_out[:,5,:,:,:] -= tumor_anomaly_map[:,1,:,:,:]
    #     organ_logits_out[:,10,:,:,:] -= tumor_anomaly_map[:,2,:,:,:]
    #     organ_logits_out[:,14,:,:,:] -= tumor_anomaly_map[:,3,:,:,:]
        
    #     organ_logits_out[:,15,:,:,:] -= tumor_anomaly_map[:,4,:,:,:]
    #     organ_logits_out[:,16,:,:,:] -= tumor_anomaly_map[:,4,:,:,:]
        
    #     organ_logits_out[:,17,:,:,:] -= tumor_anomaly_map[:,5,:,:,:]
        
    #     organ_logits_out[:,1,:,:,:] -= tumor_anomaly_map[:,6,:,:,:]
    #     organ_logits_out[:,2,:,:,:] -= tumor_anomaly_map[:,6,:,:,:]
        
    #     all_out = self.queryrefine(tumor_enconding, feats, attn_masks, organ_encondings[0])
        
    #     tumor_encodings = all_out[:, self.organ_numbers:]
    #     tumor_weight = self.out_norm_layer(self.controller(tumor_encodings))

        
    #     N = all_out.shape[1]
        
    #     tumor_logits = out.flatten(start_dim=2, end_dim=4).transpose(1, 2) @ tumor_weight.transpose(1, 2)
    #     tumor_logits_out = tumor_logits.transpose(1, 2).reshape(B, self.tumor_numbers, D, H, W)
        
    #     tumor_logits_out += tumor_anomaly_map
        
    #     logits_out = torch.cat([organ_logits_out, tumor_logits_out], dim=1)
        
    #     return logits_out