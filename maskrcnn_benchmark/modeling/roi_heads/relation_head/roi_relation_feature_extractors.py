
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from itertools import count
from tkinter.messagebox import NO
from turtle import pd
import torch
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.backbone import resnet
from maskrcnn_benchmark.modeling.poolers import Pooler
from maskrcnn_benchmark.modeling.make_layers import group_norm
from maskrcnn_benchmark.modeling.make_layers import make_fc
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_union, boxlist_intersection
from maskrcnn_benchmark.modeling.roi_heads.box_head.roi_box_feature_extractors import make_roi_box_feature_extractor
from maskrcnn_benchmark.modeling.roi_heads.attribute_head.roi_attribute_feature_extractors import make_roi_attribute_feature_extractor
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
import numpy as np
import random
import os
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.utils import cat
from torch.autograd import Variable
# HEAD = cfg.HEAD_IDS
# BODY = cfg.BODY_IDS
# TAIL = cfg.TAIL_IDS


@registry.ROI_RELATION_FEATURE_EXTRACTORS.register("RelationFeatureExtractor")
class RelationFeatureExtractor(nn.Module):
    """
    Heads for Motifs for relation triplet classification
    """
    def __init__(self, cfg, in_channels):
        super(RelationFeatureExtractor, self).__init__()
        self.cfg = cfg.clone()
        # should corresponding to obj_feature_map function in neural-motifs
        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pool_all_levels = cfg.MODEL.ROI_RELATION_HEAD.POOLING_ALL_LEVELS
        
        if cfg.MODEL.ATTRIBUTE_ON:
            self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels, half_out=True, cat_all_levels=pool_all_levels)
            self.att_feature_extractor = make_roi_attribute_feature_extractor(cfg, in_channels, half_out=True, cat_all_levels=pool_all_levels)
            self.out_channels = self.feature_extractor.out_channels * 2
        else:
            self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels, cat_all_levels=pool_all_levels)
            self.out_channels = self.feature_extractor.out_channels

        # separete spatial
        self.separate_spatial = self.cfg.MODEL.ROI_RELATION_HEAD.CAUSAL.SEPARATE_SPATIAL
        if self.separate_spatial:
            input_size = self.feature_extractor.resize_channels
            out_dim = self.feature_extractor.out_channels
            self.spatial_fc = nn.Sequential(*[make_fc(input_size, out_dim//2), nn.ReLU(inplace=True),
                                              make_fc(out_dim//2, out_dim), nn.ReLU(inplace=True),
                                            ])

        # union rectangle size
        self.rect_size = resolution * 4 -1
        self.rect_conv = nn.Sequential(*[
            nn.Conv2d(2, in_channels //2, kernel_size=7, stride=2, padding=3, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(in_channels//2, momentum=0.01),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(in_channels, momentum=0.01),
            ])
        

    def forward(self, x, proposals, rel_pair_idxs=None):
        device = x[0].device
        union_proposals = []
        rect_inputs = []
        for proposal, rel_pair_idx in zip(proposals, rel_pair_idxs):
            head_proposal = proposal[rel_pair_idx[:, 0]]
            tail_proposal = proposal[rel_pair_idx[:, 1]]
            union_proposal = boxlist_union(head_proposal, tail_proposal)
            union_proposals.append(union_proposal)

            # use range to construct rectangle, sized (rect_size, rect_size)
            num_rel = len(rel_pair_idx)
            dummy_x_range = torch.arange(self.rect_size, device=device).view(1, 1, -1).expand(num_rel, self.rect_size, self.rect_size)
            dummy_y_range = torch.arange(self.rect_size, device=device).view(1, -1, 1).expand(num_rel, self.rect_size, self.rect_size)
            # resize bbox to the scale rect_size
            head_proposal = head_proposal.resize((self.rect_size, self.rect_size))
            tail_proposal = tail_proposal.resize((self.rect_size, self.rect_size))
            head_rect = ((dummy_x_range >= head_proposal.bbox[:,0].floor().view(-1,1,1).long()) & \
                        (dummy_x_range <= head_proposal.bbox[:,2].ceil().view(-1,1,1).long()) & \
                        (dummy_y_range >= head_proposal.bbox[:,1].floor().view(-1,1,1).long()) & \
                        (dummy_y_range <= head_proposal.bbox[:,3].ceil().view(-1,1,1).long())).float()
            tail_rect = ((dummy_x_range >= tail_proposal.bbox[:,0].floor().view(-1,1,1).long()) & \
                        (dummy_x_range <= tail_proposal.bbox[:,2].ceil().view(-1,1,1).long()) & \
                        (dummy_y_range >= tail_proposal.bbox[:,1].floor().view(-1,1,1).long()) & \
                        (dummy_y_range <= tail_proposal.bbox[:,3].ceil().view(-1,1,1).long())).float()

            rect_input = torch.stack((head_rect, tail_rect), dim=1) # (num_rel, 4, rect_size, rect_size)
            rect_inputs.append(rect_input)

        # rectangle feature. size (total_num_rel, in_channels, POOLER_RESOLUTION, POOLER_RESOLUTION)
        rect_inputs = torch.cat(rect_inputs, dim=0)
        rect_features = self.rect_conv(rect_inputs)

        # union visual feature. size (total_num_rel, in_channels, POOLER_RESOLUTION, POOLER_RESOLUTION)
        union_vis_features = self.feature_extractor.pooler(x, union_proposals)
        # merge two parts
        if self.separate_spatial:
            region_features = self.feature_extractor.forward_without_pool(union_vis_features)
            spatial_features = self.spatial_fc(rect_features.view(rect_features.size(0), -1))
            union_features = (region_features, spatial_features)
        else:
            union_features = union_vis_features + rect_features
            union_features = self.feature_extractor.forward_without_pool(union_features) # (total_num_rel, out_channels)

        if self.cfg.MODEL.ATTRIBUTE_ON:
            union_att_features = self.att_feature_extractor.pooler(x, union_proposals)
            union_features_att = union_att_features + rect_features
            union_features_att = self.att_feature_extractor.forward_without_pool(union_features_att)
            union_features = torch.cat((union_features, union_features_att), dim=-1)
            
        return union_features


@registry.ROI_RELATION_FEATURE_EXTRACTORS.register("AugRelationFeatureExtractor")
class AugRelationFeatureExtractor(nn.Module):
    """
    Heads for Motifs for relation triplet classification
    """

    def __init__(self, cfg, in_channels):
        super(AugRelationFeatureExtractor, self).__init__()
        self.cfg = cfg.clone()
        # should corresponding to obj_feature_map function in neural-motifs
        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pool_all_levels = cfg.MODEL.ROI_RELATION_HEAD.POOLING_ALL_LEVELS
        self.tail_features_dict = np.load(os.path.join(cfg.MIXUP.FEAT_PATH, 'tail_feature_dict_motif.npy'), allow_pickle=True)
        if cfg.MODEL.ATTRIBUTE_ON:
            self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels, half_out=True,
                                                                    cat_all_levels=pool_all_levels)
            self.att_feature_extractor = make_roi_attribute_feature_extractor(cfg, in_channels, half_out=True,
                                                                              cat_all_levels=pool_all_levels)
            self.out_channels = self.feature_extractor.out_channels * 2
        else:
            self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels, cat_all_levels=pool_all_levels)
            self.out_channels = self.feature_extractor.out_channels
        if cfg.SELECT_DATASET == 'VG':
            self.HEAD = cfg.HEAD_IDS
            self.BODY = cfg.BODY_IDS
            self.TAIL = cfg.TAIL_IDS
            self.id_object_dict = {"0": "__background__","1": "airplane", "2": "animal", "3": "arm", "4": "bag", "5": "banana", "6": "basket", "7": "beach", "8": "bear", "9": "bed", "10": "bench", "11": "bike", "12": "bird", "13": "board", "14": "boat", "15": "book", "16": "boot", "17": "bottle", "18": "bowl", "19": "box", "20": "boy", "21": "branch", "22": "building", "23": "bus", "24": "cabinet", "25": "cap", "26": "car", "27": "cat", "28": "chair", "29": "child", "30": "clock", "31": "coat", "32": "counter", "33": "cow", "34": "cup", "35": "curtain", "36": "desk", "37": "dog", "38": "door", "39": "drawer", "40": "ear", "41": "elephant", "42": "engine", "43": "eye", "44": "face", "45": "fence", "46": "finger", "47": "flag", "48": "flower", "49": "food", "50": "fork", "51": "fruit", "52": "giraffe", "53": "girl", "54": "glass", "55": "glove", "56": "guy", "57": "hair", "58": "hand", "59": "handle", "60": "hat", "61": "head", "62": "helmet", "63": "hill", "64": "horse", "65": "house", "66": "jacket", "67": "jean", "68": "kid", "69": "kite", "70": "lady", "71": "lamp", "72": "laptop", "73": "leaf", "74": "leg", "75": "letter", "76": "light", "77": "logo", "78": "man", "79": "men", "80": "motorcycle", "81": "mountain", "82": "mouth", "83": "neck", "84": "nose", "85": "number", "86": "orange", "87": "pant", "88": "paper", "89": "paw", "90": "people", "91": "person", "92": "phone", "93": "pillow", "94": "pizza", "95": "plane", "96": "plant", "97": "plate", "98": "player", "99": "pole", "100": "post", "101": "pot", "102": "racket", "103": "railing", "104": "rock", "105": "roof", "106": "room", "107": "screen", "108": "seat", "109": "sheep", "110": "shelf", "111": "shirt", "112": "shoe", "113": "short", "114": "sidewalk", "115": "sign", "116": "sink", "117": "skateboard", "118": "ski", "119": "skier", "120": "sneaker", "121": "snow", "122": "sock", "123": "stand", "124": "street", "125": "surfboard", "126": "table", "127": "tail", "128": "tie", "129": "tile", "130": "tire", "131": "toilet", "132": "towel", "133": "tower", "134": "track", "135": "train", "136": "tree", "137": "truck", "138": "trunk", "139": "umbrella", "140": "vase", "141": "vegetable", "142": "vehicle", "143": "wave", "144": "wheel", "145": "window", "146": "windshield", "147": "wing", "148": "wire", "149": "woman", "150": "zebra"}
            self.id_to_rel_dict = {"0": "__background__","1": "above", "2": "across", "3": "against", "4": "along", "5": "and", "6": "at", "7": "attached to", "8": "behind", "9": "belonging to", "10": "between", "11": "carrying", "12": "covered in", "13": "covering", "14": "eating", "15": "flying in", "16": "for", "17": "from", "18": "growing on", "19": "hanging from", "20": "has", "21": "holding", "22": "in", "23": "in front of", "24": "laying on", "25": "looking at", "26": "lying on", "27": "made of", "28": "mounted on", "29": "near", "30": "of", "31": "on", "32": "on back of", "33": "over", "34": "painted on", "35": "parked on", "36": "part of", "37": "playing", "38": "riding", "39": "says", "40": "sitting on", "41": "standing on", "42": "to", "43": "under", "44": "using", "45": "walking in", "46": "walking on", "47": "watching", "48": "wearing", "49": "wears", "50": "with"}
        elif cfg.SELECT_DATASET == 'GQA':
            self.HEAD = cfg.GQA_HEAD_IDS
            self.BODY = cfg.GQA_BODY_IDS
            self.TAIL = cfg.GQA_TAIL_IDS
            self.id_object_dict = {"0": "__background__", "1": "window", "2": "tree", "3": "man", "4": "shirt", "5": "wall", "6": "building", "7": "person", "8": "ground", "9": "sky", "10": "leg", "11": "sign", "12": "hand", "13": "head", "14": "pole", "15": "grass", "16": "hair", "17": "car", "18": "ear", "19": "eye", "20": "woman", "21": "clouds", "22": "shoe", "23": "table", "24": "leaves", "25": "wheel", "26": "door", "27": "pants", "28": "letter", "29": "people", "30": "flower", "31": "water", "32": "glass", "33": "chair", "34": "fence", "35": "arm", "36": "nose", "37": "number", "38": "floor", "39": "rock", "40": "jacket", "41": "hat", "42": "plate", "43": "tail", "44": "leaf", "45": "face", "46": "bush", "47": "shorts", "48": "road", "49": "bag", "50": "sidewalk", "51": "tire", "52": "helmet", "53": "snow", "54": "boy", "55": "umbrella", "56": "logo", "57": "roof", "58": "boat", "59": "bottle", "60": "street", "61": "plant", "62": "foot", "63": "branch", "64": "post", "65": "jeans", "66": "mouth", "67": "cap", "68": "girl", "69": "bird", "70": "banana", "71": "box", "72": "bench", "73": "mirror", "74": "picture", "75": "pillow", "76": "book", "77": "field", "78": "glove", "79": "clock", "80": "dirt", "81": "bowl", "82": "bus", "83": "neck", "84": "trunk", "85": "wing", "86": "horse", "87": "food", "88": "train", "89": "kite", "90": "paper", "91": "shelf", "92": "airplane", "93": "sock", "94": "house", "95": "elephant", "96": "lamp", "97": "coat", "98": "cup", "99": "cabinet", "100": "street light", "101": "cow", "102": "word", "103": "dog", "104": "finger", "105": "giraffe", "106": "mountain", "107": "wire", "108": "flag", "109": "seat", "110": "sheep", "111": "counter", "112": "skis", "113": "zebra", "114": "hill", "115": "truck", "116": "bike", "117": "racket", "118": "ball", "119": "skateboard", "120": "ceiling", "121": "motorcycle", "122": "player", "123": "surfboard", "124": "sand", "125": "towel", "126": "frame", "127": "container", "128": "paw", "129": "feet", "130": "curtain", "131": "windshield", "132": "traffic light", "133": "horn", "134": "cat", "135": "child", "136": "bed", "137": "sink", "138": "animal", "139": "donut", "140": "stone", "141": "tie", "142": "pizza", "143": "orange", "144": "sticker", "145": "apple", "146": "backpack", "147": "vase", "148": "basket", "149": "drawer", "150": "collar", "151": "lid", "152": "cord", "153": "phone", "154": "pot", "155": "vehicle", "156": "fruit", "157": "laptop", "158": "fork", "159": "uniform", "160": "bear", "161": "fur", "162": "license plate", "163": "lady", "164": "tomato", "165": "tag", "166": "mane", "167": "beach", "168": "tower", "169": "cone", "170": "cheese", "171": "wrist", "172": "napkin", "173": "toilet", "174": "desk", "175": "dress", "176": "cell phone", "177": "faucet", "178": "blanket", "179": "screen", "180": "watch", "181": "keyboard", "182": "arrow", "183": "sneakers", "184": "broccoli", "185": "bicycle", "186": "guy", "187": "knife", "188": "ocean", "189": "t-shirt", "190": "bread", "191": "spots", "192": "cake", "193": "air", "194": "sweater", "195": "room", "196": "couch", "197": "camera", "198": "frisbee", "199": "trash can", "200": "paint"}
            self.id_to_rel_dict = {"0": "__background__", "1": "on", "2": "wearing", "3": "of", "4": "near", "5": "in", "6": "behind", "7": "in front of", "8": "holding", "9": "next to", "10": "above", "11": "on top of", "12": "below", "13": "by", "14": "with", "15": "sitting on", "16": "on the side of", "17": "under", "18": "riding", "19": "standing on", "20": "beside", "21": "carrying", "22": "walking on", "23": "standing in", "24": "lying on", "25": "eating", "26": "covered by", "27": "looking at", "28": "hanging on", "29": "at", "30": "covering", "31": "on the front of", "32": "around", "33": "sitting in", "34": "parked on", "35": "watching", "36": "flying in", "37": "hanging from", "38": "using", "39": "sitting at", "40": "covered in", "41": "crossing", "42": "standing next to", "43": "playing with", "44": "walking in", "45": "on the back of", "46": "reflected in", "47": "flying", "48": "touching", "49": "surrounded by", "50": "covered with", "51": "standing by", "52": "driving on", "53": "leaning on", "54": "lying in", "55": "swinging", "56": "full of", "57": "talking on", "58": "walking down", "59": "throwing", "60": "surrounding", "61": "standing near", "62": "standing behind", "63": "hitting", "64": "printed on", "65": "filled with", "66": "catching", "67": "growing on", "68": "grazing on", "69": "mounted on", "70": "facing", "71": "leaning against", "72": "cutting", "73": "growing in", "74": "floating in", "75": "driving", "76": "beneath", "77": "contain", "78": "resting on", "79": "worn on", "80": "walking with", "81": "driving down", "82": "on the bottom of", "83": "playing on", "84": "playing in", "85": "feeding", "86": "standing in front of", "87": "waiting for", "88": "running on", "89": "close to", "90": "sitting next to", "91": "swimming in", "92": "talking to", "93": "grazing in", "94": "pulling", "95": "pulled by", "96": "reaching for", "97": "attached to", "98": "skiing on", "99": "parked along", "100": "hang on"}

        obj_to_id_dict = dict(zip(self.id_object_dict.values(), self.id_object_dict.keys()))
        # separete spatial
        self.separate_spatial = self.cfg.MODEL.ROI_RELATION_HEAD.CAUSAL.SEPARATE_SPATIAL
        if self.separate_spatial:
            input_size = self.feature_extractor.resize_channels
            out_dim = self.feature_extractor.out_channels
            self.spatial_fc = nn.Sequential(*[make_fc(input_size, out_dim // 2), nn.ReLU(inplace=True),
                                              make_fc(out_dim // 2, out_dim), nn.ReLU(inplace=True),
                                              ])

        # union rectangle size
        self.rect_size = resolution * 4 - 1
        self.rect_conv = nn.Sequential(*[
            nn.Conv2d(2, in_channels // 2, kernel_size=7, stride=2, padding=3, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(in_channels // 2, momentum=0.01),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(in_channels, momentum=0.01),
        ])

    def forward(self, x, proposals, rel_pair_idxs=None, pair_labels=None, rel_labels=None, roi_features=None, training=False):
        device = x[0].device
        union_proposals = []
        rect_inputs = []
        for proposal, rel_pair_idx in zip(proposals, rel_pair_idxs):
            # print(len(proposal), rel_pair_idx[:, 0])
            head_proposal = proposal[rel_pair_idx[:, 0]]
            tail_proposal = proposal[rel_pair_idx[:, 1]]
            union_proposal = boxlist_union(head_proposal, tail_proposal)
            union_proposals.append(union_proposal)

            # use range to construct rectangle, sized (rect_size, rect_size)
            num_rel = len(rel_pair_idx)
            # print(num_rel)
            dummy_x_range = torch.arange(self.rect_size, device=device).view(1, 1, -1).expand(num_rel, self.rect_size,
                                                                                              self.rect_size)
            dummy_y_range = torch.arange(self.rect_size, device=device).view(1, -1, 1).expand(num_rel, self.rect_size,
                                                                                              self.rect_size)
            # resize bbox to the scale rect_size
            head_proposal = head_proposal.resize((self.rect_size, self.rect_size))
            tail_proposal = tail_proposal.resize((self.rect_size, self.rect_size))
            head_rect = ((dummy_x_range >= head_proposal.bbox[:, 0].floor().view(-1, 1, 1).long()) & \
                         (dummy_x_range <= head_proposal.bbox[:, 2].ceil().view(-1, 1, 1).long()) & \
                         (dummy_y_range >= head_proposal.bbox[:, 1].floor().view(-1, 1, 1).long()) & \
                         (dummy_y_range <= head_proposal.bbox[:, 3].ceil().view(-1, 1, 1).long())).float()
            tail_rect = ((dummy_x_range >= tail_proposal.bbox[:, 0].floor().view(-1, 1, 1).long()) & \
                         (dummy_x_range <= tail_proposal.bbox[:, 2].ceil().view(-1, 1, 1).long()) & \
                         (dummy_y_range >= tail_proposal.bbox[:, 1].floor().view(-1, 1, 1).long()) & \
                         (dummy_y_range <= tail_proposal.bbox[:, 3].ceil().view(-1, 1, 1).long())).float()

            rect_input = torch.stack((head_rect, tail_rect), dim=1)  # (num_rel, 4, rect_size, rect_size)
            rect_inputs.append(rect_input)

        # rectangle feature. size (total_num_rel, in_channels, POOLER_RESOLUTION, POOLER_RESOLUTION)
        rect_inputs = torch.cat(rect_inputs, dim=0)
        # print('rect_inputs', rect_inputs.shape)
        rect_features = self.rect_conv(rect_inputs)

        # union visual feature. size (total_num_rel, in_channels, POOLER_RESOLUTION, POOLER_RESOLUTION)
        union_vis_features = self.feature_extractor.pooler(x, union_proposals)
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        union_vis_features = Variable(union_vis_features, requires_grad=False)
        roi_features = Variable(roi_features, requires_grad=False)
        union_features_list = union_vis_features.split(num_rels, dim=0)
        roi_features_list = roi_features.split(num_objs, dim=0)
        if training:
            for i in range(len(pair_labels)):
                if np.any(np.array(rel_labels[i].cpu()) < 0):
                    filter_idxs = torch.where(rel_labels[i] >= 0)[0]
                    rel_labels[i] = rel_labels[i][filter_idxs]
                    rel_pair_idxs[i] = rel_pair_idxs[i][filter_idxs]
                    union_features_list[i] = union_features_list[i][filter_idxs]
                    rect_inputs[i] = rect_inputs[i][filter_idxs]
                    pair_labels[i] = pair_labels[i][filter_idxs.cpu()]

        union_vis_features = cat(union_features_list, dim=0)
        roi_features = cat(roi_features_list, dim=0)
        union_vis_features = Variable(union_vis_features, requires_grad=False)
        roi_features = Variable(roi_features, requires_grad=False)

        # merge two parts
        if self.separate_spatial:
            region_features = self.feature_extractor.forward_without_pool(union_vis_features)
            spatial_features = self.spatial_fc(rect_features.view(rect_features.size(0), -1))
            union_features = (region_features, spatial_features)
        else:
            union_features = union_vis_features + rect_features
            union_features = self.feature_extractor.forward_without_pool(
                union_features)  # (total_num_rel, out_channels)

        if self.cfg.MODEL.ATTRIBUTE_ON:
            union_att_features = self.att_feature_extractor.pooler(x, union_proposals)
            union_features_att = union_att_features + rect_features
            union_features_att = self.att_feature_extractor.forward_without_pool(union_features_att)
            union_features = torch.cat((union_features, union_features_att), dim=-1)

        return proposals, union_features, roi_features, rel_labels, rel_pair_idxs


@registry.ROI_RELATION_FEATURE_EXTRACTORS.register("AugBilvlMxiUpRelationFeatureExtractor")
class AugBilvlMxiUpRelationFeatureExtractor(nn.Module):
    """
    Heads for Motifs for relation triplet classification
    """

    def __init__(self, cfg, in_channels):
        super(AugBilvlMxiUpRelationFeatureExtractor, self).__init__()
        self.cfg = cfg.clone()
        # should corresponding to obj_feature_map function in neural-motifs
        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pool_all_levels = cfg.MODEL.ROI_RELATION_HEAD.POOLING_ALL_LEVELS
        self.fg_lambda = cfg.MIXUP.FG_LAMBDA
        self.bg_lambda = cfg.MIXUP.BG_LAMBDA
        if cfg.SELECT_DATASET == 'VG':
            self.HEAD = cfg.HEAD_IDS
            self.BODY = cfg.BODY_IDS
            self.TAIL = cfg.TAIL_IDS
            self.id_object_dict = {"0": "__background__","1": "airplane", "2": "animal", "3": "arm", "4": "bag", "5": "banana", "6": "basket", "7": "beach", "8": "bear", "9": "bed", "10": "bench", "11": "bike", "12": "bird", "13": "board", "14": "boat", "15": "book", "16": "boot", "17": "bottle", "18": "bowl", "19": "box", "20": "boy", "21": "branch", "22": "building", "23": "bus", "24": "cabinet", "25": "cap", "26": "car", "27": "cat", "28": "chair", "29": "child", "30": "clock", "31": "coat", "32": "counter", "33": "cow", "34": "cup", "35": "curtain", "36": "desk", "37": "dog", "38": "door", "39": "drawer", "40": "ear", "41": "elephant", "42": "engine", "43": "eye", "44": "face", "45": "fence", "46": "finger", "47": "flag", "48": "flower", "49": "food", "50": "fork", "51": "fruit", "52": "giraffe", "53": "girl", "54": "glass", "55": "glove", "56": "guy", "57": "hair", "58": "hand", "59": "handle", "60": "hat", "61": "head", "62": "helmet", "63": "hill", "64": "horse", "65": "house", "66": "jacket", "67": "jean", "68": "kid", "69": "kite", "70": "lady", "71": "lamp", "72": "laptop", "73": "leaf", "74": "leg", "75": "letter", "76": "light", "77": "logo", "78": "man", "79": "men", "80": "motorcycle", "81": "mountain", "82": "mouth", "83": "neck", "84": "nose", "85": "number", "86": "orange", "87": "pant", "88": "paper", "89": "paw", "90": "people", "91": "person", "92": "phone", "93": "pillow", "94": "pizza", "95": "plane", "96": "plant", "97": "plate", "98": "player", "99": "pole", "100": "post", "101": "pot", "102": "racket", "103": "railing", "104": "rock", "105": "roof", "106": "room", "107": "screen", "108": "seat", "109": "sheep", "110": "shelf", "111": "shirt", "112": "shoe", "113": "short", "114": "sidewalk", "115": "sign", "116": "sink", "117": "skateboard", "118": "ski", "119": "skier", "120": "sneaker", "121": "snow", "122": "sock", "123": "stand", "124": "street", "125": "surfboard", "126": "table", "127": "tail", "128": "tie", "129": "tile", "130": "tire", "131": "toilet", "132": "towel", "133": "tower", "134": "track", "135": "train", "136": "tree", "137": "truck", "138": "trunk", "139": "umbrella", "140": "vase", "141": "vegetable", "142": "vehicle", "143": "wave", "144": "wheel", "145": "window", "146": "windshield", "147": "wing", "148": "wire", "149": "woman", "150": "zebra"}
            self.id_to_rel_dict = {"0": "__background__","1": "above", "2": "across", "3": "against", "4": "along", "5": "and", "6": "at", "7": "attached to", "8": "behind", "9": "belonging to", "10": "between", "11": "carrying", "12": "covered in", "13": "covering", "14": "eating", "15": "flying in", "16": "for", "17": "from", "18": "growing on", "19": "hanging from", "20": "has", "21": "holding", "22": "in", "23": "in front of", "24": "laying on", "25": "looking at", "26": "lying on", "27": "made of", "28": "mounted on", "29": "near", "30": "of", "31": "on", "32": "on back of", "33": "over", "34": "painted on", "35": "parked on", "36": "part of", "37": "playing", "38": "riding", "39": "says", "40": "sitting on", "41": "standing on", "42": "to", "43": "under", "44": "using", "45": "walking in", "46": "walking on", "47": "watching", "48": "wearing", "49": "wears", "50": "with"}
        elif cfg.SELECT_DATASET == 'GQA':
            self.HEAD = cfg.GQA_HEAD_IDS
            self.BODY = cfg.GQA_BODY_IDS
            self.TAIL = cfg.GQA_TAIL_IDS
            self.id_object_dict = {"0": "__background__", "1": "window", "2": "tree", "3": "man", "4": "shirt", "5": "wall", "6": "building", "7": "person", "8": "ground", "9": "sky", "10": "leg", "11": "sign", "12": "hand", "13": "head", "14": "pole", "15": "grass", "16": "hair", "17": "car", "18": "ear", "19": "eye", "20": "woman", "21": "clouds", "22": "shoe", "23": "table", "24": "leaves", "25": "wheel", "26": "door", "27": "pants", "28": "letter", "29": "people", "30": "flower", "31": "water", "32": "glass", "33": "chair", "34": "fence", "35": "arm", "36": "nose", "37": "number", "38": "floor", "39": "rock", "40": "jacket", "41": "hat", "42": "plate", "43": "tail", "44": "leaf", "45": "face", "46": "bush", "47": "shorts", "48": "road", "49": "bag", "50": "sidewalk", "51": "tire", "52": "helmet", "53": "snow", "54": "boy", "55": "umbrella", "56": "logo", "57": "roof", "58": "boat", "59": "bottle", "60": "street", "61": "plant", "62": "foot", "63": "branch", "64": "post", "65": "jeans", "66": "mouth", "67": "cap", "68": "girl", "69": "bird", "70": "banana", "71": "box", "72": "bench", "73": "mirror", "74": "picture", "75": "pillow", "76": "book", "77": "field", "78": "glove", "79": "clock", "80": "dirt", "81": "bowl", "82": "bus", "83": "neck", "84": "trunk", "85": "wing", "86": "horse", "87": "food", "88": "train", "89": "kite", "90": "paper", "91": "shelf", "92": "airplane", "93": "sock", "94": "house", "95": "elephant", "96": "lamp", "97": "coat", "98": "cup", "99": "cabinet", "100": "street light", "101": "cow", "102": "word", "103": "dog", "104": "finger", "105": "giraffe", "106": "mountain", "107": "wire", "108": "flag", "109": "seat", "110": "sheep", "111": "counter", "112": "skis", "113": "zebra", "114": "hill", "115": "truck", "116": "bike", "117": "racket", "118": "ball", "119": "skateboard", "120": "ceiling", "121": "motorcycle", "122": "player", "123": "surfboard", "124": "sand", "125": "towel", "126": "frame", "127": "container", "128": "paw", "129": "feet", "130": "curtain", "131": "windshield", "132": "traffic light", "133": "horn", "134": "cat", "135": "child", "136": "bed", "137": "sink", "138": "animal", "139": "donut", "140": "stone", "141": "tie", "142": "pizza", "143": "orange", "144": "sticker", "145": "apple", "146": "backpack", "147": "vase", "148": "basket", "149": "drawer", "150": "collar", "151": "lid", "152": "cord", "153": "phone", "154": "pot", "155": "vehicle", "156": "fruit", "157": "laptop", "158": "fork", "159": "uniform", "160": "bear", "161": "fur", "162": "license plate", "163": "lady", "164": "tomato", "165": "tag", "166": "mane", "167": "beach", "168": "tower", "169": "cone", "170": "cheese", "171": "wrist", "172": "napkin", "173": "toilet", "174": "desk", "175": "dress", "176": "cell phone", "177": "faucet", "178": "blanket", "179": "screen", "180": "watch", "181": "keyboard", "182": "arrow", "183": "sneakers", "184": "broccoli", "185": "bicycle", "186": "guy", "187": "knife", "188": "ocean", "189": "t-shirt", "190": "bread", "191": "spots", "192": "cake", "193": "air", "194": "sweater", "195": "room", "196": "couch", "197": "camera", "198": "frisbee", "199": "trash can", "200": "paint"}
            self.id_to_rel_dict = {"0": "__background__", "1": "on", "2": "wearing", "3": "of", "4": "near", "5": "in", "6": "behind", "7": "in front of", "8": "holding", "9": "next to", "10": "above", "11": "on top of", "12": "below", "13": "by", "14": "with", "15": "sitting on", "16": "on the side of", "17": "under", "18": "riding", "19": "standing on", "20": "beside", "21": "carrying", "22": "walking on", "23": "standing in", "24": "lying on", "25": "eating", "26": "covered by", "27": "looking at", "28": "hanging on", "29": "at", "30": "covering", "31": "on the front of", "32": "around", "33": "sitting in", "34": "parked on", "35": "watching", "36": "flying in", "37": "hanging from", "38": "using", "39": "sitting at", "40": "covered in", "41": "crossing", "42": "standing next to", "43": "playing with", "44": "walking in", "45": "on the back of", "46": "reflected in", "47": "flying", "48": "touching", "49": "surrounded by", "50": "covered with", "51": "standing by", "52": "driving on", "53": "leaning on", "54": "lying in", "55": "swinging", "56": "full of", "57": "talking on", "58": "walking down", "59": "throwing", "60": "surrounding", "61": "standing near", "62": "standing behind", "63": "hitting", "64": "printed on", "65": "filled with", "66": "catching", "67": "growing on", "68": "grazing on", "69": "mounted on", "70": "facing", "71": "leaning against", "72": "cutting", "73": "growing in", "74": "floating in", "75": "driving", "76": "beneath", "77": "contain", "78": "resting on", "79": "worn on", "80": "walking with", "81": "driving down", "82": "on the bottom of", "83": "playing on", "84": "playing in", "85": "feeding", "86": "standing in front of", "87": "waiting for", "88": "running on", "89": "close to", "90": "sitting next to", "91": "swimming in", "92": "talking to", "93": "grazing in", "94": "pulling", "95": "pulled by", "96": "reaching for", "97": "attached to", "98": "skiing on", "99": "parked along", "100": "hang on"}

        if cfg.MIXUP.MIXUP_ADD_TAIL:
            self.obj_to_id_dict = dict(zip(self.id_object_dict.values(), self.id_object_dict.keys()))
            if cfg.SELECT_DATASET == 'VG':
                high_level_obj_dict_1_1_1_001_15 = {1: 0, 147: 0, 127: 0, 95: 0, 64: 1, 33: 1, 2: 1, 37: 1, 8: 1, 41: 1, 12: 1, 109: 1, 52: 1, 150: 1, 27: 1, 89: 2, 3: 2, 40: 2, 74: 2, 44: 2, 57: 2, 58: 2, 61: 2, 4: 3, 69: 3, 132: 3, 6: 3, 101: 3, 39: 3, 139: 3, 140: 3, 13: 3, 18: 3, 19: 3, 88: 3, 5: 4, 141: 4, 46: 4, 49: 4, 50: 4, 51: 4, 86: 4, 94: 4, 65: 5, 7: 5, 137: 5, 10: 5, 11: 5, 28: 5, 142: 5, 143: 5, 14: 5, 80: 5, 114: 5, 117: 5, 118: 5, 121: 5, 124: 5, 125: 5, 32: 6, 129: 6, 34: 6, 131: 6, 36: 6, 97: 6, 9: 6, 106: 6, 110: 6, 17: 6, 116: 6, 54: 6, 24: 6, 123: 6, 126: 6, 96: 7, 35: 7, 72: 7, 104: 7, 138: 7, 107: 7, 108: 7, 73: 7, 45: 7, 15: 7, 48: 7, 81: 7, 21: 7, 92: 7, 93: 7, 63: 7, 66: 8, 67: 8, 111: 8, 16: 8, 113: 8, 112: 8, 87: 8, 60: 8, 31: 8, 98: 9, 68: 9, 70: 9, 78: 9, 79: 9, 20: 9, 53: 9, 149: 9, 119: 9, 56: 9, 90: 9, 91: 9, 29: 9, 38: 10, 135: 10, 136: 10, 134: 10, 145: 10, 22: 10, 23: 10, 26: 10, 128: 11, 102: 11, 55: 11, 120: 11, 25: 11, 122: 11, 62: 11, 99: 12, 100: 12, 133: 12, 71: 12, 103: 12, 105: 12, 76: 12, 47: 12, 115: 12, 148: 12, 30: 12, 130: 13, 42: 13, 75: 13, 77: 13, 144: 13, 146: 13, 85: 13, 59: 13, 83: 14, 82: 14, 43: 14, 84: 14}
                high_level_obj_dict_150 = dict(zip(list(high_level_obj_dict_1_1_1_001_15.keys()), list(high_level_obj_dict_1_1_1_001_15.keys())))
                high_level_obj_dict_1_1_1_0_15 = {1: 0, 42: 0, 147: 0, 95: 0, 2: 1, 4: 1, 134: 1, 73: 1, 138: 1, 139: 1, 11: 1, 13: 1, 80: 1, 81: 1, 21: 1, 96: 1, 104: 1, 48: 1, 117: 1, 118: 1, 121: 1, 125: 1, 63: 1, 97: 2, 3: 2, 132: 2, 6: 2, 72: 2, 108: 2, 15: 2, 19: 2, 54: 2, 88: 2, 58: 2, 59: 2, 92: 2, 93: 2, 34: 3, 5: 3, 101: 3, 140: 3, 141: 3, 49: 3, 17: 3, 51: 3, 18: 3, 50: 3, 86: 3, 89: 3, 94: 3, 65: 4, 7: 4, 135: 4, 137: 4, 142: 4, 143: 4, 14: 4, 114: 4, 23: 4, 26: 4, 124: 4, 33: 5, 98: 5, 68: 5, 37: 5, 69: 5, 70: 5, 8: 5, 12: 5, 109: 5, 79: 5, 119: 5, 56: 5, 90: 5, 27: 5, 32: 6, 129: 6, 131: 6, 36: 6, 39: 6, 9: 6, 10: 6, 106: 6, 110: 6, 116: 6, 24: 6, 123: 6, 28: 6, 126: 6, 67: 7, 16: 7, 112: 7, 113: 7, 87: 7, 120: 7, 122: 7, 60: 7, 62: 7, 31: 7, 20: 8, 53: 8, 149: 8, 91: 8, 29: 8, 78: 8, 99: 9, 100: 9, 133: 9, 38: 9, 103: 9, 136: 9, 105: 9, 45: 9, 47: 9, 145: 9, 115: 9, 148: 9, 22: 9, 30: 9, 128: 10, 66: 10, 102: 10, 55: 10, 25: 10, 46: 10, 111: 10, 130: 11, 35: 11, 71: 11, 107: 11, 76: 11, 75: 11, 77: 11, 144: 11, 146: 11, 85: 11, 40: 12, 57: 12, 84: 12, 61: 12, 64: 13, 41: 13, 52: 13, 150: 13, 82: 14, 83: 14, 74: 14, 43: 14, 44: 14, 127: 14}
                self.high_level_obj_dict = high_level_obj_dict_1_1_1_001_15 
            elif cfg.SELECT_DATASET == 'GQA':
                self.high_level_obj_dict = {1: 0, 34: 0, 5: 0, 39: 0, 46: 0, 14: 0, 26: 0, 94: 0, 2: 1, 69: 1, 6: 1, 7: 1, 11: 1, 15: 1, 29: 1, 31: 1, 3: 2, 20: 2, 163: 2, 54: 2, 68: 2, 135: 2, 186: 2, 97: 3, 194: 3, 65: 3, 4: 3, 40: 3, 141: 3, 47: 3, 175: 3, 183: 3, 27: 3, 189: 3, 159: 3, 38: 4, 167: 4, 8: 4, 106: 4, 77: 4, 60: 4, 80: 4, 48: 4, 188: 4, 114: 4, 50: 4, 124: 4, 9: 5, 193: 5, 21: 5, 129: 6, 66: 6, 36: 6, 133: 6, 10: 6, 43: 6, 16: 6, 18: 6, 19: 6, 85: 6, 62: 6, 128: 7, 35: 7, 104: 7, 171: 7, 12: 7, 13: 7, 45: 7, 83: 7, 17: 8, 82: 8, 115: 8, 88: 8, 121: 8, 58: 8, 155: 8, 92: 8, 67: 9, 41: 9, 78: 9, 49: 9, 146: 9, 52: 9, 117: 9, 22: 9, 150: 9, 93: 9, 191: 9, 32: 10, 99: 10, 73: 10, 42: 10, 137: 10, 174: 10, 111: 10, 177: 10, 23: 10, 91: 10, 169: 11, 44: 11, 108: 11, 84: 11, 24: 11, 61: 11, 30: 11, 63: 11, 162: 12, 131: 12, 37: 12, 102: 12, 165: 12, 200: 12, 74: 12, 144: 12, 51: 12, 182: 12, 151: 12, 56: 12, 25: 12, 28: 12, 126: 12, 33: 13, 196: 13, 199: 13, 72: 13, 136: 13, 173: 13, 116: 13, 185: 13, 198: 14, 112: 14, 180: 14, 53: 14, 118: 14, 119: 14, 89: 14, 122: 14, 123: 14, 96: 15, 64: 15, 130: 15, 195: 15, 132: 15, 100: 15, 168: 15, 107: 15, 140: 15, 79: 15, 55: 15, 152: 15, 57: 15, 120: 15, 98: 16, 71: 16, 81: 16, 147: 16, 148: 16, 154: 16, 59: 16, 127: 16, 192: 17, 164: 17, 70: 17, 158: 17, 170: 17, 139: 17, 142: 17, 143: 17, 145: 17, 87: 17, 184: 17, 187: 17, 156: 17, 190: 17, 197: 18, 75: 18, 172: 18, 109: 18, 76: 18, 176: 18, 178: 18, 179: 18, 149: 18, 181: 18, 157: 18, 153: 18, 90: 18, 125: 18, 160: 19, 161: 19, 101: 19, 134: 19, 103: 19, 166: 19, 105: 19, 138: 19, 110: 19, 113: 19, 86: 19, 95: 19}

        if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL == False and cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX == True:
            self.tail_dict = np.load(os.path.join(cfg.MIXUP.FEAT_PATH, 'sgcls_tail_feature_with_proposal_dict_motif.npy'), allow_pickle=True).item()
            self.body_dict = np.load(os.path.join(cfg.MIXUP.FEAT_PATH, 'sgcls_body_feature_with_proposal_dict_motif.npy'), allow_pickle=True).item()
            self.tail_tri_map_dict = np.load(os.path.join(cfg.MIXUP.FEAT_PATH, 'sgcls_tail_tri_map.npy'), allow_pickle=True).item()
            self.body_tri_map_dict = np.load(os.path.join(cfg.MIXUP.FEAT_PATH, 'sgcls_body_tri_map.npy'), allow_pickle=True).item()
        else:
            if cfg.FG_TAIL == True:
                self.tail_features_dict = np.load(os.path.join(cfg.MIXUP.FEAT_PATH, 'sgcls_tail_sub_obj_feature_with_proposal_dict_motif.npy'), allow_pickle=True)
            if cfg.FG_BODY == True:
                self.body_features_dict = np.load(os.path.join(cfg.MIXUP.FEAT_PATH, 'sgcls_body_sub_obj_feature_with_proposal_dict_motif.npy'), allow_pickle=True)

            if cfg.MIXUP.MIXUP_ADD_TAIL == True:
                if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL == True:
                    self.tail_obj_dict = np.load(os.path.join(cfg.MIXUP.FEAT_PATH, 'tail_obj_only_feature_with_proposal_dict_motif.npy'), allow_pickle=True).item()
                    self.tail_sub_dict = np.load(os.path.join(cfg.MIXUP.FEAT_PATH, 'tail_sub_only_feature_with_proposal_dict_motif.npy'), allow_pickle=True).item()
                else:
                    self.tail_obj_dict = np.load(os.path.join(cfg.MIXUP.FEAT_PATH, 'sgcls_tail_obj_only_feature_with_proposal_dict_motif.npy'), allow_pickle=True).item()
                    self.tail_sub_dict = np.load(os.path.join(cfg.MIXUP.FEAT_PATH, 'sgcls_tail_sub_only_feature_with_proposal_dict_motif.npy'), allow_pickle=True).item()
            
        if cfg.MODEL.ATTRIBUTE_ON:
            self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels, half_out=True,
                                                                    cat_all_levels=pool_all_levels)
            self.att_feature_extractor = make_roi_attribute_feature_extractor(cfg, in_channels, half_out=True,
                                                                              cat_all_levels=pool_all_levels)
            self.out_channels = self.feature_extractor.out_channels * 2
        else:
            self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels, cat_all_levels=pool_all_levels)
            self.out_channels = self.feature_extractor.out_channels

        # separete spatial
        self.separate_spatial = self.cfg.MODEL.ROI_RELATION_HEAD.CAUSAL.SEPARATE_SPATIAL
        if self.separate_spatial:
            input_size = self.feature_extractor.resize_channels
            out_dim = self.feature_extractor.out_channels
            self.spatial_fc = nn.Sequential(*[make_fc(input_size, out_dim // 2), nn.ReLU(inplace=True),
                                              make_fc(out_dim // 2, out_dim), nn.ReLU(inplace=True),
                                              ])

        # union rectangle size
        self.rect_size = resolution * 4 - 1
        self.rect_conv = nn.Sequential(*[
            nn.Conv2d(2, in_channels // 2, kernel_size=7, stride=2, padding=3, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(in_channels // 2, momentum=0.01),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(in_channels, momentum=0.01),
        ])

    def forward(self, x, proposals, rel_pair_idxs=None, rel_binarys=None, pair_labels=None, rel_labels=None, roi_features=None, training=False, drop_pair_lens=None, mix_up_fg=False, mix_up_bg=False, mix_up_add_tail=False, image_repeats=None):
        device = x[0].device
        union_proposals = []
        rect_inputs = []
        rel_labels_one_hot = []
        all_changed_obj_idxs = []
        all_changed_pred_idxs = []
        all_filter_tail_idxs = []
        all_changed_classes = []
        all_change_to_classes = []
        dist_t = 0.7
        if training and rel_labels != None or cfg.TYPE == 'extract_cfa_feat':
            for i in range(len(rel_labels)):
                rel_label_one_hot = F.one_hot(abs(rel_labels[i]), cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES).float()
                # print(rel_label_one_hot)
                rel_labels_one_hot.append(rel_label_one_hot)
        

        for proposal, rel_pair_idx in zip(proposals, rel_pair_idxs):
            # print(len(proposal), rel_pair_idx[:, 0])
            head_proposal = proposal[rel_pair_idx[:, 0]]
            tail_proposal = proposal[rel_pair_idx[:, 1]]
            union_proposal = boxlist_union(head_proposal, tail_proposal)
            union_proposals.append(union_proposal)

            # use range to construct rectangle, sized (rect_size, rect_size)
            num_rel = len(rel_pair_idx)
            # print(num_rel)
            dummy_x_range = torch.arange(self.rect_size, device=device).view(1, 1, -1).expand(num_rel, self.rect_size,
                                                                                              self.rect_size)
            dummy_y_range = torch.arange(self.rect_size, device=device).view(1, -1, 1).expand(num_rel, self.rect_size,
                                                                                              self.rect_size)
            # resize bbox to the scale rect_size
            head_proposal = head_proposal.resize((self.rect_size, self.rect_size))
            tail_proposal = tail_proposal.resize((self.rect_size, self.rect_size))
            head_rect = ((dummy_x_range >= head_proposal.bbox[:, 0].floor().view(-1, 1, 1).long()) & \
                         (dummy_x_range <= head_proposal.bbox[:, 2].ceil().view(-1, 1, 1).long()) & \
                         (dummy_y_range >= head_proposal.bbox[:, 1].floor().view(-1, 1, 1).long()) & \
                         (dummy_y_range <= head_proposal.bbox[:, 3].ceil().view(-1, 1, 1).long())).float()
            tail_rect = ((dummy_x_range >= tail_proposal.bbox[:, 0].floor().view(-1, 1, 1).long()) & \
                         (dummy_x_range <= tail_proposal.bbox[:, 2].ceil().view(-1, 1, 1).long()) & \
                         (dummy_y_range >= tail_proposal.bbox[:, 1].floor().view(-1, 1, 1).long()) & \
                         (dummy_y_range <= tail_proposal.bbox[:, 3].ceil().view(-1, 1, 1).long())).float()

            rect_input = torch.stack((head_rect, tail_rect), dim=1)  # (num_rel, 4, rect_size, rect_size)
            rect_inputs.append(rect_input)

        # rectangle feature. size (total_num_rel, in_channels, POOLER_RESOLUTION, POOLER_RESOLUTION)

        # union visual feature. size (total_num_rel, in_channels, POOLER_RESOLUTION, POOLER_RESOLUTION)
        union_vis_features = self.feature_extractor.pooler(x, union_proposals)
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        union_vis_features = Variable(union_vis_features, requires_grad=False)
        roi_features = Variable(roi_features, requires_grad=False)
        union_features_list = list(union_vis_features.split(num_rels, dim=0))
        roi_features_list = list(roi_features.split(num_objs, dim=0))

        fg_lambda = self.fg_lambda

        if training or cfg.TYPE == 'extract_cfa_feat':

            for i in range(len(pair_labels)):
                if not (cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL == False and cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX == True):
                    # change to head
                    changed_obj_idxs= []
                    if cfg.FG_TAIL == True:
                        filter_head_idxs = np.where(rel_labels[i].cpu().numpy() < 0)[0]
                        for head_idx in filter_head_idxs:
                            head_pair_label = pair_labels[i][head_idx]
                            if head_pair_label in self.tail_features_dict.item().keys():
                                sub_bbox = proposals[i].bbox.cpu().numpy()[rel_pair_idxs[i][head_idx][0]]
                                obj_bbox = proposals[i].bbox.cpu().numpy()[rel_pair_idxs[i][head_idx][1]]      
                                curri_vector = np.array([(sub_bbox[2] + sub_bbox[0]) / 2. - (obj_bbox[2] + obj_bbox[0]) / 2., (sub_bbox[3] + sub_bbox[1]) / 2. - (obj_bbox[3] + obj_bbox[1]) / 2.]) 
                                
                                filter_sub_objs = []
                                for tail_rel_label in self.tail_features_dict.item()[head_pair_label].keys():
                                    for (rel_label, sub_feature, obj_feature, tail_sub_proposal, tail_obj_proposal, union_feature, vector, wh) in self.tail_features_dict.item()[head_pair_label][tail_rel_label]:
                                        dist = curri_vector.dot(vector) / np.linalg.norm(curri_vector) * np.linalg.norm(vector)
                                            # print(curri_vector, vector, dist)
                                        if dist > dist_t:
                                            filter_sub_objs.append((rel_label, sub_feature, obj_feature, union_feature))
                                if len(filter_sub_objs) > 0:
                                    random_idx = int(random.random() * len(filter_sub_objs))
                                    rel_label, sub_feature, obj_feature, union_feature = filter_sub_objs[random_idx]
                                    # print('tail change...............', head_pair_label, id_to_rel_dict[int(rel_label)])
                                            
                                    rel_labels[i][head_idx] = rel_label.to(device)
                                    rel_label_one_hot = F.one_hot(rel_label.to(device), cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES).float()
                                    rel_labels_one_hot[i][head_idx] = fg_lambda * rel_label_one_hot + (1 - fg_lambda) * rel_labels_one_hot[i][head_idx]
                                    # print('-----------+\n', rel_labels_one_hot[i][head_idx].shape, rel_label_one_hot.shape, lamda, rel_labels_one_hot[i][head_idx])
                                    roi_features_list[i][rel_pair_idxs[i][head_idx, 0]] = sub_feature.to(device) * fg_lambda + roi_features_list[i][rel_pair_idxs[i][head_idx, 0]] * (1 - fg_lambda)
                                    roi_features_list[i][rel_pair_idxs[i][head_idx, 1]] = obj_feature.to(device) * fg_lambda + roi_features_list[i][rel_pair_idxs[i][head_idx, 1]] * (1 - fg_lambda)
                                    union_features_list[i][head_idx] = union_feature.to(device) * fg_lambda + union_features_list[i][head_idx] * (1 - fg_lambda)
                                    changed_obj_idxs.append(rel_pair_idxs[i][head_idx, 0].item())
                                    changed_obj_idxs.append(rel_pair_idxs[i][head_idx, 1].item())

                    # change to body
                    if cfg.FG_BODY == True:
                        if np.any(np.array(rel_labels[i].cpu()) < 0):
                            filter_head_to_body_idxs = torch.where(rel_labels[i] < 0)[0]
                            for head_idx in filter_head_to_body_idxs:
                                head_pair_label = pair_labels[i][head_idx]
                                if head_pair_label in self.body_features_dict.item().keys():
                                    sub_bbox = proposals[i].bbox.cpu().numpy()[rel_pair_idxs[i][head_idx][0]]
                                    obj_bbox = proposals[i].bbox.cpu().numpy()[rel_pair_idxs[i][head_idx][1]]      
                                    curri_vector = np.array([(sub_bbox[2] + sub_bbox[0]) / 2. - (obj_bbox[2] + obj_bbox[0]) / 2., (sub_bbox[3] + sub_bbox[1]) / 2. - (obj_bbox[3] + obj_bbox[1]) / 2.]) 
                                    filter_sub_objs = []
                                    for body_rel_label in self.body_features_dict.item()[head_pair_label].keys():
                                        for (rel_label, sub_feature, obj_feature, tail_sub_proposal, tail_obj_proposal, union_feature, vector, wh) in self.body_features_dict.item()[head_pair_label][body_rel_label]:
                                            dist = curri_vector.dot(vector) / np.linalg.norm(curri_vector) * np.linalg.norm(vector)
                                            if dist > dist_t:
                                                filter_sub_objs.append((rel_label, sub_feature, obj_feature, union_feature))
                                    if len(filter_sub_objs) > 0:
                                        random_idx = int(random.random() * len(filter_sub_objs))
                                        rel_label, sub_feature, obj_feature, union_feature = filter_sub_objs[random_idx]
                                        # print('body change...............', head_pair_label, id_to_rel_dict[int(rel_label)])
                                        rel_labels[i][head_idx] = rel_label.to(device)
                                        rel_label_one_hot = F.one_hot(rel_label.to(device), cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES).float()
                                        rel_labels_one_hot[i][head_idx] = fg_lambda * rel_label_one_hot + (1 - fg_lambda) * rel_labels_one_hot[i][head_idx]
                                        roi_features_list[i][rel_pair_idxs[i][head_idx, 0]] = sub_feature.to(device) * fg_lambda+ roi_features_list[i][rel_pair_idxs[i][head_idx, 0]] * (1 - fg_lambda)
                                        roi_features_list[i][rel_pair_idxs[i][head_idx, 1]] = obj_feature.to(device) * fg_lambda + roi_features_list[i][rel_pair_idxs[i][head_idx, 1]] * (1 - fg_lambda)
                                        union_features_list[i][head_idx] = union_feature.to(device) * fg_lambda+ union_features_list[i][head_idx] * (1 - fg_lambda)
                                        changed_obj_idxs.append(rel_pair_idxs[i][head_idx, 0].item())
                                        changed_obj_idxs.append(rel_pair_idxs[i][head_idx, 1].item())

                    # filter rel label -1
                    if np.any(np.array(rel_labels[i].cpu()) < 0):
                        # print(rel_labels[i])
                        filter_idxs = torch.where(rel_labels[i] >= 0)[0]
                        rel_labels[i] = rel_labels[i][filter_idxs]
                        rel_pair_idxs[i] = rel_pair_idxs[i][filter_idxs]
                        union_features_list[i] = union_features_list[i][filter_idxs]
                        rect_inputs[i] = rect_inputs[i][filter_idxs]
                        rel_labels_one_hot[i] = rel_labels_one_hot[i][filter_idxs]
                        pair_labels[i] = pair_labels[i][filter_idxs.cpu()]

                    # mixup bg
                    if mix_up_bg == True:
                        bg_lambda = self.bg_lambda
                        filter_bg_to_tail_idxs = torch.where(rel_labels[i] == 0)[0]
                        bg_idx = []
                        for head_idx in filter_bg_to_tail_idxs:
                            bg_idx.append(head_idx)
                            head_pair_label = pair_labels[i][head_idx]
                            if head_pair_label in self.tail_features_dict.item().keys():
                                head_pair_label = pair_labels[i][head_idx]
                                sub_bbox = proposals[i].bbox.cpu().numpy()[rel_pair_idxs[i][head_idx][0]]
                                obj_bbox = proposals[i].bbox.cpu().numpy()[rel_pair_idxs[i][head_idx][1]]      
                                curri_vector = np.array([(sub_bbox[2] + sub_bbox[0]) / 2. - (obj_bbox[2] + obj_bbox[0]) / 2., (sub_bbox[3] + sub_bbox[1]) / 2. - (obj_bbox[3] + obj_bbox[1]) / 2.]) 
                            
                                filter_sub_objs = []
                                for tail_rel_label in self.tail_features_dict.item()[head_pair_label].keys():
                                    for (rel_label, sub_feature, obj_feature, tail_sub_proposal, tail_obj_proposal, union_feature, vector, wh) in self.tail_features_dict.item()[head_pair_label][tail_rel_label]:
                                        dist = curri_vector.dot(vector) / np.linalg.norm(curri_vector) * np.linalg.norm(vector)

                                        if dist > dist_t:
                                            filter_sub_objs.append((rel_label, sub_feature, obj_feature, union_feature))
                                if len(filter_sub_objs) > 0:
                                    random_idx = int(random.random() * len(filter_sub_objs))
                                    rel_label, sub_feature, obj_feature, union_feature = filter_sub_objs[random_idx]
                                    # print('body change...............', head_pair_label, id_to_rel_dict[int(rel_label)])
                                    rel_labels[i][head_idx] = rel_label.to(device)
                                    rel_label_one_hot = F.one_hot(rel_label.to(device), cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES).float()
                                    rel_labels_one_hot[i][head_idx] = bg_lambda  * rel_label_one_hot + (1 - bg_lambda ) * rel_labels_one_hot[i][head_idx]
                                    roi_features_list[i][rel_pair_idxs[i][head_idx, 0]] = sub_feature.to(device) * bg_lambda  + roi_features_list[i][rel_pair_idxs[i][head_idx, 0]] * (1 - bg_lambda)
                                    roi_features_list[i][rel_pair_idxs[i][head_idx, 1]] = obj_feature.to(device) * bg_lambda  + roi_features_list[i][rel_pair_idxs[i][head_idx, 1]] * (1 - bg_lambda)
                                    union_features_list[i][head_idx] = union_feature.to(device) * bg_lambda  + union_features_list[i][head_idx] * (1 - bg_lambda)
                                    changed_obj_idxs.append(rel_pair_idxs[i][head_idx, 0].item())
                                    changed_obj_idxs.append(rel_pair_idxs[i][head_idx, 1].item())


                    if mix_up_add_tail == True:

                        triplets_len = len(rel_labels[i])
                        repeat_p = ((image_repeats[i] - 1) / image_repeats[i]) * 0.5
                        for idx in range(triplets_len):
                            if rel_labels[i][idx] in self.TAIL and torch.sum(rel_labels_one_hot[i][idx] ** 2, dim=0) == 1:  
                                tail_rel_label = rel_labels[i][idx]
                                sub_name = pair_labels[i][idx].split('_')[0]
                                obj_name = pair_labels[i][idx].split('_')[-1]
                                sub_tail_name = sub_name + '_' + str(tail_rel_label.cpu())
                                obj_tail_name = obj_name + '_' + str(tail_rel_label.cpu())

                                # change tail obj
                                if sub_tail_name in self.tail_obj_dict.keys() and random.random() < repeat_p:
                                    sub_bbox = proposals[i].bbox.cpu().numpy()[rel_pair_idxs[i][idx][0]]
                                    obj_bbox = proposals[i].bbox.cpu().numpy()[rel_pair_idxs[i][idx][1]]      
                                    curri_vector = np.array([(sub_bbox[2] + sub_bbox[0]) / 2. - (obj_bbox[2] + obj_bbox[0]) / 2., (sub_bbox[3] + sub_bbox[1]) / 2. - (obj_bbox[3] + obj_bbox[1]) / 2.]) 
                                    filter_objs = []

                                    for tail_obj_name in self.tail_obj_dict[sub_tail_name].keys():
                                        if self.high_level_obj_dict[int(self.obj_to_id_dict[obj_name])] == self.high_level_obj_dict[int(self.obj_to_id_dict[tail_obj_name])]:
                                            for (_, obj_feature, tail_obj_proposal, _, vector, _) in self.tail_obj_dict[sub_tail_name][tail_obj_name]:
                                                dist = curri_vector.dot(vector) / np.linalg.norm(curri_vector) * np.linalg.norm(vector)

                                                if dist > dist_t:
                                                    filter_objs.append((obj_feature, tail_obj_proposal))
                                    if len(filter_objs) != 0:
                                        random_idx = int(random.random() * len(filter_objs))
                                        (obj_feature, tail_obj_proposal) = filter_objs[random_idx]

                                        proposals[i].get_field('labels')[rel_pair_idxs[i][idx][1]] = tail_obj_proposal.get_field('labels')[0]
                                        roi_features_list[i][rel_pair_idxs[i][idx, 1]] = obj_feature.to(device) 
                                        if not self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                                            proposals[i].get_field("predict_logits")[rel_pair_idxs[i][idx, 1]] = tail_obj_proposal.get_field("predict_logits")[0]

                                # change tail sub
                                if obj_tail_name in self.tail_sub_dict.keys() and random.random() < repeat_p:
                                    sub_bbox = proposals[i].bbox.cpu().numpy()[rel_pair_idxs[i][idx][0]]
                                    obj_bbox = proposals[i].bbox.cpu().numpy()[rel_pair_idxs[i][idx][1]]      
                                    curri_vector = np.array([(sub_bbox[2] + sub_bbox[0]) / 2. - (obj_bbox[2] + obj_bbox[0]) / 2., (sub_bbox[3] + sub_bbox[1]) / 2. - (obj_bbox[3] + obj_bbox[1]) / 2.]) 
                                    filter_subs = []

                                    for tail_sub_name in self.tail_sub_dict[obj_tail_name].keys():
                                        if self.high_level_obj_dict[int(self.obj_to_id_dict[sub_name])] == self.high_level_obj_dict[int(self.obj_to_id_dict[tail_sub_name])]:
                                            for (_, sub_feature, tail_sub_proposal, _, vector, _) in self.tail_sub_dict[obj_tail_name][tail_sub_name]:
                                                dist = curri_vector.dot(vector) / np.linalg.norm(curri_vector) * np.linalg.norm(vector)

                                                if dist > dist_t:
                                                    filter_subs.append((sub_feature, tail_sub_proposal))
                                    if len(filter_subs) != 0:
                                        random_idx = int(random.random() * len(filter_subs))
                                        (sub_feature, tail_sub_proposal) = filter_subs[random_idx]
                                        proposals[i].get_field('labels')[rel_pair_idxs[i][idx][0]] = tail_sub_proposal.get_field('labels')[0]
                                        roi_features_list[i][rel_pair_idxs[i][idx, 0]] = sub_feature.to(device) 
                                        if not self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                                            proposals[i].get_field("predict_logits")[rel_pair_idxs[i][idx, 0]] = tail_sub_proposal.get_field("predict_logits")[0]

                    changed_obj_idxs = list(set(changed_obj_idxs))
                    all_changed_obj_idxs.append(torch.tensor(changed_obj_idxs))

                else:  
                    all_changed_idxs = []
                    changed_obj_idxs = []
                    changed_pred_idxs = []
                    changed_classes = []
                    change_to_classes = []
                    if mix_up_fg == True:
                    # mixup head to tail
                        if self.cfg.FG_TAIL == True:
                            if np.any(np.array(rel_labels[i].cpu()) < 0):
                                filter_head_to_tail_idxs = np.where(rel_labels[i].cpu().numpy() < 0)[0]
                                proposals_change, union_features_list_change, roi_features_list_change, rel_labels_change, rel_pair_idxs_change, rel_binarys_change, rel_labels_one_hot_change, pair_labels_change, changed_idxs, changed_obj_idxs, changed_classes, change_to_classes = self.mixup(device, proposals[i], union_features_list[i], roi_features_list[i], rel_labels[i], rel_pair_idxs[i], rel_binarys[i], rel_labels_one_hot[i], pair_labels[i], filter_head_to_tail_idxs, self.tail_dict, self.tail_tri_map_dict, mixup_lambda=fg_lambda, dist_t=dist_t)
                                proposals[i], union_features_list[i], roi_features_list[i], rel_labels[i], rel_pair_idxs[i], rel_binarys[i], rel_labels_one_hot[i], pair_labels[i] = proposals_change, union_features_list_change, roi_features_list_change, rel_labels_change, rel_pair_idxs_change, rel_binarys_change, rel_labels_one_hot_change, pair_labels_change
                                all_changed_idxs.extend(changed_idxs)
                                changed_obj_idxs.extend(changed_obj_idxs)
                                changed_classes.extend(changed_classes)
                                change_to_classes.extend(change_to_classes)
                    # mixup head to body
                        if self.cfg.FG_BODY == True:
                            if np.any(np.array(rel_labels[i].cpu()) < 0):
                                filter_head_to_body_idxs = np.where(rel_labels[i].cpu().numpy() < 0)[0]
                                proposals_change, union_features_list_change, roi_features_list_change, rel_labels_change, rel_pair_idxs_change, rel_binarys_change, rel_labels_one_hot_change, pair_labels_change, changed_idxs, changed_obj_idxs, changed_classes, change_to_classes = self.mixup(device, proposals[i], union_features_list[i], roi_features_list[i], rel_labels[i], rel_pair_idxs[i], rel_binarys[i], rel_labels_one_hot[i], pair_labels[i], filter_head_to_body_idxs, self.body_dict, self.body_tri_map_dict, mixup_lambda=fg_lambda, dist_t=dist_t)
                                proposals[i], union_features_list[i], roi_features_list[i], rel_labels[i], rel_pair_idxs[i], rel_binarys[i], rel_labels_one_hot[i], pair_labels[i] = proposals_change, union_features_list_change, roi_features_list_change, rel_labels_change, rel_pair_idxs_change, rel_binarys_change, rel_labels_one_hot_change, pair_labels_change
                                all_changed_idxs.extend(changed_idxs)
                                changed_obj_idxs.extend(changed_obj_idxs)
                                changed_classes.extend(changed_classes)
                                change_to_classes.extend(change_to_classes)
                    # # mixup head to head
                        if self.cfg.FG_HEAD == True:
                            if np.any(np.array(rel_labels[i].cpu()) < 0):
                                filter_head_to_head_idxs = np.where(rel_labels[i].cpu().numpy() < 0)[0]
                                proposals[i], union_features_list[i], roi_features_list[i], rel_labels[i], rel_pair_idxs[i], rel_labels_one_hot[i], pair_labels[i], changed_idxs, changed_obj_idxs, changed_idxs, changed_obj_idxs = self.mixup(device, proposals[i], union_features_list[i], roi_features_list[i], rel_labels[i], rel_pair_idxs[i], rel_labels_one_hot[i], pair_labels[i], filter_head_to_head_idxs, self.head_dict, self.head_tri_map_dict, mixup_lambda=fg_lambda, dist_t=dist_t)
                                all_changed_idxs.extend(changed_idxs)
                                changed_obj_idxs.extend(changed_obj_idxs)

                    if mix_up_bg == True:
                        bg_lambda = self.bg_lambda
                        filter_bg_idxs = torch.where(rel_labels[i] == 0)[0].cpu().numpy()
                        # mixup bg to tail
                        if self.cfg.BG_TAIL == True:
                            filter_bg_to_tail_idxs = np.random.choice(filter_bg_idxs, int(1.0 * len(filter_bg_idxs)))
                            proposals_change, union_features_list_change, roi_features_list_change, rel_labels_change, rel_pair_idxs_change, rel_binarys_change, rel_labels_one_hot_change, pair_labels_change, changed_idxs, changed_obj_idxs, changed_classes, change_to_classes = self.mixup(device, proposals[i], union_features_list[i], roi_features_list[i], rel_labels[i], rel_pair_idxs[i], rel_binarys[i], rel_labels_one_hot[i], pair_labels[i], filter_bg_to_tail_idxs, self.tail_dict, self.tail_tri_map_dict, mixup_lambda=bg_lambda, dist_t=dist_t)
                            proposals[i], union_features_list[i], roi_features_list[i], rel_labels[i], rel_pair_idxs[i], rel_binarys[i], rel_labels_one_hot[i], pair_labels[i] = proposals_change, union_features_list_change, roi_features_list_change, rel_labels_change, rel_pair_idxs_change, rel_binarys_change, rel_labels_one_hot_change, pair_labels_change
                            all_changed_idxs.extend(changed_idxs)
                            changed_obj_idxs.extend(changed_obj_idxs)
                            changed_classes.extend(changed_classes)
                            change_to_classes.extend(change_to_classes)
                            filter_bg_idxs = np.setdiff1d(filter_bg_idxs, all_changed_idxs)
                        # mixup bg to body
                        if self.cfg.BG_BODY == True:
                            filter_bg_to_body_idxs = np.random.choice(filter_bg_idxs, int(0.5 * len(filter_bg_idxs)))
                            proposals[i], union_features_list[i], roi_features_list[i], rel_labels[i], rel_pair_idxs[i], rel_labels_one_hot[i], pair_labels[i], changed_idxs, changed_obj_idxs = self.mixup(device, proposals[i], union_features_list[i], roi_features_list[i], rel_labels[i], rel_pair_idxs[i], rel_labels_one_hot[i], pair_labels[i], filter_bg_to_body_idxs, self.body_dict, self.body_tri_map_dict, mixup_lambda=bg_lambda, dist_t=dist_t)
                            all_changed_idxs.extend(changed_idxs)
                            changed_obj_idxs.extend(changed_obj_idxs)
                            filter_bg_idxs = np.setdiff1d(filter_bg_idxs, all_changed_idxs)
                        # mixup bg to head
                        if self.cfg.BG_HEAD == True:
                            filter_bg_idxs = np.setdiff1d(filter_bg_idxs, filter_bg_to_body_idxs)
                            filter_bg_to_head_idxs = np.random.choice(filter_bg_idxs, int(0.5 * len(filter_bg_idxs)))
                            proposals[i], union_features_list[i], roi_features_list[i], rel_labels[i], rel_pair_idxs[i], rel_labels_one_hot[i], pair_labels[i], changed_idxs, changed_obj_idxs  = self.mixup(device, proposals[i], union_features_list[i], roi_features_list[i], rel_labels[i], rel_pair_idxs[i], rel_labels_one_hot[i], pair_labels[i], filter_bg_to_head_idxs, self.head_dict, self.head_tri_map_dict, mixup_lambda=bg_lambda, dist_t=dist_t)
                            all_changed_idxs.extend(changed_idxs)
                            changed_obj_idxs.extend(changed_obj_idxs)
                            filter_bg_idxs = np.setdiff1d(filter_bg_idxs, all_changed_idxs)
                    if mix_up_add_tail == True:

                        # compositional change tail
                        if self.cfg.CL_TAIL == True:
                            p = 0.5
                            filter_tail_idxs = [idx for idx in range(len(pair_labels[i])) if rel_labels[i][idx] in self.TAIL]
                            filter_tail_idxs = np.setdiff1d(np.array(filter_tail_idxs), np.array(all_changed_idxs))
                            if len(filter_tail_idxs) > 0: 
                                proposals_change, union_features_list_change, roi_features_list_change, rel_labels_change, rel_pair_idxs_change, rel_labels_one_hot_change, pair_labels_change, changed_idxs, changed_pred_idx = self.compositional_change_sub(device, proposals[i], union_features_list[i], roi_features_list[i], rel_labels[i], rel_pair_idxs[i], rel_labels_one_hot[i], pair_labels[i], image_repeats[i], filter_tail_idxs, self.tail_dict, self.tail_tri_map_dict, self.high_level_obj_dict, p=p)
                                proposals[i], union_features_list[i], roi_features_list[i], rel_labels[i], rel_pair_idxs[i], rel_labels_one_hot[i], pair_labels[i] = proposals_change, union_features_list_change, roi_features_list_change, rel_labels_change, rel_pair_idxs_change, rel_labels_one_hot_change, pair_labels_change
                                changed_pred_idxs.extend(changed_pred_idx)
                                proposals_change, union_features_list_change, roi_features_list_change, rel_labels_change, rel_pair_idxs_change, rel_labels_one_hot_change, pair_labels_change, changed_idxs, changed_pred_idx = self.compositional_change_obj(device, proposals[i], union_features_list[i], roi_features_list[i], rel_labels[i], rel_pair_idxs[i], rel_labels_one_hot[i], pair_labels[i], image_repeats[i], filter_tail_idxs, self.tail_dict, self.tail_tri_map_dict, self.high_level_obj_dict, p=p)
                                proposals[i], union_features_list[i], roi_features_list[i], rel_labels[i], rel_pair_idxs[i], rel_labels_one_hot[i], pair_labels[i] = proposals_change, union_features_list_change, roi_features_list_change, rel_labels_change, rel_pair_idxs_change, rel_labels_one_hot_change, pair_labels_change
                                changed_pred_idxs.extend(changed_pred_idx)
                        # compositional change body
                        if self.cfg.CL_BODY == True:
                            filter_body_idxs = [idx for idx in range(len(pair_labels[i])) if rel_labels[i][idx] in self.BODY]
                            filter_body_idxs = np.setdiff1d(np.array(filter_body_idxs), np.array(all_changed_idxs))
                            if len(filter_body_idxs) > 0: 
                                filter_body_idxs = np.random.choice(filter_body_idxs, int(0.5 * len(filter_body_idxs)))
                            p = 0.5
                            if len(filter_body_idxs) > 0: 
                                proposals_change, union_features_list_change, roi_features_list_change, rel_labels_change, rel_pair_idxs_change, rel_labels_one_hot_change, pair_labels_change, changed_idxs, _ = self.compositional_change_sub(device, proposals[i], union_features_list[i], roi_features_list[i], rel_labels[i], rel_pair_idxs[i], rel_labels_one_hot[i], pair_labels[i], image_repeats[i], filter_body_idxs, self.body_dict, self.body_tri_map_dict, self.high_level_obj_dict, p=p)
                                proposals[i], union_features_list[i], roi_features_list[i], rel_labels[i], rel_pair_idxs[i], rel_labels_one_hot[i], pair_labels[i] = proposals_change, union_features_list_change, roi_features_list_change, rel_labels_change, rel_pair_idxs_change, rel_labels_one_hot_change, pair_labels_change
                                proposals_change, union_features_list_change, roi_features_list_change, rel_labels_change, rel_pair_idxs_change, rel_labels_one_hot_change, pair_labels_change, changed_idxs, _ = self.compositional_change_obj(device, proposals[i], union_features_list[i], roi_features_list[i], rel_labels[i], rel_pair_idxs[i], rel_labels_one_hot[i], pair_labels[i], image_repeats[i], filter_body_idxs, self.body_dict, self.body_tri_map_dict, self.high_level_obj_dict, p=p)
                                proposals[i], union_features_list[i], roi_features_list[i], rel_labels[i], rel_pair_idxs[i], rel_labels_one_hot[i], pair_labels[i] = proposals_change, union_features_list_change, roi_features_list_change, rel_labels_change, rel_pair_idxs_change, rel_labels_one_hot_change, pair_labels_change

                        # compositional change head
                        if self.cfg.CL_HEAD == True:
                            filter_head_idxs = [idx for idx in range(len(pair_labels[i])) if rel_labels[i][idx] in self.HEAD]
                            filter_head_idxs = np.setdiff1d(np.array(filter_head_idxs), np.array(all_changed_idxs))
                            if len(filter_head_idxs) > 0:
                                filter_head_idxs = np.random.choice(filter_head_idxs, int(0.25 * len(filter_body_idxs)))
                            p = 0.005
                            if len(filter_head_idxs) > 0: 
                                proposals_change, union_features_list_change, roi_features_list_change, rel_labels_change, rel_pair_idxs_change, rel_labels_one_hot_change, pair_labels_change, changed_idxs = self.compositional_change_sub(device, proposals[i], union_features_list[i], roi_features_list[i], rel_labels[i], rel_pair_idxs[i], rel_labels_one_hot[i], pair_labels[i], image_repeats[i], filter_head_idxs, self.head_dict, self.head_tri_map_dict, self.high_level_obj_dict, p=p)
                                proposals[i], union_features_list[i], roi_features_list[i], rel_labels[i], rel_pair_idxs[i], rel_labels_one_hot[i], pair_labels[i] = proposals_change, union_features_list_change, roi_features_list_change, rel_labels_change, rel_pair_idxs_change, rel_labels_one_hot_change, pair_labels_change
                                proposals_change, union_features_list_change, roi_features_list_change, rel_labels_change, rel_pair_idxs_change, rel_labels_one_hot_change, pair_labels_change, changed_idxs = self.compositional_change_obj(device, proposals[i], union_features_list[i], roi_features_list[i], rel_labels[i], rel_pair_idxs[i], rel_labels_one_hot[i], pair_labels[i], image_repeats[i], filter_head_idxs, self.head_dict, self.head_tri_map_dict, self.high_level_obj_dict, p=p)
                                proposals[i], union_features_list[i], roi_features_list[i], rel_labels[i], rel_pair_idxs[i], rel_labels_one_hot[i], pair_labels[i] = proposals_change, union_features_list_change, roi_features_list_change, rel_labels_change, rel_pair_idxs_change, rel_labels_one_hot_change, pair_labels_change
                        # filter rel label -1
                    
                    if np.any(np.array(rel_labels[i].cpu()) < 0):

                        filter_idxs = torch.where(rel_labels[i] >= 0)[0]
                        rel_labels[i] = rel_labels[i][filter_idxs]
                        rel_pair_idxs[i] = rel_pair_idxs[i][filter_idxs]
                        union_features_list[i] = union_features_list[i][filter_idxs]
                        rect_inputs[i] = rect_inputs[i][filter_idxs]
                        rel_labels_one_hot[i] = rel_labels_one_hot[i][filter_idxs]
                        pair_labels[i] = pair_labels[i][filter_idxs.cpu()]

                    
                    filter_tail_idxs = [idx for idx in range(len(pair_labels[i])) if rel_labels[i][idx] in self.TAIL]
                    filter_tail_idxs = np.setdiff1d(np.array(filter_tail_idxs), np.array(all_changed_idxs))
                    changed_obj_idxs = list(set(changed_obj_idxs))
                    changed_pred_idxs = list(set(changed_pred_idxs))
                    all_changed_obj_idxs.append(torch.tensor(changed_obj_idxs))
                    all_changed_pred_idxs.append(torch.tensor(changed_pred_idxs))
                    all_filter_tail_idxs.append(torch.tensor(filter_tail_idxs))
                    all_changed_classes.append(torch.tensor(changed_classes))
                    all_change_to_classes.append(torch.tensor(change_to_classes))

        union_vis_features = cat(union_features_list, dim=0)
        roi_features = cat(roi_features_list, dim=0)
        union_vis_features = Variable(union_vis_features, requires_grad=False)
        roi_features = Variable(roi_features, requires_grad=False)
        rect_inputs = torch.cat(rect_inputs, dim=0)
        rect_features = self.rect_conv(rect_inputs)

        # merge two parts
        if self.separate_spatial:
            region_features = self.feature_extractor.forward_without_pool(union_vis_features)
            spatial_features = self.spatial_fc(rect_features.view(rect_features.size(0), -1))
            union_features = (region_features, spatial_features)
        else:
            union_features = union_vis_features + rect_features
            union_features = self.feature_extractor.forward_without_pool(
                union_features)  # (total_num_rel, out_channels)

        if self.cfg.MODEL.ATTRIBUTE_ON:
            union_att_features = self.att_feature_extractor.pooler(x, union_proposals)
            union_features_att = union_att_features + rect_features
            union_features_att = self.att_feature_extractor.forward_without_pool(union_features_att)
            union_features = torch.cat((union_features, union_features_att), dim=-1)

        rel_labels

        return proposals, union_features, roi_features, rel_labels, rel_pair_idxs, rel_binarys, rel_labels_one_hot, all_changed_pred_idxs, all_changed_obj_idxs, all_filter_tail_idxs, all_changed_classes, all_change_to_classes

    def mixup(self, device, proposals, union_features, roi_features, rel_labels, rel_pair_idxs, rel_binarys, rel_labels_one_hot, pair_labels, filter_idxs, changed_dict, tri_map_dict, mixup_lambda=0.5, dist_t=0.7):  
        changed_idxs = []
        changed_obj_idxs = []
        changed_classes = []
        change_to_classes = []
        for idx in filter_idxs:
            if pair_labels[idx] in tri_map_dict['pred'].keys():
                tri_keys = tri_map_dict['pred'][pair_labels[idx]]
            else:
                tri_keys = []
            if len(tri_keys) > 0:
                sub_bbox = proposals.bbox.cpu().numpy()[rel_pair_idxs[idx][0]]
                obj_bbox = proposals.bbox.cpu().numpy()[rel_pair_idxs[idx][1]]      
                curri_vector = np.array([(sub_bbox[2] + sub_bbox[0]) / 2. - (obj_bbox[2] + obj_bbox[0]) / 2., (sub_bbox[3] + sub_bbox[1]) / 2. - (obj_bbox[3] + obj_bbox[1]) / 2.]) 
                filter_pairs = []
                for tri_key in tri_keys:
                    for (rel_label, sub_feature, obj_feature, sub_proposal, obj_proposal, union_feature, vector, wh) in changed_dict[tri_key]:
                        dist = curri_vector.dot(vector) / np.linalg.norm(curri_vector) * np.linalg.norm(vector)
                        if dist > dist_t:
                            filter_pairs.append((rel_label, sub_feature, obj_feature, union_feature))
                if len(filter_pairs) > 0:
                    random_idx = int(random.random() * len(filter_pairs))
                    (rel_label, sub_feature, obj_feature, union_feature) = filter_pairs[random_idx]
                    # print('----mixup----', pair_labels[idx], 'original:', id_to_rel_dict[abs(rel_labels[idx].item())], 'new:', id_to_rel_dict[rel_label.item()])
                    changed_classes.append(rel_labels[idx])
                    if rel_labels[idx] > 0:
                        rel_binarys[rel_pair_idxs[idx, 0], rel_pair_idxs[idx, 1]] = 1
                    rel_labels[idx] = rel_label.to(device)
                    rel_label_one_hot = F.one_hot(rel_label.to(device), cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES).float()
                    rel_labels_one_hot[idx] = mixup_lambda * rel_label_one_hot + (1 - mixup_lambda) * rel_labels_one_hot[idx]
                    roi_features[rel_pair_idxs[idx, 0]] = sub_feature.to(device) * mixup_lambda + roi_features[rel_pair_idxs[idx, 0]] * (1 - mixup_lambda)
                    roi_features[rel_pair_idxs[idx, 1]] = obj_feature.to(device) * mixup_lambda + roi_features[rel_pair_idxs[idx, 1]] * (1 - mixup_lambda)
                    union_features[idx] = union_feature.to(device) * mixup_lambda + union_features[idx] * (1 - mixup_lambda)
                    changed_idxs.append(idx)
                    changed_obj_idxs.append(rel_pair_idxs[idx, 0].item())
                    changed_obj_idxs.append(rel_pair_idxs[idx, 1].item())
                    change_to_classes.append(rel_labels[idx])
        return proposals, union_features, roi_features, rel_labels, rel_pair_idxs, rel_binarys, rel_labels_one_hot, pair_labels, changed_idxs, changed_obj_idxs, changed_classes, change_to_classes

    def compositional_change_sub(self, device, proposals, union_features, roi_features, rel_labels, rel_pair_idxs, rel_labels_one_hot, pair_labels, image_repeats, filter_idxs, changed_dict, tri_map_dict, high_level_obj_dict, dist_t=0.7, p=0.5):
        repeat_p = ((image_repeats - 1) / image_repeats) * p
        changed_idxs = []
        changed_sub_idxs = []

        for i in range(len(filter_idxs)):  
            idx = filter_idxs[i]
            sub, obj = pair_labels[idx].split('_')[0], pair_labels[idx].split('_')[1]
            if str(rel_labels[idx].item()) + '_' + obj in tri_map_dict['sub'].keys():
                sub_tri_keys = tri_map_dict['sub'][str(rel_labels[idx].item()) + '_' + obj]
            else:
                sub_tri_keys = []

            if len(sub_tri_keys) > 0 and random.random() < repeat_p:
                sub_bbox = proposals.bbox.cpu().numpy()[rel_pair_idxs[idx][0]]
                obj_bbox = proposals.bbox.cpu().numpy()[rel_pair_idxs[idx][1]]      
                curri_vector = np.array([(sub_bbox[2] + sub_bbox[0]) / 2. - (obj_bbox[2] + obj_bbox[0]) / 2., (sub_bbox[3] + sub_bbox[1]) / 2. - (obj_bbox[3] + obj_bbox[1]) / 2.]) 
                filter_subs = []
                for (tri_key, new_sub) in sub_tri_keys:
                    if high_level_obj_dict[int(self.obj_to_id_dict[sub])] == high_level_obj_dict[int(self.obj_to_id_dict[new_sub])]:
                        for (rel_label, sub_feature, obj_feature, sub_proposal, obj_proposal, union_feature, vector, wh) in changed_dict[tri_key]:
                            # dist = curri_vector.dot(vector) / (np.linalg.norm(curri_vector) * np.linalg.norm(vector) + 1e-3)
                            dist = curri_vector.dot(vector) / np.linalg.norm(curri_vector) * np.linalg.norm(vector)
                            if dist > dist_t:
                                filter_subs.append((sub_feature, sub_proposal, union_feature))
                if len(filter_subs) > 0:
                    random_idx = int(random.random() * len(filter_subs))
                    (sub_feature, sub_proposal, union_feature) = filter_subs[random_idx]
                    # print(sub, obj, id_to_rel_dict[rel_label.item()], 'new_sub:', id_object_dict[str(sub_proposal.get_field('labels')[0].item())])
                    proposals.get_field('labels')[rel_pair_idxs[idx][0]] = sub_proposal.get_field('labels')[0]
                    if not self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                        proposals.get_field("predict_logits")[rel_pair_idxs[idx][0]] = sub_proposal.get_field("predict_logits")[0]
                    roi_features[rel_pair_idxs[idx, 0]] = sub_feature.to(device) 
                    # union_features[i][idx] = 0.5 * union_features[i][idx] + 0.5 * union_feature.to(device)
                    changed_idxs.append(idx)
                    changed_sub_idxs.append(i)

            return proposals, union_features, roi_features, rel_labels, rel_pair_idxs, rel_labels_one_hot, pair_labels, changed_idxs, changed_sub_idxs


    def compositional_change_obj(self, device, proposals, union_features, roi_features, rel_labels, rel_pair_idxs, rel_labels_one_hot, pair_labels, image_repeats, filter_idxs, changed_dict, tri_map_dict, high_level_obj_dict, dist_t=0.7, p=0.5):
        repeat_p = ((image_repeats - 1) / image_repeats) * p
        changed_idxs = []
        changed_obj_idxs = []
        for i in range(len(filter_idxs)):  
            idx = filter_idxs[i]
            sub, obj = pair_labels[idx].split('_')[0], pair_labels[idx].split('_')[1]

            if sub + '_' + str(rel_labels[idx].item()) in tri_map_dict['obj'].keys():
                obj_tri_keys = tri_map_dict['obj'][sub + '_' + str(rel_labels[idx].item())]
            else:
                obj_tri_keys = []

            if len(obj_tri_keys) > 0 and random.random() < repeat_p:
                sub_bbox = proposals.bbox.cpu().numpy()[rel_pair_idxs[idx][0]]
                obj_bbox = proposals.bbox.cpu().numpy()[rel_pair_idxs[idx][1]]      
                curri_vector = np.array([(sub_bbox[2] + sub_bbox[0]) / 2. - (obj_bbox[2] + obj_bbox[0]) / 2., (sub_bbox[3] + sub_bbox[1]) / 2. - (obj_bbox[3] + obj_bbox[1]) / 2.]) 
                filter_objs = []
                for (tri_key, new_obj) in obj_tri_keys:
                    if high_level_obj_dict[int(self.obj_to_id_dict[obj])] == high_level_obj_dict[int(self.obj_to_id_dict[new_obj])]:
                        for (rel_label, sub_feature, obj_feature, sub_proposal, obj_proposal, union_feature, vector, wh) in changed_dict[tri_key]:
                            dist = curri_vector.dot(vector) / np.linalg.norm(curri_vector) * np.linalg.norm(vector)
                            if dist > dist_t:
                                filter_objs.append((obj_feature, obj_proposal, union_feature))
                if len(filter_objs) > 0:
                    random_idx = int(random.random() * len(filter_objs))
                    (obj_feature, obj_proposal, union_feature) = filter_objs[random_idx]
                    # print(sub, obj, id_to_rel_dict[rel_label.item()], 'new_obj:', id_object_dict[str(obj_proposal.get_field('labels')[0].item())])
                    proposals.get_field('labels')[rel_pair_idxs[idx][0]] = obj_proposal.get_field('labels')[0]
                    if not self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                        proposals.get_field("predict_logits")[rel_pair_idxs[idx][1]] = obj_proposal.get_field("predict_logits")[0]
                    roi_features[rel_pair_idxs[idx, 1]] = obj_feature.to(device) 
                    changed_idxs.append(idx)
                    changed_obj_idxs.append(i)
                    # union_features[i][idx] = 0.5 * union_features[i][idx] + 0.5 * union_feature.to(device)

            return proposals, union_features, roi_features, rel_labels, rel_pair_idxs, rel_labels_one_hot, pair_labels, changed_idxs, changed_obj_idxs
  
  
@registry.ROI_RELATION_FEATURE_EXTRACTORS.register("UnionFeatureExtractor")
class UnionFeatureExtractor(nn.Module):
    """
    Heads for Motifs for relation triplet classification
    """

    def __init__(self, cfg, in_channels):
        super(UnionFeatureExtractor, self).__init__()
        self.cfg = cfg.clone()
        # should corresponding to obj_feature_map function in neural-motifs
        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pool_all_levels = cfg.MODEL.ROI_RELATION_HEAD.POOLING_ALL_LEVELS
        if cfg.MODEL.ATTRIBUTE_ON:
            self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels, half_out=True,
                                                                    cat_all_levels=pool_all_levels)
            self.att_feature_extractor = make_roi_attribute_feature_extractor(cfg, in_channels, half_out=True,
                                                                              cat_all_levels=pool_all_levels)
            self.out_channels = self.feature_extractor.out_channels * 2
        else:
            self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels, cat_all_levels=pool_all_levels)
            self.out_channels = self.feature_extractor.out_channels

        # separete spatial
        self.separate_spatial = self.cfg.MODEL.ROI_RELATION_HEAD.CAUSAL.SEPARATE_SPATIAL
        if self.separate_spatial:
            input_size = self.feature_extractor.resize_channels
            out_dim = self.feature_extractor.out_channels

    def forward(self, x, proposals, rel_pair_idxs=None):
        device = x[0].device
        union_proposals = []
        rect_inputs = []
        for proposal, rel_pair_idx in zip(proposals, rel_pair_idxs):
            # print(len(proposal), rel_pair_idx[:, 0])
            head_proposal = proposal[rel_pair_idx[:, 0]]
            tail_proposal = proposal[rel_pair_idx[:, 1]]
            union_proposal = boxlist_union(head_proposal, tail_proposal)
            union_proposals.append(union_proposal)

        # union visual feature. size (total_num_rel, in_channels, POOLER_RESOLUTION, POOLER_RESOLUTION)
        union_vis_features = self.feature_extractor.pooler(x, union_proposals)
        return union_vis_features

def make_roi_relation_feature_extractor(cfg, in_channels):
    func = registry.ROI_RELATION_FEATURE_EXTRACTORS[
        cfg.MODEL.ROI_RELATION_HEAD.FEATURE_EXTRACTOR
    ]
    return func(cfg, in_channels)

def make_aug_roi_relation_feature_extractor(cfg, in_channels):
    func = registry.ROI_RELATION_FEATURE_EXTRACTORS[
        "AugRelationFeatureExtractor"
    ]
    return func(cfg, in_channels)

def make_union_roi_feature_extractor(cfg, in_channels):
    func = registry.ROI_RELATION_FEATURE_EXTRACTORS[
        "UnionFeatureExtractor"
    ]
    return func(cfg, in_channels)

def make_aug_bilvl_mixup_roi_relation_feature_extractor(cfg, in_channels):
    func = registry.ROI_RELATION_FEATURE_EXTRACTORS[
        "AugBilvlMxiUpRelationFeatureExtractor"
    ]
    return func(cfg, in_channels)
