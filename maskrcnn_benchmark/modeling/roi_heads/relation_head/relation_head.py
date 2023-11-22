# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from re import S
import torch
from torch import nn
import numpy as np

from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from ..attribute_head.roi_attribute_feature_extractors import make_roi_attribute_feature_extractor
from ..box_head.roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_relation_feature_extractors import make_roi_relation_feature_extractor, make_union_roi_feature_extractor, make_aug_bilvl_mixup_roi_relation_feature_extractor, make_aug_roi_relation_feature_extractor
from .roi_relation_predictors import make_roi_relation_predictor
from .inference import make_roi_relation_post_processor, make_roi_relation_pko_post_processor
from .loss import make_roi_relation_loss_evaluator, make_roi_relation_contra_loss_evaluator
from .sampling import make_roi_relation_samp_processor
from maskrcnn_benchmark.config import cfg
from torch.nn import functional as F
from maskrcnn_benchmark.modeling.utils import cat
import copy

from maskrcnn_benchmark.modeling.roi_heads.box_head.roi_box_feature_extractors import make_roi_box_feature_extractor



class ROIRelationHead(torch.nn.Module):
    """
    Generic Relation Head class.
    """

    def __init__(self, cfg, in_channels):
        super(ROIRelationHead, self).__init__()
        self.cfg = cfg.clone()
        # same structure with box head, but different parameters
        # these param will be trained in a slow learning rate, while the parameters of box head will be fixed
        # Note: there is another such extractor in uniton_feature_extractor
        if self.cfg.TYPE == 'cfa' or self.cfg.TYPE == 'extract_cfa_feat':
            self.union_feature_extractor = make_aug_bilvl_mixup_roi_relation_feature_extractor(cfg, in_channels)
        elif self.cfg.TYPE == 'extract_aug':
            self.union_feature_extractor = make_union_roi_feature_extractor(cfg, in_channels)
        else:
            self.union_feature_extractor = make_roi_relation_feature_extractor(cfg, in_channels)
        if cfg.MODEL.ATTRIBUTE_ON:
            self.box_feature_extractor = make_roi_box_feature_extractor(cfg, in_channels, half_out=True)
            self.att_feature_extractor = make_roi_attribute_feature_extractor(cfg, in_channels, half_out=True)
            feat_dim = self.box_feature_extractor.out_channels * 2
        else:
            self.box_feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
            feat_dim = self.box_feature_extractor.out_channels
        self.predictor = make_roi_relation_predictor(cfg, feat_dim)
        self.post_processor = make_roi_relation_post_processor(cfg)
        if self.cfg.PKO == True:
            self.post_processor = make_roi_relation_pko_post_processor(cfg)
        self.loss_evaluator = make_roi_relation_loss_evaluator(cfg)
        if self.cfg.TYPE == 'cfa' and self.cfg.CONTRA == True:
            self.loss_evaluator = make_roi_relation_contra_loss_evaluator(cfg)
        else:
            self.loss_evaluator = make_roi_relation_loss_evaluator(cfg)
        self.samp_processor = make_roi_relation_samp_processor(cfg)

        self.cur_bg_c = 0.0
        self.count_dict = {}
        self.num_obj_cls = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_rel_cls = cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        # parameters
        self.use_union_box = self.cfg.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.tail_features_dict = {}
        self.cfa_features_dict = {}
        self.origial_features_dict = {}

    def forward(self, features, proposals, targets=None, logger=None, cur_iter=None):
        if cfg.SELECT_DATASET == 'VG':
            HEAD = cfg.HEAD_IDS
            BODY = cfg.BODY_IDS
            TAIL = cfg.TAIL_IDS
        elif cfg.SELECT_DATASET == 'GQA':
            HEAD = cfg.GQA_HEAD_IDS
            BODY = cfg.GQA_BODY_IDS
            TAIL = cfg.GQA_TAIL_IDS
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes. Note: it has been post-processed (regression, nms) in sgdet mode
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """
        pair_labels = None
        drop_pair_lens = None
        image_repeats = None
        all_changed_classes=None
        all_change_to_classes=None
        if self.training:
            # relation subsamples and assign ground truth label during training
            with torch.no_grad():
                if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
                        if self.cfg.TYPE == 'cfa' or self.cfg.TYPE == 'extract_cfa_feat':
                            proposals, rel_labels, rel_pair_idxs, rel_binarys, pair_labels, drop_pair_lens, image_repeats = self.samp_processor.gtbox_memory_bank_cfa(proposals, targets)
                        else:
                            proposals, rel_labels, rel_pair_idxs, rel_binarys = self.samp_processor.gtbox_relsample(proposals, targets)
                else:
                    if self.cfg.TYPE == 'cfa':
                        proposals, rel_labels, rel_pair_idxs, rel_binarys, pair_labels, image_repeats = self.samp_processor.detect_memory_bank_cfa(proposals, targets)
                    else:
                        proposals, rel_labels, rel_pair_idxs, rel_binarys = self.samp_processor.detect_relsample(proposals, targets)
        else:
            rel_labels, rel_binarys = None, None
            if self.cfg.TYPE == 'extract_aug':
                proposals, rel_labels, rel_pair_idxs, rel_binarys, pair_labels = self.samp_processor.gtbox_non_bg_memory_bank_relsample(
                            proposals, targets)
            elif self.cfg.TYPE == 'extract_cfa_feat':
                proposals, rel_labels, rel_pair_idxs, rel_binarys, pair_labels, drop_pair_lens, image_repeats = self.samp_processor.gtbox_memory_bank_cfa(proposals, targets)
            else:
                rel_pair_idxs = self.samp_processor.prepare_test_pairs(features[0].device, proposals)

        # use box_head to extract features that will be fed to the later predictor processing
        if rel_pair_idxs == None or len(rel_pair_idxs) == 0:
            return None, None, {}

        if (self.cfg.TYPE == 'cfa' and self.cfg.CONTRA != True) or self.cfg.TYPE == 'extract_cfa_feat':
            roi_features = self.box_feature_extractor.pooler(features, proposals)
            proposals, union_features, roi_features, rel_labels, rel_pair_idxs, rel_binarys, rel_labels_one_hot, changed_pred_idxs, changed_obj_idxs, _, all_changed_classes, all_change_to_classes = self.union_feature_extractor(features, proposals, rel_pair_idxs, rel_binarys, pair_labels, rel_labels, roi_features, training=self.training, drop_pair_lens=drop_pair_lens, mix_up_fg=self.cfg.MIXUP.MIXUP_FG, mix_up_bg=self.cfg.MIXUP.MIXUP_BG, mix_up_add_tail=self.cfg.MIXUP.MIXUP_ADD_TAIL, image_repeats=image_repeats)
            roi_features = self.box_feature_extractor.forward_without_pool(roi_features)
       
        elif self.cfg.TYPE == 'cfa' and self.cfg.CONTRA == True:
            roi_features = self.box_feature_extractor.pooler(features, proposals)
            # original
            if self.training:
                old_proposals, old_union_features, old_roi_features, old_rel_labels, old_rel_pair_idxs, old_rel_binarys, old_rel_labels_one_hot, _, _, filter_old_tail_idxs, all_changed_classes, all_change_to_classes = self.union_feature_extractor(copy.deepcopy(features), copy.deepcopy(proposals), copy.deepcopy(rel_pair_idxs), copy.deepcopy(rel_binarys), copy.deepcopy(pair_labels), copy.deepcopy(rel_labels), copy.deepcopy(roi_features), training=self.training, drop_pair_lens=drop_pair_lens, mix_up_fg=False, mix_up_bg=False, mix_up_add_tail=False, image_repeats=image_repeats)
                old_roi_features = self.box_feature_extractor.forward_without_pool(old_roi_features)
            # new
            proposals, union_features, roi_features, rel_labels, rel_pair_idxs, rel_binarys, rel_labels_one_hot, changed_pred_idxs, changed_obj_idxs, filter_tail_idxs, all_changed_classes, all_change_to_classes = self.union_feature_extractor(features, proposals, rel_pair_idxs, rel_binarys, pair_labels, rel_labels, roi_features, training=self.training, drop_pair_lens=drop_pair_lens, mix_up_fg=self.cfg.MIXUP.MIXUP_FG, mix_up_bg=self.cfg.MIXUP.MIXUP_BG, mix_up_add_tail=self.cfg.MIXUP.MIXUP_ADD_TAIL, image_repeats=image_repeats)
            roi_features = self.box_feature_extractor.forward_without_pool(roi_features)
        elif self.cfg.TYPE == 'extract_aug':
            roi_features = self.box_feature_extractor.pooler(features, proposals)
            union_features = self.union_feature_extractor(features, proposals, rel_pair_idxs)

            num_rels = [r.shape[0] for r in rel_pair_idxs]
            num_objs = [len(b) for b in proposals]
            union_features_list = union_features.split(num_rels, dim=0)
            roi_features_list = roi_features.split(num_objs, dim=0)
            for i in range(len(pair_labels)):
                if cfg.EXTRACT_GROUP == 'tail':
                    filter_tail_idxs = np.where(np.in1d(rel_labels[i].cpu().numpy(), TAIL))[0]
                elif cfg.EXTRACT_GROUP == 'body':
                    filter_tail_idxs = np.where(np.in1d(rel_labels[i].cpu().numpy(), BODY))[0]
                elif cfg.EXTRACT_GROUP == 'head':
                    filter_tail_idxs = np.where(np.in1d(rel_labels[i].cpu().numpy(), HEAD))[0]

                for tail_idx in filter_tail_idxs:
                    tail_pair_label = pair_labels[i][tail_idx]
                    tri_key = tail_pair_label + '_' + str(rel_labels[i][tail_idx].item())
                    if tri_key not in self.tail_features_dict.keys():
                        self.tail_features_dict[tri_key] = []
                        self.count_dict[tri_key] = 0
                    if self.count_dict[tri_key] <= 100:
                        self.count_dict[tri_key] +=1
                        tail_union_feature = union_features_list[i][tail_idx].cpu()
                        tail_rel_label = rel_labels[i][tail_idx].cpu()
                        tail_sub_feature = roi_features_list[i][rel_pair_idxs[i][tail_idx, 0]].cpu()
                        tail_obj_feature = roi_features_list[i][rel_pair_idxs[i][tail_idx, 1]].cpu()
                        tail_sub_proposal = proposals[i][rel_pair_idxs[i][tail_idx,0], None]
                        tail_obj_proposal = proposals[i][rel_pair_idxs[i][tail_idx,1], None]

                        self.tail_features_dict[tri_key].append(
                        (tail_rel_label, tail_sub_feature, tail_obj_feature, tail_sub_proposal, tail_obj_proposal, tail_union_feature))

            return self.tail_features_dict

        else:
            roi_features = self.box_feature_extractor(features, proposals)

            if self.cfg.MODEL.ATTRIBUTE_ON:
                att_features = self.att_feature_extractor(features, proposals)
                roi_features = torch.cat((roi_features, att_features), dim=-1)

            if self.use_union_box:
                union_features = self.union_feature_extractor(features, proposals, rel_pair_idxs)
            else:
                union_features = None
        
        # final classifier that converts the features into predictions
        # should corresponding to all the functions and layers after the self.context class

        if self.cfg.TYPE == 'cfa' and self.cfg.CONTRA == True:
            if self.training:
                old_refine_logits, old_relation_logits, old_add_losses = self.predictor(old_proposals, old_rel_pair_idxs, old_rel_labels, old_rel_binarys, old_roi_features, old_union_features, logger)
            refine_logits, relation_logits, add_losses = self.predictor(proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger)
        else:
            refine_logits, relation_logits, add_losses = self.predictor(proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger)
        # for test
        if self.cfg.TYPE == 'extract_cfa_feat':
            refine_logits, relation_logits, cfa_rel_feats = self.predictor(proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger)
            num_rels = [r.shape[0] for r in rel_pair_idxs]
            cfa_rel_feats_list = cfa_rel_feats.split(num_rels)
            for img_idx in range(len(cfa_rel_feats_list)):
                cfa_rel_feat = cfa_rel_feats_list[img_idx]
                for j in range(len(cfa_rel_feats_list[img_idx])):
                    rel_label = rel_labels_one_hot[img_idx][j]
                    rel_pair_idx = rel_pair_idxs[img_idx][j]
                    if rel_pair_idx[0].item() or rel_pair_idx[1].item() in changed_obj_idxs[img_idx]:
                        # cfa
                        rel_label_idxs = torch.nonzero(rel_label)
                        for rel_label_idx in rel_label_idxs:
                            if rel_label_idx > 0 and (rel_label_idx in TAIL or rel_label_idx in BODY):
                                if rel_label_idx.item() not in self.cfa_features_dict.keys():
                                    self.cfa_features_dict[rel_label_idx.item()] = []
                                if len(self.cfa_features_dict[rel_label_idx.item()])<=100:
                                    self.cfa_features_dict[rel_label_idx.item()].append(cfa_rel_feat[j].cpu())
                    else:
                        rel_label_idx = torch.nonzero(rel_label)[0]
                        if rel_label_idx > 0:
                            if rel_label_idx.item() not in self.origial_features_dict.keys():
                                self.origial_features_dict[rel_label_idx.item()] = []
                            if len(self.origial_features_dict[rel_label_idx.item()])<=200:
                                self.origial_features_dict[rel_label_idx.item()].append(cfa_rel_feat[j].cpu())
            return self.origial_features_dict, self.cfa_features_dict
        if not self.training:
            result = self.post_processor((relation_logits, refine_logits), rel_pair_idxs, proposals)
            return roi_features, result, {}

        if self.cfg.MIXUP.MIXUP_ADD_TAIL == True or self.cfg.MIXUP.MIXUP_BG == True or self.cfg.MIXUP.MIXUP_FG == True:
            rel_labels = rel_labels_one_hot
        
        if self.cfg.TYPE == 'cfa' and self.cfg.CONTRA == True:
            old_rel_labels = old_rel_labels_one_hot
            loss_relation, loss_refine, obj_contra_loss = self.loss_evaluator(proposals, rel_labels, relation_logits, refine_logits, old_proposals, old_rel_labels, old_relation_logits, old_refine_logits, cur_iter=cur_iter, rel_pair_idxs=rel_pair_idxs, changed_obj_idxs=changed_obj_idxs, changed_pred_idxs=changed_pred_idxs, filter_old_tail_idxs=filter_old_tail_idxs, filter_tail_idxs=filter_tail_idxs, all_changed_classes=all_changed_classes, all_change_to_classes=all_change_to_classes)
        else:
            loss_relation, loss_refine = self.loss_evaluator(proposals, rel_labels, relation_logits, refine_logits, cur_iter=cur_iter, rel_pair_idxs=rel_pair_idxs, all_changed_classes=all_changed_classes, all_change_to_classes=all_change_to_classes)
        if self.cfg.MODEL.ATTRIBUTE_ON and isinstance(loss_refine, (list, tuple)):
            output_losses = dict(loss_rel=loss_relation, loss_refine_obj=loss_refine[0], loss_refine_att=loss_refine[1])
        else:
            if self.cfg.TYPE == 'cfa' and self.cfg.CONTRA == True:
                output_losses = dict(loss_rel=loss_relation, loss_refine_obj=loss_refine, obj_contra_loss=obj_contra_loss)
            else:
                output_losses = dict(loss_rel=loss_relation, loss_refine_obj=loss_refine)

        output_losses.update(add_losses)

        return roi_features, proposals, output_losses


def build_roi_relation_head(cfg, in_channels):
    """
    Constructs a new relation head.
    By default, uses ROIRelationHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIRelationHead(cfg, in_channels)
