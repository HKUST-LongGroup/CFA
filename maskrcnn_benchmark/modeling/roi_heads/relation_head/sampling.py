# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch.nn import functional as F
import numpy as np
import numpy.random as npr

from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.config import cfg

# print('---------------', cfg.SELECT_DATASET)

class RelationSampling(object):
    def __init__(
        self,
        fg_thres,
        require_overlap,
        num_sample_per_gt_rel,
        batch_size_per_image,
        positive_fraction,
        use_gt_box,
        test_overlap,
    ):
        self.fg_thres = fg_thres
        self.require_overlap = require_overlap
        self.num_sample_per_gt_rel = num_sample_per_gt_rel
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.use_gt_box = use_gt_box
        self.test_overlap = test_overlap
        if cfg.SELECT_DATASET == 'VG':
            self.id_object_dict = {"0": "__background__","1": "airplane", "2": "animal", "3": "arm", "4": "bag", "5": "banana", "6": "basket", "7": "beach", "8": "bear", "9": "bed", "10": "bench", "11": "bike", "12": "bird", "13": "board", "14": "boat", "15": "book", "16": "boot", "17": "bottle", "18": "bowl", "19": "box", "20": "boy", "21": "branch", "22": "building", "23": "bus", "24": "cabinet", "25": "cap", "26": "car", "27": "cat", "28": "chair", "29": "child", "30": "clock", "31": "coat", "32": "counter", "33": "cow", "34": "cup", "35": "curtain", "36": "desk", "37": "dog", "38": "door", "39": "drawer", "40": "ear", "41": "elephant", "42": "engine", "43": "eye", "44": "face", "45": "fence", "46": "finger", "47": "flag", "48": "flower", "49": "food", "50": "fork", "51": "fruit", "52": "giraffe", "53": "girl", "54": "glass", "55": "glove", "56": "guy", "57": "hair", "58": "hand", "59": "handle", "60": "hat", "61": "head", "62": "helmet", "63": "hill", "64": "horse", "65": "house", "66": "jacket", "67": "jean", "68": "kid", "69": "kite", "70": "lady", "71": "lamp", "72": "laptop", "73": "leaf", "74": "leg", "75": "letter", "76": "light", "77": "logo", "78": "man", "79": "men", "80": "motorcycle", "81": "mountain", "82": "mouth", "83": "neck", "84": "nose", "85": "number", "86": "orange", "87": "pant", "88": "paper", "89": "paw", "90": "people", "91": "person", "92": "phone", "93": "pillow", "94": "pizza", "95": "plane", "96": "plant", "97": "plate", "98": "player", "99": "pole", "100": "post", "101": "pot", "102": "racket", "103": "railing", "104": "rock", "105": "roof", "106": "room", "107": "screen", "108": "seat", "109": "sheep", "110": "shelf", "111": "shirt", "112": "shoe", "113": "short", "114": "sidewalk", "115": "sign", "116": "sink", "117": "skateboard", "118": "ski", "119": "skier", "120": "sneaker", "121": "snow", "122": "sock", "123": "stand", "124": "street", "125": "surfboard", "126": "table", "127": "tail", "128": "tie", "129": "tile", "130": "tire", "131": "toilet", "132": "towel", "133": "tower", "134": "track", "135": "train", "136": "tree", "137": "truck", "138": "trunk", "139": "umbrella", "140": "vase", "141": "vegetable", "142": "vehicle", "143": "wave", "144": "wheel", "145": "window", "146": "windshield", "147": "wing", "148": "wire", "149": "woman", "150": "zebra"}
            self.id_to_rel_dict = {"0": "__background__","1": "above", "2": "across", "3": "against", "4": "along", "5": "and", "6": "at", "7": "attached to", "8": "behind", "9": "belonging to", "10": "between", "11": "carrying", "12": "covered in", "13": "covering", "14": "eating", "15": "flying in", "16": "for", "17": "from", "18": "growing on", "19": "hanging from", "20": "has", "21": "holding", "22": "in", "23": "in front of", "24": "laying on", "25": "looking at", "26": "lying on", "27": "made of", "28": "mounted on", "29": "near", "30": "of", "31": "on", "32": "on back of", "33": "over", "34": "painted on", "35": "parked on", "36": "part of", "37": "playing", "38": "riding", "39": "says", "40": "sitting on", "41": "standing on", "42": "to", "43": "under", "44": "using", "45": "walking in", "46": "walking on", "47": "watching", "48": "wearing", "49": "wears", "50": "with"}
        elif cfg.SELECT_DATASET == 'GQA':
            self.id_object_dict = {"0": "__background__", "1": "window", "2": "tree", "3": "man", "4": "shirt", "5": "wall", "6": "building", "7": "person", "8": "ground", "9": "sky", "10": "leg", "11": "sign", "12": "hand", "13": "head", "14": "pole", "15": "grass", "16": "hair", "17": "car", "18": "ear", "19": "eye", "20": "woman", "21": "clouds", "22": "shoe", "23": "table", "24": "leaves", "25": "wheel", "26": "door", "27": "pants", "28": "letter", "29": "people", "30": "flower", "31": "water", "32": "glass", "33": "chair", "34": "fence", "35": "arm", "36": "nose", "37": "number", "38": "floor", "39": "rock", "40": "jacket", "41": "hat", "42": "plate", "43": "tail", "44": "leaf", "45": "face", "46": "bush", "47": "shorts", "48": "road", "49": "bag", "50": "sidewalk", "51": "tire", "52": "helmet", "53": "snow", "54": "boy", "55": "umbrella", "56": "logo", "57": "roof", "58": "boat", "59": "bottle", "60": "street", "61": "plant", "62": "foot", "63": "branch", "64": "post", "65": "jeans", "66": "mouth", "67": "cap", "68": "girl", "69": "bird", "70": "banana", "71": "box", "72": "bench", "73": "mirror", "74": "picture", "75": "pillow", "76": "book", "77": "field", "78": "glove", "79": "clock", "80": "dirt", "81": "bowl", "82": "bus", "83": "neck", "84": "trunk", "85": "wing", "86": "horse", "87": "food", "88": "train", "89": "kite", "90": "paper", "91": "shelf", "92": "airplane", "93": "sock", "94": "house", "95": "elephant", "96": "lamp", "97": "coat", "98": "cup", "99": "cabinet", "100": "street light", "101": "cow", "102": "word", "103": "dog", "104": "finger", "105": "giraffe", "106": "mountain", "107": "wire", "108": "flag", "109": "seat", "110": "sheep", "111": "counter", "112": "skis", "113": "zebra", "114": "hill", "115": "truck", "116": "bike", "117": "racket", "118": "ball", "119": "skateboard", "120": "ceiling", "121": "motorcycle", "122": "player", "123": "surfboard", "124": "sand", "125": "towel", "126": "frame", "127": "container", "128": "paw", "129": "feet", "130": "curtain", "131": "windshield", "132": "traffic light", "133": "horn", "134": "cat", "135": "child", "136": "bed", "137": "sink", "138": "animal", "139": "donut", "140": "stone", "141": "tie", "142": "pizza", "143": "orange", "144": "sticker", "145": "apple", "146": "backpack", "147": "vase", "148": "basket", "149": "drawer", "150": "collar", "151": "lid", "152": "cord", "153": "phone", "154": "pot", "155": "vehicle", "156": "fruit", "157": "laptop", "158": "fork", "159": "uniform", "160": "bear", "161": "fur", "162": "license plate", "163": "lady", "164": "tomato", "165": "tag", "166": "mane", "167": "beach", "168": "tower", "169": "cone", "170": "cheese", "171": "wrist", "172": "napkin", "173": "toilet", "174": "desk", "175": "dress", "176": "cell phone", "177": "faucet", "178": "blanket", "179": "screen", "180": "watch", "181": "keyboard", "182": "arrow", "183": "sneakers", "184": "broccoli", "185": "bicycle", "186": "guy", "187": "knife", "188": "ocean", "189": "t-shirt", "190": "bread", "191": "spots", "192": "cake", "193": "air", "194": "sweater", "195": "room", "196": "couch", "197": "camera", "198": "frisbee", "199": "trash can", "200": "paint"}
            self.id_to_rel_dict = {"0": "__background__", "1": "on", "2": "wearing", "3": "of", "4": "near", "5": "in", "6": "behind", "7": "in front of", "8": "holding", "9": "next to", "10": "above", "11": "on top of", "12": "below", "13": "by", "14": "with", "15": "sitting on", "16": "on the side of", "17": "under", "18": "riding", "19": "standing on", "20": "beside", "21": "carrying", "22": "walking on", "23": "standing in", "24": "lying on", "25": "eating", "26": "covered by", "27": "looking at", "28": "hanging on", "29": "at", "30": "covering", "31": "on the front of", "32": "around", "33": "sitting in", "34": "parked on", "35": "watching", "36": "flying in", "37": "hanging from", "38": "using", "39": "sitting at", "40": "covered in", "41": "crossing", "42": "standing next to", "43": "playing with", "44": "walking in", "45": "on the back of", "46": "reflected in", "47": "flying", "48": "touching", "49": "surrounded by", "50": "covered with", "51": "standing by", "52": "driving on", "53": "leaning on", "54": "lying in", "55": "swinging", "56": "full of", "57": "talking on", "58": "walking down", "59": "throwing", "60": "surrounding", "61": "standing near", "62": "standing behind", "63": "hitting", "64": "printed on", "65": "filled with", "66": "catching", "67": "growing on", "68": "grazing on", "69": "mounted on", "70": "facing", "71": "leaning against", "72": "cutting", "73": "growing in", "74": "floating in", "75": "driving", "76": "beneath", "77": "contain", "78": "resting on", "79": "worn on", "80": "walking with", "81": "driving down", "82": "on the bottom of", "83": "playing on", "84": "playing in", "85": "feeding", "86": "standing in front of", "87": "waiting for", "88": "running on", "89": "close to", "90": "sitting next to", "91": "swimming in", "92": "talking to", "93": "grazing in", "94": "pulling", "95": "pulled by", "96": "reaching for", "97": "attached to", "98": "skiing on", "99": "parked along", "100": "hang on"}


    def prepare_test_pairs(self, device, proposals):
        # prepare object pairs for relation prediction
        rel_pair_idxs = []
        for p in proposals:
            n = len(p)
            cand_matrix = torch.ones((n, n), device=device) - torch.eye(n, device=device)
            # mode==sgdet and require_overlap
            if (not self.use_gt_box) and self.test_overlap:
                cand_matrix = cand_matrix.byte() & boxlist_iou(p, p).gt(0).byte()
            idxs = torch.nonzero(cand_matrix).view(-1,2)
            if len(idxs) > 0:
                rel_pair_idxs.append(idxs)
            else:
                # if there is no candidate pairs, give a placeholder of [[0, 0]]
                rel_pair_idxs.append(torch.zeros((1, 2), dtype=torch.int64, device=device))
        return rel_pair_idxs

    def gtbox_relsample(self, proposals, targets):
        assert self.use_gt_box
        num_pos_per_img = int(self.batch_size_per_image * self.positive_fraction)
        rel_idx_pairs = []
        rel_labels = []
        rel_sym_binarys = []
        for img_id, (proposal, target) in enumerate(zip(proposals, targets)):
            device = proposal.bbox.device
            num_prp = proposal.bbox.shape[0]

            assert proposal.bbox.shape[0] == target.bbox.shape[0]
            tgt_rel_matrix = target.get_field("relation") # [tgt, tgt]
            tgt_pair_idxs = torch.nonzero(tgt_rel_matrix > 0)
            assert tgt_pair_idxs.shape[1] == 2
            tgt_head_idxs = tgt_pair_idxs[:, 0].contiguous().view(-1)
            tgt_tail_idxs = tgt_pair_idxs[:, 1].contiguous().view(-1)
            tgt_rel_labs = tgt_rel_matrix[tgt_head_idxs, tgt_tail_idxs].contiguous().view(-1)

            # sym_binary_rels
            binary_rel = torch.zeros((num_prp, num_prp), device=device).long()
            binary_rel[tgt_head_idxs, tgt_tail_idxs] = 1
            binary_rel[tgt_tail_idxs, tgt_head_idxs] = 1
            rel_sym_binarys.append(binary_rel)
            
            rel_possibility = torch.ones((num_prp, num_prp), device=device).long() - torch.eye(num_prp, device=device).long()
            rel_possibility[tgt_head_idxs, tgt_tail_idxs] = 0
            tgt_bg_idxs = torch.nonzero(rel_possibility > 0)

            # generate fg bg rel_pairs
            if tgt_pair_idxs.shape[0] > num_pos_per_img:
                perm = torch.randperm(tgt_pair_idxs.shape[0], device=device)[:num_pos_per_img]
                tgt_pair_idxs = tgt_pair_idxs[perm]
                tgt_rel_labs = tgt_rel_labs[perm]
            num_fg = min(tgt_pair_idxs.shape[0], num_pos_per_img)

            num_bg = self.batch_size_per_image - num_fg
            perm = torch.randperm(tgt_bg_idxs.shape[0], device=device)[:num_bg]
            tgt_bg_idxs = tgt_bg_idxs[perm]

            img_rel_idxs = torch.cat((tgt_pair_idxs, tgt_bg_idxs), dim=0)
            img_rel_labels = torch.cat((tgt_rel_labs.long(), torch.zeros(tgt_bg_idxs.shape[0], device=device).long()), dim=0).contiguous().view(-1)

            rel_idx_pairs.append(img_rel_idxs)
            rel_labels.append(img_rel_labels)

        return proposals, rel_labels, rel_idx_pairs, rel_sym_binarys



    def gtbox_memory_bank_cfa(self, proposals, targets):
        assert self.use_gt_box
        num_pos_per_img = int(self.batch_size_per_image * self.positive_fraction)
        rel_idx_pairs = []
        rel_labels = []
        rel_sym_binarys = []
        filter_proposals = []
        pair_labels = []
        drop_pair_lens = []
        image_repeats = []
        for img_id, (proposal, target) in enumerate(zip(proposals, targets)):
            device = proposal.bbox.device
            num_prp = proposal.bbox.shape[0]

            assert proposal.bbox.shape[0] == target.bbox.shape[0]
            tgt_rel_matrix = target.get_field("relation")  # [tgt, tgt]
            obj_labels = target.get_field("labels")
            image_repeat = target.get_field("repeat")[0]
            tgt_pair_idxs = torch.nonzero(tgt_rel_matrix > 0)
            tgt_drop_pair_idxs = torch.nonzero(tgt_rel_matrix < 0)
            # print('tgt_drop_pair', tgt_drop_pair_idxs)
            assert tgt_pair_idxs.shape[1] == 2
            tgt_head_idxs = tgt_pair_idxs[:, 0].contiguous().view(-1)
            tgt_tail_idxs = tgt_pair_idxs[:, 1].contiguous().view(-1)
            tgt_rel_labs = tgt_rel_matrix[tgt_head_idxs, tgt_tail_idxs].contiguous().view(-1)

            tgt_drop_head_idxs = tgt_drop_pair_idxs[:, 0].contiguous().view(-1)
            tgt_drop_tail_idxs = tgt_drop_pair_idxs[:, 1].contiguous().view(-1)
            tgt_drop_labs = tgt_rel_matrix[tgt_drop_head_idxs, tgt_drop_tail_idxs].contiguous().view(-1)
            # sym_binary_rels
            binary_rel = torch.zeros((num_prp, num_prp), device=device).long()
            binary_rel[tgt_head_idxs, tgt_tail_idxs] = 1
            binary_rel[tgt_tail_idxs, tgt_head_idxs] = 1
            binary_rel[tgt_drop_head_idxs, tgt_drop_tail_idxs] = 1
            binary_rel[tgt_drop_tail_idxs, tgt_drop_head_idxs] = 1
            # rel_sym_binarys.append(binary_rel)

            rel_possibility = torch.ones((num_prp, num_prp), device=device).long() - torch.eye(num_prp,
                                                                                               device=device).long()
            rel_possibility[tgt_head_idxs, tgt_tail_idxs] = 0
            tgt_bg_idxs = torch.nonzero(rel_possibility > 0)

            # generate fg bg rel_pairs
            if tgt_pair_idxs.shape[0] > num_pos_per_img:
                perm = torch.randperm(tgt_pair_idxs.shape[0], device=device)[:num_pos_per_img]
                tgt_pair_idxs = tgt_pair_idxs[perm]
                tgt_rel_labs = tgt_rel_labs[perm]
            num_fg = min(tgt_pair_idxs.shape[0], num_pos_per_img)

            num_bg = self.batch_size_per_image - num_fg
            perm = torch.randperm(tgt_bg_idxs.shape[0], device=device)[:num_bg]
            tgt_bg_idxs = tgt_bg_idxs[perm]

            img_rel_idxs = torch.cat((tgt_pair_idxs, tgt_bg_idxs, tgt_drop_pair_idxs), dim=0)
            img_rel_labels = torch.cat((tgt_rel_labs.long(), torch.zeros(tgt_bg_idxs.shape[0], device=device).long(), tgt_drop_labs.long()),
                                       dim=0).contiguous().view(-1)
            
            # for i in range(len(img_rel_labels)):
            #     print(obj_labels[img_rel_idxs[i,0]].item())
            #     print(obj_labels[img_rel_idxs[i,1]].item())
            #     print(cfg.SELECT_DATASET)
            #     print(len(id_object_dict))
            #     print(id_object_dict[str(obj_labels[img_rel_idxs[i,0]].item())])
            #     print(id_object_dict[str(obj_labels[img_rel_idxs[i,1]].item())])

            pair_label = np.array([self.id_object_dict[str(obj_labels[img_rel_idxs[i,0]].item())] + '_' + self.id_object_dict[str(obj_labels[img_rel_idxs[i,1]].item())] for i in range(len(img_rel_labels))])
            image_repeats.append(image_repeat)
            if img_rel_idxs.shape[0] != 0:
                rel_idx_pairs.append(img_rel_idxs)
                rel_labels.append(img_rel_labels)
                filter_proposals.append(proposal)
                rel_sym_binarys.append(binary_rel)
                pair_labels.append(pair_label)
                drop_pair_lens.append(len(tgt_drop_pair_idxs))

        return filter_proposals, rel_labels, rel_idx_pairs, rel_sym_binarys, pair_labels, drop_pair_lens, image_repeats

    def gtbox_non_bg_memory_bank_relsample(self, proposals, targets):
        assert self.use_gt_box
        num_pos_per_img = int(self.batch_size_per_image * self.positive_fraction)
        rel_idx_pairs = []
        rel_labels = []
        rel_sym_binarys = []
        filter_proposals = []
        pair_labels = []

        for img_id, (proposal, target) in enumerate(zip(proposals, targets)):
            device = proposal.bbox.device
            num_prp = proposal.bbox.shape[0]

            assert proposal.bbox.shape[0] == target.bbox.shape[0]
            tgt_rel_matrix = target.get_field("relation")  # [tgt, tgt]
            obj_labels = target.get_field("labels")
            tgt_pair_idxs = torch.nonzero(tgt_rel_matrix > 0)
            assert tgt_pair_idxs.shape[1] == 2
            tgt_head_idxs = tgt_pair_idxs[:, 0].contiguous().view(-1)
            tgt_tail_idxs = tgt_pair_idxs[:, 1].contiguous().view(-1)

            tgt_rel_labs = tgt_rel_matrix[tgt_head_idxs, tgt_tail_idxs].contiguous().view(-1)

            # sym_binary_rels
            binary_rel = torch.zeros((num_prp, num_prp), device=device).long()
            binary_rel[tgt_head_idxs, tgt_tail_idxs] = 1
            binary_rel[tgt_tail_idxs, tgt_head_idxs] = 1
            # rel_sym_binarys.append(binary_rel)

            rel_possibility = torch.ones((num_prp, num_prp), device=device).long() - torch.eye(num_prp, device=device).long()
            rel_possibility[tgt_head_idxs, tgt_tail_idxs] = 0

            img_rel_idxs = tgt_pair_idxs
            img_rel_labels = tgt_rel_labs.long()

            pair_label = [self.id_object_dict[str(obj_labels[img_rel_idxs[i, 0]].item())] + '_' + self.id_object_dict[
                str(obj_labels[img_rel_idxs[i, 1]].item())] for i in range(len(img_rel_labels))]

            if img_rel_idxs.shape[0] != 0:
                rel_idx_pairs.append(img_rel_idxs)
                rel_labels.append(img_rel_labels)
                filter_proposals.append(proposal)
                rel_sym_binarys.append(binary_rel)

                pair_labels.append(pair_label)

        return filter_proposals, rel_labels, rel_idx_pairs, rel_sym_binarys, pair_labels
    
    def detect_relsample(self, proposals, targets):
        # corresponding to rel_assignments function in neural-motifs
        """
        The input proposals are already processed by subsample function of box_head,
        in this function, we should only care about fg box, and sample corresponding fg/bg relations
        Note: this function keeps a state.

        Arguments:
            proposals (list[BoxList])  contain fields: labels, predict_logits
            targets (list[BoxList]) contain fields: labels
        """
        self.num_pos_per_img = int(self.batch_size_per_image * self.positive_fraction)
        rel_idx_pairs = []
        rel_labels = []
        rel_sym_binarys = []
        for img_id, (proposal, target) in enumerate(zip(proposals, targets)):
            device = proposal.bbox.device
            prp_box = proposal.bbox
            prp_lab = proposal.get_field("labels").long()
            tgt_box = target.bbox
            tgt_lab = target.get_field("labels").long()
            tgt_rel_matrix = target.get_field("relation") # [tgt, tgt]
            # IoU matching
            ious = boxlist_iou(target, proposal)  # [tgt, prp]
            is_match = (tgt_lab[:,None] == prp_lab[None]) & (ious > self.fg_thres) # [tgt, prp]
            # Proposal self IoU to filter non-overlap
            prp_self_iou = boxlist_iou(proposal, proposal)  # [prp, prp]
            if self.require_overlap and (not self.use_gt_box):
                rel_possibility = (prp_self_iou > 0) & (prp_self_iou < 1)  # not self & intersect
            else:
                num_prp = prp_box.shape[0]
                rel_possibility = torch.ones((num_prp, num_prp), device=device).long() - torch.eye(num_prp, device=device).long()
            # only select relations between fg proposals
            rel_possibility[prp_lab == 0] = 0
            rel_possibility[:, prp_lab == 0] = 0

            img_rel_triplets, binary_rel = self.motif_rel_fg_bg_sampling(device, tgt_rel_matrix, ious, is_match, rel_possibility)
            rel_idx_pairs.append(img_rel_triplets[:, :2]) # (num_rel, 2),  (sub_idx, obj_idx)
            rel_labels.append(img_rel_triplets[:, 2]) # (num_rel, )
            rel_sym_binarys.append(binary_rel)

        return proposals, rel_labels, rel_idx_pairs, rel_sym_binarys

    def detect_memory_bank_cfa(self, proposals, targets):
        # corresponding to rel_assignments function in neural-motifs
        """
        The input proposals are already processed by subsample function of box_head,
        in this function, we should only care about fg box, and sample corresponding fg/bg relations
        Note: this function keeps a state.

        Arguments:
            proposals (list[BoxList])  contain fields: labels, predict_logits
            targets (list[BoxList]) contain fields: labels
        """
        self.num_pos_per_img = int(self.batch_size_per_image * self.positive_fraction)
        rel_idx_pairs = []
        rel_labels = []
        rel_sym_binarys = []
        pair_labels = []
        image_repeats = []
        for img_id, (proposal, target) in enumerate(zip(proposals, targets)):
            device = proposal.bbox.device
            prp_box = proposal.bbox
            prp_lab = proposal.get_field("labels").long()
            tgt_box = target.bbox
            tgt_lab = target.get_field("labels").long()
            tgt_rel_matrix = target.get_field("relation") # [tgt, tgt]
            image_repeat = target.get_field("repeat")[0]
            # IoU matching
            ious = boxlist_iou(target, proposal)  # [tgt, prp]
            is_match = (tgt_lab[:,None] == prp_lab[None]) & (ious > self.fg_thres) # [tgt, prp]
            # Proposal self IoU to filter non-overlap
            prp_self_iou = boxlist_iou(proposal, proposal)  # [prp, prp]
            if self.require_overlap and (not self.use_gt_box):
                rel_possibility = (prp_self_iou > 0) & (prp_self_iou < 1)  # not self & intersect
            else:
                num_prp = prp_box.shape[0]
                rel_possibility = torch.ones((num_prp, num_prp), device=device).long() - torch.eye(num_prp, device=device).long()
            # only select relations between fg proposals
            rel_possibility[prp_lab == 0] = 0
            rel_possibility[:, prp_lab == 0] = 0
            img_rel_triplets, binary_rel = self.motif_rel_fg_bg_sampling(device, tgt_rel_matrix, ious, is_match, rel_possibility)
            
            pair_label = np.array([self.id_object_dict[str(prp_lab[img_rel_triplets[i, 0]].item())] + '_' + self.id_object_dict[
                str(prp_lab[img_rel_triplets[i, 1]].item())] for i in range(len(img_rel_triplets))])
            
            rel_idx_pairs.append(img_rel_triplets[:, :2]) # (num_rel, 2),  (sub_idx, obj_idx)
            rel_labels.append(img_rel_triplets[:, 2]) # (num_rel, )
            pair_labels.append(pair_label)
            rel_sym_binarys.append(binary_rel)
            image_repeats.append(image_repeat)

        # return proposals, rel_labels, rel_idx_pairs, rel_sym_binarys
        return proposals, rel_labels, rel_idx_pairs, rel_sym_binarys, pair_labels, image_repeats

    def motif_rel_fg_bg_sampling(self, device, tgt_rel_matrix, ious, is_match, rel_possibility):
        """
        prepare to sample fg relation triplet and bg relation triplet
        tgt_rel_matrix: # [number_target, number_target]
        ious:           # [number_target, num_proposal]
        is_match:       # [number_target, num_proposal]
        rel_possibility:# [num_proposal, num_proposal]
        """
        tgt_pair_idxs = torch.nonzero(tgt_rel_matrix > 0)
        assert tgt_pair_idxs.shape[1] == 2
        tgt_head_idxs = tgt_pair_idxs[:, 0].contiguous().view(-1)
        tgt_tail_idxs = tgt_pair_idxs[:, 1].contiguous().view(-1)
        tgt_rel_labs = tgt_rel_matrix[tgt_head_idxs, tgt_tail_idxs].contiguous().view(-1)

        num_tgt_rels = tgt_rel_labs.shape[0]
        # generate binary prp mask
        num_prp = is_match.shape[-1]
        binary_prp_head = is_match[tgt_head_idxs] # num_tgt_rel, num_prp (matched prp head)
        binary_prp_tail = is_match[tgt_tail_idxs] # num_tgt_rel, num_prp (matched prp head)
        binary_rel = torch.zeros((num_prp, num_prp), device=device).long()

        fg_rel_triplets = []
        for i in range(num_tgt_rels):
            # generate binary prp mask
            bi_match_head = torch.nonzero(binary_prp_head[i] > 0)
            bi_match_tail = torch.nonzero(binary_prp_tail[i] > 0)

            num_bi_head = bi_match_head.shape[0]
            num_bi_tail = bi_match_tail.shape[0]
            if num_bi_head > 0 and num_bi_tail > 0:
                bi_match_head = bi_match_head.view(1, num_bi_head).expand(num_bi_tail, num_bi_head).contiguous()
                bi_match_tail = bi_match_tail.view(num_bi_tail, 1).expand(num_bi_tail, num_bi_head).contiguous()
                # binary rel only consider related or not, so its symmetric
                binary_rel[bi_match_head.view(-1), bi_match_tail.view(-1)] = 1
                binary_rel[bi_match_tail.view(-1), bi_match_head.view(-1)] = 1

            tgt_head_idx = int(tgt_head_idxs[i])
            tgt_tail_idx = int(tgt_tail_idxs[i])
            tgt_rel_lab = int(tgt_rel_labs[i])
            # find matching pair in proposals (might be more than one)
            prp_head_idxs = torch.nonzero(is_match[tgt_head_idx]).squeeze(1)
            prp_tail_idxs = torch.nonzero(is_match[tgt_tail_idx]).squeeze(1)
            num_match_head = prp_head_idxs.shape[0]
            num_match_tail = prp_tail_idxs.shape[0]
            if num_match_head <= 0 or num_match_tail <= 0:
                continue
            # all combination pairs
            prp_head_idxs = prp_head_idxs.view(-1,1).expand(num_match_head,num_match_tail).contiguous().view(-1)
            prp_tail_idxs = prp_tail_idxs.view(1,-1).expand(num_match_head,num_match_tail).contiguous().view(-1)
            valid_pair = prp_head_idxs != prp_tail_idxs
            if valid_pair.sum().item() <= 0:
                continue
            # remove self-pair
            # remove selected pair from rel_possibility
            prp_head_idxs = prp_head_idxs[valid_pair]
            prp_tail_idxs = prp_tail_idxs[valid_pair]
            rel_possibility[prp_head_idxs, prp_tail_idxs] = 0
            # construct corresponding proposal triplets corresponding to i_th gt relation
            fg_labels = torch.tensor([tgt_rel_lab]*prp_tail_idxs.shape[0], dtype=torch.int64, device=device).view(-1,1)
            fg_rel_i = cat((prp_head_idxs.view(-1,1), prp_tail_idxs.view(-1,1), fg_labels), dim=-1).to(torch.int64)
            # select if too many corresponding proposal pairs to one pair of gt relationship triplet
            # NOTE that in original motif, the selection is based on a ious_score score 
            if fg_rel_i.shape[0] > self.num_sample_per_gt_rel:
                ious_score = (ious[tgt_head_idx, prp_head_idxs] * ious[tgt_tail_idx, prp_tail_idxs]).view(-1).detach().cpu().numpy()
                ious_score = ious_score / ious_score.sum()
                perm = npr.choice(ious_score.shape[0], p=ious_score, size=self.num_sample_per_gt_rel, replace=False)
                fg_rel_i = fg_rel_i[perm]
            if fg_rel_i.shape[0] > 0:
                fg_rel_triplets.append(fg_rel_i)
        
        # select fg relations
        if len(fg_rel_triplets) == 0:
            fg_rel_triplets = torch.zeros((0, 3), dtype=torch.int64, device=device)
        else:
            fg_rel_triplets = cat(fg_rel_triplets, dim=0).to(torch.int64)
            if fg_rel_triplets.shape[0] > self.num_pos_per_img:
                perm = torch.randperm(fg_rel_triplets.shape[0], device=device)[:self.num_pos_per_img]
                fg_rel_triplets = fg_rel_triplets[perm]

        # select bg relations
        bg_rel_inds = torch.nonzero(rel_possibility>0).view(-1,2)
        bg_rel_labs = torch.zeros(bg_rel_inds.shape[0], dtype=torch.int64, device=device)
        bg_rel_triplets = cat((bg_rel_inds, bg_rel_labs.view(-1,1)), dim=-1).to(torch.int64)

        num_neg_per_img = min(self.batch_size_per_image - fg_rel_triplets.shape[0], bg_rel_triplets.shape[0])
        if bg_rel_triplets.shape[0] > 0:
            perm = torch.randperm(bg_rel_triplets.shape[0], device=device)[:num_neg_per_img]
            bg_rel_triplets = bg_rel_triplets[perm]
        else:
            bg_rel_triplets = torch.zeros((0, 3), dtype=torch.int64, device=device)

        # if both fg and bg is none
        if fg_rel_triplets.shape[0] == 0 and bg_rel_triplets.shape[0] == 0:
            bg_rel_triplets = torch.zeros((1, 3), dtype=torch.int64, device=device)

        return cat((fg_rel_triplets, bg_rel_triplets), dim=0), binary_rel


def make_roi_relation_samp_processor(cfg):
    samp_processor = RelationSampling(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_RELATION_HEAD.REQUIRE_BOX_OVERLAP,
        cfg.MODEL.ROI_RELATION_HEAD.NUM_SAMPLE_PER_GT_REL,
        cfg.MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE, 
        cfg.MODEL.ROI_RELATION_HEAD.POSITIVE_FRACTION,
        cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX,
        cfg.TEST.RELATION.REQUIRE_OVERLAP,
    )

    return samp_processor
