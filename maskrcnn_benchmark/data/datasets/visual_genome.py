from builtins import list
import os
import sys
import torch
import h5py
import json
from PIL import Image
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import random
from maskrcnn_benchmark.config import cfg

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
import copy

BOX_SCALE = 1024  # Scale at which we have the boxes

class VGDataset(torch.utils.data.Dataset):

    def __init__(self, split, img_dir, roidb_file, dict_file, image_file, transforms=None,
                filter_empty_rels=True, num_im=-1, num_val_im=5000,
                filter_duplicate_rels=True, filter_non_overlap=True, flip_aug=False, custom_eval=False, custom_path='', mode=None):
        """
        Torch dataset for VisualGenome
        Parameters:
            split: Must be train, test, or val
            img_dir: folder containing all vg images
            roidb_file:  HDF5 containing the GT boxes, classes, and relationships
            dict_file: JSON Contains mapping of classes/relationships to words
            image_file: HDF5 containing image filenames
            filter_empty_rels: True if we filter out images without relationships between
                             boxes. One might want to set this to false if training a detector.
            filter_duplicate_rels: Whenever we see a duplicate relationship we'll sample instead
            num_im: Number of images in the entire dataset. -1 for all images.
            num_val_im: Number of images in the validation set (must be less than num_im
               unless num_im is -1.)
        """
        # for debug
        # num_im = 10000
        # num_val_im = 4

        assert split in {'train', 'val', 'test'}
        self.flip_aug = flip_aug
        self.split = split
        self.img_dir = img_dir
        self.dict_file = dict_file
        self.roidb_file = roidb_file
        self.image_file = image_file
        self.filter_non_overlap = filter_non_overlap and self.split == 'train'
        self.filter_duplicate_rels = filter_duplicate_rels and self.split == 'train'
        self.transforms = transforms
        self.type = cfg.TYPE
        self.mode = mode

        self.ind_to_classes, self.ind_to_predicates, self.ind_to_attributes = load_info(dict_file) # contiguous 151, 51 containing __background__
        self.categories = {i : self.ind_to_classes[i] for i in range(len(self.ind_to_classes))}
        self.freq_dict = {'__background__': 0.0, 'above': 0.01961344922232388, 'across': 0.0007467305311877725,
         'against': 0.0005785479791184543, 'along': 0.0015001883644583176, 'and': 0.0018634626769280448,
         'at': 0.004651929390237339, 'attached to': 0.003989290135084225, 'behind': 0.037743528335396376,
         'belonging to': 0.001856735374845272, 'between': 0.001530461223830795, 'carrying': 0.003427560411172703,
         'covered in': 0.0013521877186373178, 'covering': 0.001446369947796136, 'eating': 0.0014699155050858404,
         'flying in': 1.345460416554545e-05, 'for': 0.0030138313330821806, 'from': 0.0005987298853667725,
         'growing on': 0.0004978203541251816, 'hanging from': 0.0021527366664872718, 'has': 0.1679134599860072,
         'holding': 0.024527743393789356, 'in': 0.057807706797266024, 'in front of': 0.01077377428556052,
         'laying on': 0.0015943705936171358, 'looking at': 0.00257992034874334, 'lying on': 0.0006760938593186589,
         'made of': 0.00029936494268338626, 'mounted on': 0.0007063667186911361, 'near': 0.057272886281685594,
         'of': 0.08660055971153328, 'on': 0.2898727194445939, 'on back of': 0.0008745492707604543,
         'over': 0.003649561379904203, 'painted on': 0.0004036381249663635, 'parked on': 0.0017053710779828858,
         'part of': 0.0011537323071955223, 'playing': 0.00030945589580754537, 'riding': 0.00815685377536193,
         'says': 9.754588020020451e-05, 'sitting on': 0.011655050858403746, 'standing on': 0.005869571067219202,
         'to': 0.0009149130832570906, 'under': 0.012771783004144019, 'using': 0.001113368494698886,
         'walking in': 0.0007030030676497497, 'walking on': 0.0033367418330552713, 'watching': 0.002590011301867499,
         'wearing': 0.11238967224584252, 'wears': 0.011705505624024542, 'with': 0.03192777568483935}
        # print(self.categories)
        self.custom_eval = custom_eval
        if self.custom_eval:
            self.get_custom_imgs(custom_path)
        else:
            self.split_mask, self.gt_boxes, self.gt_classes, self.gt_attributes, self.relationships = load_graphs(
                self.roidb_file, self.split, num_im, num_val_im=num_val_im,
                filter_empty_rels=filter_empty_rels,
                filter_non_overlap=self.filter_non_overlap,
            )

            self.filenames, self.img_info = load_image_filenames(img_dir, image_file) # length equals to split_mask
            self.filenames = [self.filenames[i] for i in np.where(self.split_mask)[0]]
            self.img_info = [self.img_info[i] for i in np.where(self.split_mask)[0]]

        if (self.type == 'cfa' or self.type == 'extract_cfa_feat') and self.split == 'train' and self.mode != 'statistic':
            self.resampled_gt_boxes = []
            self.resampled_gt_classes = []
            self.resampled_gt_attributes = []
            self.resampled_relationships = []
            self.resampled_filenames = []
            self.resampled_img_info = []
            self.r_list = []
            for i in range(len(self.relationships)):
                r = min([self.freq_dict[self.ind_to_predicates[rel]] for rel in self.relationships[i][:, 2]])
                r = (0.07/r)**0.5
                r = int(max(1, r))
                for _ in range(r):
                    self.resampled_gt_boxes.append(self.gt_boxes[i])
                    self.resampled_gt_classes.append(self.gt_classes[i])
                    self.resampled_gt_attributes.append(self.gt_attributes[i])
                    self.resampled_relationships.append(self.relationships[i])
                    self.resampled_filenames.append(self.filenames[i])
                    self.resampled_img_info.append(self.img_info[i])
                    self.r_list.append(r)
            self.gt_boxes = self.resampled_gt_boxes
            self.gt_classes = self.resampled_gt_classes
            self.gt_attributes = self.resampled_gt_attributes
            self.relationships = self.resampled_relationships
            self.filenames = self.resampled_filenames
            self.img_info = self.resampled_img_info

    def __getitem__(self, index):
        #if self.split == 'train':
        #    while(random.random() > self.img_info[index]['anti_prop']):
        #        index = int(random.random() * len(self.filenames))
        if self.custom_eval:
            img = Image.open(self.custom_files[index]).convert("RGB")
            target = torch.LongTensor([-1])
            if self.transforms is not None:
                img, target = self.transforms(img, target)
            return img, target, index
        
        img = Image.open(self.filenames[index]).convert("RGB")
        if img.size[0] != self.img_info[index]['width'] or img.size[1] != self.img_info[index]['height']:
            print('='*20, ' ERROR index ', str(index), ' ', str(img.size), ' ', str(self.img_info[index]['width']), ' ', str(self.img_info[index]['height']), ' ', '='*20)

        flip_img = (random.random() > 0.5) and self.flip_aug and (self.split == 'train')
        
        target = self.get_groundtruth(index, flip_img)

        if flip_img:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, index

    def get_statistics(self):
        fg_matrix, bg_matrix, fg_sub_pred_matrix, fg_obj_pred_matrix, bg_sub_pred_matrix, bg_obj_pred_matrix = get_VG_statistics(img_dir=self.img_dir, roidb_file=self.roidb_file, dict_file=self.dict_file,
                                                image_file=self.image_file, must_overlap=True)
        eps = 1e-3
        bg_matrix += 1
        fg_matrix[:, :, 0] = bg_matrix
        pred_dist = np.log(fg_matrix / fg_matrix.sum(2)[:, :, None] + eps)

        sub_matrix = fg_matrix.sum(1)   
        sub_matrix = sub_matrix / sub_matrix.sum(0)[None,:]
        sub_matrix = np.log(sub_matrix / sub_matrix.sum(1)[:,None]+ eps)
        obj_matrix = fg_matrix.sum(0)
        obj_matrix = obj_matrix / obj_matrix.sum(0)[None,:]
        obj_matrix = np.log(obj_matrix / obj_matrix.sum(1)[:,None]+ eps)

        pair_matrix = fg_matrix.reshape(-1, fg_matrix.shape[2])
        pair_matrix = pair_matrix / pair_matrix.sum(0)[None,:]
        pair_matrix = np.log(pair_matrix / pair_matrix.sum(1)[:,None] + eps)


        result = {
            'fg_matrix': torch.from_numpy(fg_matrix),
            'pred_dist': torch.from_numpy(pred_dist).float(),
            'sub_matrix': torch.from_numpy(sub_matrix).float(),
            'obj_matrix': torch.from_numpy(obj_matrix).float(),
            'pair_matrix': torch.from_numpy(pair_matrix).float(),
            'obj_classes': self.ind_to_classes,
            'rel_classes': self.ind_to_predicates,
            'att_classes': self.ind_to_attributes,
        }
        return result
    
    def get_custom_imgs(self, path):
        self.custom_files = []
        self.img_info = []
        if os.path.isdir(path):
            for file_name in tqdm(os.listdir(path)):
                self.custom_files.append(os.path.join(path, file_name))
                img = Image.open(os.path.join(path, file_name)).convert("RGB")
                self.img_info.append({'width':int(img.width), 'height':int(img.height)})
        # Expecting a list of paths in a json file
        if os.path.isfile(path):
            file_list = json.load(open(path))
            for file in tqdm(file_list):
                img = Image.open(file).convert("RGB")
                self.img_info.append({'width': int(img.width), 'height': int(img.height)})

    def get_img_info(self, index):
        # WARNING: original image_file.json has several pictures with false image size
        # use correct function to check the validity before training
        # it will take a while, you only need to do it once

        # correct_img_info(self.img_dir, self.image_file)
        return self.img_info[index]

    def get_groundtruth(self, index, evaluation=False, flip_img=False):
        img_info = self.get_img_info(index)
        w, h = img_info['width'], img_info['height']
        # important: recover original box from BOX_SCALE
        box = self.gt_boxes[index] / BOX_SCALE * max(w, h)
        box = torch.from_numpy(box).reshape(-1, 4)  # guard against no boxes
        if flip_img:
            new_xmin = w - box[:,2]
            new_xmax = w - box[:,0]
            box[:,0] = new_xmin
            box[:,2] = new_xmax
        target = BoxList(box, (w, h), 'xyxy') # xyxy

        target.add_field("labels", torch.from_numpy(self.gt_classes[index]))
        target.add_field("attributes", torch.from_numpy(self.gt_attributes[index]))

        relation = self.relationships[index].copy() # (num_rel, 3)

        if self.filter_duplicate_rels:
            old_size = relation.shape[0]
            all_rel_sets = defaultdict(list)
            for (o0, o1, r) in relation:
                all_rel_sets[(o0, o1)].append(r)
            # relation = [(k[0], k[1], np.random.choice(v)) for k, v in all_rel_sets.items()]
            relation = [(k[0], k[1], np.random.choice(all_rel_sets[k])) for k in sorted(all_rel_sets.keys())]
            relation = np.array(relation, dtype=np.int32)

            if self.type == 'my_bilvl':
                if self.split == 'train' and self.mode != 'statistic':
                    filtered_index = []
                    rel_set = list(set(relation[:, 2]))
                    non_relation = copy.deepcopy(relation)
                    non_relation[:, 2] = -1
                    r = self.r_list[index]
                    for i in range(len(rel_set)):
                        r_i = self.freq_dict[self.ind_to_predicates[rel_set[i]]]
                        r_i = (0.07/r_i)**0.5
                        drop_r = max((r - r_i) / r * 0.7, 0)
                        rel_indexs = np.where(relation[:, 2] == rel_set[i])[0]
                        # addrel_indexs = np.where(relation[:, 2] == rel_set[i])[0]
                        # print(max(int(len(rel_indexs) * (1 - drop_r)), 1))
                        filtered_index.extend(list(np.random.choice(rel_indexs, max(int(len(rel_indexs) * (1 - drop_r)), 1))))
                    non_relation[filtered_index] = relation[filtered_index]
                    relation = non_relation

            elif self.type == 'cfa' or self.type == 'extract_cfa_feat':
                if self.split == 'train' and self.mode != 'statistic':
                    filtered_index = []
                    rel_set = list(set(relation[:, 2]))
                    non_relation = copy.deepcopy(relation)
                    non_relation[:, 2] = -1 * non_relation[:, 2]
                    r = self.r_list[index]
                    for i in range(len(rel_set)):
                        r_i = self.freq_dict[self.ind_to_predicates[rel_set[i]]]
                        r_i = (0.07 / r_i) ** 0.5
                        drop_r = max((r - r_i) / r * 0.7, 0)
                        rel_indexs = np.where(relation[:, 2] == rel_set[i])[0]
                        # addrel_indexs = np.where(relation[:, 2] == rel_set[i])[0]
                        # print(max(int(len(rel_indexs) * (1 - drop_r)), 1))
                        filtered_index.extend(
                            list(np.random.choice(rel_indexs, max(int(len(rel_indexs) * (1 - drop_r)), 1))))
                    non_relation[filtered_index] = relation[filtered_index]
                    relation = non_relation
        
        # add relation to target
        num_box = len(target)
        relation_map = torch.zeros((num_box, num_box), dtype=torch.int64)
        for i in range(relation.shape[0]):
            if relation_map[int(relation[i,0]), int(relation[i,1])] > 0:
                if (random.random() > 0.5):
                    relation_map[int(relation[i,0]), int(relation[i,1])] = int(relation[i,2])
            else:
                relation_map[int(relation[i,0]), int(relation[i,1])] = int(relation[i,2])


        target.add_field("relation", relation_map, is_triplet=True)
        if self.type == 'cfa' or self.type == 'extract_cfa_feat':
            repeat_map = torch.zeros((num_box, 1), dtype=torch.int64) 
            if self.split == 'train':
                repeat_map[:] = self.r_list[index]
            target.add_field("repeat", repeat_map)

        if evaluation:
            target = target.clip_to_image(remove_empty=False)
            target.add_field("relation_tuple", torch.LongTensor(relation)) # for evaluation
            return target
        else:
            target = target.clip_to_image(remove_empty=True)
            return target

    def __len__(self):
        if self.custom_eval:
            return len(self.custom_files)
        return len(self.filenames)
    

def get_VG_statistics(img_dir, roidb_file, dict_file, image_file, must_overlap=True):
    train_data = VGDataset(split='train', img_dir=img_dir, roidb_file=roidb_file, 
                        dict_file=dict_file, image_file=image_file, num_val_im=5000, 
                        filter_duplicate_rels=False, mode='statistic')
    num_obj_classes = len(train_data.ind_to_classes)
    num_rel_classes = len(train_data.ind_to_predicates)
    fg_matrix = np.zeros((num_obj_classes, num_obj_classes, num_rel_classes), dtype=np.int64)
    bg_matrix = np.zeros((num_obj_classes, num_obj_classes), dtype=np.int64)

    fg_sub_pred_matrix = np.zeros((num_obj_classes, num_rel_classes), dtype=np.int64)
    fg_obj_pred_matrix = np.zeros((num_obj_classes, num_rel_classes), dtype=np.int64)

    bg_sub_pred_matrix = np.zeros((num_obj_classes), dtype=np.int64)
    bg_obj_pred_matrix = np.zeros((num_obj_classes), dtype=np.int64)

    for ex_ind in tqdm(range(len(train_data))):
        gt_classes = train_data.gt_classes[ex_ind].copy()
        gt_relations = train_data.relationships[ex_ind].copy()
        gt_boxes = train_data.gt_boxes[ex_ind].copy()

        # For the foreground, we'll just look at everything
        o1o2 = gt_classes[gt_relations[:, :2]]
        for (o1, o2), gtr in zip(o1o2, gt_relations[:,2]):
            fg_matrix[o1, o2, gtr] += 1
            fg_sub_pred_matrix[o1, gtr] += 1
            fg_obj_pred_matrix[o2, gtr] += 1
        # For the background, get all of the things that overlap.
        o1o2_total = gt_classes[np.array(
            box_filter(gt_boxes, must_overlap=must_overlap), dtype=int)]
        for (o1, o2) in o1o2_total:
            bg_matrix[o1, o2] += 1
            bg_sub_pred_matrix[o1] += 1
            bg_obj_pred_matrix[o2] += 1

    return fg_matrix, bg_matrix, fg_sub_pred_matrix, fg_obj_pred_matrix, bg_sub_pred_matrix, bg_obj_pred_matrix

def box_filter(boxes, must_overlap=False):
    """ Only include boxes that overlap as possible relations. 
    If no overlapping boxes, use all of them."""
    n_cands = boxes.shape[0]

    overlaps = bbox_overlaps(boxes.astype(np.float), boxes.astype(np.float), to_move=0) > 0
    np.fill_diagonal(overlaps, 0)

    all_possib = np.ones_like(overlaps, dtype=np.bool)
    np.fill_diagonal(all_possib, 0)

    if must_overlap:
        possible_boxes = np.column_stack(np.where(overlaps))

        if possible_boxes.size == 0:
            possible_boxes = np.column_stack(np.where(all_possib))
    else:
        possible_boxes = np.column_stack(np.where(all_possib))
    return possible_boxes

def bbox_overlaps(boxes1, boxes2, to_move=1):
    """
    boxes1 : numpy, [num_obj, 4] (x1,y1,x2,y2)
    boxes2 : numpy, [num_obj, 4] (x1,y1,x2,y2)
    """
    #print('boxes1: ', boxes1.shape)
    #print('boxes2: ', boxes2.shape)
    num_box1 = boxes1.shape[0]
    num_box2 = boxes2.shape[0]
    lt = np.maximum(boxes1.reshape([num_box1, 1, -1])[:,:,:2], boxes2.reshape([1, num_box2, -1])[:,:,:2]) # [N,M,2]
    rb = np.minimum(boxes1.reshape([num_box1, 1, -1])[:,:,2:], boxes2.reshape([1, num_box2, -1])[:,:,2:]) # [N,M,2]

    wh = (rb - lt + to_move).clip(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    return inter

def correct_img_info(img_dir, image_file):
    with open(image_file, 'r') as f:
        data = json.load(f)
    for i in range(len(data)):
        img = data[i]
        basename = '{}.jpg'.format(img['image_id'])
        filename = os.path.join(img_dir, basename)
        img_data = Image.open(filename).convert("RGB")
        if img['width'] != img_data.size[0] or img['height'] != img_data.size[1]:
            print('--------- False id: ', i, '---------')
            print(img_data.size)
            print(img)
            data[i]['width'] = img_data.size[0]
            data[i]['height'] = img_data.size[1]
    with open(image_file, 'w') as outfile:  
        json.dump(data, outfile)

def load_info(dict_file, add_bg=True):
    """
    Loads the file containing the visual genome label meanings
    """
    info = json.load(open(dict_file, 'r'))
    if add_bg:
        info['label_to_idx']['__background__'] = 0
        info['predicate_to_idx']['__background__'] = 0
        info['attribute_to_idx']['__background__'] = 0

    class_to_ind = info['label_to_idx']
    predicate_to_ind = info['predicate_to_idx']
    attribute_to_ind = info['attribute_to_idx']
    ind_to_classes = sorted(class_to_ind, key=lambda k: class_to_ind[k])
    ind_to_predicates = sorted(predicate_to_ind, key=lambda k: predicate_to_ind[k])
    ind_to_attributes = sorted(attribute_to_ind, key=lambda k: attribute_to_ind[k])

    return ind_to_classes, ind_to_predicates, ind_to_attributes


def load_image_filenames(img_dir, image_file):
    """
    Loads the image filenames from visual genome from the JSON file that contains them.
    This matches the preprocessing in scene-graph-TF-release/data_tools/vg_to_imdb.py.
    Parameters:
        image_file: JSON file. Elements contain the param "image_id".
        img_dir: directory where the VisualGenome images are located
    Return: 
        List of filenames corresponding to the good images
    """
    with open(image_file, 'r') as f:
        im_data = json.load(f)

    corrupted_ims = ['1592.jpg', '1722.jpg', '4616.jpg', '4617.jpg']
    fns = []
    img_info = []
    for i, img in enumerate(im_data):
        basename = '{}.jpg'.format(img['image_id'])
        if basename in corrupted_ims:
            continue

        filename = os.path.join(img_dir, basename)
        if os.path.exists(filename):
            fns.append(filename)
            img_info.append(img)
    assert len(fns) == 108073
    assert len(img_info) == 108073
    return fns, img_info


def load_graphs(roidb_file, split, num_im, num_val_im, filter_empty_rels, filter_non_overlap):
    """
    Load the file containing the GT boxes and relations, as well as the dataset split
    Parameters:
        roidb_file: HDF5
        split: (train, val, or test)
        num_im: Number of images we want
        num_val_im: Number of validation images
        filter_empty_rels: (will be filtered otherwise.)
        filter_non_overlap: If training, filter images that dont overlap.
    Return: 
        image_index: numpy array corresponding to the index of images we're using
        boxes: List where each element is a [num_gt, 4] array of ground 
                    truth boxes (x1, y1, x2, y2)
        gt_classes: List where each element is a [num_gt] array of classes
        relationships: List where each element is a [num_r, 3] array of 
                    (box_ind_1, box_ind_2, predicate) relationships
    """
    roi_h5 = h5py.File(roidb_file, 'r')
    data_split = roi_h5['split'][:]
    split_flag = 2 if split == 'test' else 0
    split_mask = data_split == split_flag

    # Filter out images without bounding boxes
    split_mask &= roi_h5['img_to_first_box'][:] >= 0
    if filter_empty_rels:
        split_mask &= roi_h5['img_to_first_rel'][:] >= 0

    image_index = np.where(split_mask)[0]
    if num_im > -1:
        image_index = image_index[:num_im]
    if num_val_im > 0:
        if split == 'val':
            image_index = image_index[:num_val_im]
        elif split == 'train':
            image_index = image_index[num_val_im:]


    split_mask = np.zeros_like(data_split).astype(bool)
    split_mask[image_index] = True

    # Get box information
    all_labels = roi_h5['labels'][:, 0]
    all_attributes = roi_h5['attributes'][:, :]
    all_boxes = roi_h5['boxes_{}'.format(BOX_SCALE)][:]  # cx,cy,w,h
    assert np.all(all_boxes[:, :2] >= 0)  # sanity check
    assert np.all(all_boxes[:, 2:] > 0)  # no empty box

    # convert from xc, yc, w, h to x1, y1, x2, y2
    all_boxes[:, :2] = all_boxes[:, :2] - all_boxes[:, 2:] / 2
    all_boxes[:, 2:] = all_boxes[:, :2] + all_boxes[:, 2:]

    im_to_first_box = roi_h5['img_to_first_box'][split_mask]
    im_to_last_box = roi_h5['img_to_last_box'][split_mask]
    im_to_first_rel = roi_h5['img_to_first_rel'][split_mask]
    im_to_last_rel = roi_h5['img_to_last_rel'][split_mask]

    # load relation labels
    _relations = roi_h5['relationships'][:]
    _relation_predicates = roi_h5['predicates'][:, 0]
    assert (im_to_first_rel.shape[0] == im_to_last_rel.shape[0])
    assert (_relations.shape[0] == _relation_predicates.shape[0])  # sanity check

    # Get everything by image.
    boxes = []
    gt_classes = []
    gt_attributes = []
    relationships = []
    for i in range(len(image_index)):
        i_obj_start = im_to_first_box[i]
        i_obj_end = im_to_last_box[i]
        i_rel_start = im_to_first_rel[i]
        i_rel_end = im_to_last_rel[i]

        boxes_i = all_boxes[i_obj_start : i_obj_end + 1, :]
        gt_classes_i = all_labels[i_obj_start : i_obj_end + 1]
        gt_attributes_i = all_attributes[i_obj_start : i_obj_end + 1, :]

        if i_rel_start >= 0:
            predicates = _relation_predicates[i_rel_start : i_rel_end + 1]
            obj_idx = _relations[i_rel_start : i_rel_end + 1] - i_obj_start # range is [0, num_box)
            assert np.all(obj_idx >= 0)
            assert np.all(obj_idx < boxes_i.shape[0])
            rels = np.column_stack((obj_idx, predicates)) # (num_rel, 3), representing sub, obj, and pred
        else:
            assert not filter_empty_rels
            rels = np.zeros((0, 3), dtype=np.int32)

        if filter_non_overlap:
            assert split == 'train'
            # construct BoxList object to apply boxlist_iou method
            # give a useless (height=0, width=0)
            boxes_i_obj = BoxList(boxes_i, (1000, 1000), 'xyxy')
            inters = boxlist_iou(boxes_i_obj, boxes_i_obj)
            rel_overs = inters[rels[:, 0], rels[:, 1]]
            inc = np.where(rel_overs > 0.0)[0]

            if inc.size > 0:
                rels = rels[inc]
            else:
                split_mask[image_index[i]] = 0
                continue

        boxes.append(boxes_i)
        gt_classes.append(gt_classes_i)
        gt_attributes.append(gt_attributes_i)
        relationships.append(rels)

    return split_mask, boxes, gt_classes, gt_attributes, relationships
