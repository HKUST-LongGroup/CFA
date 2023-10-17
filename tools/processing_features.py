import numpy as np


def gen_sub_obj_dict(feat_path, mode, group):
    obj_export_feats = {}
    sub_export_feats = {}
    export_feats = {}
    feats = np.load('{}/{}_{}_feature_dict_motif.npy'.format(feat_path, mode, group), allow_pickle=True).item()
    for k, v in feats.items():
        sub_name = k.split('_')[0]
        obj_name = k.split('_')[1]
        for item in v:
            add_rel_label, sub_feature, obj_feature, tail_sub_proposal, tail_obj_proposal, union_feature = item
            sub_bbox = tail_sub_proposal.bbox.cpu().numpy()[0]
            obj_bbox = tail_obj_proposal.bbox.cpu().numpy()[0]
            vector = np.array([(sub_bbox[2] + sub_bbox[0]) / 2. - (obj_bbox[2] + obj_bbox[0]) / 2., (sub_bbox[3] + sub_bbox[1]) / 2. - (obj_bbox[3] + obj_bbox[1]) / 2.]) 
            wh = np.array([obj_bbox[2] - obj_bbox[0], obj_bbox[3] - obj_bbox[1]])
            sp_new_key = sub_name + '_' + str(add_rel_label)

            if sp_new_key not in obj_export_feats:
                obj_export_feats[sp_new_key] = {}
            if obj_name in obj_export_feats[sp_new_key]:
                obj_export_feats[sp_new_key][obj_name].append((add_rel_label, obj_feature, tail_obj_proposal, union_feature, vector, wh))
            else:
                obj_export_feats[sp_new_key][obj_name] = [(add_rel_label, obj_feature, tail_obj_proposal, union_feature, vector, wh)]

            op_new_key = obj_name + '_' + str(add_rel_label)

            if op_new_key not in sub_export_feats:
                sub_export_feats[op_new_key] = {}
            if sub_name in sub_export_feats[op_new_key]:
                sub_export_feats[op_new_key][sub_name].append((add_rel_label, sub_feature, tail_sub_proposal, union_feature, vector, wh))
            else:
                sub_export_feats[op_new_key][sub_name] = [(add_rel_label, sub_feature, tail_sub_proposal, union_feature, vector, wh)]

            new_key = sub_name + '_' + obj_name

            if new_key not in export_feats:
                export_feats[new_key] = {}
            if sub_name in export_feats[new_key]:
                export_feats[new_key][add_rel_label].append((add_rel_label, sub_feature, obj_feature, tail_sub_proposal, tail_obj_proposal, union_feature, vector, wh))
            else:
                export_feats[new_key][add_rel_label] = [(add_rel_label, sub_feature, obj_feature, tail_sub_proposal, tail_obj_proposal, union_feature, vector, wh)]

    np.save('{}/{}_{}_sub_obj_feature_with_proposal_dict_motif.npy'.format(feat_path, mode, group), export_feats, allow_pickle=True)
    np.save('{}/{}_{}_obj_only_feature_with_proposal_dict_motif.npy'.format(feat_path, mode, group), obj_export_feats, allow_pickle=True)
    np.save('{}/{}_{}_sub_only_feature_with_proposal_dict_motif.npy'.format(feat_path, mode, group), sub_export_feats, allow_pickle=True)


def gen_tri_dict(feat_path, mode, group):
    feats = np.load('{}/{}_{}_feature_dict_motif.npy'.format(feat_path, mode, group), allow_pickle=True).item()
    export_feats = {}
    for k, v in feats.items():
        sub_name = k.split('_')[0]
        obj_name = k.split('_')[1]
        for item in v:
            add_rel_label, sub_feature, obj_feature, tail_sub_proposal, tail_obj_proposal, union_feature = item
            sub_bbox = tail_sub_proposal.bbox.cpu().numpy()[0]
            obj_bbox = tail_obj_proposal.bbox.cpu().numpy()[0]
            vector = np.array([(sub_bbox[2] + sub_bbox[0]) / 2. - (obj_bbox[2] + obj_bbox[0]) / 2., (sub_bbox[3] + sub_bbox[1]) / 2. - (obj_bbox[3] + obj_bbox[1]) / 2.]) 
            wh = np.array([sub_bbox[2] - sub_bbox[0], sub_bbox[3] - sub_bbox[1]])
            new_key = sub_name + '_' + str(add_rel_label.item()) + '_' + obj_name

            if new_key not in export_feats:
                export_feats[new_key] = {}
            if new_key in export_feats[new_key]:
                export_feats[new_key].append((add_rel_label, sub_feature, obj_feature, tail_sub_proposal, tail_obj_proposal, union_feature, vector, wh))
            else:
                export_feats[new_key] = [(add_rel_label, sub_feature, obj_feature, tail_sub_proposal, tail_obj_proposal, union_feature, vector, wh)]

    np.save('{}/{}_{}_feature_tri_with_proposal_dict_motif.npy'.format(feat_path, mode, group), export_feats, allow_pickle=True)



def gen_tri_map(feat_path, mode, group):
    feats = np.load('{}/{}_{}_feature_with_proposal_dict_motif.npy'.format(feat_path, mode, group), allow_pickle=True).item()
    pred_dict = {}
    sub_dict = {}
    obj_dict = {}

    for k, v in feats.items():
        sub_name = k.split('_')[0]
        obj_name = k.split('_')[-1]
        for item in v:
            add_rel_label, sub_feature, obj_feature, tail_sub_proposal, tail_obj_proposal, union_feature, vector, wh = item
            tri_name = sub_name + '_' + str(add_rel_label.item()) + '_' + obj_name
            pred_key = sub_name + '_' + obj_name
            sub_key= str(add_rel_label.item()) + '_' + obj_name
            obj_key = sub_name + '_' + str(add_rel_label.item())
            if pred_key not in pred_dict:
                pred_dict[pred_key] = []
            pred_dict[pred_key].append(tri_name)
            if sub_key not in sub_dict:
                sub_dict[sub_key] = []
            sub_dict[sub_key].append((tri_name, sub_name))
            if obj_key not in obj_dict:
                obj_dict[obj_key] = []
            obj_dict[obj_key].append((tri_name, obj_name))
        tri_map_dict = {'pred': pred_dict, 'sub': sub_dict, 'obj': obj_dict}
    np.save('{}/{}_{}_tri_map.npy'.format(feat_path, mode, group), tri_map_dict, allow_pickle=True)

if __name__ == "__main__":
    feat_path = 'feats'
    mode = 'predcls'
    group = 'tail'
    gen_sub_obj_dict(feat_path, mode, group)
    gen_tri_dict(feat_path, mode, group)
    gen_tri_map(feat_path, mode, group)

