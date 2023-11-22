# CFA for SGG in Pytorch

[![LICENSE](https://img.shields.io/badge/license-MIT-green)](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-3.9.7-blue.svg)](https://www.python.org/)
![PyTorch](https://img.shields.io/badge/pytorch-1.10.0-%237732a8)

Our paper [Compositional Feature Augmentation for Unbiased Scene Graph Generation](https://openaccess.thecvf.com/content/ICCV2023/html/Li_Compositional_Feature_Augmentation_for_Unbiased_Scene_Graph_Generation_ICCV_2023_paper.html) has been accepted by ICCV 2023.

## Installation

Check [INSTALL.md](INSTALL.md) for installation instructions.

## Dataset

Check [DATASET.md](DATASET.md) for instructions of dataset preprocessing.

## Extract Features
```base
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 10032 --nproc_per_node=1 tools/generate_aug_feature.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor  TEST.IMS_PER_BATCH 1 DTYPE "float16" GLOVE_DIR glove MODEL.PRETRAINED_DETECTOR_CKPT checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR exp/motif-precls MIXUP.FEAT_PATH feats TYPE extract_aug
```

## Processing Features
```base
python tools/processing_features.py
```

## Training Models with CFA

```base
# for PredCls
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port 10054 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR glove MODEL.PRETRAINED_DETECTOR_CKPT checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./exp/motifs_cfa_predcls TYPE cfa MIXUP.FEAT_PATH feats MIXUP.MIXUP_BG True MIXUP.MIXUP_FG True MIXUP.BG_LAMBDA 0.5 MIXUP.FG_LAMBDA 0.5 MIXUP.PREDICATE_LOSS_TYPE MIXUP_CE MIXUP.MIXUP_ADD_TAIL True FG_TAIL True FG_BODY True BG_TAIL True CL_TAIL True USE_PREDCLS_FEATURE True CONTRA False PKO False
```

```base
# for SGCls
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port 10054 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR glove MODEL.PRETRAINED_DETECTOR_CKPT checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./exp/motifs_cfa_sgcls TYPE cfa MIXUP.FEAT_PATH feats MIXUP.MIXUP_BG True MIXUP.MIXUP_FG True MIXUP.BG_LAMBDA 0.5 MIXUP.FG_LAMBDA 0.5 MIXUP.PREDICATE_LOSS_TYPE MIXUP_CE MIXUP.MIXUP_ADD_TAIL True FG_TAIL True FG_BODY True BG_TAIL True CL_TAIL True USE_PREDCLS_FEATURE False CONTRA True PKO False
```

```base
# for SGDet
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port 10054 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR glove MODEL.PRETRAINED_DETECTOR_CKPT checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./exp/motifs_cfa_sgdet TYPE cfa MIXUP.FEAT_PATH feats MIXUP.MIXUP_BG True MIXUP.MIXUP_FG True MIXUP.BG_LAMBDA 0.5 MIXUP.FG_LAMBDA 0.5 MIXUP.PREDICATE_LOSS_TYPE MIXUP_CE MIXUP.MIXUP_ADD_TAIL True FG_TAIL True FG_BODY True BG_TAIL True CL_TAIL True USE_PREDCLS_FEATURE False CONTRA True PKO False
```

## Test Models with Prior Knowledge
```base
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 10054 --nproc_per_node=1 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 1 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR glove MODEL.PRETRAINED_DETECTOR_CKPT checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./exp/motifs_cfa_sgcls TYPE cfa MIXUP.FEAT_PATH feats MIXUP.MIXUP_BG True MIXUP.MIXUP_FG True MIXUP.BG_LAMBDA 0.5 MIXUP.FG_LAMBDA 0.5 MIXUP.PREDICATE_LOSS_TYPE MIXUP_CE MIXUP.MIXUP_ADD_TAIL True FG_TAIL True FG_BODY True BG_TAIL True CL_TAIL True USE_PREDCLS_FEATURE False CONTRA True PKO True
```

## Comments for Parameters in Command
To make it easier for you to run our code, the Parameters in the command are explained here:

- `--master_port`: It represents the port on which the command is run.
- `CUDA_VISIBLE_DEVICES`: It means the the GPUs that you are going to use. For example, `CUDA_VISIBLE_DEVICES=0,1` use the first two GPUs.
- `--nproc_per_node`: It is the number of GPUs you are going to use.
- `SOLVER.IMS_PER_BATCH`: It is the training batch size.
- `TEST.IMS_PER_BATCH`: It is the testing batch size.
- `SOLVER.MAX_ITER`: It is the maximum iteration.
- `SOLVER.STEPS`: It is the steps where we decay the learning rate
- `SOLVER.VAL_PERIOD`: It is the period of conducting val.
- `SOLVER.CHECKPOINT_PERIOD`: It is the period of saving checkpoint.
- `MODEL.RELATION_ON` It means turning on the relationship head or not (since this is the pretraining phase for Faster R-CNN only, we turn off the relationship head), OUTPUT_DIR is the output directory to save checkpoints.
- `MODEL.ROI_RELATION_HEAD.USE_GT_BOX` and `MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL`: They used to select the protocols, (1) **PredCls**: They are all set as `True`. (2) **SGCls**: `MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL` is set to `False`, while `MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True` is set to `True`. (3) **SGDet**: They are all set to `False`.
- `MODEL.ROI_RELATION_HEAD.PREDICTOR`: It is the backbobe you are going to use., and the MOTIFS SGG backbone (`MotifPredictor`) is used by default. 
- `MIXUP.FEAT_PATH`: It refers to the path through which features are extracted and processed.
- `EXTRACT_GROUP`: It represents the predicate in which group to extract. The options are `head`, `body`, `tail`, or their combinations, separated by commas.  
- `TYPE`: The type of operation. If it set to 'cfa', it represents training with cfa. If it set to 'extract_aug', it represents feature extraction operation
- `FG_HEAD/FG_BODY/FG_TAIL`: It represents whether the Etrinsic-CFA operation for the group's foreground is performed.
- `BG_HEAD/BG_BODY/BG_TAIL`: It represents whether the Etrinsic-CFA operation for the group's background is performed.
- `CL_HEAD/CL_BODY/CL_TAIL`: It represents whether the Intrinsic-CFA operation for the group is performed.
- `CONTRA`: It implies whether to use the contrastive loss.
- `PKO`: It implies whether to use the prior knowledge during the inference.

## Models and Generated Files
For the Motifs-CFA, we provide the trained models (checkpoint) for verification purpose. Please download from [here*](https://mega.nz/folder/4ZImwLCZ#Hcd7tFBCzfjuuplrrAZeww) and unzip to checkpoints. Besides, we provide the extracted feature files, you can download from [here*](https://mega.nz/folder/NNAw3L6K#H-JmO5sVCvLMMbmDSnDkwA).


## Citations

If you find this project helps your research, please kindly consider citing our paper in your publications.

```
@InProceedings{Li_2023_ICCV,
    author    = {Li, Lin and Chen, Guikun and Xiao, Jun and Yang, Yi and Wang, Chunping and Chen, Long},
    title     = {Compositional Feature Augmentation for Unbiased Scene Graph Generation},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {21685-21695}
}
```
## Credits

Our codebase is based on [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch).
