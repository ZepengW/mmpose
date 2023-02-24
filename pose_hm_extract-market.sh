CUDA_VISIBLE_DEVICES='1' python demo/top_down_img_demo_with_mmdet_outhm.py \
    ../mmdetection/configs/yolox/yolox_x_8x8_300e_coco.py \
    https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_x_8x8_300e_coco/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth \
    configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py \
    https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth \
    --img-root ../dataset/market1501/bounding_box_train/  \
    --out-img-root ../dataset/market1501/bounding_box_train-vis_mmpose_yolox_hrnet/ \
    --out_kp_root ../dataset/market1501/bounding_box_train-npy_mmpose_yolox_hrnet/

CUDA_VISIBLE_DEVICES='1' python demo/top_down_img_demo_with_mmdet_outhm.py \
    ../mmdetection/configs/yolox/yolox_x_8x8_300e_coco.py \
    https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_x_8x8_300e_coco/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth \
    configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py \
    https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth \
    --img-root ../dataset/market1501/bounding_box_test/  \
    --out-img-root ../dataset/market1501/bounding_box_test-vis_mmpose_yolox_hrnet/ \
    --out_kp_root ../dataset/market1501/bounding_box_test-npy_mmpose_yolox_hrnet/

CUDA_VISIBLE_DEVICES='1' python demo/top_down_img_demo_with_mmdet_outhm.py \
    ../mmdetection/configs/yolox/yolox_x_8x8_300e_coco.py \
    https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_x_8x8_300e_coco/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth \
    configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py \
    https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth \
    --img-root ../dataset/market1501/query/  \
    --out-img-root ../dataset/market1501/query-vis_mmpose_yolox_hrnet/ \
    --out_kp_root ../dataset/market1501/query-npy_mmpose_yolox_hrnet/

CUDA_VISIBLE_DEVICES='1' python demo/top_down_img_demo_with_mmdet_outhm.py \
    ../mmdetection/configs/yolox/yolox_x_8x8_300e_coco.py \
    https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_x_8x8_300e_coco/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth \
    configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py \
    https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth \
    --img-root ../dataset/market1501/gt_query/  \
    --out-img-root ../dataset/market1501/gt_query-vis_mmpose_yolox_hrnet/ \
    --out_kp_root ../dataset/market1501/gt_query-npy_mmpose_yolox_hrnet/