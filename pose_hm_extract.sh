CUDA_VISIBLE_DEVICES='1' python demo/top_down_img_demo_with_mmdet_outhm.py \
    ../mmdetection/configs/yolox/yolox_x_8x8_300e_coco.py \
    https://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_d53_320_273e_coco/yolov3_d53_320_273e_coco-421362b6.pth \
    configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py \
    https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth \
    --img-root ../dataset/cuhk_03/images_detected/  \
    --out-img-root ../dataset/cuhk_03/images_detected-vis_mmpose_yolox_hrnet/ \
    --out_kp_root ../dataset/cuhk_03/images_detected-npy_mmpose_yolox_hrnet/

CUDA_VISIBLE_DEVICES='1' python demo/top_down_img_demo_with_mmdet_outhm.py \
    ../mmdetection/configs/yolox/yolox_x_8x8_300e_coco.py \
    https://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_d53_320_273e_coco/yolov3_d53_320_273e_coco-421362b6.pth \
    configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py \
    https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth \
    --img-root ../dataset/cuhk_03/images_labeled/  \
    --out-img-root ../dataset/cuhk_03/images_labeled-vis_mmpose_yolox_hrnet/ \
    --out_kp_root ../dataset/cuhk_03/images_labeled-npy_mmpose_yolox_hrnet/