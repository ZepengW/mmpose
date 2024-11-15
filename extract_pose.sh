
CUDA_VISIBLE_DEVICES='0' python demo/topdown_demo_with_mmdet_dir.py \
    demo/mmdetection_cfg/rtmdet_m_8xb32-300e_coco.py \
    https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_m_8xb32-300e_coco/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth \
    configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-large_8xb64-210e_coco-256x192.py \
    https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-large_8xb64-210e_coco-256x192-53609f55_20230314.pth \
    --input ../dataset/cuhk_03/images_labeled  \
    --output-root ../dataset/cuhk_03/images_labeled-mmpose/ \
    --save-predictions --draw-heatmap


    