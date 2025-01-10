import os
import time
import argparse
import pdb
import glob

import cv2
import json_tricks as json
import numpy as np

import mmcv
from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, split_instances
from mmpose.utils import adapt_mmdet_pipeline

from mmdet.apis import inference_detector, init_detector


def process_one_image(args,
                      img,
                      detector,
                      pose_estimator,
                      visualizer=None,
                      show_interval=0):
    """Visualize predicted keypoints (and heatmaps) of one image."""

    # predict bbox
    det_result = inference_detector(detector, img)
    pred_instance = det_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate((pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    bboxes = bboxes[np.logical_and(pred_instance.labels == args.det_cat_id,
                                   pred_instance.scores > args.bbox_thr)]
    bboxes = bboxes[nms(bboxes, args.nms_thr), :4]

    # predict keypoints
    pose_results = inference_topdown(pose_estimator, img, bboxes)
    data_samples = merge_data_samples(pose_results)

    # show the results
    if isinstance(img, str):
        img = mmcv.imread(img, channel_order='rgb')
    elif isinstance(img, np.ndarray):
        img = mmcv.bgr2rgb(img)
        
    if visualizer is not None:
        visualizer.add_datasample(
            'result',
            img,
            data_sample=data_samples,
            draw_gt=False,
            draw_heatmap=args.draw_heatmap,
            draw_bbox=args.draw_bbox,
            show_kpt_idx=args.show_kpt_idx,
            skeleton_style=args.skeleton_style,
            show=args.show,
            wait_time=show_interval,
            kpt_thr=args.kpt_thr)
        
    # if there is no instance detected, return None
    return data_samples.get('pred_instances', None)


def main(args):
    assert args.show or (args.output_root != ''), 'please either show or save image'
    assert args.input != '', 'input can not be empty'

    if args.input == 'webcam':
        input_type = 'webcam'
    else:
        if os.path.isdir(args.input):
            input_type = 'folder'
        else:
            file_extension = args.input.split('.')[-1]
            if file_extension in ['jpg', 'jpeg', 'png']:
                input_type = 'image'
            elif file_extension in ['mp4']:
                input_type = 'video'
            else:
                raise ValueError(f'Unknown file extension {file_extension}')
    
    output_file = None
    if args.output_root:
        if not os.path.exists(args.output_root):
            os.mkdir(args.output_root)

    if args.save_predictions:
        assert args.output_root != ''
        args.pred_save_path = f'{args.output_root}/results_' \
            f'{os.path.splitext(os.path.basename(args.input))[0]}.json'

    # build detector
    detector = init_detector(args.det_config, args.det_checkpoint, device=args.device)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    # build pose estimator
    pose_estimator = init_pose_estimator(args.pose_config,
                                         args.pose_checkpoint,
                                         device=args.device,
                                         cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=args.draw_heatmap)))
                                         )

    # build visualizer
    pose_estimator.cfg.visualizer.radius = args.radius
    pose_estimator.cfg.visualizer.alpha = args.alpha
    pose_estimator.cfg.visualizer.line_width = args.thickness
    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_pose_estimator
    visualizer.set_dataset_meta(pose_estimator.dataset_meta, skeleton_style=args.skeleton_style)


    if input_type == 'image':
        # inference
        pred_instances = process_one_image(args, args.input, detector,
                                           pose_estimator, visualizer)

        if args.save_predictions:
            pred_instances_list = split_instances(pred_instances)

        file_extension = '.'+args.input.split('.')[-1]
        output_file = '.'.join(args.input.split('.')[:-1])+'_result'+file_extension
        output_file = os.path.join(args.output_root, output_file)
        if output_file:
            img_vis = visualizer.get_image()
            mmcv.imwrite(mmcv.rgb2bgr(img_vis), output_file)
            
    elif input_type == 'folder':
        image_list = glob.glob(f'{args.input}/*.jpg')+glob.glob(f'{args.input}/*.jpeg')+glob.glob(f'{args.input}/*.png')
        for image in image_list:
            if 'result' in image:
                continue
            pred_instances = process_one_image(args, image, detector,
                                               pose_estimator, visualizer)
            if args.save_predictions:
                pred_instances_list = split_instances(pred_instances)

            file_extension = '.'+image.split('.')[-1]
            output_file = '.'.join(image.split('.')[:-1])+'_result'+file_extension
            output_file = os.path.join(args.output_root, output_file)
            if output_file:
                img_vis = visualizer.get_image()
                mmcv.imwrite(mmcv.rgb2bgr(img_vis), output_file)
            

    elif input_type in ['webcam', 'video']:
        if args.input == 'webcam':
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(args.input)
        output_file = os.path.join(args.output_root, 'webcam.mp4')

        video_writer = None
        pred_instances_list = []
        frame_idx = 0

        while cap.isOpened():
            success, frame = cap.read()
            frame_idx += 1

            if not success:
                break
            
            # topdown pose estimation
            pred_instances = process_one_image(args, frame, detector,
                                               pose_estimator, visualizer,
                                               0.001)

            if args.save_predictions:
                # save prediction results
                pred_instances_list.append(
                    dict(
                        frame_id=frame_idx,
                        instances=split_instances(pred_instances)))

            # output videos
            if output_file:
                frame_vis = visualizer.get_image()

                if video_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    # the size of the image with visualization may vary
                    # depending on the presence of heatmaps
                    video_writer = cv2.VideoWriter(output_file,
                                                   fourcc,
                                                   25,  # saved fps
                                                   (frame_vis.shape[1], frame_vis.shape[0]))

                video_writer.write(mmcv.rgb2bgr(frame_vis))

            if args.show:
                cv2.imshow('Display Window', mmcv.rgb2bgr(frame_vis))
                
                # press ESC to exit
                if cv2.waitKey(100) & 0xFF == 27:
                    break

                time.sleep(args.show_interval)

        if video_writer:
            video_writer.release()

        cap.release()
        cv2.destroyAllWindows()

    else:
        args.save_predictions = False
        raise ValueError(
            f'file {os.path.basename(args.input)} has invalid format.')

    if args.save_predictions:
        with open(args.pred_save_path, 'w') as f:
            json.dump(
                dict(
                    meta_info=pose_estimator.dataset_meta,
                    instance_info=pred_instances_list),
                f,
                indent='\t')
        print(f'predictions have been saved at {args.pred_save_path}')

    if output_file:
        print(f'the output has been saved at {output_file}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument('--input', type=str, default='', help='Image/Video file')
    parser.add_argument('--show', action='store_true', default=False, help='whether to show img')
    parser.add_argument('--output-root', type=str, default='', help='root of the output img file.')
    parser.add_argument('--save-predictions', action='store_true', default=False, help='whether to save results')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--det-cat-id', type=int, default=0, help='Category id for bounding box detection model')
    parser.add_argument('--bbox-thr', type=float, default=0.3, help='Bounding box score threshold')
    parser.add_argument('--nms-thr', type=float, default=0.3, help='IoU threshold for bounding box NMS')
    parser.add_argument('--kpt-thr', type=float, default=0.3, help='Visualizing keypoint thresholds')
    parser.add_argument('--draw-heatmap', action='store_true', default=False, help='Draw heatmap predicted by the model')
    parser.add_argument('--show-kpt-idx', action='store_true', default=False, help='Whether to show the index of keypoints')
    parser.add_argument('--skeleton-style', default='mmpose', type=str, choices=['mmpose', 'openpose'], help='Skeleton style selection')
    parser.add_argument('--radius', type=int, default=3, help='Keypoint radius for visualization')
    parser.add_argument('--thickness', type=int, default=1, help='Link thickness for visualization')
    parser.add_argument('--show-interval', type=int, default=0, help='Sleep seconds per frame')
    parser.add_argument('--alpha', type=float, default=0.8, help='The transparency of bboxes')
    parser.add_argument('--draw-bbox', action='store_true', help='Draw bboxes of instances')

    return parser.parse_args()
        

if __name__ == '__main__':
    args = parse_args()
    main(args)
"""
** single image
python topdown_demo_with_mmdet.py pretrained_weight/detection/rtmdet_tiny_8xb32-300e_coco.py pretrained_weight/detection/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth pretrained_weight/detection/reg_mobilenetv2_rle_b256_420e_aic-coco-192x192.py pretrained_weight/detection/rle_mobilenet_0.75x_192x192_mmpose_202404121425.pth --input 'demos/image31.jpeg' --output-root '.' --draw-bbox

** folder
python topdown_demo_with_mmdet.py pretrained_weight/detection/rtmdet_tiny_8xb32-300e_coco.py pretrained_weight/detection/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth pretrained_weight/detection/simcc_mobilenetv2_wo-deconv_b128_420e_aic-coco-192x192.py pretrained_weight/detection/simcc_mobilenet_0.75x_192x192_202412301530.pth --input 'demos/images' --output-root '.' --draw-bbox

** video
python topdown_demo_with_mmdet.py pretrained_weight/detection/rtmdet_tiny_8xb32-300e_coco.py pretrained_weight/detection/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth pretrained_weight/detection/reg_mobilenetv2_rle_b256_420e_aic-coco-192x192.py pretrained_weight/detection/rle_mobilenet_0.75x_192x192_mmpose_202404121425.pth --input 'demos/jazz_pinkvenom.mp4' --output-root '.' --draw-bbox

python topdown_demo_with_mmdet.py pretrained_weight/detection/rtmdet_tiny_8xb32-300e_coco.py pretrained_weight/detection/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth pretrained_weight/detection/simcc_mobilenetv2_wo-deconv_b128_420e_aic-coco-192x192.py pretrained_weight/detection/simcc_mobilenet_0.75x_192x192_202412301530.pth --input 'demos/jazz_pinkvenom.mp4' --output-root '.' --draw-bbox
"""
