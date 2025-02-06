# MMPose Clean Version
A Clean Version of MMPose for Human Pose Estimation

#### Overview
MMPose_Clean is a streamlined version of the popular MMPose framework, designed for human pose estimation tasks. It removes complex dependencies, optimizes performance, and offers an easy-to-use interface for researchers and developers interested in pose estimation, and it is compatible with my own model quantization frame(for some reason currently this quantization tool is private in my github, sorry). Whether you are working on real-time applications or research projects, MMPose_Clean provides an efficient and flexible solution.<br>
Original repo https://github.com/open-mmlab/mmpose<br>

#### Key Features
- Cleaned-up version of the original MMPose framework, no complex package needed.
- **Super light-weighted** pre-trained models provided(Number of parameters is **less than 1.5M**)
- Flexible and extendable architecture
- Easy integration with other machine learning frameworks
- TODO
- [ ] remove all dependency in single image/video demo

#### Selected Performance
All Pre-trained models are available in work_dirs/ , find them by using log name.<br>
For MMPose official, you may find these models in MMPose offical repo also by using log name.
| Method  | Input Size(H*W) | Backbone | # of Param | mAP(COCO)| log name |
| --------| ----------------| ---------|-----------:|----------|----------|
| RLE  | 192x192  | MobileNetV2-0.5x  | 774,788   | 48.81 | 202403131203 |
| RLE  | 192x192  | MobileNetV2-0.75x | 1,442,532 | 57.20 | 202404121425 |
| RLE<br>(MMPose official)  | 256x192  | MobileNetV2-1.0x | 2,310,980 | 59.30 | 39b73bd5_20220922 <br>(MMPose official log) |
| SimCC  | 192x192  | MobileNetV2-0.75x | 1,405,617 | 58.60 | 202412301530 |
| SimCC  | 256x192  | MobileNetV2-0.75x | 1,421,105 | 60.44 | 202405131104 |
| SimCC<br>(MMPose official)  | 256x192  | MobileNetV2-1.0x  | 2,289,553 | 62.0 | 4b0703bb_20221010 <br> (MMPose official log) |

#### Usage
You can use MMPose_Clean to run pose estimation on images or video files. For topdown algorithm, you need detection model to find person first and then run kepoint model for keypoint detection.<br>
Here is an example of how to use the framework:<br>
```rb
python topdown_demo_with_mmdet.py <detection model config> <detection model weight> <leypoint model config> <keypoint model weight> --input <image file> --output-root <output directory> --draw-bbox
```
for example
```rb
python topdown_demo_with_mmdet.py pretrained_weight/detection/rtmdet_tiny_8xb32-300e_coco.py pretrained_weight/detection/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth pretrained_weight/detection/reg_mobilenetv2_rle_b256_420e_aic-coco-192x192.py pretrained_weight/detection/rle_mobilenet_0.75x_192x192_mmpose_202404121425.pth --input 'demos/image31.jpeg' --output-root '.' --draw-bbox
```
#### 

#### Prepare Dataset
**COCO**: https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset<br>
**AIC**: https://pan.baidu.com/s/1THPLarF9A2njs7FsCA_QoQ  password: 9t16<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; You only need ai_challenger_keypoint_validation_20170911.zip and ai_challenger_keypoint_train_20170909.zip<br>

#### Training
You may reproduce all my results by following steps,
1. Prepare your dataset.
2. Modify the configuration file to match your dataset's structure and hyperparameters.
3. Train the model using the following command:
```rb
python train.py --cfg_file configs/my_custom/<config_file>
```
for example
```rb
python train.py --cfg_file configs/my_custom/reg_mobilenetv2_rle_b256_420e_aic-coco-192x192.py
```

#### Code explanation
**train.py**: Main entrance of training.<br>
**val.py**: Main entrance of evaluation after training.<br>
**codec**: Define how you encode and decode the keypoint, different algorithms differ from each other.<br>
**configs**: Different configs to train different model with different algorithms.<br>
**datasets**: Define how the dataset will be loaded, including dataloader, combining different dataset and image augmentation. Be aware that image normalization is done in model/data_preprocess.<br>
**evaluation**: All the evaluation code for evaluating model with different metrics.<br>
**models**: Define the models. A typical keypoint detection model is divided into: backbone, neck, head and loss functions. All of them can be combined together to form a keypoint model.<br>
**optim**: Define the optimizer and scheduler.<br>
**pretrained_weight**: Pretrained weight that you can use to initialize model and train on new datasets.<br>
**structures**: Define data structures that we used in datasets.<br>
**optim**: Define the optimizer and scheduler.<br>
**utils**: Define some helper functions for training.<br>
