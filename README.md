## This is a clean version of MMPose, no complex dependency.<br>
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
