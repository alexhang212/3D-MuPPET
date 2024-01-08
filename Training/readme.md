# Training
We also provide scripts for training all models in the paper. To download the dataset used in the manuscript, please download the "N6000" folder from the [3D-POP dataset](https://doi.org/10.17617/3.HPBBC7) repository, and paste it in `TrainingData/` directory, which is the default directory. You can also specify the path to the training data in each respective config file.


## 1. Training KPRCNN
You can modify the [KPRCNN Config File](./Configs/KPRCNN_3dpop.yaml) to change model paths, pretrained models, data augmentation parameters and model training hyper parameters.

```bash
##run following:
python Training/KPRCNN_train.py --config ./Configs/KPRCNN_3dpop.yaml
```

## 2. Training YOLO
Training data from 3DPOP is also formatted in YOLO format, so to retrain the model, you can just run the following script. You can modify the [YOLO Confg File](./Configs/Pigeon_YOLO.yaml) to the appropriate path.

```bash
##run following:
python Training/YOLO_Train.py --preTrained ./Weights/yolov8l.pt --config Pigeon_YOLO.yaml
```

## 3. Training DeepLabCut
We also provide training data in DeepLabCut format (for cropped pigeons). Please run the following, with the path to DeepLabCut project folder. The config file can be found within the DeepLabCut folder structure. For more info and help, please refer to [DeepLabCut Documentation](https://deeplabcut.github.io/DeepLabCut/README.html#)

```bash
##run following:
python Training/DLC_Train.py --path ./TrainingData/N6000/DLC/ --batch [batch_size]
```

## 4. Training ViTPose
We also provide training data in COCO format (for cropped pigeons) to train the ViTPose model. Please run the following. The training pipeline is implented using the MMPose framework. Please refer to the [VitPose repository](https://github.com/ViTAE-Transformer/ViTPose) for more information.

```bash
##run following:
python Training/VitPose_Train.py Configs/ViTPose_huge_3dpop_256x192.py
```



## 5. Training ltohp
Finally, we provide modified scripts to training the volumentric model from the [learnable triangulation for human posture](https://github.com/karfly/learnable-triangulation-pytorch) framework for the 3D-POP dataset. Run the following

```bash
##run following:
python --config ./Configs/ltohp_pigeonConfig.yaml --logdir [output_directory]

```