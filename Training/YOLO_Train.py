"""Train YOLO for pigeons"""

from ultralytics import YOLO
import argparse


def TrainYOLO(PreTrained, Config):
    model = YOLO(PreTrained)

    model.train(data=Config, batch=-1, imgsz=3840,epochs=500)  # train the model
    model.val()
    model.export()

def ParseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument("--preTrained",
                        type=str,
                        default= "Weights/yolov8l.pt",
                        help="Path to pre-trained weight for YOLO model")
    parser.add_argument("--config",
                        type=str,
                        default= "Configs/",
                        help="Path to YOLO config yaml file")

    arg = parser.parse_args()

    return arg


if __name__ == "__main__":
    args = ParseArgs()
    PreTrained = args.preTrained
    Config = args.configs
    TrainYOLO(PreTrained, Config)