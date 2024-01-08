
<!-- TODO: - Upload N6000 and wild muppet - Upload Weights -->

# 3D-MuPPET: 3D Multi-Pigeon Pose Estimation and Tracking
## News:
- **03/01/2024:** Preprint updated, code updated, datasets and weights uploaded!
- **18/10/2023:** Officially launched git repository. Please hang on for the datasets and weights


## Description
This repository contains the code for the preprint: ["3D-MuPPET: 3D Multi-Pigeon Pose
Estimation and Tracking"](https://arxiv.org/abs/2308.15316). For more details and qualitative results,
please visit our [project page](https://alexhang212.github.io/3D-MuPPET/). We offer code and scripts for: 1)
Inference 2) Tracking Evaluation 3) Model Training on the 3DPOP dataset.

For evaluations on mice and cowbird datasets, please refer to
[i-MuPPET](https://github.com/urs-waldmann/i-muppet/#i-muppet-interactive-multi-pigeon-pose-estimation-and-tracking).

Also check out our [Hugging Face demo](https://huggingface.co/spaces/alexhang/PigeonEverywhere) for pigeon posture tracking in any environments!

**Abstract:**\
Markerless methods for animal posture tracking have been rapidly developing recently, but frameworks and benchmarks for tracking large animal groups in 3D are still lacking. To overcome this gap in the literature, we present 3D-MuPPET, a framework to estimate and track 3D poses of up to 10 pigeons at interactive speed using multiple camera views. We train a pose estimator to infer 2D keypoints and bounding boxes of multiple pigeons, then triangulate the keypoints to 3D. For identity matching of individuals in all views, we first dynamically match 2D detections to global identities in the first frame, then use a 2D tracker to maintain IDs across views in subsequent frames. We achieve comparable accuracy to a state of the art 3D pose estimator in terms of median error and Percentage of Correct Keypoints. Additionally, we benchmark the inference speed of 3D-MuPPET, with up to 9.45 fps in 2D and 1.89 fps in 3D, and perform quantitative tracking evaluation, which yields encouraging results. Finally, we showcase two novel applications for 3D-MuPPET. First, we train a model with data of single pigeons and achieve comparable results in 2D and 3D posture estimation for up to 5 pigeons. Second, we show that 3D-MuPPET also works in outdoors without additional annotations from natural environments. Both use cases simplify the domain shift to new species and environments, largely reducing annotation effort needed for 3D posture tracking. To the best of our knowledge we are the first to present a framework for 2D/3D animal posture and trajectory tracking that works in both indoor and outdoor environments for up to 10 individuals. We hope that the framework can open up new opportunities in studying animal collective behaviour and encourages further developments in 3D multi-animal posture tracking.

## Prerequisites

Before starting, clone the following repositories into `Repositories/`: 
- [sort](https://github.com/abewley/sort) 
- [3DPOP](https://github.com/alexhang212/Dataset-3DPOP)
- [DeepLabCut-live](https://github.com/DeepLabCut/DeepLabCut-live) 
- [i-muppet](https://github.com/urs-waldmann/i-muppet/#i-muppet-interactive-multi-pigeon-pose-estimation-and-tracking) 
- [Learnable Triangulation of Human Postures](https://github.com/karfly/learnable-triangulation-pytorch)
- [ViTPose](https://github.com/ViTAE-Transformer/ViTPose)

## Dataset and Weights
All new datasets used in the paper can be found here:
- Sampled images from 3DPOP for multi and single pigeon can be downloaded within the 3D-POP repository [here](https://doi.org/10.17617/3.HPBBC7)
- Wild-MuPPET dataset can be found [here](https://doi.org/10.17617/3.ENDMTI)

All weights used for evaluation in the paper can be downloaded [here](https://doi.org/10.5281/zenodo.10453890)


## Inference on 3DPOP

To perform inference and evaluation on the 3D-POP dataset: 1. Download
the [3D-POP dataset](https://github.com/alexhang212/Dataset-3DPOP). 2.
Download the pretrained weights from [this link](https://doi.org/10.5281/zenodo.10453890) and
place them in the `/Weights` directory.

### Setting up the Environment

Create a conda environment using:

``` bash
conda env create -f /Conda/muppet3D.yml
```

### Inference Scripts
If you run the scripts with the following commands, this will use the pre-trained weights from 3D-MuPPET. If you want to specify custom weights/ checkpoints, you can input it as argument. Use ` -h ` to find out what argument names to use.

1.  **KPRCNN**:

``` bash
# 2D Inference
python Inference/KPRCNN_2DInference.py --input [input_video]
# 3D Inference
python Inference/KPRCNN_3DInference.py --dataset [3dpop_path] --seq [3dpop_sequence] 
```

2. **DLC\***:
> Note: The DeepLabCut weight parameter is the directory for the exported model.
```bash
# 2D Inference
python Inference/YOLODLC_2DInference.py --input [input_video] 
# 3D Inference
python Inference/YOLODLC_3DInference.py --dataset [3dpop_path] --seq [3dpop_sequence]
```

3. **ViTPose\***
```bash
# 2D Inference
python Inference/VitPose_2DInference.py --input [input_video] 
# 3D Inference
python Inference/VitPose_3DInference.py --dataset [3dpop_path] --seq [3dpop_sequence]
```


4.  **ltohp baseline**: 
> Note: ltohp requires a different conda
    environment. Use `conda create -f Conda/ltohp.yaml` to set it up. The default config
    is already provided in the `config` folder.

``` bash
python Inference/ltohp_inference.py --dataset [3dpop_path] --seq [3dpop_sequence]
```

5.  **Pigeons in the Wild**:
> Note: This also requires a separate
    conda environment, which can be set up using
    `conda create -f Conda/WildPigeon.yaml`.

``` bash
python Inference/PigeonWild_2DInference.py --input [input_video] --DLCweight [path_to_DLCweight]
```


## Training

Scripts for training are provided. For detailed instructions, please
refer to the [training documentation](./Training/readme.md).

## Tracking Evaluation

Scripts for tracking and evaluation on the test videos of the 3D POP
dataset are available. The scripts perform inference using the VitPose\*
pipeline and then format the output to do benchmarking [here](https://github.com/JonathonLuiten/TrackEval/blob/master/docs/MOTChallenge-Official/Readme.md).

```bash
##Run evaluation:
python Tracking/TrackingInference.py --dataset [3dpop_path] --seq [3dpop_sequence] --OutDir [path_to_save_datafiles]

###Organize outputs:
python Tracking/PrepareTrackingBenchmark.py --dataset [3dpop_path] --seq [3dpop_sequence] --input [directory_of_tracking_output] --output [directory_for_benchmark_files]

```

## Contact
For any questions regarding the code, please contact Alex Chan: hoi-hang.chan [at] uni-konstanz.de

## Cite Us
```
@article{waldmann20233d,
    title={3D-MuPPET: 3D Multi-Pigeon Pose Estimation and Tracking},
    author={Waldmann, Urs and Chan, Alex Hoi Hang and Naik, Hemal and Nagy, M{\'a}t{\'e} and Couzin, Iain D and Deussen, Oliver and Goldluecke, Bastian and Kano, Fumihiro},
    journal={arXiv preprint arXiv:2308.15316},
    year={2023}
}                           
```
