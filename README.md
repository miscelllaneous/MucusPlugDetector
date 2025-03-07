# Mucus Plug Detector

In our Institutional Review Board (IRB) approved retrospective study, we developed a model to detect mucus plugs from HRCT images. The ground_truth directory contains corresponding masks for cases with mucus plugs among 14 HRCT files from LIDC-IDRI dataset. Cases without files indicate the absence of mucus plugs.

## Usage

Download the models from the following link and place the directory in the root directory of the repository:
https://drive.google.com/drive/folders/1mGL0g-UyrwLhiaPMtvBl5tcRu9CN9oWe?usp=sharing

Please create an image and run a container following the provided Dockerfile. Please mount this directory.
Place your nii.gz files in the inference_input directory.
We recommend using HRCT data with finer resolution than 0.5mm in x and y directions, and thinner than 1mm resolution in z direction.

### Inference Mucus + Airway Mask
This will process nii.gz files in the inference_input directory and output a single mask containing both airways and mucus plugs (including thickened bronchi) to the inference_output directory. Airways will be predicted regardless of their distance from the lung periphery.
The directory structure should be as follows. The process works on CPU as well.


```bash
.
└── inference_input
    ├── 000.nii.gz
    ├── 001.nii.gz
    └── ...
```

```bash
python src/inference.py
```

### Generate Mucus Prediction
Based on the nii.gz files in both inference_input and inference_output directories, this will output mucus plug predictions to the inference_mucus directory. Airways will be predicted regardless of their distance from the lung periphery.


```bash
python src/detect_mucus.py
```

### Train Model
This will train the model using nii.gz files from train_input and train_airway directories.
Please place lung field masks corresponding to train_input files in the train_lungmask directory.
The process will generate weight maps, resize the data, and perform training.
The directory structure should be as follows.


```bash
.
├── train_input
│   ├── 000.nii.gz
│   ├── 001.nii.gz
│   └── ...
├── train_lungmask
│   ├── 000.nii.gz
│   ├── 001.nii.gz
│   └── ...
└── train_airway
    ├── 000.nii.gz
    ├── 001.nii.gz
    └── ...
```


```bash
python src/create_weightmap.py
python src/change_resolution_with_weightmap.py
python src/train.py
```

### Create Lung Mask
This will predict lung fields within 2cm from the lung periphery.


```bash
python src/pipeline.py
```

### Acknowledgement

This code and the following paper utilize these tools and datasets:

- [ATM'22](https://atm22.grand-challenge.org/)
- [LIDC-IDRI](https://www.cancerimagingarchive.net/collection/lidc-idri/)
- [lungmask](https://github.com/JoHof/lungmask)
- [TotalSegmentator](https://github.com/wasserth/TotalSegmentator)



### Citation

Please cite our work if you find it helpful.

```bibtex
@article{,
  title={},
  author={Yuki, S., et al.},
  year={2025}
}
```
