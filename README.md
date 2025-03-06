# Mucus Plug Detector

In our Institutional Review Board (IRB) approved retrospective study, we developed a model to detect mucus plugs from HRCT images. The ground_truth directory contains corresponding masks for cases with mucus plugs among the first 14 HRCT files from LIDC-IDRI dataset.

## Usage

Download the models from the following link and place the directory in the root directory of the repository:
https://drive.google.com/drive/folders/1mGL0g-UyrwLhiaPMtvBl5tcRu9CN9oWe?usp=sharing

Please create and run a container following the provided Dockerfile.
Place your nii.gz files in the inference_input directory.
We recommend using HRCT data with finer resolution than 0.5mm in x and y directions, and thinner than 1mm resolution in z direction.

### Inference Mucus + Airway Mask
This will process nii.gz files in the inference_input directory and output a single mask containing both airways and mucus plugs (including thickened bronchi) to the inference_output directory.

```bash
python inference.py
```

### Generate Mucus Prediction
Based on the nii.gz files in both inference_input and inference_output directories, this will output mucus plug predictions to the inference_mucus directory.

```bash
python detect_mucus.py
```

### Train Model
This will train the model using nii.gz files from train_input and train_mask directories.
Please place lung field masks corresponding to train_input files in the train_lungmask directory.
The process will generate weight maps, resize the data, and perform training.

```bash
python create_weightmap.py
python change_resolution_with_weightmap.py
python train.py
```

### Create Lung Mask
This generates lung field masks for the regions where mucus plugs should be detected. As a secondary output, it also generates masks for the entire lung fields.

```bash
python pipeline.py
```

### Citation

Please cite our work if you find it helpful.

```bibtex
@article{mucus_plug_detector,
  title={Mucus Plug Detector},
  author={Yuki, S., et al.},
  year={2025}
}
```
