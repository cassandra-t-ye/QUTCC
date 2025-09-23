# QUTCC🤗: Quantile Uncertainty Training and Conformal Calibration
The official implementation of QUTCC: Quantile Uncertainty Training and Conformal Calibration for Imaging Inverse Problems

## Setup: 
Dependencies can be installed using
```
conda env create -f environment.yml
source activate qutcc
```

## Data Processing
We evaluate QUTCC on 6 different inverse tasks, which we provide support for below. To train your model for a specific task, please follow the instructions for the task you're interested in. Once downloaded and processed, please proceed to the training portion.  

**Available Tasks:**
- [MRI dataset](#fastmri-dataset)
- [Gaussian/Poisson/Real Noise Dataset](#gaussianpoissonreal-noise-dataset)
- [CT dataset](#ct-dataset)
- [QPI Dataset](#qpi-dataset)

---

### FastMRI dataset
* Download ```knee_singlecoil_train``` from the [FastMRI](https://fastmri.med.nyu.edu/) dataset. 
* In the QUTCC directory, run 
```
python datasets/fastmri/data_processing.py -d [path/to/fast-mri/dataset]
```

### Gaussian/Poisson/Real Noise Dataset
* Download the [FMD dataset](https://github.com/yinhaoz/denoising-fluorescence).
* [TO-DO: Add something about changing the path]

### CT dataset
* Download the [LIDC-IDRI dataset](https://www.cancerimagingarchive.net/collection/lidc-idri/).
* In the QUTCC directory, run
```
python datasets/Lidcldri/data_processing.py -d [path/to/lidcldri/dataset]
```

### QPI Dataset
* Download the [BSCCM Dataset](https://github.com/Waller-Lab/BSCCM), following the instructions from the repo. 

## Training

## Evaluation
