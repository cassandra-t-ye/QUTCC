# QUTCC🤗: Quantile Uncertainty Training and Conformal Calibration
The official implementation of QUTCC: Quantile Uncertainty Training and Conformal Calibration for Imaging Inverse Problems

## Setup: 
Dependencies can be installed using
```
conda env create -f environment.yml
source activate qutcc
```

## Data Processing:
We evaluate QUTCC on 6 different inverse tasks, which we provide support for below. To train your model for a specific task, please follow the instructions for the task you're interested in. When loading data, make sure that the path ``` /path/to/your/data ``` in the training bash scripts correctly points to your data. Once downloaded and processed, please proceed to the training portion.  

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
* For the Gaussian or Poisson task, run
```
python datasets/fmd/data_processing.py -d [path/to/fmd/dataset]
```
* For the Real Noise task, run
```
python datasets/fmd/data_processing_realnoise.py -d [path/to/fmd/dataset]
```

### QPI Dataset
* Download the [BSCCM Dataset](https://github.com/Waller-Lab/BSCCM), following the instructions from the repo. Make sure to download the full code repository, not the tiny version. 

## Training:
To train models, use ``train.py`` with the relevant parameters (for learning rate and weight decay, they are already correctly initialized). An example of Gaussian is included below. We recommend running all experiments on GPUs (for all results in the paper, A6000s were used).

```
python -u train.py \
    --net unet_im2im \
    --transform "center_crop" \
    --epochs 50 \
    --experiment-type "Denoising" \
    --data-root /PATH/TO/DATA/HERE \
    --in-channels 1 \
    --noise-type "gaussian" \
    --sigma 0.5 \
    --exp-name "gaussian" \
    --ckpt-freq 5 \
    --batch-size 4 \
    --plot-freq 1000
```

## Calibration:
To calibrate your model, first go to ```analysis.py``` and fill in BEST_RUNS with the experiment of interest, model, and paths to the saved model checkpoints. 

Then go to ```calibration_sweep.py``` and do the following. 
1. Line 123: Choose which experiment you want to calibrate (can do multiple)
2. Line 126: Choose which model you want to calibrate (can do multiple)
3. Line 138: Specify which GPU partition you would like to calibrate on.

Afterwards, run 
```
python analysis_sweep.py
```
This will sweep through all the model checkpoints you have saved and calibrate them to the specified alpha.


## Evaluation
To recreate table 2 (image reconstruction errors) in the paper, use ``calculate_statistics`` in ``analysis.py`` with all model paths being updated. This calculates image reconstruction performances over MSE, SSIM, PSNR, and LPIPS. To recreate table 3 (quantile crossings in $\{0.1, 0.2, \dots, 0.9 \}$) in the paper, use ``check_crossings`` with the default quantiles. 