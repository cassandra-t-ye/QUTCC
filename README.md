# [QUTCC🤗: Quantile Uncertainty Training and Conformal Calibration](https://arxiv.org/abs/2507.14760)
The official implementation of the [QUTCC: Quantile Uncertainty Training and Conformal Calibration for Imaging Inverse Problems 🤗](https://arxiv.org/abs/2310.16102)

## Setup: 

Clone this project using:

```
git clone https://github.com/cassandra-t-ye/QUTCC.git
```

Dependencies can be installed using

```
conda env create -f environment.yml
source activate qutcc
```

## Reproducing the results

### FastMRI dataset
* Download ```knee_singlecoil_train``` from the [FastMRI](https://fastmri.med.nyu.edu/) dataset. 
* [TO-DO: Add something about changing the path]

### Gaussian/Poisson/Real Noise Dataset
* Download the [FMD](https://github.com/yinhaoz/denoising-fluorescence) dataset.
* [TO-DO: Add something about changing the path]

## Paper 
[QUTCC: Quantile Uncertainty Training and Conformal Calibration for Imaging Inverse Problems 🤗]([https://arxiv.org/abs/2310.16102]
```
@article{ye2025qutcc,
  title={QUTCC: Quantile Uncertainty Training and Conformal Calibration for Imaging Inverse Problems},
  author={Ye, Cassandra Tong and Li, Shamus and King, Tyler and Monakhova, Kristina},
  journal={arXiv preprint arXiv:2507.14760},
  year={2025}
}
```
