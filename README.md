# pytorch-jet-tagging

This project uses a fully-connected neural network for classify jets.

## Setup

Create conda environemt

```bash
conda env create -f environment.yml
conda activate jet-tagging
```

### Data

Numpy files are proveded through [sharepoint](https://fermicloud-my.sharepoint.com/:f:/r/personal/jcampos_services_fnal_gov/Documents/jet-tagging-data?csf=1&web=1&e=A9V7k6) for easy setup. Data is taken from [hls4ml LHC Jet dataset](https://paperswithcode.com/dataset/hls4ml-lhc-jet-dataset). Review the README.md under `data` for further details.

## Training

Run the training scipt with default parameters (use the `--help` flag for available options)

```bash
python train.py
```

## Model Summary

```text
Net(
  (fc1): Linear(in_features=16, out_features=64, bias=True)
  (relu): ReLU()
  (fc2): Linear(in_features=64, out_features=32, bias=True)
  (relu): ReLU()
  (fc3): Linear(in_features=32, out_features=32, bias=True)
  (relu): ReLU()
  (fc4): Linear(in_features=32, out_features=5, bias=True)
  (softmax): Softmax(dim=-1)
)
```
