# mri_super_resolution
## Introduction:
This is the code of paper named " [Cross-Modality High-Frequency Transformer for
MR Image Super-Resolution](https://arxiv.org/abs/2203.15314)". In this work, we make an early effort to build a Transformer-based MR image super-resolution framework, with careful designs on exploring valuable domain prior knowledge. Specifically, we consider two-fold domain priors including the high-frequency structure prior and the inter-modality context prior, and establish a novel Transformer architecture, called Cross-modality high-frequency Transformer (Cohf-T), to introduce such priors into super-resolving the low-resolution (LR) MR images. Experiments on two datasets indicate that Cohf-T achieves new state-of-the-art performance.
## Requirement:
python == 3.7  
PyTorch>=1.10       
nibabel     
os  
numpy  
math  
cv2

## Examples:
git clone https://github.com/langlang3/mri_sr   
cd mri_sr  
for training:
python main.py --upscale 4 --batch_size 4 --args.data_train <path of the train data>   --verbose
for testing:
python test.py
The data directory should follow the pattern below:   
|-traindir   
  |--MRI_volume1.nii.gz   
  |--MRI_volume2.nii.gz   
  |--MRI_volume3.nii.gz   
  |--...   
|-testdir   
  |--MRI_volume1.nii.gz   
  |--MRI_volume2.nii.gz   
  |--MRI_volume3.nii.gz   
  |--...   

the pretrained model can be obtained from this(IQ1W):https://pan.baidu.com/s/146XZ9rldjY2GjJ3lMZ69ZQ 

## Citation:
If you find this work or code is helpful in your research, please cite:   

@inproceedings{liang1visible,
  title={Cross-Modality High-Frequency Transformer for MR Image Super-Resolution},
  author={Fang, Chaowei and Zhang, Dingwen and Wang, Liang and Zhang, Yulun and Cheng, Lechao and Han, Junwei },
  booktitle={Proceedings of the 29th ACM International Conference on Multimedia},
  year={2022}
}
