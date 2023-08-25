# Transfer Learning for Microstructure Segmentation with CS-UNet: A Hybrid Algorithm with Transformer and CNN Encoders
## News

## Introduction

## Pretrained microscopy models

| Swin-T architecture |Depth | Pre-training method |Top-1 accuracy|top-5 accuracy | Download|
| --- | --- | --- | --- | --- | --- |
| Original | [2,2,6,2]|MicroLite | 84.23 | 95.91 |[ckp](https://drive.google.com/file/d/1SZsdAYgQXDUHRoxENoUICL_SxcToKzd0/view?usp=sharing)  |
| Original | [2,2,6,2] |ImageNet → MicroLite  | 84.63 | 96.35  | [ckp](https://drive.google.com/file/d/1ksqnjN1aiM133ASSg4PEswOnZtklg7Jb/view?usp=sharing) |
| Intermediate| [2,2,2,2] | MicroLite | 84.0 | 96.91  | [ckp](https://drive.google.com/file/d/11iuqZUfZEDmKJ_2UimDRFZraqPoCDWMu/view?usp=sharing) |
| Intermediate| [2,2,2,2]| ImageNet → MicroLite | 84.45 | 97.83 |[ckp](https://drive.google.com/file/d/1uNRH0DjAQiRPRdIvEraZdLpS6P1kDQUw/view?usp=sharing)  |


## Dataset

## Citation
<code>@misc{yang2022focal,
      title={Focal Modulation Networks}, 
      author={Jianwei Yang and Chunyuan Li and Xiyang Dai and Jianfeng Gao},
      journal={Advances in Neural Information Processing Systems (NeurIPS)},
      year={2022}
}
</code>




