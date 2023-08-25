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
<pre class="notranslate"><code>@inproceedings{liu2021Swin,
  title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2021}
}
</code></pre>




