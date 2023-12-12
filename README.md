# CS-UNet: A Generalizable and Flexible Segmentation Algorithm


## News

## Introduction
Feel free to check out our preprint on [arXiv](https://arxiv.org/abs/2308.13917) &#8291;


![alt text](https://github.com/Kalrfou/SwinT-pretrained-microscopy-models/blob/main/figures/NewFig1.jpeg)
The encoder-decoder architecture for microstructure segmentation with transferring learning, where the CNN and Swin-T models
are pre-trained on ImageNet and microscopy images. The weights of the pre-trained CNN and Swin-T models are used to initialize the
encoders while the weights of the Swin-T models are used to initialize the decoders.
## Pretrained microscopy models
You can download the pretrained MicroLite Swin-T encoders, which utilize transfer learning from classification models trained on an extensive dataset of microscopy images containing over 50,000 images.

| Swin-T architecture |Depth | Pre-training method |Top-1 accuracy|top-5 accuracy | Download|
| --- | --- | --- | --- | --- | --- |
| Original | [2,2,6,2]|MicroLite | 84.23 | 95.91 |[ckp](https://drive.google.com/file/d/1SZsdAYgQXDUHRoxENoUICL_SxcToKzd0/view?usp=sharing)  |
| Original | [2,2,6,2] |ImageNet → MicroLite  | 84.63 | 96.35  | [ckp](https://drive.google.com/file/d/1ksqnjN1aiM133ASSg4PEswOnZtklg7Jb/view?usp=sharing) |
| Intermediate| [2,2,2,2] | MicroLite | 84.0 | 96.91  | [ckp](https://drive.google.com/file/d/11iuqZUfZEDmKJ_2UimDRFZraqPoCDWMu/view?usp=sharing) |
| Intermediate| [2,2,2,2]| ImageNet → MicroLite | 84.45 | 97.83 |[ckp](https://drive.google.com/file/d/1uNRH0DjAQiRPRdIvEraZdLpS6P1kDQUw/view?usp=sharing)  |


## Dataset
For image segmentation, the datasets used in this repository were obtained from NASA GitHub ([pretrained-microscopy-models](https://github.com/nasa/pretrained-microscopy-models)). They consist of 7 microscopy datasets derived from two materials:
Nickel-based superalloys (Super): These datasets have 3 classes: matrix, secondary, and tertiary.
Environmental barrier coatings (EBC): These datasets have 2 classes: oxide layer and background (non-oxide) layer.

## Citation
### Citing Swin Transformer
<pre class="notranslate"><code>@inproceedings{liu2021Swin,
  title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2021}
}
</code></pre>
### Citing Swin-Unet
<pre class="notranslate"><code>@inproceedings{cao2022swin,
  title={Swin-unet: Unet-like pure transformer for medical image segmentation},
  author={Cao, Hu and Wang, Yueyue and Chen, Joy and Jiang, Dongsheng and Zhang, Xiaopeng and Tian, Qi and Wang, Manning},
  booktitle={European conference on computer vision},
  pages={205--218},
  year={2022},
  organization={Springer}
}
</code></pre>
### Citing Transdeeplab
<pre class="notranslate"><code>@inproceedings{azad2022transdeeplab,
  title={Transdeeplab: Convolution-free transformer-based deeplab v3+ for medical image segmentation},
  author={Azad, Reza and Heidari, Moein and Shariatnia, Moein and Aghdam, Ehsan Khodapanah and Karimijafarbigloo, Sanaz and Adeli, Ehsan and Merhof, Dorit},
  booktitle={International Workshop on PRedictive Intelligence In MEdicine},
  pages={91--102},
  year={2022},
  organization={Springer}
}
</code></pre>
### Citing HiFormer
<pre class="notranslate"><code>@inproceedings{heidari2023hiformer,
  title={Hiformer: Hierarchical multi-scale representations using transformers for medical image segmentation},
  author={Heidari, Moein and Kazerouni, Amirhossein and Soltany, Milad and Azad, Reza and Aghdam, Ehsan Khodapanah and Cohen-Adad, Julien and Merhof, Dorit},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={6202--6212},
  year={2023}
}
</code></pre>
### Microstructure segmentation with deep learning encoders pre-trained on a large microscopy dataset
<pre class="notranslate"><code>@article{stuckner2022microstructure,
  title={Microstructure segmentation with deep learning encoders pre-trained on a large microscopy dataset},
  author={Stuckner, Joshua and Harder, Bryan and Smith, Timothy M},
  journal={NPJ Computational Materials},
  volume={8},
  number={1},
  pages={200},
  year={2022},
  publisher={Nature Publishing Group UK London}
}
</code></pre>






