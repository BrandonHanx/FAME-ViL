<div align="center">

# FAME-ViL: Multi-Tasking Vision-Language Model for Heterogeneous Fashion Tasks

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://mmf.sh/"><img alt="MMF" src="https://img.shields.io/badge/MMF-0054a6?logo=meta&logoColor=white"></a>
[![Conference](http://img.shields.io/badge/CVPR-2023(Highlight)-6790AC.svg)](https://cvpr.thecvf.com/)
[![Paper](http://img.shields.io/badge/Paper-arxiv.2303.02483-B31B1B.svg)](https://arxiv.org/abs/2303.02483)

</div>

## Updates
- :heart_eyes: (21/03/2023) Our FAME-ViL is selected as a **highlight paper** at CVPR 2023! (**Top 2.5%** of 9155 submissions)
- :blush: (12/03/2023) Code released!

Our trained model is available at [Google Drive](https://drive.google.com/drive/folders/17YflGKqt4sLbsfCSKZTGzdwaP9JcO7aN?usp=sharing).

Please refer to [FashionViL repo](https://github.com/BrandonHanx/mmf#data-preparation) for the dataset preparation.

Test on FashionIQ
```
python mmf_cli/run.py \
config=projects/fashionclip/configs/mtl_wa.yaml \
model=fashionclip \
datasets=fashioniq \
checkpoint.resume_file=save/backup_ckpts/fashionclip_512.pth \
run_type=test \
model_config.fashionclip.adapter_config.bottleneck=512
```
