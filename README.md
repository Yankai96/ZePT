# ZePT: Zero-Shot Pan-Tumor Segmentation via Query-Disentangling and Self-Prompting
[Yankai Jiang](https://scholar.google.com/citations?user=oQKcL_oAAAAJ), [Zhongzhen Huang](https://scholar.google.com/citations?user=LrZdFHgAAAAJ), [Rongzhao Zhang](https://scholar.google.com/citations?user=NMp31uMAAAAJ), [Xiaofan Zhang](https://scholar.google.com/citations?user=30e95fEAAAAJ), [Shaoting Zhang](https://scholar.google.com/citations?user=oiBMWK4AAAAJ)

  <p align="center">
    <a href='https://openaccess.thecvf.com/content/CVPR2024/html/Jiang_ZePT_Zero-Shot_Pan-Tumor_Segmentation_via_Query-Disentangling_and_Self-Prompting_CVPR_2024_paper.html'>
      <img src='https://img.shields.io/badge/Paper-PDF-green?style=flat&logo=arXiv&logoColor=green' alt='arXiv PDF'>
    </a>
    <a href='https://github.com/Yankai96/ZePT/tree/main'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=flat&logo=webpack' alt='Project Page'>
    </a>
    <a href='https://www.youtube.com/watch?v=JMAra8YVEFY'>
      <img src='https://img.shields.io/badge/Video-YouTube-red?style=flat&logo=YouTube' alt='Video'>
    </a>
  </p>
<br />

## 🎉 News
- **\[2024/03\]** ZePT is accepted to CVPR 2024!
- **\[2024/12\]** The codes and model weights of ZePT are released!
## 🛠️ Quick Start

### Installation

- It is recommended to build a Python-3.9 virtual environment using conda

  ```bash
  git clone https://github.com/Yankai96/ZePT.git
  cd ZePT
  conda env create -f env.yml

### Dataset Preparation
- 01 [Multi-Atlas Labeling Beyond the Cranial Vault - Workshop and Challenge (BTCV)](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789)
- 02 [Pancreas-CT TCIA](https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT)
- 03 [Combined Healthy Abdominal Organ Segmentation (CHAOS)](https://chaos.grand-challenge.org/Combined_Healthy_Abdominal_Organ_Segmentation/)
- 04 [Liver Tumor Segmentation Challenge (LiTS)](https://competitions.codalab.org/competitions/17094#learn_the_details)
- 05 [Kidney and Kidney Tumor Segmentation (KiTS)](https://kits21.kits-challenge.org/participate#download-block)
- 06 [Liver segmentation (3D-IRCADb)](https://www.ircad.fr/research/data-sets/liver-segmentation-3d-ircadb-01/)
- 07 [WORD: A large scale dataset, benchmark and clinical applicable study for abdominal organ segmentation from CT image](https://github.com/HiLab-git/WORD)
- 08 [AbdomenCT-1K](https://github.com/JunMa11/AbdomenCT-1K)
- 09 [Multi-Modality Abdominal Multi-Organ Segmentation Challenge (AMOS)](https://amos22.grand-challenge.org)
- 10 [Decathlon (Liver, Lung, Pancreas, HepaticVessel, Spleen, Colon](https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2)
- 11 [CT volumes with multiple organ segmentations (CT-ORG)](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=61080890)
- 12 [AbdomenCT 12organ](https://zenodo.org/records/7860267)
### Dataset Pre-Process
1. Please refer to [CLIP-Driven](https://github.com/ljwztc/CLIP-Driven-Universal-Model) to organize the downloaded datasets.
2. Modify [ORGAN_DATASET_DIR](https://github.com/zongzi3zz/CAT/blob/2146b2e972d0570956c52317a75c823891a4df2c/label_transfer.py#L51) and [NUM_WORKER](https://github.com/zongzi3zz/CAT/blob/2146b2e972d0570956c52317a75c823891a4df2c/label_transfer.py#L53) in label_transfer.py  
3. `python -W ignore label_transfer.py`

### Text Prompts
We provide the text prompts used for Query-Knowledge Alignment.
These texts contain detailed knowledge of each [class name].

### Model Weights
The weights used for  zero-shot inference are provided in [GoogleDrive](https://drive.google.com/file/d/1NYhBcVsvi1zCZocwbk6oTfws31eCeqpV/view?usp=drive_link)

### Zero-Shot Evaluation

- **Evaluation**
  ```shell
  bash scripts/test.sh
  ```

## Citation
If you find ZePT useful, please cite using this BibTeX:
```bibtex
@inproceedings{jiang2024zept,
  title={Zept: Zero-shot pan-tumor segmentation via query-disentangling and self-prompting},
  author={Jiang, Yankai and Huang, Zhongzhen and Zhang, Rongzhao and Zhang, Xiaofan and Zhang, Shaoting},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11386--11397},
  year={2024}
}
```
## Acknowledgement
The [CLIP-Driven-Universal-Model](https://github.com/ljwztc/CLIP-Driven-Universal-Model) served as the foundational codebase for our work and provided us with significant inspiration!
