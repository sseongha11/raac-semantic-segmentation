[![License](https://img.shields.io/badge/license-MIT-green)](../raac-semantic-segmentation/LICENSE.md)

# RAAC Semantic Segmentation

## Overview
This project, developed by [Seongha Hwang](https://www.linkedin.com/in/seongha-hwang-a478a068/) at Loughborough University, presents a sophisticated Crack Detection Model. The model is based on various semantic segmantation deep learning architectures. It's designed for efficient and accurate detection and segmentation of cracks in surfaces or materials.

## Dataset

You can check the dataset details using the command:

```bash
python check-dataset.py
```
```bash
python check-augmented-dataset.py
```

## Dependencies
We provide a user-friendly configuring method via [Conda](https://docs.conda.io/en/latest/) system, and you can create a new Conda environment using the command:

```bash
conda env create -f environment.yml
```

## Training
Before the training, please download the dataset and copy it into the folder `datasets`.   
Then, you can train the model using the command:

```bash
python train.py --models unet --output output/unet/
# python train.py --models unet++ --output output/unetpp/
# python train.py --models deeplabv3+ --output output/deeplabv3p/
# python train.py --models fpn --output output/fpn/
```

## Inference


## Evaluation


## Acknowledgements


## Citation
If you take use of our datasets or code, please cite our papers:

```
@article{},
  title={},
  author={},
  journal={},
  volume={},
  pages={},
  year={},
  doi={}
}
```

If you have any questions, please contact me without hesitation (s.hwang3@lboro.ac.uk).

## License
See `LICENSE.md` for more information.


