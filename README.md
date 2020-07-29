# PANDA Challenge
AI powered Prostate cancer ISUP grading. Team **rähmä.ai** solution for the [Prostate cANcer graDe Assessment (PANDA) Challenge](https://www.kaggle.com/c/prostate-cancer-grade-assessment). We ranked 24th (top 3%) in the competition. The models included in this repository get **0.930** QWK in the private test set and **0.904** QWK in the public test set.

## Team rähmä.ai

- [Mikko Tukiainen](https://github.com/mjkvaak)
- [Joni Juvonen](https://github.com/jpjuvo)
- [Antti Karlsson](https://github.com/AnttiKarlsson)

-----------------------------------------------------

## Installation

1. Download [PANDA dataset](https://www.kaggle.com/c/prostate-cancer-grade-assessment/data)

2. Clone this repository.

3. cd into cloned repository and install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip3 install --upgrade pip
pip3 install -r requirements.txt
```

4. (OPTIONAL) Install ResNeSt pre-trained models package

```
pip install git+https://github.com/zhanghang1989/ResNeSt
```

## Preprocessing

1. Find all serial section replicates by running [Detect_serial_sections.ipynb](./preprocessing/Detect_serial_sections.ipynb)

2. Sample tissue parts of the training data ot generate tile training sets.

- [Level 1 6x6 256-tiles from 256 slide size](./ptrprocessing/tissue_mosaic_generation_lvl1_256_6x6_256.ipynb)
- [Level 1 6x6 256-tiles from 384 slide size](./ptrprocessing/tissue_mosaic_generation_lvl1_384_6x6_256.ipynb)
- [Level 1 5x5 299-tiles from 299 slide size](./ptrprocessing/tissue_mosaic_generation_lvl1_299_5x5_299.ipynb)

## Training

Train using [Train template notebook](./training/Train-template.ipynb).

#### Competition models

We trained our `256`, `299` and `384`  models that we used in the final submission with these scripts (in order).

**256 model**

1. `train_256_ordinal_0.py`
2. `train_256_ordinal_1.py`

**384 model**

1. `train_384_ordinal_0.py`
2. `train_384_ordinal_1.py`
3. `train_384_ordinal_2.py`

**299 model**

1. `train_299_0.py`
2. `train_299_1.py`
3. `train_299_2.py`
4. `train_299_3.py`

## Inference

Check out our [inference notebook](https://www.kaggle.com/qitvision/panda-r-hm-ai-private-score-0-93) that uses `256` and `384` models.