<div align="center">

# TransPross: 1D transformer for predicting protein secondary structure prediction

![TransPross Architecture](https://github.com/BioinfoMachineLearning/TransPro/blob/main/img/TransPross_Architecture.png)

</div>

## Description
1D transformer for predicting protein structural features (secondary structure)


## Installation
```bash
git clone https://github.com/BioinfoMachineLearning/TransPro.git
cd TransPro
mkdir env
python3.6 -m venv env/ss_virenv
source env/ss_virenv/bin/activate
pip install --upgrade pip
pip install -r requirments.txt
```

## Training data
The training protein targets were extracted from the Protein Data Bank(PDB) before May 2019 with the the sequence identity < 90%. The sequence length range: [50, 500]

All the required data for training are provided as below and avaiable at [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6762376.svg)](https://doi.org/10.5281/zenodo.6762376):
* Protein sequences in fasta file (fasta.tar.gz)
* Target id list for training
* MSA in a3m file (a3m.tar.gz is too large, stored at /bml/TransPro/a3m.tar.gz)
* True ss labels in 3 states (ss_3.tar.gz)
* True 3D structures in pdb file (atom.tar.gz)
* 5 trained TransPross models (model.tar.gz)

## Testing data
All the testing data for evaluation are provided as below:
* CASP test sets(CASP13, CASP14)

## Training
```bash
python MSA_transformer2_train.py --model_num 1 --N 6 --max_positions 1500  --BATCH_SIZE 5 --data_dir <train> --dataset <custom>

model_num: training list model
N: number of attention layers
max_positions: maximum number of sequences allowed in the input MSA
BATCH_SIZE: batch size
data_dir: folder path for storing data
dataset: training set name
```
## Inference
**Predicting with the single a3m file as the input:**
```bash
python MSA_transformer2_predict_batch.py -i <a3m_file>
e.g. python MSA_transformer2_predict_batch.py -i T1026.a3m
```

**Predicting multiple targets in one time:**
```bash
python MSA_transformer2_predict_batch.py --data_dir <test> --dataset <casp13>

If you want to predict multiple targets, you can create a test.lst file under the path /data_dir/dataset/test.lst in the format: <target_id> length
e.g test/casp13/test.lst

data_dir: folder path for storing data
dataset: testing set name
```
