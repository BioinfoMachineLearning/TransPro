<div align="center">

# TransPross: 1D transformer for predicting protein secondary structure prediction

![TransPross Architecture](https://github.com/BioinfoMachineLearning/TransPro/blob/main/img/TransPross_Architecture.png)

</div>

## Description
1D transformer for predicting protein structural features (secondary structure)


## Training data
The training protein targets were extracted from the Protein Data Bank(PDB) before May 2019 with the the sequence identity < 90%. The sequence length range: [50, 500]

All the required data for training are provided as below:
* Protein sequences in fasta file (fasta.tar.gz)
* Target id list for training
* MSA in a3m file (a3m.tar.gz)
* True ss labels (ss.tar.gz)
* True 3D structures in pdb file (atom.tar.gz)
* 5 trained TransPross models (model.tar.gz)

## Testing data
All the testing data for evaluation are provided as below:
* CASP test set
* CAMEO test set

