# SMFM-master
The repository is developed for enhancer sequences identification using Stacked Multivariate Fusion Framework (SMFM) 

# Requirements
- python == 3.6.2

- torch==1.10.0+cu113

- tensorflow==2.5.1

- transformers==4.12.5

- deep-forest==0.1.5

- Ubuntu 18.04 (64-bit)

# Usage
## Extract dynamic semantic information using EnhancerBERT
You can extract dynamic semantic information by following a few steps:
- Download EnhancerBERT models and requirement.txt in http://39.104.69.176:5010/.
- Create virtual environment with following command:

>***1)*** conda create env -n EnhancerBERT.  
>***2)*** conda activate EnhancerBERT.  
>***3)*** pip install requirement.txt.  

Revise the path where the EnhancerBERT locate and run extract_information.py
After running extract_information.py, the progress of learning implicit relations and long-distance dependicies can be performed by running dl_based_sequence_network.py
Finally you can get a .npy file when you have finished the whole process

## Generate multi-source biological features
You can generate multi-source biological features by following a few steps:

- Activate the virtual environment described above and run the multi-source_biological_features.py.
- When run the code, you should be aware of the following points:

>***1)*** The sequences should be strored as a .fasta or a .txt file.  
>***2)*** The labels of sequences should be stored independently as a .txt file.  
>***3)*** After the code has finished running, the features will be store as a .csv file, and please upload the fulldataset instead of optimumdataset.

## Predict sequences using dynamic semantic information and multi-source biological features 
Run ensemble_classification.py to get the prediction results for the sequences.
