# Run BPNet

Supported python version is 3.6. After installing anaconda ([download page](https://www.anaconda.com/download/)) or miniconda ([download page](https://conda.io/miniconda.html)), create a new bpnet environment by executing the following code:

```
# create 'motif' conda environment
conda env create -f conda-env.yml

# Activate the conda environment
source activate motif
conda install -c bioconda pybedtools bedtools pybigwig pysam genomelake
pip install git+https://github.com/kundajelab/DeepExplain.git
pip install tensorflow~=1.0 # or tensorflow-gpu if you are using a GPU
pip install bpnet
echo 'export HDF5_USE_FILE_LOCKING=FALSE' >> ~/.bashrc
```
## Step

Train a model on BigWig tracks specified in *dataspec.yml* using an existing architecture bpnet:

```bash
bpnet train --premade=bpnet9 dataspec.yml
```

Compute contribution scores for regions specified in the `dataspec.yml` file and store them into `contrib.scores.h5`

```bash
bpnet contrib . --method=deeplift contrib.scores.h5
```

Export BigWig tracks containing model predictions and contribution scores

```bash
bpnet export-bw . --regions=intervals.bed --scale-contribution bigwigs/
```

Discover motifs with TF-MoDISco using contribution scores stored in `contrib.scores.h5`, premade configuration [modisco-50k](bpnet/premade/modisco-50k.gin) and restricting the number of seqlets per metacluster to 20k:

```bash
bpnet modisco-run contrib.scores.h5 --premade=modisco-50k modisco/
```


# Run SMFM

## Install deeplift and tfmodisco
DeepLIFT and tfmodisco are on pypi, which can be installed using pip:
```unix
pip install deeplift
pip install modisco
```

While DeepLIFT does not require your models to be trained with any particular library, we have provided autoconversion functions to convert models trained using Keras into the DeepLIFT format. If you used a different library to train your models, you can still use DeepLIFT if you recreate the model using DeepLIFT layers.

This implementation of DeepLIFT was tested with tensorflow 1.7, and autoconversion was tested using keras 2.0.

After training and save the deep learning-based sequence network, using deeplift to convert the model to appropriate format. TF-MoDISco then uses the contrib.scores.h5 obtained by deeplift for motif detection (see above).

```python
import deeplift
from deeplift.conversion import kerasapi_conversion as kc

deeplift_model =\
    kc.convert_model_from_saved_files(
        saved_hdf5_file_path,
        nonlinear_mxts_mode=deeplift.layers.NonlinearMxtsMode.DeepLIFT_GenomicsDefault) 
        
find_scores_layer_idx = 0


#Compile the function that computes the contribution scores
deeplift_contribs_func = deeplift_model.get_target_contribs_func(
                            find_scores_layer_idx=find_scores_layer_idx,
                            target_layer_idx=-1)
#compute scores on inputs                        
scores = np.array(deeplift_contribs_func(task_idx=0,
                                         input_data_list=[X],
                                         batch_size=10,
                                         progress_update=1000))
```


