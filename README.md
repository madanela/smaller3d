# smaller3d
### smaller models for 3D Semantic Segmentation using Minkowski engine and Knowledge Distillation methods

Paper is available at: ```paper``` folder

arxiv: [arxiv pdf](http://arxiv.org/abs/2305.03188) soon.


## Structure

Project structure is under ```config_scripts/README.md```

## Dependencies
Project was tested on 
```yaml
cuda: 10.1
python: 3.7.7 # suggested to use python>=3.7.7 and python<3.8 as some dependencies are not correct
poetry: 1.2.2
neptune-client: >=0.14.2 # for monitoring
```
## Installation



```yaml
# install python evironment
pyenv install 3.7.7
pyenv virtualenv 3.7.7 smaller3d
pyenv local smaller3d
pyenv activate

# install all dependencies
python -m pip install poetry # ==1.2.2
python -m poetry install
```

After sucessfully installing poetry installing is one line command only
```yaml
poetry install
```





## Preprocessing
Dataset should be dropped under ```data/raw/*/``` and preprocessed dataset after script running under ```data/processed/*/```. For our case ```"*" = scannet"```  

After the dependencies are installed, it is important to run the preprocessing scripts. They will bring scannet   datasets to a single format. By default, the script expect to find datsets in the data/raw/ folder. Check scripts/preprocess_*.bash for more details.

```yaml
dvc repro scannet
``` 
This command will run the preprocessing for scannet and will save the result using the dvc data versioning system.


## Training

You have to train any model (for example mix3d model with a voxel size of 5cm and place it under ```  saved/baseline/*  ```)

After training the model and placing it. You can train here Distillation model, which should look like this

```yaml
poetry run train # by changing checkpoint_teacher to your model checkpoint inside  "smaller3d/conf/conf.yaml"
```

The parameters of suggested Loss are under ```  smaller3d/conf/loss/DistillationLoss.yaml  ```

## Additional
For more information check ```config_scripts/README.md```