# Info

The following is README file of smaller3d paper. It is using Knowledge Distillation methods over 3d Semantic Segmentation task, to test KD methods on 3D spatio-temporal convolutions



# Structure

Project is using hydra configs. Structure is similar to mix3d structure [mix3d](https://kumuji.github.io/mix3d/)


# Dependencies
Project was tested on 
```
cuda: 10.1
python: 3.7.7 # suggested to use python>=3.7.7 and python<3.8 as some dependencies are not correct
poetry: 1.2.2
neptune-client: >=0.14.2 # for monitoring
```
# Installation



```
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
```
poetry install
```





# Preprocessing
Dataset should be dropped under ```data/raw/*/``` and preprocessed dataset after script running under ```data/processed/*/```. For our case ```"*" = scannet"```  

After the dependencies are installed, it is important to run the preprocessing scripts. They will bring scannet   datasets to a single format. By default, the script expect to find datsets in the data/raw/ folder. Check scripts/preprocess_*.bash for more details.

```
dvc repro scannet
``` 
This command will run the preprocessing for scannet and will save the result using the dvc data versioning system.


# Training

You have to train any model (for example mix3d model with a voxel size of 5cm and place it under ```  saved/baseline/*  ```)

After training the model and placing it. You can train here Distillation model, which should look like this

```
poetry run train # by changing checkpoint_teacher to your model checkpoint inside  "smaller3d/conf/conf.yaml"
```

The parameters of suggested Loss are under ```  smaller3d/conf/loss/DistillationLoss.yaml  ```

Default tests have been done with 

```

temperature: 2
alpha: 1

```