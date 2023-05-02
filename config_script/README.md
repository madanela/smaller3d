The following is code structe, that was used to test smaller3d.
Project is using hydra configs. Structure is similar to mix3d structure [mix3d](https://kumuji.github.io/mix3d/), with additional functionallities to satisfy the Knowledge Distillation problem.



### Code structure

```
├── smaller3d
│   ├── __init__.py
│   ├── __main__.py     <- the main file
│   ├── conf            <- hydra configuration files 
│   ├── datasets
│   │   ├── preprocessing       <- folder with preprocessing scripts
│   │   ├── semseg.py       <- indoor dataset
│   │   └── utils.py        <- code for mixing point clouds
│   ├── loss
│   │   ├── KD_loss.py      <- code for Knowledge Distillation Networks.
│   ├── logger
│   ├── models      <- MinkowskiNet models
│   ├── trainer
│   │   ├── __init__.py
│   │   └── trainer.py      <- train loop
│   └── utils
├── data
│   ├── processed       <- folder for preprocessed datasets
│   └── raw     <- folder for raw datasets
├── config_scripts
│   ├── README.md   <- details of test run and code structure
├── dvc.lock
├── dvc.yaml        <- dvc file to reproduce the data
├── poetry.lock
├── pyproject.toml      <- project dependencies
├── README.md
├── saved       <- folder that stores models and logs
```

To run base KnowledgeDistillation models, chech out versions we have tested
```yaml
#   for Smooth label KD methods Loss
  - teacher_model: baseline_simple_loss
  - student_model: half_simple_loss

#   for Smooth label KD methods + Encoder Loss
  - teacher_model: baseline_additional_loss
  - student_model: half_with_encoder_loss

#   for Smooth label KD methods + Encoder + Decoder Loss
  - teacher_model: baseline_additional_loss
  - student_model: half_with_encoder_decoder_loss

```


Default tests have been done with following, but for more smoother labels, you can increase temperature.
```yaml
temperature: 1
alpha: 1
```