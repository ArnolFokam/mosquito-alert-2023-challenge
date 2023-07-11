## Requirements
- Python ```3.9.16```
- Pip ```23.1.2```

## Setup Environment

This process assumes you have create an python virtual environment with [conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) or [virtualenv](https://docs.python.org/3/library/venv.html)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Dataset

[Create an account](https://www.aicrowd.com/participants/sign_up) and [register](https://www.aicrowd.com/challenges/mosquitoalert-challenge-2023) for the challenge. Then, run the following commands.

```bash
aicrowd login

cd # [repository directory]
mkdir data && aicrowd dataset download --challenge mosquitoalert-challenge-2023 -o data/
unzip -qq data/test_images_phase1.zip -d test_images_phase1/
unzip -qq data/train_images.zip -d train_images/
```

Note: you can also download the dataset manually at this [link](https://www.aicrowd.com/challenges/mosquitoalert-challenge-2023/dataset_files).


## HowTo

### Perform a Training Experiment

1. Create a `yaml` configuration for the training experiment in [exps](/exps/). Then, run this command.

```python train.py --config-path=exps --config-name=[config filename]```

See [tips](#tips) for shorcut command

### Create a Pytorch Dataset Module

All datasets should be created in this [folder](/mosquito/datasets/). Here is a template to get you started.

```python
class MosquitoAlertDataset(BaseDataset):
    def __init__(self, 
                 cfg,
                 transform: Optional[callable] = None) -> None:
        super().__init__(cfg)
    # dataset's logic...
```

Here are the required methods  and property. See [BaseDataset](/mosquito/datasets/base.py#L6).

- `__init__`: Loads the data using the configurations in `cfg` and transform pipeline (see [section](#create-a-transform-pipeline)).
- `num_classes`(property): Number of classes in our dataset
- `__getitem__`: Returns a single data sample for model training.
- `__len__`: Returns the number of samples in the data.
- `get_train_and_val_dataset`(staticmethod): Returns a tuple of dataset objects (from your class). One for *training* and the other for *validation*. You can return a null value on the second element of the tuple if you do not wish to have a *validation* dataset.
    ```python
    @staticmethod
    def get_train_and_val_dataset(cfg, transfrom):
        train_dataset = # dataset object creation logic
        return train dataset, None
    ```
- `collate_fn` (staticmethod): Returns a collated version of the batch of samples. Check this [link](https://pytorch.org/docs/stable/data.html#dataloader-collate-fn) to learn more.

Note:  You are free to add additional methods to your classes.

Do not forget to add your dataset class [here](/mosquito/datasets/__init__.py).

### Create a Transform Pipeline

All transforms should be created in this [folder](/mosquito/transforms/). Here is a template to get you start.

```python
class MosquitoAlertTransform(BaseTransform):
    def __init__(self, 
                 cfg,) -> None:
        super().__init__(cfg)
    # transform's logic...
```

Here are the required methods. See [BaseTransform](/mosquito/transforms/base.py#L6).

- `__init__`: Creates the transform pipeline (you could use [torchvision.transforms](https://pytorch.org/vision/stable/transforms.html) or [albumentations](https://albumentations.ai/)). The choice is up to you.
- `__call__`: Receives a Numpy image array and a bounding box to return a tuple consisting of a transformed tensor image and the target values according to the requirement of the model you created using this [section](#create-a-model).

Note:  You are free to add additional methods to your transforms.

Do not forget to add your transform class [here](/mosquito/transforms/__init__.py).

### Create a Model

All models should be created in this [folder](/mosquito/models/). Here is a template to get you start.

```python
class MosquitoAlertModel(BaseModel):
    def __init__(self, 
                 cfg,) -> None:
        super().__init__(cfg)
    # model's logic...
```

Here are the required methods. See [BaseModel](/mosquito//models/base.py#).

- `__init__`: Creates the architecture of the model.
- `forward`: Receives a batch of tensor images and bounding box and returns a dictionary of losses (See [website](https://pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html#torchvision.models.detection.fasterrcnn_resnet50_fpn) for help).
- `configure_optimizers`: Returns a list of optimizers. You can just return a list with a single optimizer if all the parameters of the model should be optimized the same way.

Note:  You are free to add additional methods to your model.

Do not forget to add your model class [here](/mosquito/models/__init__.py).

## TIPS
- You can easily add -> commit -> push with this helper command `make push commit="your commit messsage"`
- You can prevent any file or folder in you local version of the repository from being push to the remote version by adding the suffix ```_private``` to its name.
- You can use the following shorcut command to train model `make train exp=[configurarion filename]`.
