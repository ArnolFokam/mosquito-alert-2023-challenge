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
mkdir data && cd data && aicrowd dataset download --challenge mosquitoalert-challenge-2023
unzip -qq test_images_phase1.zip -d test_images_phase1/ && unzip -qq train_images.zip -d train_images/
```

Note: you can also download the dataset manually at this [link](https://www.aicrowd.com/challenges/mosquitoalert-challenge-2023/dataset_files).


## HowTo

### Create a Pytorch Dataset Module

All datasets should be created in this [folder](/mosquito/datasets/). Here is a template to get you start.

```python
class MosquitoAlert:
    def __init__(self, cfg) -> None:
        raise NotImplementedError
    
    def __getitem__(self) -> Any:
        raise NotImplementedError
    
    def __len__(self) -> int:
        raise NotImplementedError
    
    @staticmethod
    def get_train_and_val_dataset(cfg):
        raise NotImplementedError
```

Here is the specifications for each required datasets method.

- `__init__`: Loads the data using the configurations in `cfg`.
- `__getitem__`: Returns a single data sample for model training.
- `__len__`: Returns the number of samples in the data.
- `get_train_and_val_dataset`: Returns a tuple of dataset objects (from your class). One for *training* and the other for *validation*. You can return a null value on the second element of the tuple if you do not wish to have a *validation* dataset.
    ```python
    @staticmethod
    def get_train_and_val_dataset(cfg):
        train_dataset = # dataset object creation logic
        return train dataset, None
    ```

Note:  You are free to add additional methods to your classes.


## TIPS
- You can easily add -> commit -> push with this helper command `make push commit="your commit messsage"`
- You can prevent any file or folder in you local version of the repository from being push to the remote version by adding the suffix ```_private``` to its name.