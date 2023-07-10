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
```

Note: you can also download the dataset manually at this [link](https://www.aicrowd.com/challenges/mosquitoalert-challenge-2023/dataset_files).


## HowTo

### Create a Pytorch Dataset Module

All datasets should be created in this [folder](/mosquito/datasets/). Here is a template to get you start.

```python
class MosquitoAlert:
    class MosquitoAlert:
    def __init__(self, cfg) -> None:
        raise NotImplementedError
    
    def __getitem__(self) -> Any:
        raise NotImplementedError
    
    def __len__(self) -> int:
        raise NotImplementedError
    
    @staticmethod
    def get_train_and_test_dataset(cfg):
        raise NotImplementedError
```

Here is the specifications for each required datasets method.

- `__init__`: Should load the data using the configurations in `cfg`.
- `__getitem__`: Should return a single data sample used to train the model.
- `__len__`: Returns the number of samples in the data.
- `get_train_and_test_dataset`: Returns two dataset object (from your class). One for *training* and the other for *validation*.

Note: you can add additional methods to your class if you wish. Just make sure the required methods are also implemented.


## TIPS
- You can easily add -> commit -> push with this helper command `make push commit="your commit messsage"`