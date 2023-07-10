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


## TIPS
- You can easily add -> commit -> push with this helper command `make push "you commit message"`