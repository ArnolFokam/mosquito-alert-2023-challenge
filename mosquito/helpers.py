import os


def get_dir(*paths) -> str:
    directory = os.path.join(*paths)
    os.makedirs(directory, exist_ok=True)
    return directory