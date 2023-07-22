from mosquito.helpers import get_new_run_dir_params


def main():
    # get additional params for new runs
    params = get_new_run_dir_params()

    # set text equal to those params
    text = ""
    for k, v in params.items():
        text += f" {k}={v}"

    return text


if __name__ == "__main__":
    print(main())