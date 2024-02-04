import argparse
import json
from collections import (
    deque,
)  # just for fun using dequeue instead of just a list for faster appends
from pprint import pprint


class LoadFromFile(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        data_dict = json.load(values)
        arg_list = deque()
        action_dict = {e.option_strings[0]: e for e in parser._actions}
        for i, e in enumerate(data_dict):
            arg_list.extend(self.__build_parse_arge__(e, data_dict, action_dict))
        parser.parse_args(arg_list, namespace=namespace)

    def __build_parse_arge__(self, arg_key, arg_dict, file_action):
        arg_name = f"--{arg_key}"
        arg_val = str(arg_dict[arg_key]).replace(
            "'", '"'
        )  # list of text need to be modified so they can be parsed properly
        try:
            file_action[arg_name].required = False
        except:
            raise KeyError(
                f"The Key {arg_name} is not an expected parameter. Delete it from config or update build_args method in helper_utils.configs.py"
            )
        return arg_name, arg_val


def parse_bool(s: str):
    return eval(s) == True


def build_args():
    """Parses args. Must include all hyperparameters you want to tune.

    Special Note:
        Since i entirely expect to load from config files the behavior is
        1.
    """
    parser = argparse.ArgumentParser(description="Confguration for png extraction")
    parser.add_argument(
        "--DICOMHome",
        required=True,
        type=str,
        help="Path to where dicom images are stored",
    )  # TODO: uPDATE README TO EXPLAIN CONFI OF PICKLE FILE
    parser.add_argument(
        "--ConfigPath",
        required=False,
        type=open,
        action=LoadFromFile,
        help="Path to your config file",
    )
    parser.add_argument(
        "--OutputDirectory",
        required=True,
        type=str,
        help="Directory to Store dicomes",
    )
    parser.add_argument(
        "--SaveBatchSize",
        required=True,
        type=int,
        default=2000,
        help="Save the metadata in in batches of N as we extracted them ",
    )
    parser.add_argument(
        "--SavePNGs", required=True, type=bool, help="Save Images as PNGs"
    )
    parser.add_argument("--NumProcesses", type=int, required=True)
    parser.add_argument("--PublicHeadersOnly", type=bool, required=True, default=True)
    parser.add_argument("--SpecificHeadersOnly", type=str, required=True, default=False)
    return parser


def get_params():
    args = build_args()
    my_args = args.parse_args()
    arg_dict = vars(my_args)
    return arg_dict
