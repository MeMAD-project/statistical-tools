# import pickle
# import pandas as pd
import argparse
from pandas.io.json import json_normalize  # package for flattening json in pandas df
import analysis_funs as va


def flatten_json(data, record_path):
    flattened_data = json_normalize(data, record_path)
    return flattened_data


def main(args):
    data = va.load_data(args.json_file)

    if args.data_root:
        print("Setting data root to {}".format(args.data_root))
        data = data[args.data_root]

    print("Flattening JSON file and storing as data frame, using {} as record_path".
          format(args.record_path))

    df = flatten_json(data, args.record_path)

    print("Saving data to {}".format(args.output_path))

    df.to_feather(args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file', type=str, default='',
                        help='Path to json file holding records')
    parser.add_argument('--record_path', type=str, default=None,
                        help="name of the record that will correspond to 1 row in pandas dataset")
    parser.add_argument('--data_root', type=str, default=None,
                        help='for example if we want root to be data[\'images\'] pass \'images\'')
    parser.add_argument('--output_path', type=str, default=None,
                        help='path to save files to')
    args = parser.parse_args()

    main(args=args)
