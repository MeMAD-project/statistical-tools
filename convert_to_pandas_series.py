import pickle
import pandas as pd
import argparse
import sys
import os


def main(args):
    print("Loading {}".format(args.pickle_file))
    input = pickle.load(open(args.pickle_file, 'rb'))
    if isinstance(input, pd.Series):
        print("Pickled file already a Pandas series. Do nothing.")
        sys.exit(0)

    print("Converting to pandas dataframe...")
    series = pd.Series(input, name=args.series_name)

    if args.format == 'pickle':
        output_filename = os.path.splitext(args.pickle_file)[0] + '.pandas.pkl'
        print("Saving results to {}".format(output_filename))

        with open(output_filename, 'wb') as f:
            pickle.dump(series, f)

    elif args.format == 'feather':
        output_filename = os.path.splitext(args.pickle_file)[0] + '.pandas.feather'
        print("Writing to {}".format(output_filename))
        series.to_frame().to_feather(output_filename)

    else:
        print("Invalid file format specified")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pickle_file', type=str, default='',
                        help='Path to pickle which should be converted to pandas series')
    parser.add_argument('--series_name', type=str, default='',
                        help="name of the series")
    parser.add_argument('--format', type=str, default='pickle',
                        help='file format to save into can be pickle or feather')
    args = parser.parse_args()

    main(args=args)
