import warnings
import argparse
import json
from pandas.io.json import json_normalize
from categories import Categories
import sys
from nltk.corpus import wordnet as wn
import pandas as pd
from tqdm import tqdm
import os

warnings.filterwarnings(
    "ignore", message="numpy.dtype size changed, may indicate binary incompatibility. Expected 96, got 88")
tqdm.pandas()


def get_hypernyms(synset):
    """Returns top level category for the synset given"""

    term_list = []

    for s in synset.hypernyms():
        term_list += [s.name()]
        h = get_hypernyms(s)
        if len(h):
            term_list += h

    return (set(term_list) | set([synset.name()]))


def synset_in_category(synset, category):
    hypernyms = get_hypernyms(wn.synset(synset))
    return hypernyms & category


def get_category_synsets_for_row(phrase, category):
    """Takes a phrase as a list of (word, synset) tuples and outputs a
    a list of (word, synset, category_synset) tuples or None if no matches found"""
    results = []
    for (word, synset) in phrase:
        result = synset_in_category(synset, category)
        if result:
            results.append((word, synset, result))

    if not results:
        return None
    else:
        return results


def dataset_to_pandas(data_file, dataset):
    """Accept raw JSON/TXT/other data that is then formatted into a same
    Pandas data frame based on the dataset-specific rules"""

    if dataset == 'vg-regions':
        with open(data_file) as f:
            data = json.load(f)
        # Creates Pandas data frame with 'regions' object as a root:
        df = json_normalize(data, 'regions')
    elif dataset == 'coco':
        df = None
    else:
        print("ERROR: Unknown dataset: {}.".format(dataset))
        sys.exit(1)

    return df


def main(args):
    print("Loading {} caption data from {}".format(args.dataset, args.data_file))
    df = dataset_to_pandas(args.data_file, args.dataset)

    print("Loading sentence synsets from {}".format(args.synset_file))
    sentence_syns = pd.read_json(args.synset_file, typ='series')

    print("Loading category '{}'".format(args.category))
    category = set(Categories[args.category])

    print("Getting data rows matching category '{}'".format(args.category))
    category_synsets = sentence_syns.progress_apply(
        lambda x: get_category_synsets_for_row(x, category))

    print("Adding results to master data")
    df[args.category] = category_synsets

    file_name = os.path.basename(args.synset_file)
    file_name_no_ext = os.path.splitext(file_name)[0]
    output_file_name_json = '{}_{}.json'.format(file_name_no_ext, args.category)
    output_file_name_pandas = '{}_{}.pandas.pkl'.format(file_name_no_ext, args.category)
    output_file_json = os.path.join(args.output_path, output_file_name_json)
    output_file_pandas = os.path.join(args.output_path, output_file_name_pandas)

    print("Saving results to {} for user analysis".format(output_file_json))
    df.to_json(output_file_json, orient='records')
    print("Saving results to {} for further steps in the pipeline.".format(output_file_pandas))
    df.to_pickle(output_file_pandas)
    print("DONE!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='vg-regions',
                        help='Dataset to use')
    parser.add_argument('--data_file', type=str,
                        help='original dataset in JSON/TXT/other format')
    parser.add_argument('--synset_file', type=str,
                        help='JSON file containing synsets for the dataset. '
                        'Should have the same number of entries as the dataset file.')
    parser.add_argument('--category', type=str, default='location',
                        help='Which category to do the stats for')
    parser.add_argument('--output_path', type=str, default='output/',
                        help='path where to save output data (both JSON and serialized Pandas')

    args = parser.parse_args()

    main(args=args)
