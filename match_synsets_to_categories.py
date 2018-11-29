import warnings
import argparse
from collections import namedtuple
from categories import Categories
import sys
from nltk.corpus import wordnet as wn
import pandas as pd
from tqdm import tqdm
import os

warnings.filterwarnings(
    "ignore", message="numpy.dtype size changed, may indicate binary incompatibility. Expected 96, got 88")
tqdm.pandas()

DatasetConfig = namedtuple('DatasetConfig', 'captions, sentence_syns')


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


def main(args):
    dataset_config = {
        'vg-regions': DatasetConfig('pickles/vg_region_descriptions.feather',
                                    'pickles/vg_sentences_tagged_syns.pandas.pkl'),
        'coco': DatasetConfig('pickles/coco_captions_train2014.feather',
                              'pickles/coco_sentences_tagged_syns.pandas.pkl')
    }

    config = dataset_config.get(args.dataset)
    if not config:
        print('Invalid dataset specified')
        sys.exit(1)

    print("Loading caption data from {}".format(config.captions))
    df = pd.read_feather(config.captions)

    print("Loading sentence synsets from {}".format(config.sentence_syns))
    sentence_syns = pd.read_pickle(config.sentence_syns)

    print("Loading category '{}'".format(args.category))
    category = set(Categories[args.category])

    print("Getting data rows matching category '{}'".format(args.category))
    category_synsets = sentence_syns.progress_apply(
        lambda x: get_category_synsets_for_row(x, category))

    print("Adding results to master data")
    df[args.category] = category_synsets

    file_name = os.path.basename(config.sentence_syns)
    file_name_no_ext = os.path.splitext(file_name)[0]
    output_file_name = '{}_{}.pkl'.format(file_name_no_ext, args.category)
    output_path = os.path.join(args.output_path, output_file_name)

    print("Saving results to {}".format(output_path))
    df.to_pickle(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='vg-regions',
                        help='Dataset to use')
    parser.add_argument('--category', type=str, default='location',
                        help='Which category to do the stats for')
    parser.add_argument('--output_path', type=str, default='pickles/',
                        help='path where to save output images and csvs')

    args = parser.parse_args()

    main(args=args)
