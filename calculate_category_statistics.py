import pandas as pd
import argparse
from collections import Counter
import analysis_funs as va


def flatten_list(li):
    """Flatten shallow nested list"""
    return [element for sub_list in li for element in sub_list]


def get_counts(results):
    words, synsets, categories = zip(*results)
    words = flatten_list(words)
    synsets = flatten_list(synsets)
    categories = flatten_list(categories)

    word_cnt = Counter(words)
    synset_cnt = Counter(synsets)
    categories_cnt = Counter(categories)

    return word_cnt, synset_cnt, categories_cnt


def get_element(match_list):
    """Get an element from a list of triplets of a type (word, synset, category),
    position indicated the index of the element in the triplet"""
    words, synsets, categories = zip(*match_list)
    return words, synsets, categories


def aggregate_category_in_group(group):
    """Return a set of words, synsets, category synsets in each group"""
    tuples = group.apply(lambda x: get_element(x))
    word_list, synset_list, category_list = zip(*tuples)
    words = list(set([element for tupl in word_list for element in tupl]))
    synsets = list(set([element for tupl in synset_list for element in tupl]))
    categories = list(
        set([element for tupl in category_list for _set in tupl for element in _set]))

    return words, synsets, categories


def filter_by_category(df, category, group_by):
    print("Removing rows that have no category matches")
    df_filtered = df[df[category].str.len() > 0]
    n_entries = len(df_filtered[group_by].unique())
    return df_filtered, n_entries


def group_results(df, group_by, category):
    """Creates groups of rows grouped by group_by (for example image_id)"""
    print("Grouping results by {}".format(group_by))
    grouped = df.groupby(group_by)[category]

    return grouped


def process_data(df, category, group_by):
    df, n_entries = filter_by_category(df, category, group_by)
    grouped = group_results(df, group_by, category)

    print("Aggregating category results by {}".format(group_by))
    results = grouped.apply(lambda x: aggregate_category_in_group(x))

    # counts is a nested list of length 3 containing following Counter objects:
    # counts[0]: Counter for words that occured in captions
    # counts[1]: -"- Synsets corresponding to the above words
    # counts[2]: -"- Parent synsets that were matched to the actual category
    # (location, temporal, etc)
    counts = get_counts(results)

    return counts, n_entries


def main(args):
    # Load data frame with categories matched
    df = pd.read_pickle(args.data_file)
    counts, n_unique_images = process_data(df, args.category, args.group_by)

    print("-" * 80)

    print("Synsets that belong to category {} found in {} unique images".format(
        args.category, n_unique_images))

    print("=" * 80)

    va.plot_and_output_csv(counters=[counts[0]],
                           names=['words'], maxnum=45,
                           title='{} {} - Words'.format(
                               args.output_name, args.category),
                           filename_base='{}_{}'.format(args.output_prefix,
                                                        args.category),
                           batch=True)

    print("=" * 80)

    va.plot_and_output_csv(counters=[counts[1]],
                           names=['synsets'], maxnum=45,
                           title='{} {} - Synsets'.format(
                               args.output_name, args.category),
                           filename_base='{}_{}'.format(args.output_prefix,
                                                        args.category),
                           batch=True)

    print("=" * 80)

    va.plot_and_output_csv(counters=[counts[2]],
                           names=['category_synsets'], maxnum=45,
                           title='{} {} - Category Synsets'.format(args.output_name,
                                                                   args.category),
                           filename_base='{}_{}'.format(args.output_prefix,
                                                        args.category),
                           batch=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file', nargs='?', type=str, default='',
                        help='Data file to load')
    parser.add_argument('--output_name', type=str,
                        help='Name to use in the output images, '
                        'should be dataset specific.')
    parser.add_argument('--output_prefix', type=str,
                        help='prefix to put in front of output files, '
                        'should be dataset specific')
    parser.add_argument('--category', type=str, default='location',
                        help='Name of the field we are interested in')
    parser.add_argument('--group_by', type=str, default='image_id',
                        help='name of the column to group by - '
                             'we usually want data grouped by images')
    # NOTE:
    # The below --output_path parameter is not currently in use. In order to enable it
    # analysis_funs.py needs to be modified so that all functions that perform
    # plotting and storing to CSV take one optional parameter output_path,
    # which is then used as a prefix where all the files are stored to.
    # The above functions all take parameter "filename", so it's easy to find them!

    # parser.add_argument('--output_path', type=str, default='results/',
    #                   help='path where to save output images and csvs')

    args = parser.parse_args()

    main(args=args)
