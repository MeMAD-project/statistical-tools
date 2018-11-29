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

    counts = get_counts(results)

    return counts, n_entries


def main(args):
    # Load data frame with categories matched
    df = pd.read_pickle(args.data_file)
    word_cnt, synset_cnt, category_cnt = process_data(df, args.category, args.groupby)
    va.plot_bar_counts(word_cnt)


# TODO supply names for plots!
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file', nargs='?', type=str, default='',
                        help='Data file to load')
    parser.add_argument('--category', type=str, default='location',
                        help='Name of the field we are interested in')
    parser.add_argument('--group_by', type=str, default='image_id',
                        help='name of the column to group by - '
                             'we usually want data grouped by images')
    parser.add_argument('--output_path', type=str, default='results/',
                        help='path where to save output images and csvs')

    args = parser.parse_args()

    main(args=args)
