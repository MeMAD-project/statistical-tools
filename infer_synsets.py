"""Disambiguate word senses"""
# import warnings
# warnings.filterwarnings("ignore", message="numpy.dtype size changed, may indicate binary incompatibility. Expected 96, got 88")
# warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import argparse
import sys
from datetime import datetime
import os
import json
from tqdm import tqdm
import random
from nltk.corpus import wordnet as wn
import pandas as pd
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
tqdm.pandas()


# Load a pickle file, run it through a word sense disambiguator
# Save pickle file with most likely senses

# A mapping between penn treebank and WordNet
# WordNet POS tags are: NOUN = 'n', ADJ = 's', VERB = 'v', ADV = 'r', ADJ_SAT = 'a'
tag_map = {
    'CC': None,  # coordin. conjunction (and, but, or)
    'CD': 'n',  # cardinal number (one, two)
    'DT': None,  # determiner (a, the)
    'EX': 'r',  # existential ‘there’ (there)
    'FW': None,  # foreign word (mea culpa)
    'IN': 'r',  # preposition/sub-conj (of, in, by)
    'JJ': ['a', 's'],  # adjective (yellow)
    'JJR': ['a', 's'],  # adj., comparative (bigger)
    'JJS': ['a', 's'],  # adj., superlative (wildest)
    'LS': None,  # list item marker (1, 2, One)
    'MD': None,  # modal (can, should)
    'NN': 'n',  # noun, sing. or mass (llama)
    'NNS': 'n',  # noun, plural (llamas)
    'NNP': 'n',  # proper noun, sing. (IBM)
    'NNPS': 'n',  # proper noun, plural (Carolinas)
    'PDT': ['a', 's'],  # predeterminer (all, both)
    'POS': None,  # possessive ending (’s )
    'PRP': None,  # personal pronoun (I, you, he)
    'PRP$': None,  # possessive pronoun (your, one’s)
    'RB': 'r',  # adverb (quickly, never)
    'RBR': 'r',  # adverb, comparative (faster)
    'RBS': 'r',  # adverb, superlative (fastest)
    'RP': ['a', 's'],  # particle (up, off)
    'SYM': None,  # symbol (+,%, &)
    'TO': None,  # “to” (to)
    'UH': None,  # interjection (ah, oops)
    'VB': 'v',  # verb base form (eat)
    'VBD': 'v',  # verb past tense (ate)
    'VBG': 'v',  # verb gerund (eating)
    'VBN': 'v',  # verb past participle (eaten)
    'VBP': 'v',  # verb non-3sg pres (eat)
    'VBZ': 'v',  # verb 3sg pres (eats)
    'WDT': None,  # wh-determiner (which, that)
    'WP': None,  # wh-pronoun (what, who)
    'WP$': None,  # possessive (wh- whose)
    'WRB': None,  # wh-adverb (how, where)
    '$': None,  # dollar sign ($)
    '#': None,  # pound sign (#)
    '“': None,  # left quote (‘ or “)
    '”': None,  # right quote (’ or ”)
    '(': None,  # left parenthesis ([, (, {, <)
    ')': None,  # right parenthesis (], ), }, >)
    ',': None,  # comma (,)
    '.': None,  # sentence-final punc (. ! ?)
    ':': None,  # mid-sentence punc (: ; ... – -)
    '``': None,
    "''": None
}


def sentence_to_synsets_top(phrase_tagged):
    syns = []

    if not phrase_tagged:
        return syns

    for (word, tag) in phrase_tagged:
        # If WordNet has a corresponding tag:
        pos = tag_map[tag]
        if pos:
            if is_word_pos(word, pos):
                syn = top_synset(word, pos)
                if syn:
                    word = wordnet_lemmatizer.lemmatize(word.lower())
                    syn = syn.name()
                    syns.append((word, syn))

    return syns


def is_word_pos(word, pos):
    is_pos = False
    if wn.synsets(word, pos):
        is_pos = True

    return is_pos


def max_dict(d):
    """Return dictionary key with the maximum value"""
    if d:
        return max(d, key=lambda key: d[key])
    else:
        return None


def top_synset(word, tags):
    synsets = wn.synsets(word, tags)

    top_synset = None
    if synsets:
        top_synset = synsets[0]

    return top_synset


def main(args):
    print("Loading POS tagged sentences from: {}".format(args.pos_tagged_sentences))
    sentence_data = json.load(open(args.pos_tagged_sentences, 'rb'))

    if args.max_sentences is not None and args.max_sentences < len(sentence_data):
        print("WARNING: Limiting the number of sentences being processed to {}".format(
            args.max_sentences))
        sentence_data = sentence_data[:args.max_sentences]

    sentences = pd.Series(sentence_data)
    num_sentences = len(sentences)

    print("Loaded {} sentences...".format(num_sentences))

    # Added index at 34222 which is known to have adjectives we need:
    test_indices = [random.randint(0, num_sentences) for i in range(5)]

    print("5 random sentences at indices: {}".format(test_indices))
    for i, idx in enumerate(test_indices):
        print('{}: {}'.format(i + 1, sentences.iloc[idx]))

    begin = datetime.now()
    print("Starting infering synsets at {}".format(begin))

    sentence_syns = sentences.progress_apply(lambda x: sentence_to_synsets_top(x))
    sentence_syns.name = 'sentence_syns'

    end = datetime.now()

    print("Synset inference ended at {}. Total inference time: {}.".format(end, end - begin))
    print("Obtained most likely synsets for {} sentences".format(len(sentence_syns)))

    print("5 random sentences with synsets: {}".format(test_indices))
    for i, idx in enumerate(test_indices):
        print('{}: {}'.format(i + 1, sentence_syns.iloc[idx]))

    file_name = os.path.basename(args.pos_tagged_sentences)
    file_name_no_ext = os.path.splitext(file_name)[0]
    output_file_name = '{}_syns.json'.format(file_name_no_ext)
    output_path = os.path.join(args.output_path, output_file_name)

    print("Saving results to {}".format(output_path))

    sentence_syns.to_json(output_path)

    print("DONE!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('pos_tagged_sentences', nargs='?', type=str, default='',
                        help='Path to JSON file with tagged sentences')
    parser.add_argument('--output_path', type=str, default='output/',
                        help='path where to save output pickles')
    parser.add_argument('--max_sentences', type=int,
                        help='Used for debugging - set cap on the number of sentences')

    args = parser.parse_args()

    main(args=args)
