import pickle
import os
import argparse
from pprint import pprint


def main(args):
    syns = args.sentences_pickle
    tags = '_'.join(os.path.splitext(syns)[0].split('_')[:-2]) + '.pkl'

    print("Loading synsets from {}".format(syns))
    sentence_syns = pickle.load(open(syns, 'rb'))
    print("Loading tagged sentences from {}".format(tags))
    sentence_tags = pickle.load(open(tags, 'rb'))

    assert len(sentence_syns) == len(sentence_tags)

    for i, sent_tags in enumerate(sentence_tags):
        print('Sentence {}'.format(i))
        if not sent_tags:
            continue
        sent, _ = zip(*sent_tags)
        sent = ' '.join(sent)
        print(sent)
        pprint(sentence_tags[i])
        pprint(sentence_syns[i])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sentences_pickle', type=str, default='',
                        help='Pickle file with sentence synsets')
    args = parser.parse_args()

    main(args=args)
