# Import own functions for image analysis:
import analysis_funs as va
from nltk import word_tokenize
from nltk.tag import StanfordPOSTagger
from multiprocessing import Pool
import pickle
import math
from itertools import chain
import random
import argparse
import sys
import os


# Define Stanford POS Tagger as a global variable:
tagger = StanfordPOSTagger(
    'stanford-postagger-2017-06-09/models/english-bidirectional-distsim.tagger',
    path_to_jar='stanford-postagger-2017-06-09/stanford-postagger.jar',
    java_options='-mx4000m')


def collect_sentences_vg(data):
    """Collect all sentences from the VisualGenome regions and tag them"""
    sentences = []
    for img in data:
        for region in img['regions']:
            sentences.append(region['phrase'])

    return sentences


def collect_sentences_coco(data):
    """Collect all sentences from the VisualGenome regions and tag them"""
    sentences = []
    for ann in data['annotations']:
        sentences.append(ann['caption'])

    return sentences


def collect_sentences_vist_dii(data):
    """Collect all sentences from the VisualGenome regions and tag them"""
    sentences = []
    for ann in data['annotations']:
        sentences.append(ann[0]['text'])

    return sentences


def tag_sentences(sentences):
    global tagger
    sentences_tagged = tagger.tag_sents(word_tokenize(sent) for sent in sentences)

    return sentences_tagged


def tag_sentences_in_batch(batch):
    global tagger

    num_sentences = len(sentences)

    batch_size = math.floor(num_sentences / num_batches)

    if batch == 0:
        sentences_to_tag = sentences[:batch_size]
    elif batch == num_batches - 1:
        sentences_to_tag = sentences[batch_size * batch:]
    else:
        sentences_to_tag = sentences[batch_size * batch:batch_size * batch + batch_size]

    tagged_sentences = tag_sentences(tagger, sentences_to_tag)

    print("Finished batch {}".format(batch))

    return tagged_sentences


def main(args):
    if args.dataset is None:
        print("ERROR: No dataset specified. \n"
              "HINT: Please specify dataset for example: --dataset vg-regions")
        sys.exit(1)

    if args.input_file is None or not os.path.isfile(args.input_file):
        print('ERROR: Invalid input file. Please make sure you specify a valid input file '
              'corresponding to the dataset. \n'
              'HINT: Use --input_file /path/to/captions-file.json')
        sys.exit(1)

    if args.output_file is None:
        print('ERROR: No output file specified. '
              'Please make sure you specify a valid output file.\n'
              'HINT: Use --output_file /path/to/tagged_captions.txt')
        sys.exit(1)

    if os.path.isfile(args.output_file):
        print('ERROR: File {} already exists. '
              'Please rename or delete it'.format(args.output_file))
        sys.exit(1)

    global sentences
    global num_batches

    print("Loading tagger...")

    if args.dataset == 'vg-regions':
        collect_sentences = collect_sentences_vg
        raw_data = va.load_json(args.input_file)
    elif args.dataset == 'coco':
        collect_sentences = collect_sentences_coco
        raw_data = va.load_json(args.input_file)
    elif args.dataset == 'vist-dii':
        collect_sentences = collect_sentences_vist_dii
        raw_data = va.load_json(args.input_file)
    else:
        print("Invalid dataset {}".format(args.dataset))
        sys.exit(1)

    print("Collecting all sentences...")
    sentences = collect_sentences(raw_data)
    print('{} sentences collected from dataset {} load from {}'.format(
        len(sentences), args.dataset, args.input_file))

    random_indices = [random.randint(0, len(sentences)) for i in range(5)]

    print("5 random raw sentences at indices: {}".format(random_indices))
    for i, idx in enumerate(random_indices):
        print('{}: {}'.format(i + 1, sentences[idx]))

    num_batches = 15
    tasks = [x for x in range(num_batches)]
    # Number of threads to use:
    num_workers = 4

    print("Starting tagging, using {} batches and {} workers"
          .format(num_batches, num_workers))

    with Pool(num_workers) as p:
        sentences_tagged = p.map(tag_sentences_in_batch, tasks)

    sentences_tagged = list(chain(*sentences_tagged))

    print('{} tagged sentences produced...'.format(len(sentences_tagged)))

    print("5 random tagged sentences:")
    for i, idx in enumerate(random_indices):
        print('{}: {}'.format(i + 1, sentences_tagged[idx]))

    with open(args.output_path, 'wb') as f:
        pickle.dump(sentences_tagged, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        help='Dataset from which to extract tagged sentences. '
                        'Possible values: vg-regions, coco, vist-dii')
    parser.add_argument('--input_file', type=str,
                        help='File containing the text that we want tagged')
    parser.add_argument('--output_file', type=str,
                        help='Output file containing tagged sentences')

    args = parser.parse_args()

    main(args)
