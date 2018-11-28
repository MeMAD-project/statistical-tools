from nltk import word_tokenize
from nltk.tag import StanfordPOSTagger
from multiprocessing import Pool
import json
import math
from itertools import chain
import random
import argparse
import sys
import os
from datetime import datetime


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

    tagged_sentences = tag_sentences(sentences_to_tag)

    print("Finished batch {}".format(batch))

    return tagged_sentences


def main(args):
    global tagger
    global sentences
    global num_batches

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

    num_batches = args.num_batches

    if args.dataset == 'vg-regions':
        collect_sentences = collect_sentences_vg
    elif args.dataset == 'coco':
        collect_sentences = collect_sentences_coco
    elif args.dataset == 'vist-dii':
        collect_sentences = collect_sentences_vist_dii
    else:
        print("Invalid dataset {}".format(args.dataset))
        sys.exit(1)

    print("Loading tagger...")
    tagger = StanfordPOSTagger(
        'stanford-postagger-2017-06-09/models/english-bidirectional-distsim.tagger',
        path_to_jar='stanford-postagger-2017-06-09/stanford-postagger.jar',
        java_options='-mx4000m')

    print("Loading dataset {}".format(args.dataset))
    with open(args.input_file) as f:
        raw_data = json.load(f)

    print("Collecting all sentences...")
    sentences = collect_sentences(raw_data)

    # used for debugging:
    if args.max_sentences is not None and args.max_sentences < len(sentences):
        sentences = sentences[:args.max_sentences]
        print("WARNING: Tagging only first {} sentences.".format(args.max_sentences))

    if num_batches > len(sentences):
        print("WARNING: There are more batches than sentences. "
              "Making the number of batches equal to the number of sentences.")
        num_batches = len(sentences)

    print('{} sentences collected from dataset {} loaded from {}'.format(
        len(sentences), args.dataset, args.input_file))

    random_indices = [random.randint(0, len(sentences) - 1) for i in range(5)]

    print("5 random raw sentences at indices: {}".format(random_indices))
    for i, idx in enumerate(random_indices):
        print('{}: {}'.format(i + 1, sentences[idx]))

    begin = datetime.now()
    print("Starting tagging at {}, using {} batches and {} workers".format(
        begin, num_batches, args.num_workers))

    # Perform the actual tagging using multiple threads:
    with Pool(args.num_workers) as p:
        sentences_tagged = p.map(tag_sentences_in_batch, range(num_batches))

    end = datetime.now()

    print('Tagging ended at {}. Total tagging time: {}.'.format(end, end - begin))

    sentences_tagged = list(chain(*sentences_tagged))

    print('{} tagged sentences produced...'.format(len(sentences_tagged)))

    print("5 random tagged sentences:")
    for i, idx in enumerate(random_indices):
        print('{}: {}'.format(i + 1, sentences_tagged[idx]))

    print("Storing tagged sentenes in {}".format(args.output_file))
    path = os.path.dirname(args.output_file)
    os.makedirs(path, exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(sentences_tagged, f)

    print("DONE!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        help='Dataset from which to extract tagged sentences. '
                        'Possible values: vg-regions, coco, vist-dii')
    parser.add_argument('--input_file', type=str,
                        help='File containing the text that we want tagged')
    parser.add_argument('--output_file', type=str,
                        help='Output file containing tagged sentences')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of Java workers doing the tagging')
    parser.add_argument('--num_batches', type=int, default=16,
                        help='Number of batches to use - more batches - less memory needed')
    parser.add_argument('--max_sentences', type=int,
                        help='Set a maximum number of sentences to tag (used for debugging)')

    args = parser.parse_args()

    main(args)
