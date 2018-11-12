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
    sentences_tagged = st.tag_sents(word_tokenize(sent) for sent in sentences)

    return sentences_tagged


def tag_sentences_in_batch(batch):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='vg-regions',
                        help='Dataset from which to extract tagged sentences')

    args = parser.parse_args()

    print("Loading tagger...")
    st = StanfordPOSTagger('stanford-postagger-2017-06-09/models/english-bidirectional-distsim.tagger',
                           path_to_jar='stanford-postagger-2017-06-09/stanford-postagger.jar',
                           java_options='-mx4000m')

    if args.dataset == 'vg-regions':
        json_file = '/scratch/cs/imagedb/picsom/databases/visualgenome/download/' + \
                    '1.2/VG/1.2/region_descriptions.json'
        data = va.load_data(json_file)
        assert len(data) == 108077
        print("Collecting all sentences...")
        sentences = collect_sentences_vg(data)
        print('{} sentences collected'.format(len(sentences)))
        pickle_path = 'pickles/vg_sentences_tagged.pkl'
    elif args.dataset == 'coco':
        json_file = '/scratch/cs/imagedb/picsom/databases/COCO/download/' + \
                    'annotations/captions_train2014.json'
        data = va.load_data(json_file)
        # assert len(data) == 108077
        print("Collecting all sentences...")
        sentences = collect_sentences_coco(data)
        print('{} sentences collected'.format(len(sentences)))
        pickle_path = 'pickles/coco_sentences_tagged.pkl'
    elif args.dataset == 'vist-dii':
        json_file = '/scratch/cs/imagedb/picsom/databases/vist/download/' + \
                    'data/dii/train.description-in-isolation.json'
        data = va.load_data(json_file)
        print("Collecting all sentences...")
        sentences = collect_sentences_vist_dii(data)
        print('{} sentences collected'.format(len(sentences)))
        pickle_path = 'pickles/vist_dii_sentences_tagged.pkl'
    else:
        print("Invalid dataset")
        sys.exit(1)

    random_indices = [random.randint(0, len(sentences)) for i in range(5)]

    print("5 random raw sentences at indices: {}".format(random_indices))
    for i, idx in enumerate(random_indices):
        print('{}: {}'.format(i + 1, sentences[idx]))

    num_batches = 15
    tasks = [x for x in range(num_batches)]
    # Number of threads to use:
    num_workers = 5

    print("Starting tagging, using {} batches and {} workers"
          .format(num_batches, num_workers))

    with Pool(num_workers) as p:
        sentences_tagged = p.map(tag_sentences_in_batch, tasks)

    sentences_tagged = list(chain(*sentences_tagged))

    print('{} tagged sentences produced...'.format(len(sentences_tagged)))

    print("5 random tagged sentences:")
    for i, idx in enumerate(random_indices):
        print('{}: {}'.format(i + 1, sentences_tagged[idx]))

    with open(pickle_path, 'wb') as f:
        pickle.dump(sentences_tagged, f)
