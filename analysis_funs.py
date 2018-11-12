import pandas as pd
import json
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
from matplotlib.patches import Rectangle
import matplotlib.image as mpimg
import re
from collections import Counter
import numpy as np
from nltk.corpus import wordnet as wn
import pickle
import marshal
import re
import os

DATA_DIR = '/scratch/cs/imagedb/picsom/databases/visualgenome/download/'


def test_reload():
    """
    Call this function from Jupyter to check that autoreload has been set up correctly!
    """
    print('bar')


# Data loadin and caching:

def load_json(json_file):
    d = None

    with open(json_file) as f:
        d = json.load(f)

    return d


def load_data(json_file, reload=False):
    """
        Loads json_file from pickle if json_file.pickle is found, else loads the actual json file 
    and creates a new pickle.
    reload = True forces this behavior
    """
    data = None
    file_name = os.path.basename(json_file)
    file_name_no_ext = os.path.splitext(file_name)[0]
    pickle_file = os.path.join('pickles', file_name_no_ext + ".marshal")

    if os.path.exists(pickle_file) and not reload:
        print(f"Loading {pickle_file}...")
        data = marshal.load(open(pickle_file, "rb"))
    else:
        print(f"Loading {json_file}...")
        data = load_json(json_file)
        marshal.dump(data, open(pickle_file, "wb"))

    return data

# Helper functions:


def get_all_humans():
    """Return an exhaustive list of WordNet synsets that descend
    from person.n.01 and human.n.01"""

    person_hyponyms = get_hyponyms(wn.synset('person.n.01'))
    human_hyponyms = get_hyponyms(wn.synset('human.n.01'))
    human_synsets = set(person_hyponyms + human_hyponyms +
                        ['person.n.01', 'human.n.01'])

    return human_synsets


def sort_counter(counter):
    sorted_counter = sorted(((value, key) for (key, value) in counter.items()))
    return sorted_counter


def get_hyponyms(root):
    """root: wordnet synset whose children we want to traverse"""
    term_list = []
    for s in root.hyponyms():
        term_list += [s.name()]
        h = get_hyponyms(s)
        if len(h):
            term_list += h

    return term_list


def is_human(synset_name):
    if synset_name in human_synsets:
        return True
    elif wn.synset(synset_name).lemmas()[0].synset().name() in human_synsets:
        return True
    else:
        return False


def is_verb(syn_name):
    re_verb = '\.v\.'

    if re.search(re_verb, syn_name):
        return True
    else:
        return False


# Attributes stuff:

def count_attributes_per_synset(data, query_synset):
    word_cnt = Counter()  # count individual attribute values
    attr_shared_cnt = Counter()  # count full phrases
    attr_cnt = Counter()  # count total number of attributes per query_word
    query_cnt = 0  # count number of query

    print(f"Counting attributes for synsets matching {query_synset}")

    for i, row in enumerate(data):
        for attr in row['attributes']:
            for syn in attr['synsets']:
                # query_word detected
                if syn == query_synset or query_synset == '*':
                    query_cnt += 1
                    if 'attributes' in attr.keys():
                        attrs = attr['attributes']
                        attr_cnt[str(len(attrs))] += 1
                        attr_shared_cnt[str(sorted(attrs))] += 1
                        for attr in attrs:
                            word_cnt[attr.strip()] += 1
                    else:
                        attrs = 'no attributes'
                        attr_cnt[str(0)] += 1
                    break

                    # print(f"Entry {i} contains match '{s}' with attributes {attrs}")
    print(
        f"Done. {len(data)} rows processed, {query_cnt} intances of '{query_synset}' found")

    return word_cnt, attr_shared_cnt, attr_cnt, query_cnt


def count_attribute_synsets(data, synset_names):
    synset_attr_cnt = Counter()  # count individual synsets
    name_attr_cnt = Counter()  # count names associated with synsets
    synset_img_cnt = Counter()  # count individual synsets
    name_img_cnt = Counter()  # count names associated with synsets
    matches = 0  # count number of matching synsets
    rows = 0  # count number of rows that had matching synsets

    for i, row in enumerate(data):
        row_matched = False
        image_synsets = set()
        image_names = set()
        for attr in row['attributes']:
            for syn_name in attr['synsets']:
                if is_human(syn_name):
                    synset_attr_cnt[syn_name] += 1
                    matches += 1

                    if syn_name not in image_synsets:
                        image_synsets.add(syn_name)
                        synset_img_cnt[syn_name] += 1

                    for name in attr['names']:
                        name_attr_cnt[name] += 1
                        if name not in image_names:
                            image_names.add(name)
                            name_img_cnt[name] += 1

                    row_matched = True

        if row_matched:
            rows += 1

    print(f"Done. {len(data)} rows processed, {matches} instances of found in {rows} rows")

    return synset_attr_cnt, name_attr_cnt, synset_img_cnt, name_img_cnt, matches, rows


# Object stuff:

def humans_in_objects(data):
    humans_objects = []

    return humans_objects


# Relationship stuff:

def count_relationships(data, synset_names, verbs=False):
    """
    TODO:
    Count relationships of the synset_names in data based on 4 types of counts:
            1 = count the relationships where subjects  match synset_names
            2 = count the relationships where objects match synset_names
            3 = count the cases where at least subject or object match humans
            4 = count the cases where both subjects and objects match synset names

    verbs: when True, only count relations where relation is a verb
    """

    # TODO: setup 3 dicts for storing 4 types of counters:

    subj = {}
    obj = {}

    subj_or_obj = {}
    subj_and_obj = {}

    rel_name_cnt = Counter()  # count relation names where human is a subject
    rel_syn_cnt = Counter()  # count relation synsets where human is a subject

    subj_name_cnt = Counter()  # count subject names where human is a subject
    subj_syn_cnt = Counter()  # count subject synsets where human is a subject

    obj_name_cnt = Counter()  # count object names where human is a subject
    obj_syn_cnt = Counter()  # count object synsets where human is a subject

    matches = 0  # count the number of matching relations
    rows = 0  # count the number of matching images (data rows in json)

    for i, row in enumerate(data):
        row_matched = False

        for rel in row['relationships']:
            subject_added = False

            # Check if we are in verb mode or not:
            if verbs and len(rel['synsets']) and not is_verb(rel['synsets'][0]):
                # skip if non-verb when verbs = True and relationship synsets not empty
                continue

            for s_syn in rel['subject']['synsets']:
                if is_human(s_syn):
                    subj_syn_cnt[s_syn] += 1
                    # count all objects and relationship only once
                    if not subject_added:
                        # Count subject name:
                        subj_name_cnt[rel['subject']['name']] += 1

                        # Count relationships
                        rel_name_cnt[rel['predicate']] += 1
                        for r_syn in rel['synsets']:
                            rel_syn_cnt[r_syn] += 1

                        # Count objects
                        obj_name_cnt[rel['object']['name']] += 1
                        for o_syn in rel['object']['synsets']:
                            obj_syn_cnt[o_syn] += 1

                        matches += 1
                        subject_added = True
                        row_matched = True

        if row_matched:
            rows += 1

    print(f"Done. {len(data)} rows processed, {matches} instances {'with verbs' if verbs else ''}found in {rows} rows")

    return (rel_name_cnt, rel_syn_cnt), (subj_name_cnt, subj_syn_cnt), (obj_name_cnt, obj_syn_cnt)


def stats_on_humans_in_relationships(data):
    """
    Output the following numbers:
    - Relationships where Subject is human and Object is human
    - Relationships where Subject is human
    - Relationships where Object is human
    - Repeat above with verbs only
    """

    index = []  # index of all human rows

    # Dict of counters:
    cnt = {}

    cnt['rels'] = {}
    cnt['imgs'] = {}

    # We are counting relationships with 
    #  - All relationships in dataset
    #  - All relationships with human subjects
    #  - All relationships with human objects
    #  - All relationships with human subjects and objects
    #  - Perform above also with unique image counts

    cnt['rels']['all'] = Counter()
    cnt['rels']['verbs'] = Counter()

    cnt['imgs']['all'] = Counter()
    cnt['imgs']['verbs'] = Counter()

    for i, row in enumerate(data):

        human_found = False

        subj_human_image = False
        obj_human_image = False
        subj_and_obj_human_image = False

        subj_human_verb_image = False
        obj_human_verb_image = False
        subj_and_obj_human_verb_image = False

        image_has_verb = False

        # Simple image counter:
        cnt['imgs']['all']['all'] += 1

        for rel in row['relationships']:

            subj_human = False
            obj_human = False
            relationship_is_verb = False

            # Total count of relationships:
            cnt['rels']['all']['all'] += 1

            if len(rel['synsets']) and is_verb(rel['synsets'][0]):
                relationship_is_verb = True
                image_has_verb = True
                cnt['rels']['verbs']['all'] += 1

            for s_syn in rel['subject']['synsets']:
                if is_human(s_syn):
                    subj_human = True
                    break

            for o_syn in rel['object']['synsets']:
                if is_human(o_syn):
                    obj_human = True
                    break

            if subj_human:
                subj_human_image = True
                cnt['rels']['all']['subj'] += 1
                human_found = True
                if relationship_is_verb:
                    cnt['rels']['verbs']['subj'] += 1
                    subj_human_verb_image = True

            if obj_human:
                obj_human_image = True
                cnt['rels']['all']['obj'] += 1
                human_found = True
                if relationship_is_verb:
                    cnt['rels']['verbs']['obj'] += 1
                    obj_human_verb_image = True

            if subj_human and obj_human:
                subj_and_obj_human_image = True
                cnt['rels']['all']['subj_and_obj'] += 1
                if relationship_is_verb:
                    cnt['rels']['verbs']['subj_and_obj'] += 1
                    subj_and_obj_human_verb_image = True

        if subj_human_image: cnt['imgs']['all']['subj'] += 1
        if obj_human_image: cnt['imgs']['all']['obj'] += 1
        if subj_and_obj_human_image: cnt['imgs']['all']['subj_and_obj'] += 1

        if subj_human_verb_image: cnt['imgs']['verbs']['subj'] += 1
        if obj_human_verb_image: cnt['imgs']['verbs']['obj'] += 1
        if subj_and_obj_human_verb_image: cnt['imgs']['verbs']['subj_and_obj'] += 1
        
        if image_has_verb: cnt['imgs']['verbs']['all'] += 1

        if human_found: index.append(i)

    # Print the stats:
    print("Summary statistics - Relationships")
    print("=" * 30)
    print("All relationships:")
    print("-" * 20)
    print(f"Relations with human subjects:\t{cnt['rels']['all']['subj']}")
    print(f"Relations with human objects:\t{cnt['rels']['all']['obj']}")
    print(
        f"Relations with both human subjects and objects:\t{cnt['rels']['all']['subj_and_obj']}\n")
    print(f"All relationships (human and not):\t{cnt['rels']['all']['all']}\n")
    print("=" * 30)
    print("Relationships with verbs:")
    print("-" * 20)
    print(f"Relations with human subjects:\t{cnt['rels']['verbs']['subj']}")
    print(f"Relations with human objects:\t{cnt['rels']['verbs']['obj']}")
    print(
        f"Relations with both human subjects and objects:\t{cnt['rels']['verbs']['subj_and_obj']}\n")
    print(
        f"All relationships with verbs (human and not):\t{cnt['rels']['verbs']['all']}")

    print("\n\n\n")

    print("Summary statistics - Images")
    print("=" * 30)
    print("All images:")
    print("-" * 20)
    print(f"Images with human subjects:\t{cnt['imgs']['all']['subj']}")
    print(f"Images with human objects:\t{cnt['imgs']['all']['obj']}")
    print(
        f"Images with both human subjects and objects:\t{cnt['imgs']['all']['subj_and_obj']}\n")
    print(f"All images (human and not):\t{cnt['imgs']['all']['all']}\n")
    print("=" * 30)
    print("Images with verbs:")
    print("-" * 20)
    print(f"Images with human subjects:\t{cnt['imgs']['verbs']['subj']}")
    print(f"Images with human objects:\t{cnt['imgs']['verbs']['obj']}")
    print(
        f"Images with both human subjects and objects:\t{cnt['imgs']['verbs']['subj_and_obj']}\n")
    print(
        f"All images with verbs (human and not):\t{cnt['imgs']['verbs']['all']}")

    return cnt, index


def indexed_images_relationship_match(data, indices, synset_matches):
    """
    For each image in the data matched with with indices, see if there is a synset_match
    data - relationship json
    indices - indices of images that we are interested in (in the first use cas this is humans / persons)
    synset_matches - callable, outputs whether sysnset matches some top level synset
    """

    index_matches = []

    for i in indices:
        for rel in data[i]:
            for syn in rel['subject']['synsets'] + rel['object']['synsets']:
                if synset_matches(syn):
                    index_matches.append((i, rel['relationship_id']))

    return index_matches

# Synset conversion functions:

# TODO


# Output functions:

def plot_venn(counts, labels, title, filename = None, batch = False):
    if batch:
        plt.switch_backend('agg')

    fig = plt.figure(figsize=(20, 10))

    A = counts['subj'] - counts['subj_and_obj']
    B = counts['obj'] - counts['subj_and_obj']
    AB = counts['subj_and_obj']
    not_AB = counts['all'] - A - B - AB
    total_humans = A + B - AB
    total_all = counts['all']

    _ = plt.title(title, fontsize=20)
    v = venn2(subsets=(A, B, AB),
              set_labels=labels[:-1])

    for text in v.set_labels:
        text.set_fontsize(14)
    for text in v.subset_labels:
        text.set_fontsize(16)

    _ = plt.text(-0.65, 0.52, labels[-1], fontsize=14)
    _ = plt.text(0.65, 0.52, str(not_AB), fontsize=16)
    _ = plt.text(-0.65, -0.72,
                 f"Total human: {total_humans}", fontsize=16)
    _ = plt.text(-0.65, -0.82,
                 f"Total: {total_all}", fontsize=16)

    if filename:
        save_to = 'plots/' + filename + '.png'
        print(f"Saving plot to {save_to}")
        plt.savefig(save_to, bbox_inches='tight',
                    facecolor='white', transparent=False)

    plt.show()

def plot_relationship_venn(counts, filename=None, batch=False):
    """
    Plot counters as venn diagram:
    counts - counters of the form counts['all|verbs']['subj|obj|subj_and_obj|rels']
    filename - if defined, the file is saved there
    batch - batch mod = True means that plots are saved to disk, no output to screen
    """
    if batch:
        plt.switch_backend('agg')

    fig = plt.figure(figsize=(20, 20))

    _ = plt.subplot(2, 1, 1)

    A1 = counts['all']['subj'] - counts['all']['subj_and_obj']
    B1 = counts['all']['obj'] - counts['all']['subj_and_obj']
    AB1 = counts['all']['subj_and_obj']
    not_AB1 = counts['all']['rels'] - A1 - B1 - AB1
    total_humans1 = A1 + B1 - AB1
    total_all1 = counts['all']['rels']

    _ = plt.title('Relations with humans as subject or object', fontsize=20)
    v = venn2(subsets=(A1, B1, AB1),
              set_labels=('Human Subjects', 'Human Objects'))

    for text in v.set_labels:
        text.set_fontsize(14)
    for text in v.subset_labels:
        text.set_fontsize(16)

    #plt.gca().set_axis_on()
    _ = plt.text(-0.65, 0.52, 'Other relations', fontsize=14)
    _ = plt.text(0.65, 0.52, str(not_AB1), fontsize=16)
    _ = plt.text(-0.65, -0.72,
                 f"Total human relationships: {total_humans1}", fontsize=16)
    _ = plt.text(-0.65, -0.82,
                 f"Total relationships: {total_all1}", fontsize=16)

    ax = plt.subplot(2, 1, 2)

    A2 = counts['verbs']['subj'] - counts['all']['subj_and_obj']
    B2 = counts['verbs']['obj'] - counts['all']['subj_and_obj']
    AB2 = counts['verbs']['subj_and_obj']
    not_AB2 = counts['verbs']['rels'] - A2 - B2 - AB2
    total_humans2 = A2 + B2 - AB2
    total_all2 = counts['verbs']['rels']

    _ = plt.title('Relations with humans as subject or object\nrelations are verbs', 
        fontsize=20)
    v = venn2(subsets=(counts['verbs']['subj'], counts['verbs']['obj'], counts['verbs']['subj_and_obj']),
              set_labels=('Human Subjects', 'Human Objects'))

    for text in v.set_labels:
        text.set_fontsize(14)
    for text in v.subset_labels:
        text.set_fontsize(16)

    #plt.gca().set_axis_on()
    _ = plt.text(-0.65, 0.52, 'Other relations', fontsize=14)
    _ = plt.text(0.65, 0.52, str(not_AB2), fontsize=16)
    _ = plt.text(-0.65, -0.72,
                 f"Total human relationships: {total_humans2}", fontsize=16)
    _ = plt.text(-0.65, -0.82,
                 f"Total relationships: {total_all2}", fontsize=16)

    plt.subplots_adjust(hspace=0.5)

    if filename:
        save_to = 'plots/' + filename + '.png'
        print(f"Saving plot to {save_to}")
        plt.savefig(save_to, bbox_inches='tight',
                    facecolor='white', transparent=False)

    plt.show()


def plot_bar_counts(counter, maxnum=20, title=None, filename=None, batch=False):
    if batch:
        print(f"Batch mode. Saving plot to file.")
        plt.switch_backend('agg')

    print(f"Plotting {title}...")

    if not len(counter):
        print("Empty counter, exiting")
        return

    # sort the counter by values
    sorted_counter = counter.most_common(maxnum)

    x = np.array(sorted_counter)[:, 0][::-1]
    y = np.uint32(np.array(sorted_counter)[:, 1])[::-1]

    if maxnum >= 20:
        plt.figure(figsize=(10, 20))
    else:
        plt.figure(figsize=(10, maxnum))

    if title:
        plt.title(title, fontsize=15)

    for i, v in enumerate(y):
        plt.text(25, i - .15, str(v), color='black',
                 fontweight='bold', fontsize=15)

    plt.yticks(fontsize=20)
    plt.barh(x, y)

    if filename:
        save_to = 'plots/' + filename + '.png'
        print(f"Saving plot to {save_to}")
        plt.savefig(save_to, bbox_inches='tight')
    
    if not batch: plt.show()


def plot_bar_counts_side_by_side(counters, names, maxnum=20, title=None, filename=None, batch=False):
    if batch:
        plt.switch_backend('agg')

    print(f"Plotting {title}...")

    if not len(counters[0]) and not len(counters[1]):
        print("Both counters empty, exiting")
        return

    # sort the counters by values
    sorted_counter1 = counters[0].most_common(maxnum)
    sorted_counter2 = counters[1].most_common(maxnum)

    x1 = np.array(sorted_counter1)[:, 0][::-1]
    y1 = np.uint32(np.array(sorted_counter1)[:, 1])[::-1]
    x2 = np.array(sorted_counter2)[:, 0][::-1]
    y2 = np.uint32(np.array(sorted_counter2)[:, 1])[::-1]

    fig = plt.figure(figsize=(30, 20), facecolor='white')

    if title:
        fig.suptitle(title, fontsize=35)

    # Plot the first subplot
    plt.subplot(1, 2, 1)
    plt.title(names[0], fontsize=20)
    for i, v in enumerate(y1):
        plt.text(25, i - .15, str(v), color='black',
                 fontweight='bold', fontsize=15)

    plt.yticks(fontsize=20)
    plt.barh(x1, y1)

    # Plot the second subplot
    plt.subplot(1, 2, 2)
    plt.title(names[1], fontsize=20)
    for i, v in enumerate(y2):
        plt.text(25, i - .15, str(v), color='black',
                 fontweight='bold', fontsize=15)

    plt.yticks(fontsize=20)
    plt.barh(x2, y2)

    plt.subplots_adjust(left=0.2, wspace=0.25, top=0.94)

    if filename:
        save_to = 'plots/' + filename + '.png'
        print(f"Saving plot to {save_to}")
        plt.savefig(save_to, bbox_inches='tight',
                    facecolor='white', transparent=False)
    if not batch:
        plt.show()


def counter_to_csv(counter, countername, filename, desc=None):
    """
    counter - counter object (may be unsorted)
    filename - partial filename of the form "type/name", base dir and extension added 
            automatically by this function
    countername - name of the column with labels of the counts (e.g 'synset' or 'name')
    desc - optional description that is added as a comment on the first line of the CSV
    """
    base_dir = 'csv/'
    sorted_counter = counter.most_common()
    col_names = [countername, 'count']
    df = pd.DataFrame(sorted_counter, columns=col_names)
    df = df[df['count'] >= 100]

    filename = base_dir + filename + '.csv'

    with open(filename, 'w') as f:
        print(f"Creating CSV with {desc}")
        print(f"Saving to {filename} ")
        if desc:
            f.write('#' + desc + '\n')

        df.to_csv(f, index=False)


def plot_and_output_csv(counters, names, maxnum, title, filename_base, batch=False):
    """
    Runs plotting and csv saving functions. Designed to plot either 1 or 2 plots depending on the
    number of counters given (currently accepts 1 or 2 counters).
    """
    if len(counters) == 1:
        plot_bar_counts(counters[0], maxnum, title,
                        filename_base + '_' + names[0], batch)
        counter_to_csv(counters[0], names[0],
                         filename_base + '_' + names[0], title)
    else:
        plot_bar_counts_side_by_side(
            counters, names, maxnum, title, filename_base, batch)
        counter_to_csv(counters[0], names[0],
                       filename_base + '_' + names[0], title)
        counter_to_csv(counters[1], names[1],
                       filename_base + '_' + names[1], title)


def img_id_to_filename(img_id, dataset, base_path, prefix=None):
    filepath = None
    if dataset == 'vg':
        filepath = os.path.join(base_path, img_id + '.jpg')
    elif dataset == 'coco':
        filename = '{}{:012d}.jpg'.format(prefix, img_id)
        filepath = os.path.join(base_path, filename)

    return filepath


def display_image_and_caption(img_path, captions, keywords):
    img = mpimg.imread(img_path)
    imgplot = plt.imshow(img)

    for (caption, keyword_tuples) in zip(captions, keywords):
        words, synsets, categories = zip(*keyword_tuples)
        print("Caption: {}".format(caption))
        print("Words: {}".format(words))
        print("Synsets: {}".format(synsets))
        print("Category synsets {}".format(categories))


def sample_entries(df, n_samples, group_by):
    grouped = df.sample(n_samples).groupby(group_by)
    return grouped


def show_samples(grouped, group_by, caption_field_name, category_name):
    pass

# Initialize the list of all known human synsets:
human_synsets = get_all_humans()

if __name__ == '__main__':
    # run tests
    non_verb = 'along.r.01'
    verb = 'wear.v.01'

    assert is_verb(verb)
    assert not is_verb(non_verb)
