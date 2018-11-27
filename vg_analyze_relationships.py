import sys
import analysis_funs as va
import argparse


def main(args):
    print("Loading relationships data from: {}".format(args.relationships_json))
    data = va.load_json(args.relationships_json)

    assert len(data) == 108077

    print("=" * 80)

    if args.rel_counts:
        print("Count all relationships (verbs and non-verbs):")
        print("=" * 80)
        rels, subjs, objs = va.count_relationships(data, va.human_synsets)

        va.plot_and_output_csv(rels, ['relationship name', 'relationship synset'], 40,
                               "Relationships with people as subjects",
                               'relationships/rel_subj_people', batch=True)

        va.plot_and_output_csv(subjs, ['subject name', ' subject synset'], 40,
                               "Subjects with people as subjects",
                               'relationships/subj_subj_people', batch=True)

        va.plot_and_output_csv(objs, ['object name', 'object synset'], 40,
                               "Objects with people as subjects",
                               'relationships/obj_subj_people', batch=True)

        print("=" * 80)
        print("Count verb-only relationships:")
        print("=" * 80)
        rels, subjs, objs = va.count_relationships(data, va.human_synsets, verbs=True)

        va.plot_and_output_csv(rels, ['relationship name', 'relationship synset'], 40,
                               "Relationships with people as subjects, verbs only",
                               'relationships/rel_subj_people_verbs', batch=True)

        va.plot_and_output_csv(subjs, ['subject name', ' subject synset'], 40,
                               "Subjects with people as subjects, verbs only",
                               'relationships/subj_subj_people_verbs', batch=True)

        va.plot_and_output_csv(objs, ['object name', 'object synset'], 40,
                               "Objects with people as subjects, verbs only",
                               'relationships/obj_subj_people_verbs', batch=True)

    print("=" * 80)
    counts, indices = va.stats_on_humans_in_relationships(data)
    print("=" * 80)
    print("Plotting venn diagrams...")
    print("=" * 80)

    va.plot_venn(counts['rels']['all'], ['Human subjects', 'Human objects', 'Other'],
                 'Relationships with humans as subjects or objects',
                 filename='relationships/venn_rels_all', batch=True)

    va.plot_venn(counts['rels']['verbs'], ['Human subjects', 'Human objects', 'Other'],
                 "Relationships with humans as subjects or objects,\nverbs only",
                 filename='relationships/venn_rels_verbs', batch=True)

    va.plot_venn(counts['imgs']['all'], ['Human subjects', 'Human objects', 'Other'],
                 'Images with humans as subjects or objects',
                 filename='relationships/venn_imgs_all', batch=True)
    va.plot_venn(counts['imgs']['verbs'], ['Human subjects', 'Human objects', 'Other'],
                 "Images with humans as subjects or objects,\nverbs only",
                 filename='relationships/venn_imgs_verbs', batch=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--relationships_json', type=str,
                        help='location of Visual Genome relationships JSON file')
    parser.add_argument('--rel_counts', action='store_true',
                        help='Count different types of relationships, '
                        'otherwise show summary stats only')

    args = parser.parse_args()

    main(args=args)
