import analysis_funs as va
import argparse


def main(args):
    print("Loading attributes data from: {}".format(args.attributes_json))
    data = va.load_json(args.attributes_json)

    assert len(data) == 108077

    synset = 'person.n.01'
    print("=" * 80)
    print("Counting attributes for synset: {}".format(synset))
    (word_cnt, attr_shared_cnt,
        attr_cnt, query_cnt) = va.count_attributes_per_synset(data, synset)

    va.plot_and_output_csv(word_cnt, 'synset', 25,
                           "Attribute counts for: {}".format(synset),
                           'attributes/count_{}'.format(synset),
                           batch=True)

    print("-" * 80)

    va.plot_and_output_csv(attr_shared_cnt, 'synset', 25,
                           "Attribute combinations counts: {}".format(synset),
                           'attributes/combi_count_{}'.format(synset),
                           batch=True)

    print("-" * 80)

    va.plot_and_output_csv(attr_cnt, 'synset', 25,
                           "Object attribute counts: {}".format(synset),
                           'attributes/raw_numbers_{}'.format(synset),
                           batch=True)

    print("=" * 80)
    print("Counting attributes for synsets that refer to human beings defined by:")
    print(va.human_synsets)

    (synset_attr_cnt, name_attr_cnt, synset_img_cnt, name_img_cnt,
      matches, rows) = va.count_attribute_synsets(data, va.human_synsets)

    va.plot_and_output_csv(synset_attr_cnt, 'synset', 40,
                           "Attributes with people synsets in attribute data",
                           'attributes/people_syns_attributes', batch=True)

    print("-" * 80)

    va.plot_and_output_csv(name_attr_cnt, 'name', 40,
                           "Attributes with people names in attribute data",
                           'attributes/people_names_attributes', batch=True)

    print("-" * 80)

    print("Total numbers of attributes matched: {}".format(matches))

    print("-" * 80)

    va.plot_and_output_csv(synset_img_cnt, 'synset', 40,
                           "Images with people synsets in attribute data",
                           'attributes/people_syns_images', batch=True)

    print("-" * 80)

    va.plot_and_output_csv(name_img_cnt, 'name', 40,
                           "Images with people names in attribute data",
                           'attributes/people_names_images', batch=True)

    print("-" * 80)

    print("Total numbers of images matched: {}".format(rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--attributes_json', type=str,
                        help='location of Visual Genome attributes JSON file')

    args = parser.parse_args()

    main(args=args)
