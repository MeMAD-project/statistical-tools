import analysis_funs as va

print(f"Loading attributes data..")
data = va.load_json(va.DATA_DIR + 'attributes.json')

assert len(data) == 108077

synset = 'person.n.01'
word_cnt, attr_shared_cnt, attr_cnt, query_cnt = va.count_attributes_per_synset(data, synset) 

va.plot_and_output_csv(word_cnt, 'synset', 25, f"Attribute counts for: {synset}", 'attributes/count_' + synset, batch = True)
va.plot_and_output_csv(attr_shared_cnt, 'synset', 25, f"Attribute combinations counts: {synset}", 'attributes/combi_count_' + synset, batch = True)
va.plot_and_output_csv(attr_cnt, 'synset', 25, f"Object attribute counts: {synset}", 'attributes/raw_numbers_' + synset, batch = True)

synset_cnt, name_cnt, _ , _ = va.count_synsets(data, va.human_synsets)

va.plot_and_output_csv(synset_cnt, 'synset', 40, f"Images with people synsets in attribute data", 'attributes/people_syns')
va.plot_and_output_csv(name_cnt, 'name', 40, f"Images with people names in attribute data", 'attributes/people_names' ) 
