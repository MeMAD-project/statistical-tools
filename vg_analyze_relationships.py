import sys
import analysis_funs as va


data = va.load_data('relationships.json')

assert len(data) == 108077

if sys.argv[1] == '--help':
	print("Usage:\n--rel_counts : count different types of relationships\n--summary_stats : print summary statistics")

if sys.argv[1] == '--rel_counts':
	# Result tuple has the following order:
	# r[rel_name_cnt, rel_syn_cnt], [subj_name_cnt, subj_syn_cnt], [obj_name_cnt, obj_syn_cnt]

	names = ['name','synset']

	rels, subjs, objs = va.count_relationships(data, va.human_synsets)

	va.plot_and_output_csv(rels, names, 40, f"Relations with people as subjects", 'relationships/rel_subj_people' , batch = True) 
	va.plot_and_output_csv(subjs, names, 40, f"Subjects with people as subjects", 'relationships/subj_subj_people' , batch = True) 
	va.plot_and_output_csv(objs, names, 40, f"Objects with people as subjects", 'relationships/obj_subj_people' , batch = True) 

	rels, subjs, objs = va.count_relationships(data, va.human_synsets, verbs = True)

	va.plot_and_output_csv(rels, names, 40, f"Relations with people as subjects, verbs only", 'relationships/rel_subj_people_verbs' , batch = True) 
	va.plot_and_output_csv(subjs, names, 40, f"Subjects with people as subjects, verbs only", 'relationships/subj_subj_people_verbs' , batch = True) 
	va.plot_and_output_csv(objs, names, 40, f"Objects with people as subjects, verbs only", 'relationships/obj_subj_people_verbs' , batch = True) 

if sys.argv[1] == '--summary_stats':

	count, indices = stats_on_humans_in_relationships(data)