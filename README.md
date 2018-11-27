# statistical-tools
Tools for creating dataset stats

## Package requirements

* numpy
* matplotlib
* matplotlib-venn
* pandas
* nltk, after installing nltk run `nltk.download('wordnet')` in Python shell

## Running instructions

### Visual Genome dataset analysis

*Human synsets in Visual Genome attribute data*

`python3 vg_analyze_attributes.py --attributes_json /proj/mediaind/picsom/databases/visualgenome/download/1.2/VG/1.2/attributes.json`

*Human synsets in Visual Genome relationships data*

`python3 vg_analyze_relationships.py --relationships_json /proj/mediaind/picsom/databases/visualgenome/download/1.2/VG/1.2/relationships.json --rel_counts`

### Generic linguistic analysis