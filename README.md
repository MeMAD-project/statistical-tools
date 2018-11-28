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

`python3 vg_analyze_attributes.py \
    --attributes_json /proj/mediaind/picsom/databases/visualgenome/download/1.2/VG/1.2/attributes.json`

*Human synsets in Visual Genome relationships data*

`python3 vg_analyze_relationships.py  --rel_counts \
    --relationships_json /proj/mediaind/picsom/databases/visualgenome/download/1.2/VG/1.2/relationships.json`

### Generic linguistic analysis

First you will need to download and unzip Stanford Part-Of-Speech (POS) tagger. To perform this run the following command inside the project folder:

`bash download_pos_tagger.sh`

Next, please make sure that your Java runtime environment is at least version 1.8:

`java -version`

If the version is lower than 1.8, you will need to make sure that you update your environment to use Java `>= 1.8`. On CSC _Taito_ environment this can be done with:

`module load java/oracle/1.8`

*Visual Genome region descriptions*

`python3 tag_sentences.py --dataset vg-regions --num_workers 4 --num_batches 16 \
    --input_file /proj/mediaind/picsom/databases/visualgenome/download/1.2/VG/1.2/region_descriptions.json \
    --output_file output/pos_tags_vg_regions.json`