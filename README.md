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

### Extracting information on words belonging to different categories, such as time, location, etc

First you will need to download and unzip Stanford Part-Of-Speech (POS) tagger. To perform this run the following command inside the project folder:

`bash download_pos_tagger.sh`

Next, please make sure that your Java runtime environment is at least version 1.8:

`java -version`

If the version is lower than 1.8, you will need to make sure that you update your environment to use Java `>= 1.8`. On _CSC Taito_ environment this can be done with:

`module load java/oracle/1.8`

#### Example: Analyze Visual Genome region descriptions and COCO captions

Stage 1: Preprocess that data

1.1) Perform POS tagging, store result in JSON file:

_Visual Genome_:

`python3 tag_sentences.py --dataset vg-regions --num_workers 4 --num_batches 32 \
    --input_file /proj/mediaind/picsom/databases/visualgenome/download/1.2/VG/1.2/region_descriptions.json \
    --output_file output/pos_tags_vg_regions.json`

_COCO_:

`python3 tag_sentences.py --dataset coco --num_workers 4 --num_batches 32 \
    --input_file COCOPATH \
    --output_file output/pos_tags_coco.json`

The above command should work ok with 16 GB of allocated memory, and takes around *40 minutes* with 4 CPU cores on Taito for Visual Genome region descriptions (total 5,408,689 captions).

1.2) Infer synsets based on POS tags, store result in JSON file:

_Visual Genome_:

`python3 infer_synsets.py output/pos_tags_vg_regions.json \
    --output_path output`

_COCO_:

`python3 infer_synsets.py output/pos_tags_coco.json \
    --output_path output`

Total synset inference time for Visual Genome regions dataset is around *15 minutes* on Taito CPU instance (only one core is used for this task).

Stage 2: Match categories and extract stats

2.1) Match synsets to categories

_Visual Genome_:

`python3 match_synsets_to_categories.py --dataset vg-regions \
    --data_file /proj/mediaind/picsom/databases/visualgenome/download/1.2/VG/1.2/region_descriptions.json \
    --synset_file output/pos_tags_vg_regions_syns.json \
    --category location --output_path output`

_COCO_:

`python3 match_synsets_to_categories.py --dataset vg-regions \
    --data_file COCOPATH \
    --synset_file output/pos_tags_coco_syns.json \
    --category location --output_path output`

The above command tries to match synsets that match a user specified category. Synsets that are part of a given category are defined in a dictionary inside `categories.py`. 

The above script outputs 2 files: JSON containing both the original captions and the inferred categories, as well as same data inside a serialized Pandas object that is used for calculating the actual statistics in the next and final step of the pipeline.

2.2) Calculate statistics

_Visual Genome_:

`python3 `

_COCO_:

`python3`


### Extending generic analysis

* Different data set handlers can be added to:
    *  `tag_sentences.py` to produce per sentence POS tagging for an arbitrary list of sentences
    * `match_synsets_to_categories.py`.

* One can add more category types to `categories.py`. Right now only `location` and `temporal` types are supported.
