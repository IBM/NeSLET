# NeSLET
This repo contains source code to run the experiments described in our paper "Zero-shot entity linking with less data". Some of the code is a modification of https://github.com/facebookresearch/BLINK.

The paper was published at the Findings of NAACL 2022. It can be found at https://aclanthology.org/2022.findings-naacl.127/ 

Please cite our paper if you use code from this repository or the ideas and results discussed in our paper. 

```
@inproceedings{bhargav-etal-2022-zero,
    title = "Zero-shot Entity Linking with Less Data",
    author = "Bhargav, G P Shrivatsa  and
      Khandelwal, Dinesh  and
      Dana, Saswati  and
      Garg, Dinesh  and
      Kapanipathi, Pavan  and
      Roukos, Salim  and
      Gray, Alexander  and
      Subramaniam, L Venkata",
    booktitle = "Findings of the Association for Computational Linguistics: NAACL 2022",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-naacl.127",
    pages = "1681--1697",
    abstract = "Entity Linking (EL) maps an entity mention in a natural language sentence to an entity in a knowledge base (KB). The Zero-shot Entity Linking (ZEL) extends the scope of EL to unseen entities at the test time without requiring new labeled data. BLINK (BERT-based) is one of the SOTA models for ZEL. Interestingly, we discovered that BLINK exhibits diminishing returns, i.e., it reaches 98{\%} of its performance with just 1{\%} of the training data and the remaining 99{\%} of the data yields only a marginal increase of 2{\%} in the performance. While this extra 2{\%} gain makes a huge difference for downstream tasks, training BLINK on large amounts of data is very resource-intensive and impractical. In this paper, we propose a neuro-symbolic, multi-task learning approach to bridge this gap. Our approach boosts the BLINK{'}s performance with much less data by exploiting an auxiliary information about entity types. Specifically, we train our model on two tasks simultaneously - entity linking (primary task) and hierarchical entity type prediction (auxiliary task). The auxiliary task exploits the hierarchical structure of entity types. Our approach achieves superior performance on ZEL task with significantly less training data. On four different benchmark datasets, we show that our approach achieves significantly higher performance than SOTA models when they are trained with just 0.01{\%}, 0.1{\%}, or 1{\%} of the original training data. Our code is available at https://github.com/IBM/NeSLET.",
}
```

# Environment Setup
We have run all our experiments on power machines. The conda env file for ppc machines is `./code/blink/blinkppc.yml`.

The x86 env file `./code/blink/environment.yml` has not been thoroughly tested.

# Running the Code
There are multiple steps involved in training our model NeSLET. The scripts in `./code/blink/automation` automatically run all the training steps and generate the results and accuracy on the validation and test sets. 

NeSLET-F can be run using `./code/blink/automation/run_pipeline_typed_4.sh` and NeSLET-G and NeSLET-L can be run using `./code/blink/automation/run_pipeline_typed_5.sh`. 

The scripts themselves say what inputs are required. Please make sure the path are given correctly.

`./code/blink/automation/pipeline_typed_job_submitter.py` can be used to automatically schedule all the experiments in Table 16, 17 and 18 of our paper. `./code/blink/automation/gather_type_model_results.py` can be used to gather the accuracies of all the models trained by `./code/blink/automation/pipeline_typed_job_submitter.py`.  

# Data and Other Resources 
We are unable to release the data and other resources that we have created due to licensing issues. However, we have provided information about the format and the content of each resource and data file. 

The file templates are here: https://ibm.box.com/s/gcmrp05w7ccy5j9b45d0adcafbmqd8nl

## Description of the Files

- Training and validation sets should be built as per the file named `dataset.jsonl`. Our target knowledgebase is Wikipedia, but we use the type hierarchy from DBpedia. Each Wikipedia entity is mapped to DBpedia in order to get its types.
- `entity.jsonl` is a list of dictionaries. Each dictionary will correspond to a single entity.
- `dbpedia_2020_type_map.json` is a mapping between an integer id (0, 1, 2, 3, ... k) and a DBpedia type. 
- `dbpedia_ontology_node_parent.json` is a list. It maps every type to its parent. The id of the parent of type i is written in the ith position of the list. -1 indicates that the parent is root. These ids will correspond to the ids in `dbpedia_2020_type_map.json`.
- `dbpedia_2020_types.t7` has the embedding for each type. Row i will have the embedding of type i. `./code/blink/data_preparation/create_bert_embedding_for_types.py` can be used to create these embeddings. It will take as input.
- `entity_token_ids_128.t7` contains the token ids of the entity descriptions. For each entity, create a sequence `[CLS]Entity title[SEP]Entity description[SEP]`. Then tokenize the sequence as suitable for BERT. The ith row of this matrix will correspond to the ith entity in `entity.jsonl`.

Additional pointers:
- `entity.jsonl` and `entity_token_ids_128.t7` can be downloaded from https://github.com/facebookresearch/BLINK
- A large dataset having mappings from mention (and context) to wikipedia entity can be obtained from https://github.com/facebookresearch/KILT#additional-data
