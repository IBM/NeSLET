import os

# path to all data files
data_path  = '/type_api/data'

entity_catalogue = os.path.join(data_path, 'entity.jsonl')
ontology_nodes_parent_file_path = os.path.join(data_path,'dbpedia_ontology_node_parent.json')
type_dict_path = os.path.join(data_path,'dbpedia_2020_type_map.json')
type_co_occurrence_prob_file_path = os.path.join(data_path,'types_co_occurrence_matrix.json')
entity_to_type_file_path = os.path.join(data_path,'wiki_BLINK_entities_5.9M_with_categories_dbpedia_desc_ances.jsonl')
cached_id_to_type_dict_path = os.path.join(data_path,'cached_id_to_type_dict.json')
cached_id_to_fine_type_dict_path = os.path.join(data_path,'cached_id_to_fine_type_dict.json')
cached_title2id = os.path.join(data_path,'title2id.json')
cached_id2title = os.path.join(data_path,'id2title.json')
cached_wikipedia_id2local_id = os.path.join(data_path,'wikipedia_id2local_id.json')
cached_id2text = os.path.join(data_path,'id2text.json')


types_key='fine_types_id'
max_tree_depth = 7