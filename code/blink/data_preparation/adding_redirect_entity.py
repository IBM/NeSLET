
import json
import sys
import os
from itertools import chain
import re
from collections import defaultdict
import jsonlines
import pickle
import json
import requests


with open('/shared_data/entity_wikiredirect_2020.pkl','rb')as handle:
  entities_redirects = dict(pickle.load(handle))

print(entities_redirects['<http://dbpedia.org/resource/83_(film)>'])

Entity_file_blink="/ZEL/code/blink/blink/models/entity.jsonl"
entity_page_id_mapping='/ZEL/code/blink/blink/data/20210301_dump/wiki_title_pageid_mapping_20210301_dump.json'

with open(entity_page_id_mapping) as f:
  entity_pageid_map = json.load(f)


dict_entity = defaultdict(dict)
idx2title={}
with jsonlines.open(Entity_file_blink) as reader:
  for obj in reader:
    idx= obj["idx"]
    title = obj["title"]
    dict_entity[title] = obj['text']
    idx2title[idx] =title

with open('/shared_data/entity_wikiredirect_2020.pkl','rb')as handle:
  entities_redirects = dict(pickle.load(handle, encoding="utf-8"))

out_dict={"text":"","idx":"none","title":"","entity":""}

redirects_target_not_present = []
count_duplicate_idx = []
for key, value in entities_redirects.items():
      # source_url= bytes(key).decode('utf-8')
      # target_url= bytes(value).decode('utf-8')
      source_entity =  key[29:len(key) - 1].replace("_", " ")
      target_entity = value[29:len(value) - 1].replace("_", " ")
      if source_entity not in dict_entity and source_entity in entity_pageid_map and target_entity in dict_entity and len(source_entity)>1:
        idx_red = "https://en.wikipedia.org/wiki?curid=" + str(entity_pageid_map[source_entity])
        #count_duplicate_idx = []
        if idx_red not in idx2title:
            out_dict["text"] = dict_entity[target_entity]
            out_dict["title"] = source_entity
            print(source_entity,target_entity)
            out_dict["entity"] = source_entity
            out_dict["idx"] = idx_red
            idx2title.update({idx_red: out_dict["title"]})
            #out_dict.update(out_dict)
            #writer.write(out_dict)
        else:
            count_duplicate_idx.append(idx_red)
            continue
      else:
        redirects_target_not_present.append({key:value})
#
print("dbpedia entity_redirects count", len( entities_redirects))
print("target entity not present count:",len(redirects_target_not_present))
print("duplicate idx count:", len(count_duplicate_idx))
print("length of idx 2title dict:", len(idx2title))
print("done")
