import pandas as pd
import json



df = pd.read_csv('/data/all_wikipedia_dump/20210301_dump/enwiki-20210301-pages-articles-multistream.csv', quotechar='|', index_col = False)



df.loc[df['page_title']== " "]

title_pageid_mapping=df.set_index('page_title')['page_id'].to_dict()

with open('/data/all_wikipedia_dump/dump_from_blink_github_link/wikipedia_title__pageid_mapping_blink_dump.json', 'w', encoding='utf8') as json_file:
   json.dump(title_pageid_mapping, json_file, indent=4)

len(title_pageid_mapping)
print("done")