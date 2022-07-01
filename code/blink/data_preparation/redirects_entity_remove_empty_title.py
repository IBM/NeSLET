import json
import jsonlines

entity_file= '/raw_data/redirects_from_20210301_dump/redirects_entity.jsonl'

out_dict={"text":"","idx":"none","title":"","entity":""}
count_entity_enpty=[]

for line in open(entity_file,'r', encoding = 'utf-8'):
        each_line=json.loads(line)
        #count_entity_enpty=[]
        if len(each_line['entity']) <=2:
            print(each_line['idx'])
            count_entity_enpty.append(each_line['idx'])
            continue
        else:
            out_dict['text'] = each_line['text']
            out_dict['idx']= each_line['idx']
            out_dict['entity'] =each_line['entity']
            out_dict['title']=each_line['title']
            #out_dict.update(out_dict)
            #writer.write(out_dict)
print("total empty entity", len(count_entity_enpty))
print("done")