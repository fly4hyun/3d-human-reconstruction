





import json

new_data = {}

with open('./woman/datainfo.json', 'r', encoding='cp949') as f:
    data = json.load(f)
    
    key_arr = data.keys()
    
    
    
    for keyss in key_arr:
        if keyss[:2] == '8_':
            new_data[keyss] = data[keyss]
    

    with open('./woman/data_info.json', 'w') as ff:
        json.dump(new_data, ff, ensure_ascii = False, indent = 4)





