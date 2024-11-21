import json
import glob
import re

json_file_list = glob.glob('../dataset/MOA_dataset/*.json')
df_list = []

prefix = '{"foo" : ['
suffix = ']}'

for path in json_file_list :
    with open(path, 'r') as f :
        print(path + " Read")
        content = f.read()
        content = content.replace(']}',']},')
        content = re.sub(r',$', '', content)
        new_content = prefix + content + suffix


    with open(path, 'w') as f :
        print(path + " Write")
        f.write(new_content)

 
