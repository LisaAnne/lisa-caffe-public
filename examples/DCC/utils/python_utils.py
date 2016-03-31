import json

def open_txt(f):
  open_f = open(f).readlines()
  return [f.strip() for f in open_f]

def read_json(t_file):
  j_file = open(t_file).read()
  return json.loads(j_file)

def save_json(json_dict, save_name):
  with open(save_name, 'w') as outfile:
    json.dump(json_dict, outfile)


