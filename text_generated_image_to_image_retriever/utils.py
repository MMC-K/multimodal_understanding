import os
import json
import csv
import re

# configs
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
EM_GROUP_TABLE_PATH = os.path.join(PROJECT_PATH, "data/em_group_table.json")
FASHION_TEST_QUERY_PATH = os.path.join(PROJECT_PATH, "data/fashion_test_queries.csv")
FASHION_TRAINING_QUERY_PATH = os.path.join(PROJECT_PATH, "../../fashion_retriever/data/fashion_query/ko/im2query_max2_train.json")

# gobal variables
id_to_tags_dict = {}
print("[*] loading data for id_to_tags")
em_group_table_dict = json.load(open(EM_GROUP_TABLE_PATH, mode="r"))

# import routines
def build_id_to_tags_dict():
    global em_group_table_dict, id_to_tags_dict
    for group in em_group_table_dict['imgs_group']:
        group_tags = {tag for tag in group['tags']}
        for id in group['f_list']:
            id_to_tags_dict[id] = group_tags



print("[*] build data for id_to_tags")
build_id_to_tags_dict()


# util functions for importers
def id_to_tags(id):
    global id_to_tags_dict
    return id_to_tags_dict[id]

def load_test_data():
    csv_reader = csv.reader(open(FASHION_TEST_QUERY_PATH, mode="r"), delimiter=",")
    next(csv_reader, None)
    data_list = []
    for row in csv_reader:
        natural_language_description = row[1]
        included_tags = {t.strip() for t in re.split('\n|,', row[2]) if len(t.strip()) > 0}
        clothes_tags = {t.strip() for t in re.split('\n|,', row[3]) if len(t.strip()) > 0}
        type = row[4]
        url = row[5]
        data_list.append([natural_language_description, included_tags, clothes_tags, type, url])
    
    return data_list

def load_training_data(p = FASHION_TRAINING_QUERY_PATH, natural_description_key="global"):
    json_object = json.load(open(p, mode="r"))
    data_list = []
    for k, v in json_object.items():
        # print(k, v)
        if len(v["global"]) == 0:
            print(f"[*] Warning, load_training_data, no tags found for input id {k}")
            continue
        if natural_description_key == "global_en":
            natural_language_description = v[natural_description_key]["query"][0]
        elif natural_description_key == "global":
            natural_language_description = v[natural_description_key][-1]["query"]
        else:
            assert(False)
            
        included_tags = set(v["global"][-1]["tags"])
        clothes_tags = included_tags
        type = None
        for t in ["상의", "하의", "아우터", "원피스"]:
            if len(v[t]) > 0:
                type = t
                break
        url = f"https://storage.googleapis.com/k_fashion_images/k_fashion_images/{k}.jpg"
        data_list.append([natural_language_description, included_tags, clothes_tags, type, url])
        

    return data_list

if __name__ == "__main__":
    # print(id_to_tags('1162001'))
    load_training_data()
