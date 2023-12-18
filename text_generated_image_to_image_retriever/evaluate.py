import os
import sys
assert(sys.argv[1] in ['text_image_retrieval',
                       'text_image_retrieval_v2',
                       'text_image_retrieval_multiple_images_prompt_engineered_v3',
                       'text_image_retrieval_multiple_images_prompt_engineered_controlnet_v3',
                       'text_retrieval',
                       'text_image_retrieval_multiple_images_prompt_engineered_training_data_v3',
                       'text_retrieval_with_aux_classifier']) 
import json

import numpy as np
from ko_to_en_translation import translate_ko2en, google_translate_ko2en
from text_to_image_generation import generate_image, generate_controlnet_image
from image_to_image_retrieval import retrieve_image_with_image, retrieve_image_with_text, retrieve_image_with_multiple_images
import utils
import tqdm


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(PROJECT_DIR, f"tmp/{sys.argv[1]}")

os.makedirs(LOG_PATH, exist_ok=True)

data_list = utils.load_test_data()
# data_list = utils.load_training_data()


COUNT = 0
def text_image_retrieval_multiple_images_prompt_engineered_training_data_v3(text_list):
    global COUNT
    korean = text_list[0]
    english = google_translate_ko2en(text_list)[0]
    generated_images = generate_image([english], batch_size=4)[0]
    generated_images[0].save(LOG_PATH+"/"+"{:03d}".format(COUNT)+".jpg")
    [generated_image.save(LOG_PATH+"/"+"{:03d}".format(COUNT)+f"_{i}.jpg") for i, generated_image in enumerate(generated_images)]
    retrieved_results = retrieve_image_with_multiple_images([generated_images])

    aux_result = {"english":english}
    return retrieved_results, aux_result

def text_image_retrieval_multiple_images_prompt_engineered_v3(text_list):
    global COUNT
    korean = text_list[0]
    english = google_translate_ko2en(text_list)[0]
    generated_images = generate_image([english], batch_size=4)[0]
    generated_images[0].save(LOG_PATH+"/"+"{:03d}".format(COUNT)+".jpg")
    [generated_image.save(LOG_PATH+"/"+"{:03d}".format(COUNT)+f"_{i}.jpg") for i, generated_image in enumerate(generated_images)]
    retrieved_results = retrieve_image_with_multiple_images([generated_images])

    aux_result = {"english":english}
    return retrieved_results, aux_result

def text_image_retrieval_multiple_images_prompt_engineered_controlnet_v3(text_list):
    global COUNT
    korean = text_list[0]
    english = google_translate_ko2en(text_list)[0]
    generated_images = generate_controlnet_image([english], batch_size=4)[0]
    generated_images[0].save(LOG_PATH+"/"+"{:03d}".format(COUNT)+".jpg")
    [generated_image.save(LOG_PATH+"/"+"{:03d}".format(COUNT)+f"_{i}.jpg") for i, generated_image in enumerate(generated_images)]
    retrieved_results = retrieve_image_with_multiple_images([generated_images])

    aux_result = {"english":english}
    return retrieved_results, aux_result

def text_image_retrieval_v2(text_list):
    global COUNT
    korean = text_list[0]
    english = google_translate_ko2en(text_list)[0]
    generated_image = generate_controlnet_image([english])[0][0]
    
    generated_image.save(LOG_PATH+"/"+"{:03d}".format(COUNT)+".jpg")
    
    retrieved_results = retrieve_image_with_image([generated_image])

    aux_result = {"english":english}
    return retrieved_results, aux_result

def text_image_retrieval(text_list):
    global COUNT
    korean = text_list[0]
    english = translate_ko2en(text_list)[0]
    generated_image = generate_image([english])[0][0]
    
    generated_image.save(LOG_PATH+"/"+"{:03d}".format(COUNT)+".jpg")
    
    retrieved_results = retrieve_image_with_image([generated_image])

    aux_result = {"english":english}
    return retrieved_results, aux_result

def text_retrieval(text_list):    
    retrieved_results = retrieve_image_with_text(text_list)
    aux_result = {}
    return retrieved_results, aux_result

def text_retrieval_with_aux_classifier(text_list):    
    retrieved_results = retrieve_image_with_text(text_list, FVECS_DIR="../fashion_retriever/result/simple_query_label/hf_model/fvecs",\
                                                            HF_PATH="../fashion_retriever/result/simple_query_label/hf_model",\
                                                            MARKDOWN_OUT="../fashion_retriever/result/simple_query_label/md")
    aux_result = {}
    return retrieved_results, aux_result

def calc_score(target_tags, retrieved_tags):
    len_target = len(target_tags)
    len_retrieved = len(retrieved_tags)
    len_intersection = len(target_tags.intersection(retrieved_tags))

    precision = len_intersection / len_retrieved
    recall = len_intersection / len_target

    return precision, recall

precision_list = []
recall_list = []

eval_func = eval(sys.argv[1])

for d in tqdm.tqdm(data_list):
    COUNT+=1

    description = d[0]
    tags = d[1]
    if len(tags) == 0:
        continue

    result, aux_result = eval_func([description])
    result = result[0]

    precision_sub_list = []
    recall_sub_list = []
    retrieved_tags_sub_list = []
    for result_i in result:
        retrieved_id = result_i['image_url'].split("/")[-1].split(".jpg")[0]
        # print('retrieved_id', retrieved_id)
        retrieved_tags = utils.id_to_tags(retrieved_id) 

        p, r = calc_score(tags, retrieved_tags)
        # print(p, r)
        precision_sub_list.append(p)
        recall_sub_list.append(r)
        retrieved_tags_sub_list.append(list(retrieved_tags))

    retrieved_urls = [ r['image_url'] for r in result]
    # print(d, len(d))
    reference_url = d[4]

    ### per item result
    json_result = {"korean":description, "request_tags":list(tags), "retrieved_tags_list":retrieved_tags_sub_list,
                    "recall_list":recall_sub_list, "precision_list":precision_sub_list, "reference_url":reference_url, "retrieved_urls":retrieved_urls}

    for k,v in aux_result.items():
        json_result[k] = v

    json.dump(json_result,\
                open(LOG_PATH+"/"+"{:03d}".format(COUNT)+".json", mode="w"), ensure_ascii=False, indent=4)


    
    precision_list.extend(precision_sub_list)
    recall_list.extend(recall_sub_list)

## total precision recall

precision_list = np.array(precision_list)
recall_list = np.array(recall_list)
print(np.mean(precision_list), np.mean(recall_list))

print("[*] Mean Precision:{:.04f}, Mean Recall:{:.04f}".format(np.mean(np.array(precision_list)), np.mean(np.array(recall_list))))

np.save(LOG_PATH+"/precision_list.npy", precision_list)
np.save(LOG_PATH+"/recall_list.npy", recall_list)