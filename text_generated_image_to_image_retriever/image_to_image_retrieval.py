# Copyright 2022 san kim
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../")
import csv
import time
import json
import shutil
import logging
import hashlib
import functools

import numpy as np
from numpy.core.numeric import indices
import tqdm

import torch


import torch.utils.data
import torch.utils.data.distributed
import torch.nn.parallel

from transformers import AutoTokenizer, ViTFeatureExtractor

from index_scorer import FaissScorerExhaustiveGPU, FaissScorerExhaustiveMultiGPU, FaissScorer
from data_utils import DatasetForImages
from modeling_encoder import (
    VisionT5SimpleBiEncoder,
    VisionT5MeanBiEncoder,
    VisionT5SimpleBiEncoderHN,
    VisionT5MeanBiEncoderHN,
)

from training_retriever import (
    create_directory_info, 
    MODEL_CLS)



logger = logging.getLogger(__name__)

faiss_scorer = None
image_tokenizer = None
text_tokenizer = None
model = None
ref_data = None


def retrieve_image_with_image(image_query_list, FVECS_DIR="result/simple_query_ko/fvecs", HF_PATH="result/simple_query_ko/hf_model", MARKDOWN_OUT="result/simple_query_ko/md"):
    global faiss_scorer, image_tokenizer, model, ref_data, text_tokenizer
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--data_path",
                        default="cc12m_filtered.tsv", type=str)
    # parser.add_argument("--query_path",
    #                     default="query.json", type=str)
    parser.add_argument("--fvecs_dir",
                        default=None, type=str)
    parser.add_argument("--index_path",
                        default=None, type=str)
    parser.add_argument("--index_str",
                        default="IVF65536,Flat", type=str)
    parser.add_argument("--nprobe",
                        default=4, type=int)

    # model
    parser.add_argument("--vision_model",
                        default="google/vit-base-patch16-384", type=str)
    parser.add_argument("--language_model",
                        default="KETI-AIR/ke-t5-base", type=str)

    parser.add_argument("--model_cls", default="VisionT5MeanBiEncoder", 
                        choices=["VisionT5SimpleBiEncoder", 
                                "VisionT5MeanBiEncoder"],
                        type=str, help="model class")
    parser.add_argument("--dir_suffix",
                        default=None, type=str)
    parser.add_argument("--output_dir",
                        default="output", type=str)
    parser.add_argument("--markdown_out",
                        default="md", type=str)

    # resume
    parser.add_argument("--hf_path", default=None, type=str,
                        help="path to score huggingface model")
    
    parser.add_argument("--topk", default=10,
                        type=int, help="top k")
    parser.add_argument("--image_size", default=180,
                        type=int, help="image size for html formatting")

    # default settings for training, evaluation
    parser.add_argument("--batch_size", default=16,
                        type=int, help="mini batch size")
    parser.add_argument("--num_workers", default=0, type=int,
                        help="number of workers")

    # distributed setting
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--local_world_size", type=int, default=1,
                        help="The size of the local worker group.")
    parser.add_argument("--rank", type=int, default=0,
                        help="The rank of the worker within a worker group.")
    parser.add_argument("--world_size", type=int, default=1,
                        help="world size. (num_nodes*num_dev_per_node)")
    parser.add_argument("--distributed", action='store_true',
                        help="is distributed training")
    
    parser.add_argument('--model_gpu',
                        default=0, type=int)
    parser.add_argument('--scorer_gpus', nargs="+",
                        default=[0, 1, 2, 3], type=int)

    # --data_path ../kfashion_images_group.tsv --fvecs_dir result/simple_query_ko/fvecs --hf_path result/simple_query_ko/hf_model --query_path query.json --markdown_out result/simple_query_ko/md --model_cls VisionT5MeanBiEncoder
    args = parser.parse_args(["--data_path", "../kfashion_images_group.tsv",\
                            "--fvecs_dir", FVECS_DIR, \
                            "--hf_path", HF_PATH,\
                            "--markdown_out", MARKDOWN_OUT,\
                            "--model_cls", "VisionT5MeanBiEncoder",\
                            "--scorer_gpus", "0"])

    # print(args.scorer_gpus)
    # print(args.fvecs_dir)

    
    path_info = create_directory_info(args, create_dir=False)

    if args.fvecs_dir is None:
        args.fvecs_dir = os.path.join(path_info["model_dir"], "fvecs")

    if args.hf_path.lower()=='default':
        args.hf_path = os.path.join(path_info["model_dir"], "hf")

    model_device = torch.device('cuda:{}'.format(args.model_gpu))


    

    if faiss_scorer is None:
        faiss_scorer = FaissScorerExhaustiveMultiGPU(
                fvec_root=args.fvecs_dir,
                gpu_list=args.scorer_gpus
            )
        # get model class
        model_cls_cfg = MODEL_CLS[args.model_cls]
        model_cls = model_cls_cfg["model_cls"]

        # load model
        model = model_cls.from_pretrained(args.hf_path)
        model = model.to(model_device)

        # get tokenizer
        text_tokenizer = AutoTokenizer.from_pretrained(args.language_model)
        image_tokenizer = ViTFeatureExtractor.from_pretrained(args.vision_model)
        ref_data = [
        item for item in tqdm.tqdm(csv.DictReader(
            open(args.data_path, "r"), 
            delimiter="\t", 
            quoting=csv.QUOTE_MINIMAL, 
            fieldnames=['path', 'image_url']
        ), desc="loading item...")
    ]

        model.eval()

    markdown_out_dir = args.markdown_out
    if not os.path.isdir(markdown_out_dir):
        os.makedirs(markdown_out_dir, exist_ok=True)

    with torch.no_grad():
        """
        text_feature = text_tokenizer(text_query, return_tensors="pt", truncation='longest_first', padding=True)
        
        q_vecs = model.encode_text({
            "input_ids":text_feature["input_ids"].to(model_device),
            "attention_mask":text_feature["attention_mask"].to(model_device),})
        q_vecs = q_vecs.cpu().numpy()
        """
        
        image_features = image_tokenizer(image_query_list, return_tensors="pt").to(model_device)
        q_vecs = model.encode_image(image_features)
        q_vecs = q_vecs.cpu().numpy()
        scores, indice = faiss_scorer.get_topk(q_vecs, args.topk)

        result_list = []

        for t, score, index in zip(range(len(image_query_list)), scores, indice):
            result = [ {
                    "k": k+1,
                    "score": s,
                    "image_url": ref_data[i]["image_url"]
                } for k, s, i in zip(range(args.topk), score, index)]
            result_list.append(result)
    
        return result_list


def retrieve_image_with_multiple_images(multiple_image_query_list, FVECS_DIR="result/simple_query_ko/fvecs", HF_PATH="result/simple_query_ko/hf_model", MARKDOWN_OUT="result/simple_query_ko/md"):
    global faiss_scorer, image_tokenizer, model, ref_data, text_tokenizer
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--data_path",
                        default="cc12m_filtered.tsv", type=str)
    # parser.add_argument("--query_path",
    #                     default="query.json", type=str)
    parser.add_argument("--fvecs_dir",
                        default=None, type=str)
    parser.add_argument("--index_path",
                        default=None, type=str)
    parser.add_argument("--index_str",
                        default="IVF65536,Flat", type=str)
    parser.add_argument("--nprobe",
                        default=4, type=int)

    # model
    parser.add_argument("--vision_model",
                        default="google/vit-base-patch16-384", type=str)
    parser.add_argument("--language_model",
                        default="KETI-AIR/ke-t5-base", type=str)

    parser.add_argument("--model_cls", default="VisionT5MeanBiEncoder", 
                        choices=["VisionT5SimpleBiEncoder", 
                                "VisionT5MeanBiEncoder"],
                        type=str, help="model class")
    parser.add_argument("--dir_suffix",
                        default=None, type=str)
    parser.add_argument("--output_dir",
                        default="output", type=str)
    parser.add_argument("--markdown_out",
                        default="md", type=str)

    # resume
    parser.add_argument("--hf_path", default=None, type=str,
                        help="path to score huggingface model")
    
    parser.add_argument("--topk", default=10,
                        type=int, help="top k")
    parser.add_argument("--image_size", default=180,
                        type=int, help="image size for html formatting")

    # default settings for training, evaluation
    parser.add_argument("--batch_size", default=16,
                        type=int, help="mini batch size")
    parser.add_argument("--num_workers", default=0, type=int,
                        help="number of workers")

    # distributed setting
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--local_world_size", type=int, default=1,
                        help="The size of the local worker group.")
    parser.add_argument("--rank", type=int, default=0,
                        help="The rank of the worker within a worker group.")
    parser.add_argument("--world_size", type=int, default=1,
                        help="world size. (num_nodes*num_dev_per_node)")
    parser.add_argument("--distributed", action='store_true',
                        help="is distributed training")
    
    parser.add_argument('--model_gpu',
                        default=0, type=int)
    parser.add_argument('--scorer_gpus', nargs="+",
                        default=[0, 1, 2, 3], type=int)

    # --data_path ../kfashion_images_group.tsv --fvecs_dir result/simple_query_ko/fvecs --hf_path result/simple_query_ko/hf_model --query_path query.json --markdown_out result/simple_query_ko/md --model_cls VisionT5MeanBiEncoder
    args = parser.parse_args(["--data_path", "../kfashion_images_group.tsv",\
                            "--fvecs_dir", FVECS_DIR, \
                            "--hf_path", HF_PATH,\
                            "--markdown_out", MARKDOWN_OUT,\
                            "--model_cls", "VisionT5MeanBiEncoder",\
                            "--scorer_gpus", "0"])

    # print("[*] args.scorer_gpus", args.scorer_gpus)
    # print("[*] args.fvecs_dir", args.fvecs_dir)

    
    path_info = create_directory_info(args, create_dir=False)

    if args.fvecs_dir is None:
        args.fvecs_dir = os.path.join(path_info["model_dir"], "fvecs")

    if args.hf_path.lower()=='default':
        args.hf_path = os.path.join(path_info["model_dir"], "hf")

    model_device = torch.device('cuda:{}'.format(args.model_gpu))


    

    if faiss_scorer is None:
        faiss_scorer = FaissScorerExhaustiveMultiGPU(
                fvec_root=args.fvecs_dir,
                gpu_list=args.scorer_gpus
            )
        # get model class
        model_cls_cfg = MODEL_CLS[args.model_cls]
        model_cls = model_cls_cfg["model_cls"]

        # load model
        model = model_cls.from_pretrained(args.hf_path)
        model = model.to(model_device)

        # get tokenizer
        text_tokenizer = AutoTokenizer.from_pretrained(args.language_model)
        image_tokenizer = ViTFeatureExtractor.from_pretrained(args.vision_model)
        ref_data = [
        item for item in tqdm.tqdm(csv.DictReader(
            open(args.data_path, "r"), 
            delimiter="\t", 
            quoting=csv.QUOTE_MINIMAL, 
            fieldnames=['path', 'image_url']
        ), desc="loading item...")
    ]

        model.eval()

    markdown_out_dir = args.markdown_out
    if not os.path.isdir(markdown_out_dir):
        os.makedirs(markdown_out_dir, exist_ok=True)

    with torch.no_grad():
        q_vecs = []
        for image_query_list in multiple_image_query_list:
            image_features = image_tokenizer(image_query_list, return_tensors="pt").to(model_device)
            q_vec = model.encode_image(image_features)
            q_vec = q_vec.mean(dim=0, keepdim=True).cpu().numpy()
            q_vecs.append(q_vec)
        q_vecs = np.concatenate(q_vecs, axis=0)
        # print(np.shape(q_vecs))
        scores, indice = faiss_scorer.get_topk(q_vecs, args.topk)

        result_list = []

        for t, score, index in zip(range(len(multiple_image_query_list)), scores, indice):
            result = [ {
                    "k": k+1,
                    "score": s,
                    "image_url": ref_data[i]["image_url"]
                } for k, s, i in zip(range(args.topk), score, index)]
            result_list.append(result)
    
        return result_list
    

def retrieve_image_with_text(text_query_list, FVECS_DIR="result/simple_query_ko/fvecs", HF_PATH="result/simple_query_ko/hf_model", MARKDOWN_OUT="result/simple_query_ko/md"):
    global faiss_scorer, image_tokenizer, model, ref_data, text_tokenizer
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--data_path",
                        default="cc12m_filtered.tsv", type=str)
    # parser.add_argument("--query_path",
    #                     default="query.json", type=str)
    parser.add_argument("--fvecs_dir",
                        default=None, type=str)
    parser.add_argument("--index_path",
                        default=None, type=str)
    parser.add_argument("--index_str",
                        default="IVF65536,Flat", type=str)
    parser.add_argument("--nprobe",
                        default=4, type=int)

    # model
    parser.add_argument("--vision_model",
                        default="google/vit-base-patch16-384", type=str)
    parser.add_argument("--language_model",
                        default="KETI-AIR/ke-t5-base", type=str)

    parser.add_argument("--model_cls", default="VisionT5MeanBiEncoder", 
                        choices=["VisionT5SimpleBiEncoder", 
                                "VisionT5MeanBiEncoder"],
                        type=str, help="model class")
    parser.add_argument("--dir_suffix",
                        default=None, type=str)
    parser.add_argument("--output_dir",
                        default="output", type=str)
    parser.add_argument("--markdown_out",
                        default="md", type=str)

    # resume
    parser.add_argument("--hf_path", default=None, type=str,
                        help="path to score huggingface model")
    
    parser.add_argument("--topk", default=10,
                        type=int, help="top k")
    parser.add_argument("--image_size", default=180,
                        type=int, help="image size for html formatting")

    # default settings for training, evaluation
    parser.add_argument("--batch_size", default=16,
                        type=int, help="mini batch size")
    parser.add_argument("--num_workers", default=0, type=int,
                        help="number of workers")

    # distributed setting
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--local_world_size", type=int, default=1,
                        help="The size of the local worker group.")
    parser.add_argument("--rank", type=int, default=0,
                        help="The rank of the worker within a worker group.")
    parser.add_argument("--world_size", type=int, default=1,
                        help="world size. (num_nodes*num_dev_per_node)")
    parser.add_argument("--distributed", action='store_true',
                        help="is distributed training")
    
    parser.add_argument('--model_gpu',
                        default=0, type=int)
    parser.add_argument('--scorer_gpus', nargs="+",
                        default=[0, 1, 2, 3], type=int)

    # --data_path ../kfashion_images_group.tsv --fvecs_dir result/simple_query_ko/fvecs --hf_path result/simple_query_ko/hf_model --query_path query.json --markdown_out result/simple_query_ko/md --model_cls VisionT5MeanBiEncoder
    args = parser.parse_args(["--data_path", "../kfashion_images_group.tsv",\
                            "--fvecs_dir", FVECS_DIR, \
                            "--hf_path", HF_PATH,\
                            "--markdown_out", MARKDOWN_OUT,\
                            "--model_cls", "VisionT5MeanBiEncoder",\
                            "--scorer_gpus", "0"])

    # print(args.scorer_gpus)
    # print(args.fvecs_dir)

    
    path_info = create_directory_info(args, create_dir=False)

    if args.fvecs_dir is None:
        args.fvecs_dir = os.path.join(path_info["model_dir"], "fvecs")

    if args.hf_path.lower()=='default':
        args.hf_path = os.path.join(path_info["model_dir"], "hf")

    model_device = torch.device('cuda:{}'.format(args.model_gpu))


    

    if faiss_scorer is None:
        faiss_scorer = FaissScorerExhaustiveMultiGPU(
                fvec_root=args.fvecs_dir,
                gpu_list=args.scorer_gpus
            )
        # get model class
        model_cls_cfg = MODEL_CLS[args.model_cls]
        model_cls = model_cls_cfg["model_cls"]

        # load model
        model = model_cls.from_pretrained(args.hf_path)
        model = model.to(model_device)

        # get tokenizer
        text_tokenizer = AutoTokenizer.from_pretrained(args.language_model)
        image_tokenizer = ViTFeatureExtractor.from_pretrained(args.vision_model)
        ref_data = [
        item for item in tqdm.tqdm(csv.DictReader(
            open(args.data_path, "r"), 
            delimiter="\t", 
            quoting=csv.QUOTE_MINIMAL, 
            fieldnames=['path', 'image_url']
        ), desc="loading item...")
    ]

        model.eval()

    markdown_out_dir = args.markdown_out
    if not os.path.isdir(markdown_out_dir):
        os.makedirs(markdown_out_dir, exist_ok=True)

    with torch.no_grad():
        # print("[BG] text_query_list:", text_query_list)
        text_feature = text_tokenizer(text_query_list, return_tensors="pt", truncation='longest_first', padding=True)
        
        q_vecs = model.encode_text({
            "input_ids":text_feature["input_ids"].to(model_device),
            "attention_mask":text_feature["attention_mask"].to(model_device),})
        q_vecs = q_vecs.cpu().numpy()
        

        scores, indice = faiss_scorer.get_topk(q_vecs, args.topk)

        result_list = []

        for t, score, index in zip(range(len(text_query_list)), scores, indice):
            result = [ {
                    "k": k+1,
                    "score": s,
                    "image_url": ref_data[i]["image_url"]
                } for k, s, i in zip(range(args.topk), score, index)]
            result_list.append(result)
    
        return result_list



if __name__ == "__main__":
    from PIL import Image
    image_list = [Image.open("text_generated_image_to_image_retriever/test_1.jpg")]
    result = retrieve_image_with_image(image_list)
    print(result)


# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python retrieve_images.py \
# --data_path ../downloaded_data/cc12m/cc12m_filtered_new.tsv \
# --fvecs_dir fvecs_cc12m_freeze_lm \
# --hf_path output/VisionT5MeanBiEncoder-google_vit-base-patch16-384-KETI-AIR_ke-t5-base_freeze_lm/hf


# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python retrieve_images.py \
# --data_path ../downloaded_data/cc12m/cc12m_filtered_new.tsv \
# --fvecs_dir fvecs_cc12m_hn \
# --hf_path output/VisionT5MeanBiEncoderHN-google_vit-base-patch16-384-KETI-AIR_ke-t5-base/hf
