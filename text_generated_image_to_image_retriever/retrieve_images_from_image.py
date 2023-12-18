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

import os
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
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch import optim

import torch.distributed as dist
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.parallel
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import AutoTokenizer, ViTFeatureExtractor
from torch.utils.tensorboard import SummaryWriter

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



def main():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--data_path",
                        default="cc12m_filtered.tsv", type=str)
    parser.add_argument("--query_path",
                        default="query.json", type=str)
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
                        default=[1,2,3,4], type=int)

    args = parser.parse_args()

    print(args.scorer_gpus)
    print(args.fvecs_dir)

    
    path_info = create_directory_info(args, create_dir=False)

    if args.fvecs_dir is None:
        args.fvecs_dir = os.path.join(path_info["model_dir"], "fvecs")

    if args.hf_path.lower()=='default':
        args.hf_path = os.path.join(path_info["model_dir"], "hf")

    model_device = torch.device('cuda:{}'.format(args.model_gpu))

    faiss_scorer = FaissScorerExhaustiveMultiGPU(
            fvec_root=args.fvecs_dir,
            gpu_list=args.scorer_gpus
        )
    # faiss_scorer = FaissScorer(
    #     index_path=args.index_path,
    #     fvec_root=args.fvecs_dir,
    #     index_str=args.index_str,
    #     nprobe=args.nprobe,
    # )
    
    ref_data = [
            item for item in tqdm.tqdm(csv.DictReader(
                open(args.data_path, "r"), 
                delimiter="\t", 
                quoting=csv.QUOTE_MINIMAL, 
                fieldnames=['path', 'image_url']
            ), desc="loading item...")
        ]

    # get model class
    model_cls_cfg = MODEL_CLS[args.model_cls]
    model_cls = model_cls_cfg["model_cls"]

    # load model
    model = model_cls.from_pretrained(args.hf_path)
    model = model.to(model_device)

    # get tokenizer
    text_tokenizer = AutoTokenizer.from_pretrained(args.language_model)
    image_tokenizer = ViTFeatureExtractor.from_pretrained(args.vision_model)
    
    model.eval()

    markdown_out_dir = args.markdown_out
    if not os.path.isdir(markdown_out_dir):
        os.makedirs(markdown_out_dir, exist_ok=True)

    # text_query = json.load(open(args.query_path, "r"))
    
    from PIL import Image
    text_ko_description_list = ["코듀로이 소재의 베이지 조거팬츠", 
                                "에스닉 무늬의 짧은 기장 원피스",
                                "어깨끈 라인이 두줄씩 있는 니트 브라탑과 코듀로이 소재의 베이지 조거팬츠",
                                "터틀넥에 스트링 디테일이 들어간 연핑크 브라탑와 화려하고 하늘하늘한 롱 기장 스커트",
                                "화려하고 하늘하늘한 롱 기장 스커트",]
    
    text_en_description_list = ["Beige jogger pants made of corduroy material",
                                "Short length dress with ethnic pattern",
                                "Knit bra top with two shoulder strap lines and beige jogger pants made of corduroy material",
                                "A light pink bra top with a turtleneck and string details, and a gorgeous, flowy long-length skirt",
                                "Gorgeous and airy long length skirt",
                                ] 
    
    IMAGE_QUERY_DIR = "/mnt/nfs4/byunggill/multimodal/k_fashion/VL-KE-T5/image_queries/"
    #IMAGE_QUERY_DIR = "../../../"

    image_query_path_list = [IMAGE_QUERY_DIR+"00000-2608241296.png",
                             IMAGE_QUERY_DIR+"ethnic_2.png",
                             IMAGE_QUERY_DIR+"knit_bra_jogger_pants.png",
                             IMAGE_QUERY_DIR+"light_pink_bra_top_turtleneck_1.png",
                             IMAGE_QUERY_DIR+"long_air_skirt_1.png",
                             ]
    image_query_list = [Image.open(p).convert("RGB") for p in image_query_path_list]
    
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

        img_size=args.image_size

        for query, result, ko_des, en_des in zip(image_query_path_list, result_list, text_ko_description_list, text_en_description_list):
            print(f"query: {query}\nresults\n"+'-'*40)
            print(result)
            print('-'*40+'\n\n')

            md5_hash = hashlib.md5(ko_des.encode("utf-8"))
            hash_str = md5_hash.hexdigest()
            markdown_path = os.path.join(markdown_out_dir, hash_str+"_image.md")


            HTML_STR = f"""
<table>

<tr>
    <td>KO Query</td>
    <td colspan="3">{ko_des}</td>
</tr>
<tr>
    <td>KO -> EN</td>
    <td colspan="3">{en_des}</td>
</tr>
<tr>
    <td>EN -> Generated Image</td>
    <td colspan="3"> <img height="{img_size}" width="{img_size}"
            src="../../../image_queries/{os.path.basename(query)}"></td>
</tr>
<tr>
    <td>Top 1</td>
    <td>Top 2</td>
    <td>Top 3</td>
    <td>Top 4</td>
</tr>
<tr>
    <td>
        <img height="{img_size}" width="{img_size}"
            src="{result[0]['image_url']}"
            alt="score: {result[0]['score']:.2f}">
    </td>
    <td>
        <img height="{img_size}" width="{img_size}"
            src="{result[1]['image_url']}"
            alt="score: {result[1]['score']:.2f}">
    </td>
    <td>
        <img height="{img_size}" width="{img_size}"
            src="{result[2]['image_url']}"
            alt="score: {result[2]['score']:.2f}">
    </td>
    <td>
        <img height="{img_size}" width="{img_size}"
            src="{result[3]['image_url']}"
            alt="score: {result[3]['score']:.2f}">
    </td>
</tr>
<tr>
    <td>Top 5</td>
    <td>Top 6</td>
    <td>Top 7</td>
    <td>Top 8</td>
</tr>
<tr>
    <td>
        <img height="{img_size}" width="{img_size}"
            src="{result[4]['image_url']}"
            alt="score: {result[4]['score']:.2f}">
    </td>
    <td>
        <img height="{img_size}" width="{img_size}"
            src="{result[5]['image_url']}"
            alt="score: {result[5]['score']:.2f}">
    </td>
    <td>
        <img height="{img_size}" width="{img_size}"
            src="{result[6]['image_url']}"
            alt="score: {result[6]['score']:.2f}">
    </td>
    <td>
        <img height="{img_size}" width="{img_size}"
            src="{result[7]['image_url']}"
            alt="score: {result[7]['score']:.2f}">
    </td>
</tr>
</table>
            """
            with open(markdown_path, "w") as f:
                f.write(HTML_STR)



if __name__ == "__main__":
    logging.basicConfig(level = logging.INFO)
    main()


# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python retrieve_images.py \
# --data_path ../downloaded_data/cc12m/cc12m_filtered_new.tsv \
# --fvecs_dir fvecs_cc12m_freeze_lm \
# --hf_path output/VisionT5MeanBiEncoder-google_vit-base-patch16-384-KETI-AIR_ke-t5-base_freeze_lm/hf


# CUDA_VISIBLE_DEVICES="0,1,2,3,4" python retrieve_images.py \
# --data_path ../downloaded_data/cc12m/cc12m_filtered_new.tsv \
# --fvecs_dir fvecs_cc12m_hn \
# --hf_path output/VisionT5MeanBiEncoderHN-google_vit-base-patch16-384-KETI-AIR_ke-t5-base/hf
