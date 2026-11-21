import json
import pandas as pd
import yaml
import os
from termcolor import cprint

behaviour_dataset_path = 'omnigibson_data/data/og_dataset/objects'


with open("data/dataset/behaviour_partnet_cate_uid_dict.json", 'r') as f:
    cate_uid_dict:dict = json.load(f)

with open("data/dataset/behaviour_partnet_uid_cate_dict.json", 'r') as f:
    uid_cate_dict:dict = json.load(f)

with open("data/dataset/behaviour_partnet_aabb.json", 'r') as f:
    uid_aabb_dict:dict = json.load(f)

if os.path.exists("data/dataset/objaverse/objaverse_text_uids_dict.json"):
    # dict[text: list[uids]]
    with open("data/dataset/objaverse/objaverse_text_uids_dict.json", 'r') as f:
        objaverse_text_uids_dict = json.load(f)
else:
    objaverse_text_uids_dict = {}


def get_object_description(uid:str)->str:
    if uid.isdigit() or len(uid) == 6:
        captions = pd.read_csv('data/dataset/behaviour_partnet_captions.csv', header=None)
    else:
        captions = pd.read_csv('data/dataset/objaverse/objaverse_captions.csv', header=None)
    captions = captions.dropna()
    idx = captions[captions.isin([uid]).any(axis=1)].index
    if len(idx) == 0:
        return None
    description = captions.iloc[idx,1].values[0]
    return description.replace('"', '')


def get_object_joint_semantics(uid)->dict:
    with open('data/dataset/behaviour_partnet_joint_semantics.yaml', 'r') as file:
        semantics_dict = yaml.safe_load(file)

    if str(uid) in semantics_dict:
        return semantics_dict[str(uid)]
    else:
        return None

def get_object_link_semantics(uid:str)->dict:
    with open('data/dataset/behaviour_partnet_link_semantics.yaml', 'r') as file:
        semantics_dict = yaml.safe_load(file)

    if str(uid) in semantics_dict:
        return semantics_dict[str(uid)]
    else:
        return None

def get_model_path(uid:str)->str:
    if uid.isdigit():   # partnet
        usd_path = f"data/dataset/partnet/objects/{uid}/mobility_base.usd"
        if os.path.exists(usd_path):
            return usd_path, 'partnet'
        else: return None, None
    
    elif not uid.isdigit() and len(uid) == 6: # behaviour
        cate = uid_cate_dict[uid].replace(' ', '_')
        usd_path = behaviour_dataset_path + f'/{cate}/{uid}/usd/{uid}.usd'
        if os.path.exists(usd_path):
            return usd_path, 'behaviour'
        else: return None, None
    
    else:
        urdf_path = f'data/dataset/objaverse/objects/{uid}/material.urdf'
        if os.path.exists(urdf_path):
            usd_path = urdf_path.replace('.urdf', '.usd')

            if not os.path.exists(usd_path):
                cprint(f"{usd_path} donot exists, you must convert **urdf** to **usd**", "yellow")

            return os.path.abspath(usd_path), 'objaverse'
        return None, None
