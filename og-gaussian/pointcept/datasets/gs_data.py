import os
import glob
import numpy as np
from torch.utils.data import Dataset as TorchDataset

from pointcept.utils.logger import get_root_logger
from .builder import DATASETS
from .transform import Compose
import pickle
import gzip
import random
import copy
from tqdm import tqdm
import h5py


def load_nested_dict_from_h5(h5file, path):
    result = {}
    for key in h5file[path]:
        full_path = f"{path}/{key}"
        if isinstance(h5file[full_path], h5py.Group):
            result[key] = load_nested_dict_from_h5(h5file, full_path)
        else:
            data = h5file[full_path][()]
            if isinstance(data, bytes):
                data = data.decode('utf-8')
            result[key] = data
    return result


@DATASETS.register_module()
class Dataset(TorchDataset):
    def __init__(
        self,
        data_root,
        clip_emb_path="data/clip_emb/v1",
        split="train",
        transform=None,
        activated=False
    ):
        super(Dataset, self).__init__()
        self.data_root = data_root
        self.clip_emb_path = clip_emb_path
        self.split = split
        self.transform = Compose(transform)
        self.sample_epoch = dict()

        self.data_list = self.get_data_list()
        self.load_text_emb()
        logger = get_root_logger()
        logger.info(
            "Totally {} samples.".format(len(self.data_list))
        )
        if activated:
            self.act_gs_feat_name = "gs_feat_activated"
            self.act_goal_gs_feat_name = "gs_goal_feat_activated"
            self.anc_gs_feat_name = "gs_feat_activated"
        else:
            self.act_gs_feat_name = "gs_feat"
            self.act_goal_gs_feat_name = "gs_goal_feat"
            self.anc_gs_feat_name = "gs_feat"
        self.pre_load()
    def pre_load(self):
        self.preload_data_list = []
        logger = get_root_logger()
        logger.info("Pre-loading data into RAM...")
        for data_path in tqdm(self.data_list):
            with h5py.File(data_path, "r") as f:
                h5_data = load_nested_dict_from_h5(f, "/")
            self.preload_data_list.append(h5_data)
            
    def _process_raw_data(self, data):
        anchor_data = data['anchor']
        
        anchor_feat = anchor_data[self.anc_gs_feat_name]
        anchor_vfeat = anchor_data['f_distilled']
        anchor_desc = anchor_data['desc']
        
        if self.od_aug is not None and anchor_desc in self.od_aug:
            anchor_desc = random.choice(self.od_aug[anchor_desc])
        
        if anchor_desc in self.od_emb:
            stfeat = self.od_emb[anchor_desc].copy()
        else:
            print(f'object name [{anchor_desc}] not exist!')
            raise ValueError
        
        anchor_tfeat = np.tile(stfeat, (anchor_vfeat.shape[0], 1))
        
        action_data = data['action']
        action_feat = action_data[self.act_gs_feat_name]
        action_goal_feat = action_data[self.act_goal_gs_feat_name]
        action_vfeat = action_data['f_distilled']
        action_desc = action_data['desc']
        
        if self.od_aug is not None and action_desc in self.od_aug:
            action_desc = random.choice(self.od_aug[action_desc])
            
        if action_desc in self.od_emb:
            otfeat = self.od_emb[action_desc].copy()
        else:
            print(f'object name [{action_desc}] not exist!')
            raise ValueError
            
        action_tfeat = np.tile(otfeat, (action_feat.shape[0], 1))
        
        skill_name = data['skill_name']
        if self.sn_aug is not None and skill_name in self.sn_aug:
            skill_name = random.choice(self.sn_aug[skill_name])
            
        if skill_name in self.sn_emb:
            skill_emb = self.sn_emb[skill_name].copy()
        else:
            print(f'skill [{skill_name}] not exist!')
            raise ValueError
        move_link = action_data['move_link'] if "move_link" in action_data else False
        data_dict = dict(
            anchor_feat=anchor_feat,
            anchor_vfeat=anchor_vfeat.copy(),
            anchor_tfeat=anchor_tfeat,
            anchor_init_pose=anchor_data['init_pose'],
            
            action_feat=action_feat,
            action_goal_feat=action_goal_feat,
            action_vfeat=action_vfeat.copy(),
            action_tfeat=action_tfeat,
            action_init_pose=action_data['init_pose'].copy(),
            action_goal_pose=action_data['goal_pose'].copy(),
            
            move_link=move_link,

            skill_emb=skill_emb,
        )
        return data_dict
    
    def load_text_emb(self):
        # read objects_description
        objects_description_path = os.path.join(self.data_root, "objects_description.pkl.gz")
        if not os.path.exists(objects_description_path):
            objects_description_path = os.path.join(self.clip_emb_path, "objects_description.pkl.gz")
        assert os.path.exists(objects_description_path)

        od_emb_path = os.path.join(self.data_root, "objects_description_emb.pkl.gz")
        if not os.path.exists(od_emb_path):
            od_emb_path = os.path.join(self.clip_emb_path, "objects_description_emb.pkl.gz")
        assert os.path.exists(od_emb_path)

        with gzip.open(objects_description_path, 'rb') as f:
            objects_description = pickle.load(f)
        with gzip.open(od_emb_path, 'rb') as f:
            od_emb = pickle.load(f)
        self.od_emb = dict()
        for i, od in enumerate(objects_description):
            self.od_emb[od] = od_emb[i]

        od_aug_path = os.path.join(self.data_root, "objects_description_augment.pkl.gz")
        if not os.path.exists(od_aug_path):
            od_aug_path = os.path.join(self.clip_emb_path, "objects_description_augment.pkl.gz")
        if os.path.exists(od_aug_path):
            with gzip.open(od_aug_path, 'rb') as f:
                self.od_aug:dict = pickle.load(f)
        else:
            self.od_aug = None

        # read skills_name
        skills_name_path = os.path.join(self.data_root, "skills_name.pkl.gz")
        if not os.path.exists(skills_name_path):
            skills_name_path = os.path.join(self.clip_emb_path, "skills_name.pkl.gz")
        assert os.path.exists(skills_name_path)

        sn_emb_path = os.path.join(self.data_root, "skills_name_emb.pkl.gz")
        if not os.path.exists(sn_emb_path):
            sn_emb_path = os.path.join(self.clip_emb_path, "skills_name_emb.pkl.gz")
        assert os.path.exists(sn_emb_path)

        with gzip.open(skills_name_path, 'rb') as f:
            skills_name = pickle.load(f)
        with gzip.open(sn_emb_path, 'rb') as f:
            sn_emb = pickle.load(f)
        self.sn_emb = dict()
        for i, sn in enumerate(skills_name):
            self.sn_emb[sn] = sn_emb[i]
        
        sn_aug_path = os.path.join(self.data_root, "skills_name_augment.pkl.gz")
        if not os.path.exists(sn_aug_path):
            sn_aug_path = os.path.join(self.clip_emb_path, "skills_name_augment.pkl.gz")
        if os.path.exists(sn_aug_path):
            with gzip.open(sn_aug_path, 'rb') as f:
                self.sn_aug:dict = pickle.load(f)
        else:
            self.sn_aug = None

    def get_data_list(self):
        if self.split == "train":
            data_list = glob.glob(os.path.join(self.data_root, "train/*"))
            data_list.sort()
            return data_list
        elif self.split == "val":
            data_list = glob.glob(os.path.join(self.data_root, "val/*"))
            data_list.sort()
            return data_list
        elif self.split == "test":
            data_list = glob.glob(os.path.join(self.data_root, f"test/*"))
            data_list.sort()
            assert len(data_list) > 0
            return data_list
        else:
            raise NotImplementedError

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        with h5py.File(data_path, "r") as f:
            data = load_nested_dict_from_h5(f, "/")
        
        # get anchor data
        anchor_data: dict = data['anchor']
        
        anchor_feat = anchor_data[self.anc_gs_feat_name]
        anchor_vfeat = anchor_data['f_distilled']
        anchor_desc = anchor_data['desc']
        if self.od_aug is not None and anchor_desc in self.od_aug:
            anchor_desc = random.choice(self.od_aug[anchor_desc])
        if anchor_desc in self.od_emb:
            stfeat = self.od_emb[anchor_desc].copy()
        else:
            print(f'object name [{anchor_desc}] not exist!')
            raise ValueError
        anchor_tfeat = np.tile(stfeat, (anchor_data['f_distilled'].shape[0], 1))
        
        # get action data
        action_data: dict = data['action']
        action_feat = action_data[self.act_gs_feat_name]
        action_goal_feat = action_data[self.act_goal_gs_feat_name]
        action_vfeat = action_data['f_distilled']
        action_desc = action_data['desc']
        if self.od_aug is not None and action_desc in self.od_aug:
            action_desc = random.choice(self.od_aug[action_desc])
        if action_desc in self.od_emb:
            otfeat = self.od_emb[action_desc].copy()
        else:
            print(f'object name [{action_desc}] not exist!')
            raise ValueError
        action_tfeat = np.tile(otfeat, (action_data[self.act_gs_feat_name].shape[0], 1))
        # get skill embedding
        skill_name = data['skill_name']
        if self.sn_aug is not None and skill_name in self.sn_aug:
            skill_name = random.choice(self.sn_aug[skill_name])
        if skill_name in self.sn_emb:
            skill_emb = self.sn_emb[skill_name].copy()
        else:
            print(f'skill [{skill_name}] not exist!')
            raise ValueError
        data_dict = dict(
            anchor_feat=anchor_feat,
            anchor_vfeat=anchor_vfeat.copy(),
            anchor_tfeat=anchor_tfeat,
            anchor_init_pose=anchor_data['init_pose'],
            
            action_feat=action_feat,
            action_goal_feat=action_goal_feat,
            action_vfeat=action_vfeat.copy(),
            action_tfeat=action_tfeat,
            action_init_pose=action_data['init_pose'].copy(),
            action_goal_pose=action_data['goal_pose'].copy(),

            skill_emb=skill_emb,

            sample_name=self.get_data_name(idx)
        )

        return data_dict

    def get_data_from_ram(self, idx):
        raw_data = self.preload_data_list[idx % len(self.data_list)]
        
        raw_data_copy = copy.deepcopy(raw_data)
        
        data_dict = self._process_raw_data(raw_data_copy)
        
        data_dict['sample_name'] = self.get_data_name(idx)
        return data_dict

    def get_data_name(self, idx):
        return os.path.basename(self.data_list[idx % len(self.data_list)]).split(".")[0]

    def prepare_data(self, idx):
        # load data
        data_dict = self.get_data_from_ram(idx)
        data_dict = self.transform(data_dict)
        return data_dict

    def __getitem__(self, idx):
        return self.prepare_data(idx)

    def __len__(self):
        return len(self.data_list)


@DATASETS.register_module()
class RepeatedDataset(Dataset):
    def __init__(
        self,
        data_root,
        clip_emb_path="data/clip_emb/v1",
        split="train",
        transform=None,
        repeat=1,
        activated=False
    ):
        super(RepeatedDataset, self).__init__(
            data_root,
            clip_emb_path,
            split,
            # test_mode,
            transform,
            activated
        )
        self.repeat = repeat


    def __len__(self):
        return len(self.data_list) * self.repeat
