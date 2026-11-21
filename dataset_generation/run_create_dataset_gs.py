import argparse
import copy
import json
import os
import random
import subprocess
from pathlib import Path
from typing import List, Any

import yaml
from termcolor import cprint

import constants as cs
from data.dataset.utils import uid_cate_dict, cate_uid_dict, uid_aabb_dict, \
    get_object_link_semantics, get_model_path


UID_SPLIT_PATH = Path(__file__).resolve().parent / 'data/dataset/train_test_split.json'


def validate_uid_split(uid_split):
    if not isinstance(uid_split, dict):
        raise ValueError('The saved UID split must be a dictionary.')
    for split in ('train', 'test'):
        pools = uid_split.get(split)
        if not isinstance(pools, dict) or set(pools) != set(cate_uid_dict):
            raise ValueError(f'Invalid categories in the saved {split} UID split.')

    all_train, all_test = set(), set()
    for category, uids in cate_uid_dict.items():
        available = {str(uid) for uid in uids}
        train = uid_split['train'][category]
        test = uid_split['test'][category]
        if not all(isinstance(pool, list) and all(isinstance(uid, str) for uid in pool)
                   for pool in (train, test)):
            raise ValueError(f'Invalid UID lists for {category}.')
        train_set, test_set = set(train), set(test)
        if (not train or not test or len(train_set) != len(train)
                or len(test_set) != len(test) or train_set & test_set
                or train_set | test_set != available
                or len(train) != len(available) // 2):
            raise ValueError(f'Invalid 50/50 UID split for {category}.')
        all_train.update(train_set)
        all_test.update(test_set)
    if all_train & all_test:
        raise ValueError('Training and test UID pools overlap across categories.')


def load_or_create_uid_split(split_path=UID_SPLIT_PATH, seed=42):
    split_path = Path(split_path)
    if split_path.exists():
        with split_path.open('r', encoding='utf-8') as file:
            uid_split = json.load(file)
        validate_uid_split(uid_split)
        return uid_split

    rng = random.Random(seed)
    uid_split = {'seed': seed, 'train': {}, 'test': {}}
    for category in sorted(cate_uid_dict):
        uids = sorted({str(uid) for uid in cate_uid_dict[category]})
        if len(uids) < 2:
            raise ValueError(f'{category} needs at least two UIDs for a train/test split.')
        rng.shuffle(uids)
        midpoint = len(uids) // 2
        uid_split['train'][category] = uids[:midpoint]
        uid_split['test'][category] = uids[midpoint:]
    validate_uid_split(uid_split)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with split_path.open('x', encoding='utf-8') as file:
            json.dump(uid_split, file, indent=4)
            file.write('\n')
    except FileExistsError:
        return load_or_create_uid_split(split_path, seed)
    return uid_split


def scene_matches_uid_pool(scene_yaml, uid_pool):
    if not isinstance(scene_yaml, list):
        return False
    objects = [item for item in scene_yaml if isinstance(item, dict) and 'name' in item]
    return bool(objects) and all(
        str(item.get('uid')) in uid_pool.get(uid_cate_dict.get(str(item.get('uid'))), [])
        for item in objects
    )


def get_correct_skill_name(scene_yaml_path):
    with open(scene_yaml_path, 'r') as file:
        scene_yaml: list = yaml.safe_load(file)
    correct_skill_name = None
    for yaml_item in scene_yaml:
        if 'correct_skill_name' in yaml_item:
            correct_skill_name = yaml_item['correct_skill_name']
    return correct_skill_name


def _sort_recursively(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _sort_recursively(obj[k]) for k in sorted(obj)}
    if isinstance(obj, list):
        return [_sort_recursively(it) for it in obj]
    return obj  # str / int / float / None / bool

def yaml_equal(path_a: str | Path,
               path_b: str | Path,
               ignore_order: bool = False) -> bool:
    with open(path_a, "r") as fa, open(path_b, "r") as fb:
        data_a = yaml.safe_load(fa)
        data_b = yaml.safe_load(fb)

    if ignore_order:
        data_a, data_b = _sort_recursively(data_a), _sort_recursively(data_b)

    return data_a == data_b


def augmentation_scene_model(
    scene_model_path:str, 
    number:int=0, 
    probability=cs.augment_object_probability, 
    size_thresh=5, 
    overwrite=False,
    avoid_uid:list=None,
    save_path=None,
    attempt_times=100,
    split='train',
    split_path=UID_SPLIT_PATH,
):
    if split not in ('train', 'test'):
        raise ValueError(f'Unknown split: {split}')
    if number < 0 or (number < cs.augment_scene_count) != (split == 'train'):
        raise ValueError(f'Scene {number} is outside the {split} scene-number range.')
    uid_pool = load_or_create_uid_split(split_path)[split]

    if save_path is None:
        save_folder = os.path.join(os.path.dirname(scene_model_path), 'augmentation')
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        save_path = save_folder + '/{:04d}_scene_model.yaml'.format(number)

    if not overwrite and os.path.exists(save_path):
        with open(save_path, 'r') as file:
            existing_scene = yaml.safe_load(file)
        if not scene_matches_uid_pool(existing_scene, uid_pool):
            cprint(f'{save_path} does not match the saved {split} UID split', 'yellow')
            return None
        return save_path

    # read yaml
    with open(scene_model_path, 'r') as file:
        scene_yaml: list = yaml.safe_load(file)
    
    success = True
    for _ in range(attempt_times):
        augment_yaml = copy.deepcopy(scene_yaml)

        # record spatial_relationship inside
        inside_relationships = {}
        for item in augment_yaml:
            if 'spatial_relationship' in item and 'Inside' in item['spatial_relationship']:
                relationship = item['spatial_relationship'].split(',')
                relationship = [i.strip() for i in relationship]
                if len(relationship) == 3:
                    inside_relationships[item['name']] = relationship
                    
        for item in augment_yaml:
            if 'name' not in item: continue

            ifaugment = random.random() < probability
            if not ifaugment: continue

            uid:str = item['uid']
            if uid.isdigit() or len(uid) == 6:  # behaviour/partnet
                cate = uid_cate_dict[uid]
            else:
                continue
            
            # sample new obj with same category
            uids = list(uid_pool[cate])
            # new_uid = random.choice(uids)
            random.shuffle(uids)

            avoid_uids_ = list(getattr(cs, 'avoid_uids', []))
            if avoid_uid is not None:
                assert isinstance(avoid_uid, List)
                avoid_uids_ += avoid_uid

            augment = False
            for new_uid in uids:
                if new_uid in avoid_uids_:
                    continue

                # check size
                if not uid.isdigit() and len(uid) == 6: # behaviour
                    size = uid_aabb_dict[uid]
                    new_size = uid_aabb_dict[new_uid]
                    ratio_xy = (size[0]/size[1]) / (new_size[0]/new_size[1])
                    ratio_yz = (size[1]/size[2]) / (new_size[1]/new_size[2])
                    ratio_xz = (size[0]/size[2]) / (new_size[0]/new_size[2])
                    if max([ratio_xy, ratio_yz, ratio_xz]) > size_thresh:
                        continue
                    if min([ratio_xy, ratio_yz, ratio_xz]) < 1./size_thresh:
                        continue

                # set joint
                link_semantics:dict = get_object_link_semantics(uid)
                new_link_semantics = get_object_link_semantics(new_uid)
                joint_values:dict = item['joint_values']
                link_pair = None    # {old_link_name: new_link_name}
                new_joint_values = {}
                if 'operated_link_name' in item and item['operated_link_name'] != 'none':
                    operated_link_name = item['operated_link_name']
                    if new_link_semantics is None:
                        continue
                    
                    # set operated_link_name
                    same_semantic_links = []
                    for lname, sem in new_link_semantics.items():
                        if sem == link_semantics[operated_link_name]:
                            same_semantic_links.append(lname)
                    if len(same_semantic_links) == 0:
                        continue
                    new_link = random.choice(same_semantic_links)
                    item['operated_link_name'] = new_link

                    # set joint_values
                    new_joint_values = {new_link: joint_values[operated_link_name]}
                    link_pair = {operated_link_name: new_link}
                    joint_values.pop(operated_link_name)
                    for new_lname, new_sem in new_link_semantics.items():
                        if new_lname == new_link:
                            continue
                        # search same semantic link in joint_values
                        for lname in random.sample(list(link_semantics.keys()), len(link_semantics)):
                            if lname == operated_link_name: continue
                            if new_sem == link_semantics[lname]:
                                new_joint_values[new_lname] = joint_values[lname]
                                link_pair[lname] = new_lname
                                break
                
                elif joint_values != 'none':
                    if max(list(joint_values.values())) == 1 and new_link_semantics is None:
                        continue
                    
                    if new_link_semantics is not None:
                        new_joint_values = {}
                        link_pair = {}
                        # search same semantic link in joint_values
                        fail = False
                        for lname in random.sample(list(link_semantics.keys()), len(link_semantics)):
                            if joint_values[lname] != -1:
                                found = True
                                if joint_values[lname] == 1:
                                    found = False
                                for new_lname, new_sem in random.sample(list(new_link_semantics.items()), len(new_link_semantics)):
                                    if new_sem == link_semantics[lname] and new_lname not in new_joint_values:
                                        new_joint_values[new_lname] = joint_values[lname]
                                        link_pair[lname] = new_lname
                                        found = True
                                        break
                                if not found:
                                    fail = True
                                    break
                        
                        if fail:
                            continue

                
                # fill in missing joints
                if new_link_semantics is not None:
                    for new_lname, new_sem in new_link_semantics.items():
                        if new_lname not in new_joint_values:
                            new_joint_values[new_lname] = -1
                else:
                    new_joint_values = 'none'

                # check important_link
                important_link = None
                for n, rela in inside_relationships.items():
                    if item['name'] == rela[1]:
                        important_link = rela[2]

                        if link_pair is None:
                            continue
                        elif important_link not in link_pair:
                            # set important_link
                            set_important_link = False
                            for new_lname, new_sem in new_link_semantics.items():
                                if new_sem == link_semantics[important_link]:
                                    new_joint_values[new_lname] = joint_values[important_link]
                                    link_pair[important_link] = new_lname
                                    set_important_link = True
                            if not set_important_link:
                                continue                
                        inside_relationships[n][2] = link_pair[important_link]

                augment = True
                break
            
            if not augment:
                return None

            # update yaml
            item['uid'] = new_uid
            item['model'] = get_model_path(new_uid)[0]
            item['joint_values'] = new_joint_values
            # set scale
            if new_uid.isdigit():   # partnet
                ori_size = uid_aabb_dict[uid]
                new_size = uid_aabb_dict[new_uid]
                scale = min([ori_size[i]/new_size[i] for i in range(3)])
                item['scale'] *= scale
            
            print(f"augment {item['name']}: {uid} -> {new_uid}")

        # update spatial_relationship
        for item in augment_yaml:
            if 'name' in item and item['name'] in inside_relationships:
                item['spatial_relationship'] = ', '.join(inside_relationships[item['name']])

        if not scene_matches_uid_pool(augment_yaml, uid_pool):
            return None

        # save yaml
        with open(save_path, 'w', encoding='utf-8') as file:
            yaml.dump(augment_yaml, file, allow_unicode=True, default_flow_style=False)

        # Test scenes may reuse the same instances and object combinations.
        if split == 'train':
            for ii in range(number):
                folder = os.path.join(os.path.dirname(scene_model_path), 'augmentation')
                pre_path = folder + '/{:04d}_scene_model.yaml'.format(ii)
                if os.path.exists(pre_path) and yaml_equal(save_path, pre_path, ignore_order=True):
                    os.remove(save_path)
                    success = False
                    break
    
        if success:
            break
    
    if success:    
        print('>> save augmentation yaml to', save_path)
        return save_path
    else:
        return None


def create_task_dataset(
        dataset_path, 
        augment_scene_count:int, 
        augment_state_count:int, 
        overwrite=False, 
        start_scene_num=None,
        start_state_num=0,
        split='train',
        split_path=UID_SPLIT_PATH,
    ):
    """

    mode: train/test
    """
    if not os.path.exists(dataset_path):
        cprint(f"{dataset_path} donot exists", "yellow")
        return

    scene_path = os.path.dirname(os.path.dirname(dataset_path)) + '/scene_model.yaml'
    if not os.path.exists(scene_path):
        cprint(f"{scene_path} donot exists", "yellow")
        return

    if start_scene_num is None:
        start_scene_num = 0 if split == 'train' else cs.augment_scene_count

    print('='*100)
    print(scene_path)
    print("="*100)

    for i in range(start_scene_num+augment_scene_count)[start_scene_num:]:
        augmented_scene_model_path = augmentation_scene_model(
            scene_path, number=i, overwrite=overwrite, split=split, split_path=split_path)
        if augmented_scene_model_path is None:
            print('!! can not augmentation scene_model for', scene_path)
            break
        dataset_folder = os.path.dirname(os.path.dirname(augmented_scene_model_path)) + '/dataset'
        # check done
        done = True
        for j in range(start_state_num+augment_state_count)[start_state_num:]:
            # check exist
            scene_num = os.path.basename(augmented_scene_model_path).split('_')[0]
            augment_num = "{:04d}".format(j)
            dataset_path = dataset_folder + f'/{scene_num}-{augment_num}_dataset.pkl'
            if not overwrite:
                if os.path.exists(dataset_path):
                    continue
                keep_dataset_path = dataset_folder + f'/(keep){scene_num}-{augment_num}_dataset.pkl'
                if os.path.exists(keep_dataset_path):
                    continue
                done = False
                break
            else:
                done = False
                break

        if done:
            print('scene {:04d} done!'.format(i))
            continue

        try:
            command = [
                "python", "create_dataset_sample_gs.py",
                "--scene_yaml_path", augmented_scene_model_path,
                "--state_count", str(start_state_num + augment_state_count),
                "--save_keystate"
            ]
            if overwrite:
                command.append("--overwrite")
            subprocess.run(command, check=True)
        except:
            pass


def create_task_test_envcfg(
        scene_path, 
        augment_scene_count:int, 
        augment_state_count:int, 
        overwrite=False, 
        start_scene_num=cs.augment_scene_count,
        start_state_num=0,
        save_path=None,
        split_path=UID_SPLIT_PATH,
    ):
    """

    mode: train/test
    """
    if not os.path.exists(scene_path):
        cprint(f"{scene_path} donot exists", "yellow")
        return
    
    assert save_path is not None

    print('='*100)
    print(scene_path)
    print("="*100)

    correct_skill_name = get_correct_skill_name(scene_path)

    for i in range(start_scene_num+augment_scene_count)[start_scene_num:]:
        augmented_scene_model_path = augmentation_scene_model(
            scene_path, number=i, overwrite=overwrite, split='test', split_path=split_path)
        if augmented_scene_model_path is None:
            print('!! can not augmentation scene_model for', scene_path)
            break
        # check done
        done = True
        for j in range(start_state_num+augment_state_count)[start_state_num:]:
            # check exist
            folder = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(augmented_scene_model_path))))
            save_name = '{obj_name}-{correct_skill_name}-{scene_num}-{state_num}.json'.format(
                obj_name=folder.split('_')[0].replace('-' , '_'),
                correct_skill_name=correct_skill_name.replace(' ', '_'),
                scene_num=os.path.basename(augmented_scene_model_path).split('_')[0],
                state_num="{:04d}".format(j),
            )
            dataset_path = os.path.join(save_path, save_name)
            if not overwrite:
                if not os.path.exists(dataset_path):
                    done = False
                    break
            else:
                done = False
                break
        if done:
            print('scene {:04d} done!'.format(i))
            continue

        try:
            command = [
                "python", "create_dataset_sample.py",
                "--scene_yaml_path", augmented_scene_model_path,
                "--state_count", str(start_state_num + augment_state_count),
                "--save_simenvcfg",
                "--simenvcfg_save_path", save_path
            ]
            if overwrite:
                command.append("--overwrite")
            subprocess.run(command, check=True)
        except:
            pass


def create_task_test_envcfg_and_pkl(
        scene_path, 
        augment_scene_count:int, 
        augment_state_count:int, 
        overwrite=False, 
        start_scene_num=cs.augment_scene_count,
        start_state_num=0,
        save_path=None,
        split_path=UID_SPLIT_PATH,
    ):
    if not os.path.exists(scene_path):
        cprint(f"{scene_path} donot exists", "yellow")
        return
    
    assert save_path is not None

    print('='*100)
    print(scene_path)
    print("="*100)

    correct_skill_name = get_correct_skill_name(scene_path)

    for i in range(start_scene_num+augment_scene_count)[start_scene_num:]:
        augmented_scene_model_path = augmentation_scene_model(
            scene_path, number=i, overwrite=overwrite, split='test', split_path=split_path)
        if augmented_scene_model_path is None:
            print('!! can not augmentation scene_model for', scene_path)
            break
        # check done
        done = True
        for j in range(start_state_num+augment_state_count)[start_state_num:]:
            # check exist
            folder = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(augmented_scene_model_path))))
            scene_num = os.path.basename(augmented_scene_model_path).split('_')[0]
            state_num = "{:04d}".format(j)
            save_name = '{scene_num}-{state_num}.json'.format(
                scene_num=scene_num,
                state_num=state_num,
            )
            envcfg_path = os.path.join(save_path, save_name)
            dataset_folder = os.path.join(os.path.dirname(os.path.dirname(augmented_scene_model_path)), 'dataset')
            keystate_path = os.path.join(dataset_folder, f'{scene_num}-{state_num}_dataset.pkl')
            keep_keystate_path = os.path.join(dataset_folder, f'(keep){scene_num}-{state_num}_dataset.pkl')
            
            has_json = os.path.exists(envcfg_path)
            has_pkl = os.path.exists(keystate_path) or os.path.exists(keep_keystate_path)

            if not overwrite:
                if has_json and (not has_pkl):
                    try:
                        os.remove(envcfg_path)
                        print(f"remove stale json: {envcfg_path}")
                    except Exception as e:
                        print(f"failed to remove {envcfg_path}: {e}")
                    done = False
                    continue

                if (not has_json) or (not has_pkl):
                    done = False
            else:
                done = False

        if done:
            print('scene {:04d} done!'.format(i))
            continue

        try:
            command = [
                "python", "create_dataset_sample_gs_pkl_cfg.py",
                "--scene_yaml_path", augmented_scene_model_path,
                "--state_count", str(start_state_num + augment_state_count),
                "--save_keystate",
                "--save_simenvcfg",
                "--simenvcfg_save_path", save_path
            ]
            if overwrite:
                command.append("--overwrite")
            subprocess.run(command, check=True)
        except:
            pass



def create_alltask_dataset(augment_scene_count:int, augment_state_count:int, overwrite=False,
                          split='train', split_path=UID_SPLIT_PATH):
    folders = os.listdir(cs.task_meta_path)
    folders.sort()
    for folder in folders:
        if folder.startswith('(abandon)'):
            continue
        folder_path = cs.task_meta_path + '/' + folder
        if not os.path.isdir(folder_path):
            continue
        tasks = os.listdir(folder_path)
        tasks.sort()
        for task in tasks:
            if not task.startswith('(keep)'):
                continue
            
            base_dataset_path = os.path.join(folder_path, task, 'dataset/dataset.pkl')
            if not os.path.exists(base_dataset_path):
                continue

            scene_path = os.path.join(folder_path, task, 'scene_model.yaml')
            if not os.path.exists(scene_path):
                continue

            create_task_dataset(base_dataset_path, augment_scene_count, augment_state_count,
                                overwrite=overwrite, split=split, split_path=split_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=('train', 'test'), default='test')
    parser.add_argument('--dataset-path', default=(
        'generated_tasks_v2_source/Kettle/Kettle-lift_kettle_and_pour_water_v5/dataset/dataset.pkl'))
    parser.add_argument('--split-file', type=Path, default=UID_SPLIT_PATH)
    args = parser.parse_args()

    load_or_create_uid_split(args.split_file)
    is_train = args.split == 'train'

    create_task_dataset(
        dataset_path=args.dataset_path,
        augment_scene_count=cs.augment_scene_count if is_train else cs.test_scene_count,
        augment_state_count=cs.augment_state_count if is_train else cs.test_state_count,
        start_scene_num=0 if is_train else cs.augment_scene_count,
        start_state_num=0,
        overwrite=False,
        split=args.split,
        split_path=args.split_file,
    )
    
