import argparse
import os
import yaml
from scene.scene_gs import SceneEnv
from run_create_dataset_gs import get_correct_skill_name


parser = argparse.ArgumentParser()
parser.add_argument('--scene_yaml_path', type=str, default=None)   # path of scene_model.yaml
parser.add_argument('--state_count', type=int, default=None)
parser.add_argument('--overwrite', action='store_true')
parser.add_argument('--save_keystate', action='store_true', help="save keystate")
parser.add_argument('--save_simenvcfg', action='store_true', help="save sim env")
parser.add_argument('--simenvcfg_save_path', type=str, help="only set for save sim env")
args = parser.parse_args()

print("Parsed arguments:")
for k, v in vars(args).items():
    print(f"  {k}: {v}")

assert args.save_keystate or args.save_simenvcfg
assert not (args.save_keystate and args.save_simenvcfg)

correct_skill_name = get_correct_skill_name(args.scene_yaml_path)
if correct_skill_name is None:
    raise ValueError("correct_skill_name do not exist!")

base_dataset = os.path.dirname(os.path.dirname(args.scene_yaml_path)) + '/dataset/dataset.pkl'
scene = None
fixed_base = True

for i in range(args.state_count):
    if args.save_keystate:
        # check exist
        scene_num = os.path.basename(args.scene_yaml_path).split('_')[0]
        augment_num = "{:04d}".format(i)
        save_dataset_path = os.path.dirname(base_dataset)
        dataset_path = save_dataset_path + f'/{scene_num}-{augment_num}_dataset.pkl'
        if not args.overwrite:
            if os.path.exists(dataset_path):
                print(f'{scene_num}-{augment_num}_dataset exist!')
                continue
            keep_dataset_path = save_dataset_path + f'/(keep){scene_num}-{augment_num}_dataset.pkl'
            if os.path.exists(keep_dataset_path):
                print(f'{scene_num}-{augment_num}_dataset exist!')
                continue

        if scene is None:
            scene = SceneEnv(args.scene_yaml_path, ifaugment=True, fixed_base=fixed_base)

        scene.place_objects(ifaugment=True)
        success = scene.augment_keystate(base_dataset, augment_number=i)

        if not success:
            break
    
    elif args.save_simenvcfg:
        assert args.simenvcfg_save_path is not None

        if not os.path.exists(args.simenvcfg_save_path):
            os.makedirs(args.simenvcfg_save_path)

        folder = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(args.scene_yaml_path))))
        save_name = '{obj_name}-{correct_skill_name}-{scene_num}-{state_num}.json'.format(
            obj_name=folder.split('_')[0].replace('-' , '_'),
            correct_skill_name=correct_skill_name.replace(' ', '_'),
            scene_num=os.path.basename(args.scene_yaml_path).split('_')[0],
            state_num="{:04d}".format(i),
        )
        dataset_path = os.path.join(args.simenvcfg_save_path, save_name)

        if not args.overwrite and os.path.exists(dataset_path):
            print(f'{save_name} exist!')
            continue

        if scene is None:
            scene = SceneEnv(args.scene_yaml_path, ifaugment=True, fixed_base=fixed_base)

        scene.place_objects(ifaugment=True)
        scene.save_env(dataset_path)
        print(f"====== Saved dataset-env to {dataset_path} ======")