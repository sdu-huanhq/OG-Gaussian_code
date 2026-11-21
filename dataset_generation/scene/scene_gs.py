
import omnigibson as og
import omnigibson.utils.transform_utils as T
import omnigibson.lazy as lazy
from omnigibson import object_states
from omnigibson.macros import gm
from omnigibson.utils.ui_utils import KeyboardEventHandler
from omnigibson.utils.ui_utils import choose_from_options, draw_line, clear_debug_drawing
from omnigibson.utils.sampling_utils import raytest_batch
from omnigibson.objects import DatasetObject, USDObject
from omnigibson.controllers import ControlType
from omnigibson.utils.sim_utils import check_collision

import numpy as np
import torch
import torch as th
import random
from common import visual as vsl
from common.contact import compute_contact_similarity, calculate_contact_distribusion_consistency, \
    compute_contact_similarity_v2
import cv2
import os
import h5py
import open3d as o3d
import pyvista as pv
import time
import json
import copy
import constants as cs
import pyspacemouse
import threading
from typing import Union
from termcolor import cprint
import common.transformation as tf
from tqdm import tqdm
import pickle
from sklearn.neighbors import NearestNeighbors
import yaml
from data.dataset.utils import uid_cate_dict, \
    get_object_joint_semantics, get_object_link_semantics, uid_aabb_dict, \
    get_object_description, get_model_path



gm.ENABLE_HQ_RENDERING = True
gm.ENABLE_FLATCACHE = True
gm.HEADLESS = False


class SceneEnv:
    def __init__(
        self, 
        scene_yaml_path:str, 
        ifaugment=False, 
        fixed_base=False, 
        random_place_objects=True, 
        fixed_base_operate=False,
        restore_cfg_path=None,
        istest=False,
        ground_range=3
    ):
        """
        teleop_device: ['spacemouse', 'keyboard']
        teleop_frame: ['camera', 'object']
        fixed_base: if True, all fixed_base; if false, follow 'movable'
        restore_cfg_path: saved env using og.sim.save(json_paths=['/path'])
        """
        self.scene_yaml_path = scene_yaml_path
        self.ifaugment = ifaugment
        self.fixed_base = fixed_base
        self.fixed_base_operate = fixed_base_operate
        self.restore_cfg_path = restore_cfg_path
        self.istest = istest

        self.objects_uids_dict = dict()
        self.objects_nativeBbox_dict = dict()
        self.objects_nativeBbox_offset_dict = dict()
        self.objects_jointValues_dict = dict()
        if ifaugment:
            self.contact_indices_dict = dict()
        self.first_time = True

        self.env = None
        self.load_scene(scene_yaml_path, ifaugment, fixed_base, fixed_base_operate, restore_cfg_path=restore_cfg_path, ground_range=ground_range)

        if (not ifaugment) and (random_place_objects) and (restore_cfg_path is None):
            self.place_objects(ifaugment)
        
        for _ in range(10):
            self.env.step([])
        
    
    def init_save_render_key(self):
        print("type 's' to save image")
        print("type 'q' to quit")
        KeyboardEventHandler.initialize()
        KeyboardEventHandler.add_keyboard_callback(key=lazy.carb.input.KeyboardInput.S, callback_fn=self.save_render)
        KeyboardEventHandler.add_keyboard_callback(key=lazy.carb.input.KeyboardInput.Q, callback_fn=self.stop)


    def save_render(self):
        obs, info = self.viewer_camera.get_obs()
        rgb = obs['rgb'].cpu().numpy()[..., (2,1,0)]
        cv2.imwrite(f'{int(time.time()*1000)}.png', rgb)
        print('render and save imagae')

    
    def save_env(self, path):
        og.sim.save(json_paths=[path])


    def reload_scene(self, scene_yaml_path, restore_cfg_path=None):
        self.load_scene(scene_yaml_path, self.ifaugment, self.fixed_base, self.fixed_base_operate, restore_cfg_path)


    def load_scene(
            self, 
            scene_yaml_path:str, 
            ifaugment=False, 
            fixed_base=False, 
            fixed_base_operate=False,
            restore_cfg_path=None,
            ground_range=3
        ):
        """
        scene_yaml_path: path of scene_model.yaml
        ifaugment: if the yaml is augmented
        restore_cfg_path: saved env using og.sim.save(json_paths=['/path'])
        ground_range: range of the ground plane
        """
        with open(scene_yaml_path, 'r') as file:
            scene_yaml: list = yaml.safe_load(file)

        # parse yaml
        self.correct_skill_name = None
        self.task_name, self.skill_name, self.tele_obj_cfg, objects_cfg, self.objects_relationships = \
            self.parse_scene_yaml(scene_yaml, ifaugment, fixed_base, fixed_base_operate)
        assert self.correct_skill_name is not None

        cfg = dict()
        cfg["scene"] = {
            "type": "Scene",
            "floor_plane_visible": True,
        }

        self.objects_name = list(self.objects_uids_dict.keys()) + ['ground',]

        if restore_cfg_path is None:
            # Update their spawn positions so they don't collide immediately
            for i, obj_cfg in enumerate(objects_cfg):
                obj_cfg["position"] = [100 + i, 100 + i, 100 + i]

            # save objects_cfg
            self.objects_cfg = copy.deepcopy(objects_cfg) 

            ground = {
                "type": "DatasetObject",
                "name": "ground",
                "category": "floors",
                "model": "aoaasb",
                "position": [0, 0, 0.01],
                "bounding_box": [ground_range, ground_range, 0.02],
                "fixed_base": True,
                "visible": True,
                # "visual_only": True
            }
            objects_cfg.append(ground)
            cfg["objects"] = objects_cfg

            print('='*30, 'objects_cfg', '='*30)
            for obj_cfg in objects_cfg:
                yaml_str = yaml.dump(obj_cfg, default_flow_style=False)
                print(yaml_str)
            print('='*100)
        
        else:
            cfg["scene"]['scene_file'] = restore_cfg_path



        # set camera
        camera_modalities = ["rgb", "depth_linear", "seg_semantic", "seg_instance", "seg_instance_id"]
        sensor_kwargs = {
            "image_height": cs.image_height, 
            "image_width": cs.image_width, 
            "focal_length": cs.focal_length
            }
        external_sensors_config = [
            {
                "sensor_type": "VisionSensor",
                "name": "camera1",
                "relative_prim_path": "/camera1",
                "modalities": camera_modalities,
                "sensor_kwargs": sensor_kwargs,
                "position": [-1.4015, -1.4858,  2.0775],
                "orientation": [ 0.3790, -0.1509, -0.3376,  0.8483],
            },
            {
                "sensor_type": "VisionSensor",
                "name": "camera2",
                "relative_prim_path": "/camera2",
                "modalities": camera_modalities,
                "sensor_kwargs": sensor_kwargs,
                "position": [1.4482, 1.4170, 2.0622],
                "orientation": [0.1609, 0.3909, 0.8381, 0.3449],
            },
            {
                "sensor_type": "VisionSensor",
                "name": "camera3",
                "relative_prim_path": "/camera3",
                "modalities": camera_modalities,
                "sensor_kwargs": sensor_kwargs,
                "position": [1.5622, -1.6144, 1.9203],
                "orientation": [0.3874, 0.1574, 0.3419, 0.8416],
            },
            {
                "sensor_type": "VisionSensor",
                "name": "camera4",
                "relative_prim_path": "/camera4",
                "modalities": camera_modalities,
                "sensor_kwargs": sensor_kwargs,
                "position": [-1.3320,  1.3416,  2.1274],
                "orientation": [-0.1368,  0.3290,  0.8628, -0.3587],
            }
        ]
        cfg["env"] = {
            "rendering_frequency": 60,  # for HQ rendering
            "external_sensors": external_sensors_config
            }
        cfg["render"] = {"viewer_width": cs.image_width, "viewer_height": cs.image_height}

        # Create the environment
        if self.env is None:
            self.env = og.Environment(configs=cfg)

        else:
            og.clear()
            self.env.reload(cfg, overwrite_old=True)
            og.sim.play()
            self.env.post_play_load()

        self.first_time = False
        self.env.step([])

        # get native bbox
        self.objects_nativeBbox_dict, self.objects_nativeBbox_offset_dict = self.get_native_bbox()
        # set contact id
        self.objects_contact_id_dict = self.set_objects_contact_id()

        self.viewer_camera = og.sim.viewer_camera
        self.cameras = {
            'camera1': self.env.external_sensors['camera1'], 
            'camera2': self.env.external_sensors['camera2'], 
            'camera3': self.env.external_sensors['camera3'], 
            'camera4': self.env.external_sensors['camera4'], 
            'camerav': self.viewer_camera
        }

        # set viewer_camera
        for modality in camera_modalities:
            if modality not in self.viewer_camera.modalities:
                self.viewer_camera.add_modality(modality)
        
        print('load scene done')
        for _ in range(10):
            self.env.step([])


    def place_objects(self, ifaugment):
        """place objs"""
        # reset to sky
        for i, obj_cfg in enumerate(self.objects_cfg):
            obj = self.env.scene.object_registry("name", obj_cfg['name'])
            pos = [100 + i, 100 + i, 100 + i]
            qua = [0, 0, 0, 1]
            obj.set_position_orientation(position=pos, orientation=qua)
            obj.keep_still()
        og.sim.step_physics()
        og.sim.step_physics()
        og.sim.step_physics()
        og.sim.step_physics()
        og.sim.step_physics()

        relationships = copy.deepcopy(self.objects_relationships)
        placed_objs = ['ground',]
        while len(relationships) > 0:
            for i, relationship in enumerate(relationships):
                if relationship[2] in placed_objs:
                    obj_name = relationship[0]
                    print(f'>>>>>>>>> place {relationship}')
                    success = self.place_object(*relationship)
                    if not success:
                        if relationship[0] == self.tele_obj_cfg['obj_name']:
                            if not ifaugment:
                                skill_path = os.path.dirname(self.scene_yaml_path)
                                os.rename(skill_path, skill_path.replace('/(keep)', '/(abandon)'))
                            raise RuntimeError(f'place {relationship} failed')
                        else:
                            print('replace to ground')
                            success = self.place_object(relationship[0], 'OnTop', 'ground')
                    print('success =', success)
                    self.env.step([])
                    self.env.step([])
                    self.env.step([])
                    placed_objs.append(relationship[0])
                    relationships.pop(i)

                    # set joint value
                    jointValues = self.objects_jointValues_dict[obj_name]
                    if jointValues != 'none':
                        print('set jointValues: ', jointValues)
                        obj = self.env.scene.object_registry("name", obj_name)
                        for lname, value in jointValues.items():
                            joint = self.get_jointPrim_w_link(obj, lname)
                            print('set joint: ', joint.joint_name)
                            if joint.control_type != ControlType.POSITION:
                                joint.set_control_type(ControlType.POSITION, kp=cs.joint_position_kp)
                            if value == 'random':
                                # value = random.uniform(-1, 1)
                                value = random.uniform(0.4, 0.85)
                            obj.keep_still()
                            jnt_success = self.set_jnt_position(joint, value)
                            # if not jnt_success:
                            #     raise RuntimeError(f'set {obj_name} joint failed')
                    break 

        for _ in range(10):
            self.env.step([])
    

    def get_objects_poses(self) -> dict:
        objects_poses = dict()
        obj_names = list(self.objects_uids_dict.keys()) + ['ground',]
        for name in obj_names:
            obj = self.env.scene.object_registry("name", name)
            pos, qua = obj.get_position_orientation()
            objects_poses[name] = pos.tolist() + qua.tolist()
        return objects_poses
        

    def get_observations(self, with_table=False) -> dict:
        data = dict()
        cams = ['camera1', 'camera2', 'camera3', 'camera4']
        for cam_name in cams:
            cam = self.cameras_[cam_name]
            obs, info = cam.get_obs()
            camv_pose = tf.PosQua_to_TransMat(*cam.get_position_orientation())
            diff = tf.PosEuler_to_TransMat((0, 0, 0), (np.pi, 0, 0))
            camv_pose = camv_pose @ diff
            data[cam_name] = dict(
                rgb=obs['rgb'].cpu().numpy()[..., :3],
                depth=obs['depth_linear'].cpu().numpy(),
                mask=obs['seg_instance'].cpu().numpy(),
                K=cam.intrinsic_matrix.numpy(),
                pose=camv_pose
            )
        # get obj_id
        allObjsname_id_dict = {n:i for i,n in info['seg_instance'].items()}
        obj_id = allObjsname_id_dict[self.tele_obj_cfg['obj_name']]
        # get objs_id_cate
        # uids = ['zdvgol', '103208']
        # obj_id = allObjsname_id_dict['Kettle']
        objs_id_cate = dict()
        for id, name in info['seg_instance'].items():
            if id not in objs_id_cate and name in self.objects_uids_dict:
                if not with_table and name.lower() == 'table':
                    continue
                uid = self.objects_uids_dict[name]
                # if uid in uids:
                objs_id_cate[id] = uid_cate_dict[uid] if uid in uid_cate_dict \
                    else get_object_description(uid)

        data['meta_info'] = dict(
            operate_obj_id=obj_id,
            objs_id_cate=objs_id_cate,
        )

        print('objs_id_cate =', objs_id_cate)

        if self.correct_skill_name is not None:
            data['meta_info']['skill_name'] = self.correct_skill_name

        return data


    def collect_task_demo(self, teleop_device='spacemouse', reference_frame='world'):
        """
        collect data of task, saved in parent path of self.scene_yaml_path

        args:
        - teleop_device (str): ['spacemouse', 'keyboard']
        - reference_frame (str): ['world', 'object']




        ```dict
        skill_name:
            {name}: {maskid}
            ...
        objs:
            name: 
                uid
            camera_name: (3, 3)
            spatial_relationship: 
                OnTop, obj_name
                Inside, obj_name
                Inside, obj_name, link_name, joint_pos
            obj_pose: (7, )  xyz/xyzw
            objs_pose: (only for objs w/ relevance = True)
                name: (7, )  xyz/xyzw
            links_pose:
                {link_name}: (7,)  xyz/xyzw
                    contact_id: int
                    uid:
                    pose: (7,) xyz/xyzw
                camera_name:
                    rgb: (H, W, 3)
                    depth: (H, W)
                    instance: (H, W)
                    instanceid: (H, W)
                    pose: (4,4)
        ```

        """

        # init tele-operation system
        self.obj_actions = torch.zeros((6,), dtype=torch.float32)
        self.jnt_action = 0.
        if teleop_device == 'keyboard':
            self.obj_move_delta = 0.02  # m
            self.obj_rota_delta = 10./180.*torch.pi
            self.jnt_delta = 0.2   # norm
            self.init_teleop_keyboard()
        elif teleop_device == 'spacemouse':
            self.obj_move_delta = 0.04 * cs.obj_control_ratio  # m
            self.obj_rota_delta = (20./180.*torch.pi) * cs.obj_control_ratio
            self.jnt_delta = 0.2   # norm
            self.init_teleop_spacemouse()
            self.spacemouse_thread.start()
        else:
            raise ValueError("teleop_device must be chosen in ['keyboard', 'spacemouse']")

        # init data
        data_dict = dict()
        objs_data = {n: dict(uid=v) for n, v in self.objects_uids_dict.items()}
        
        # get camera intrinsic_matrix
        for cam_name, cam in self.cameras.items():
            K = None
            for _ in range(20):
                try:
                    _ = cam.camera_parameters

                    og.sim.render()
                    og.sim.render()
                    og.sim.render()
                    og.sim.render()

                    K = cam.intrinsic_matrix
                    K = K.cpu().numpy() if hasattr(K, "cpu") else K.numpy()

                    if K[0, 0] > 1e-6 and K[1, 1] > 1e-6:
                        break
                except Exception:
                    pass
        
        camera_intrinsic = dict()
        for cam_name, cam in self.cameras.items():
            camera_intrinsic[cam_name] = cam.intrinsic_matrix.numpy()
        data_dict['camera_intrinsic'] = camera_intrinsic

        print('='*100)
        print('Task name: ', self.task_name)
        print('Skill name: ', self.correct_skill_name)
        print('='*100)

        # get teleop obj and link
        tele_obj = self.env.scene.object_registry("name", self.tele_obj_cfg['obj_name'])
        tele_jnt = None
        if self.tele_obj_cfg['operated_link_name'] != 'none':
            tele_jnt = self.get_jointPrim_w_link(tele_obj, self.tele_obj_cfg['operated_link_name'])
        data_dict['obj_name'] = tele_obj.name
        objs_data[tele_obj.name]['relevance'] = True
        
        if tele_jnt is not None:
            print('>>> Tele-object: {}, link: {}, joint: {}'.format(
                tele_obj.name, self.tele_obj_cfg['operated_link_name'], tele_jnt.joint_name))
            print('>>> Joint init pos:', tele_jnt.get_state(normalized=True)[0])
            if tele_jnt.control_type != ControlType.POSITION:
                tele_jnt.set_control_type(ControlType.POSITION, kp=cs.joint_position_kp)
            links_name = self.links_from_joint(tele_obj, tele_jnt)
            data_dict['operated_link_name'] = self.tele_obj_cfg['operated_link_name']
            data_dict['links_name_maskid'] = dict()
        else:
            print('>>> Tele-object: {}'.format(tele_obj.name))
            data_dict['operated_link_name'] = None

        # run
        self.esc = False
        self.iskeystate = False
        self.obj_sleep = True
        tele_obj.sleep()
        keystate_num = 0
        while not self.esc:
            if tele_obj.is_asleep and not self.obj_sleep:
                tele_obj.wake()
            if not tele_obj.is_asleep and self.obj_sleep:
                tele_obj.sleep()

            self.draw_object_axes(tele_obj, reference_frame=reference_frame)

            #* ================= read action =================
            if torch.max(torch.abs(self.obj_actions)) > 0 and tele_jnt is None:
                # reset obj pose
                pos, qua = self.step_obj(tele_obj, self.obj_actions, reference_frame)
                tele_obj.set_position_orientation(position=pos, orientation=qua)
                tele_obj.keep_still()
                if teleop_device == 'keyboard':
                    self.obj_actions *= 0
                clear_debug_drawing()
            
            if self.jnt_action != 0 and tele_jnt is not None:
                # reset joint
                self.step_jnt(tele_jnt, self.jnt_action)
                print('-- joint value:', tele_jnt.get_state(normalized=True)[0])
                if teleop_device == 'keyboard':
                    self.jnt_action = 0
                clear_debug_drawing()

            if not tele_obj.is_asleep:
                clear_debug_drawing()

            #* ================= step =================
            self.env.step([])
             
            #* ================= save data =================
            if not self.iskeystate:
                time.sleep(0.02)
                continue

            print('record keystate:', keystate_num)
            key_state = dict()

            # camera data
            camera_data = dict()
            for cam_name, cam in self.cameras.items():
                cam_pose = tf.PosQua_to_TransMat(*cam.get_position_orientation())
                diff = tf.PosEuler_to_TransMat((0, 0, 0), (np.pi, 0, 0))
                cam_pose = cam_pose @ diff
                obs, info = cam.get_obs()
                camera_data[cam_name] = dict(
                    rgb=obs['rgb'].cpu().numpy()[..., :3],
                    depth=obs['depth_linear'].cpu().numpy(),
                    instance=obs['seg_instance'].cpu().numpy(),
                    instanceid=obs['seg_instance_id'].cpu().numpy(),
                    pose=cam_pose
                )
                
                # record maskid
                for i, n in info['seg_instance'].items():
                    if n in self.objects_uids_dict and 'maskid' not in objs_data[n]:
                        objs_data[n]['maskid'] = i

                # update links_name_maskid
                if tele_jnt is not None:
                    allLinksname_id_dict = {}
                    for i,n in info['seg_instance_id'].items():
                        if 'scene_0' in n:
                            allLinksname_id_dict[n.split('/')[3] + '-' + n.split('/')[4]] = i
                    for link_name in links_name:
                        obj_link_name = tele_obj.name + '-' + link_name
                        if link_name not in data_dict['links_name_maskid'] and obj_link_name in allLinksname_id_dict:
                            data_dict['links_name_maskid'][link_name] = allLinksname_id_dict[obj_link_name]
            key_state['camera_data'] = camera_data

            # record obj/link pose, spatial_relationship
            if tele_jnt is None:
                # record obj pose
                opos, oqua = tele_obj.get_position_orientation()
                key_state['obj_pose'] = opos.tolist() + oqua.tolist()

                # record contact
                points, contact_data, contact_objs = self.get_contact_state(
                    tele_obj, shooting_offset=cs.contact_shooting_offset, visual=False)
                key_state['contact'] = contact_data
                print('contact_objs =', contact_objs)
                assert len(contact_objs) == 1

                #* visual contact_points, points with contact is red, else is blue
                colors = np.zeros((points.shape[0], 3), dtype=np.float32)
                colors[:, 2] = 1
                colors[contact_data[tele_obj.name] > 0] = np.array([1, 0, 0])                
                vsl.draw_pcl(points, colors)

                # record spatial_relationship
                for contact_obj in contact_objs:
                    print('contact_obj =', contact_obj.name)
                    if contact_obj.name in objs_data:
                        objs_data[contact_obj.name]['relevance'] = True
                    # check Inside
                    inside_success = tele_obj.states[object_states.Inside].get_value(contact_obj)
                    # check inside which link
                    if inside_success:
                        links = contact_obj.links
                        for lname, link in links.items():
                            inside_succ = self.check_inside_link(tele_obj, link)
                            jnt = self.get_jointPrim_w_link(contact_obj, lname)
                            jpos = str(float(jnt.get_state(normalized=True)[0])) if jnt is not None else 'none'
                            if inside_succ and ('root' not in lname) and ('base' not in lname):
                                key_state['spatial_relationship'] = f"Inside, {contact_obj.name}, {lname}, {jpos}"
                                print('spatial_relationship =', f"Inside, {contact_obj.name}, {lname}, {jpos}")
                                break
                        if 'spatial_relationship' not in key_state:
                            key_state['spatial_relationship'] = f"Inside, {contact_obj.name}"
                            print('spatial_relationship =', f"Inside, {contact_obj.name}")
                        break
                    # check OnTop
                    ontop_success = tele_obj.states[object_states.OnTop].get_value(contact_obj)
                    if ontop_success:
                        key_state['spatial_relationship'] = f"OnTop, {contact_obj.name}"
                        print('spatial_relationship =', f"OnTop, {contact_obj.name}")
                        break
                if 'spatial_relationship' not in key_state:
                    key_state['spatial_relationship'] = None
                    cprint("[Warn]: spatial_relationship not set", color="yellow")

            else:
                # record links_pose
                links_pose = dict()
                for link_name in links_name:
                    lpos, lqua = tele_obj.links[link_name].get_position_orientation()
                    links_pose[link_name] = lpos.tolist() + lqua.tolist()
                key_state['links_pose'] = links_pose
                key_state['joint'] = float(tele_jnt.get_state(normalized=True)[0])
                key_state['spatial_relationship'] = None

            data_dict['key_state_{:03d}'.format(keystate_num)] = key_state
            keystate_num += 1
            self.iskeystate = False

        assert keystate_num >= 1
        data_dict['length'] = keystate_num
        data_dict['objs'] = objs_data
        data_dict['skill_name'] = self.correct_skill_name
        og.sim.stop()

        # save dataset
        save_path = os.path.dirname(self.scene_yaml_path) + '/dataset'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        with open(save_path+'/dataset.pkl', 'wb') as f:
            pickle.dump(data_dict, f)

        print(f"====== Saved dataset to {save_path} ======")


    def stop(self):
        og.clear()


    def augment_keystate(self, base_dataset_path:str, augment_number:int, save_dataset=True, save_path=None):
        """
        base_dataset_path: dataset.pkl
        """
        # load base dataset
        with open(base_dataset_path, 'rb') as f:
            base_data = pickle.load(f)

        obj_name = base_data['obj_name']
        obj = self.env.scene.object_registry("name", obj_name)
        obj.sleep_threshold = 10000.0
        link_name = self.tele_obj_cfg['operated_link_name']
        if link_name != 'none':
            joint = self.get_jointPrim_w_link(obj, link_name)
            link = obj.links[link_name]

        data_dict = dict()
        data_dict['obj_name'] = obj_name
        data_dict['length'] = base_data['length'] + 1
        objs_data = {n: dict(uid=v) for n, v in self.objects_uids_dict.items()}
        check_mask_names = []
        for n,v in objs_data.items():
            if 'relevance' in base_data['objs'][n] and base_data['objs'][n]['relevance']:
                objs_data[n]['relevance'] = True
                check_mask_names.append(n)
        
        #==================================
        if len(check_mask_names) == 1:
            obj_anc = self.env.scene.object_registry("name", obj_name)
        else:
            check_mask_names_ = check_mask_names.copy()
            check_mask_names_.remove(obj_name)
            anc_name = check_mask_names_[0]
            obj_anc = self.env.scene.object_registry("name", anc_name)
        #==================================
        
        # get camera intrinsic_matrix
        camera_intrinsic = dict()
        for cam_name, cam in self.cameras.items():
            camera_intrinsic[cam_name] = cam.intrinsic_matrix
        data_dict['camera_intrinsic'] = camera_intrinsic

        print('='*100)
        print('Task name: ', self.task_name)
        print('Skill name: ', self.correct_skill_name)
        print('='*100)

        if link_name != 'none':
            print('>>> object: {}, link: {}, joint: {}'.format(obj_name, link_name, joint.joint_name))
            if joint.control_type != ControlType.POSITION:
                joint.set_control_type(ControlType.POSITION, kp=cs.joint_position_kp)
            links_name = self.links_from_joint(obj, joint)
            data_dict['operated_link_name'] = link_name
            data_dict['links_name_maskid'] = dict()
        else:
            print('>>> Tele-object: {}'.format(obj_name))
            data_dict['operated_link_name'] = None
            data_dict['links_name_maskid'] = None

        # collect data
        for idx in range(data_dict['length']):
            print(f'>>> keystate: {idx}:')

            if idx > 0:
                base_keystate: dict = base_data['key_state_{:03d}'.format(idx-1)]
                base_obj_uid = base_data['objs'][base_data['obj_name']]['uid']

                if link_name != 'none':
                    # set joint
                    print(f"set joint: set {link_name} to {base_keystate['joint']}")
                    jnt_success = self.set_jnt_position(joint, base_keystate['joint'])
                    if not jnt_success:
                        return False

                else:
                    # set obj pose
                    print(f"set {obj_name} pose")
                    print('relationship:', base_keystate['spatial_relationship'])


                    base_contact: dict = base_keystate['contact']  # load contact
                    base_contact2_data = None
                    for n,v in base_contact.items():
                        if n != obj_name:
                            base_contact2_data = v
                            base_contact2_data['name'] = n
                    assert base_contact2_data is not None

                    if base_keystate['spatial_relationship'] is None:
                        if base_contact[obj_name].max() == 0:
                            cprint('Do not augment the state of the object hanging in the air!', 'yellow')
                            skill_path = os.path.dirname(os.path.dirname(base_dataset_path))
                            os.rename(skill_path, skill_path.replace('/(keep)', '/(noAugment)(keep)'))
                            return False
                        else:
                            obj_pcd = self.get_loaded_obj_pcd(obj_name) # obj frame
                            # get contact_indices
                            if obj_name in self.contact_indices_dict:
                                contact_indices = self.contact_indices_dict[obj_name]
                            else:
                                base_contact_indices = np.where(base_contact[obj_name] > 0)[0]
                                if self.objects_uids_dict[obj_name] != base_obj_uid:
                                    base_obj_pcd = self.get_obj_pcd(base_obj_uid)   # obj frame
                                    base_obj_pcd['points'] = vsl.class_registration(
                                        base_obj_pcd, obj_pcd)
                                    contact_indices = self.contact_transfer(
                                        base_obj_pcd, base_contact_indices, obj_pcd)
                                else:
                                    contact_indices = base_contact_indices
                                    if contact_indices.ndim == 0:
                                        contact_indices = np.expand_dims(contact_indices, axis=0)
                                self.contact_indices_dict[obj_name] = contact_indices

                            contact_center = np.mean(obj_pcd['points'][contact_indices], axis=0)    # obj frame

                            #* visual contact
                            visual = False
                            if visual:
                                plotter = pv.Plotter()
                                # contact center
                                plotter.add_points(contact_center[None, ...], scalars=np.array([[1., 0., 0.],]), rgb=True, point_size=30)
                                # objA
                                plotter.add_points(obj_pcd['points'], scalars=obj_pcd['colors'], rgb=True, point_size=10)
                                # contact
                                color_contact_pts = np.array([[1., 0., 0.],]).repeat(contact_indices.shape[0], axis=0)
                                plotter.add_points(obj_pcd['points'][contact_indices], scalars=color_contact_pts, rgb=True, point_size=15)
                                plotter.show()

                            obj2_pcd = self.get_loaded_obj_pcd(base_contact2_data['name']) # obj frame
                            # get contact2_indices
                            if base_contact2_data['name'] in self.contact_indices_dict:
                                contact2_indices = self.contact_indices_dict[base_contact2_data['name']]
                            else:
                                if self.objects_uids_dict[base_contact2_data['name']] != base_contact2_data['uid']:
                                    base_obj2_pcd = self.get_obj_pcd(base_contact2_data['uid'])   # obj frame
                                    base_obj2_pcd['points'] = vsl.class_registration(
                                        base_obj2_pcd, obj2_pcd)
                                    contact2_indices = self.contact_transfer(
                                        base_obj2_pcd, base_contact2_data['contact_idx'], obj2_pcd)
                                else:
                                    contact2_indices = base_contact2_data['contact_idx']
                                    if contact2_indices.ndim == 0:
                                        contact2_indices = np.expand_dims(contact2_indices, axis=0)
                                self.contact_indices_dict[base_contact2_data['name']] = contact2_indices

                            obj2 = self.env.scene.object_registry("name", base_contact2_data['name'])
                            T_w_o2 = T.pose2mat(obj2.get_position_orientation()).numpy()
                            contact2_pts = tf.transPts_T(obj2_pcd['points'][contact2_indices], T_w_o2)
                            contact2_direction = tf.transVecs_T(-base_contact2_data['contact_direction'], T_w_o2)
                            contact2_distances = np.tile(
                                np.array(base_contact2_data['contact_distances'])[:, None], 
                                (1, 3)
                            )
                            contact2_end = contact2_pts + contact2_direction * contact2_distances
                            contact2_center = np.mean(contact2_end, axis=0)

                            #* visual contact
                            visual = False
                            if visual:
                                plotter = pv.Plotter()
                                # contact line
                                lines_start = np.arange(contact2_direction.shape[0]).reshape((contact2_direction.shape[0], 1))  # (n,1)
                                lines_end = lines_start + contact2_direction.shape[0] # (n,1)
                                lines_end = lines_end.reshape((contact2_direction.shape[0], 1))
                                lines = np.concatenate((lines_start, lines_end), axis=1)    # (n, 2)
                                rays_points = np.concatenate((contact2_pts, contact2_end), axis=0)
                                lines_extra = np.ones((lines.shape[0], 1), dtype=np.uint8)*2
                                lines = np.concatenate((lines_extra, lines), axis=1).ravel()
                                poly = pv.PolyData()
                                poly.points = rays_points
                                poly.lines = lines
                                plotter.add_mesh(poly, color="red", line_width=5)
                                # contact center
                                plotter.add_points(contact2_center[None, ...], scalars=np.array([[1., 0., 0.],]), rgb=True, point_size=30)
                                # objB
                                # obj2_pcd['colors'][:] = np.array([0, 1, 0])
                                plotter.add_points(
                                    tf.transPts_T(obj2_pcd['points'], T_w_o2), 
                                    scalars=obj2_pcd['colors'], rgb=True, point_size=10)
                                plotter.show()
                            
                            obj2_rmat = T.quat2mat(obj2.get_position_orientation()[1]).numpy()
                            obj2_rmat_ = tf.quaternion_to_rotation_matrix(base_contact2_data['pose'][3:])
                            obj_rmat_ = tf.quaternion_to_rotation_matrix(base_keystate['obj_pose'][3:])
                            obj_rmat = (obj2_rmat @ obj2_rmat_.T) @ obj_rmat_

                            pos_o_w = contact_center - obj_rmat.T @ contact2_center
                            T_w_o = np.linalg.inv(tf.PosRmat_to_TransMat(pos_o_w, obj_rmat.T))
                            pos, qua = tf.TransMat_to_PosQua(T_w_o)

                            #* visual contact
                            visual = False
                            if visual:
                                plotter = pv.Plotter()
                                # contact line
                                # lines_start = np.arange(contact2_direction.shape[0]).reshape((contact2_direction.shape[0], 1))  # (n,1)
                                # lines_end = lines_start + contact2_direction.shape[0] # (n,1)
                                # lines_end = lines_end.reshape((contact2_direction.shape[0], 1))
                                # lines = np.concatenate((lines_start, lines_end), axis=1)    # (n, 2)
                                # rays_points = np.concatenate((contact2_pts, contact2_end), axis=0)
                                # lines_extra = np.ones((lines.shape[0], 1), dtype=np.uint8)*2
                                # lines = np.concatenate((lines_extra, lines), axis=1).ravel()
                                # poly = pv.PolyData()
                                # poly.points = rays_points
                                # poly.lines = lines
                                # plotter.add_mesh(poly, color="red", line_width=5)
                                # contact center
                                plotter.add_points(contact2_center[None, ...], scalars=np.array([[1., 0., 0.],]), rgb=True, point_size=30)
                                # objB
                                plotter.add_points(
                                    tf.transPts_T(obj2_pcd['points'], T_w_o2), 
                                    scalars=obj2_pcd['colors'], rgb=True, point_size=10)
                                
                                # contact center
                                plotter.add_points(
                                    tf.transPts_T(contact_center[None, ...], T_w_o), 
                                    scalars=np.array([[1., 0., 0.],]), rgb=True, point_size=30)
                                # objA
                                plotter.add_points(
                                    tf.transPts_T(obj_pcd['points'], T_w_o), 
                                    scalars=obj_pcd['colors'], 
                                    rgb=True, 
                                    point_size=10
                                )
                                # contact
                                # color_contact_pts = np.array([[1., 0., 0.],]).repeat(contact_indices.shape[0], axis=0)
                                # plotter.add_points(
                                #     obj_pcd['points'][contact_indices], 
                                #     scalars=color_contact_pts, rgb=True, point_size=15
                                # )

                                plotter.show()


                            obj.disable_gravity()
                            poss = vsl.generate_uniform_spherical_expansion(
                                origin=pos, num_layers=10, points_per_layer=100, max_radius=0.1)

                            success = False
                            for pos in poss:
                                if not obj.is_asleep:
                                    obj.sleep()
                                obj.set_position_orientation(position=pos, orientation=qua)
                                obj.keep_still()
                                for i in range(10):
                                    og.sim.step_physics()
                                pose = obj.get_position_orientation()
                                if max(pose[0] - pos) > 1e-4:
                                    continue
                                contact_list = obj.contact_list()
                                if len(contact_list) != 0:
                                    continue
                                success = True
                                break
                            if not success: return False

                            # for debugging
                            # points, contact_data, contact_objs = self.get_contact_state(
                            #     obj, shooting_offset=cs.contact_shooting_offset, visual=True)
                    
                    # ontop
                    elif base_keystate['spatial_relationship'].startswith('OnTop'):
                        obj_pcd = self.get_loaded_obj_pcd(obj_name) # obj frame
                        # get contact_indices
                        if obj_name in self.contact_indices_dict:
                            contact_indices = self.contact_indices_dict[obj_name]
                        else:
                            base_contact_indices = np.where(base_contact[obj_name] > 0)[0]
                            if self.objects_uids_dict[obj_name] != base_obj_uid:
                                base_obj_pcd = self.get_obj_pcd(base_obj_uid)   # obj frame
                                base_obj_pcd['points'] = vsl.class_registration(
                                    base_obj_pcd, obj_pcd)
                                contact_indices = self.contact_transfer(
                                    base_obj_pcd, base_contact_indices, obj_pcd)
                            else:
                                contact_indices = base_contact_indices
                                if contact_indices.ndim == 0:
                                    contact_indices = np.expand_dims(contact_indices, axis=0)
                            self.contact_indices_dict[obj_name] = contact_indices

                            #! visual
                            # obj_pcd['colors'][:] = np.array([1, 0, 0])
                            # obj_pcd['colors'][contact_indices] = np.array([0, 0, 1])
                            # vsl.draw_pcl(obj_pcd['points'], obj_pcd['colors'], coordinate_frame_size=0.02)
                            
                        contact_center = np.mean(obj_pcd['points'][contact_indices], axis=0)    # obj frame

                        obj2_pcd = self.get_loaded_obj_pcd(base_contact2_data['name']) # obj frame
                        # get contact2_indices
                        if base_contact2_data['name'] in self.contact_indices_dict:
                            contact2_indices = self.contact_indices_dict[base_contact2_data['name']]
                        else:
                            if self.objects_uids_dict[base_contact2_data['name']] != base_contact2_data['uid']:
                                base_obj2_pcd = self.get_obj_pcd(base_contact2_data['uid'])   # obj frame
                                base_obj2_pcd['points'] = vsl.class_registration(
                                    base_obj2_pcd, obj2_pcd)
                                contact2_indices = self.contact_transfer(
                                    base_obj2_pcd, base_contact2_data['contact_idx'], obj2_pcd)
                            else:
                                contact2_indices = base_contact2_data['contact_idx']
                                if contact2_indices.ndim == 0:
                                    contact2_indices = np.expand_dims(contact2_indices, axis=0)
                            self.contact_indices_dict[base_contact2_data['name']] = contact2_indices

                            #! visual
                            # obj2_pcd['colors'][:] = np.array([1, 0, 0])
                            # obj2_pcd['colors'][contact2_indices] = np.array([0, 0, 1])
                            # vsl.draw_pcl(obj2_pcd['points'], obj2_pcd['colors'], coordinate_frame_size=0.02)
                            
                        obj2 = self.env.scene.object_registry("name", base_contact2_data['name'])
                        T_w_o2 = T.pose2mat(obj2.get_position_orientation()).numpy()
                        contact2_pts = tf.transPts_T(obj2_pcd['points'][contact2_indices], T_w_o2)
                        contact2_direction = tf.transVecs_T(-base_contact2_data['contact_direction'], T_w_o2)
                        contact2_distances = np.tile(
                            np.array(base_contact2_data['contact_distances'])[:, None], 
                            (1, 3)
                        )
                        contact2_center = np.mean(
                            contact2_pts + contact2_direction * contact2_distances, 
                            axis=0
                        )
                        
                        obj_rmat = T.quat2mat(obj.get_position_orientation()[1]).numpy()
                        pos_o_w = contact_center - obj_rmat.T @ contact2_center
                        T_w_o = np.linalg.inv(tf.PosRmat_to_TransMat(pos_o_w, obj_rmat.T))
                        pos, qua = tf.TransMat_to_PosQua(T_w_o)
                        obj.wake()
                        poss = vsl.generate_2d_spiral_with_constraints(
                            origin=pos,
                            point_spacing=0.05,
                            max_radius=0.5,
                            max_points=500
                        )
                        success = False
                        for pos in poss:
                            pos[2] += 0.02
                            obj.set_position_orientation(position=pos, orientation=qua)

                            obj.keep_still()
                            # Step until either (a) max steps is reached (total of 0.5 second in sim time) or (b) contact is made, then
                            # step until (a) max steps is reached (restarted from 0) or (b) velocity is below some threshold
                            n_steps_max = int(0.5 / og.sim.get_physics_dt())
                            i = 0
                            while len(obj.states[object_states.ContactBodies].get_value()) == 0 and i < n_steps_max:
                                og.sim.step_physics()
                                i += 1
                            obj.keep_still()
                            # Step a few times so velocity can become non-zero if the objects are moving
                            for i in range(5):
                                og.sim.step_physics()
                            i = 0
                            while th.norm(obj.get_linear_velocity()) > 1e-3 and i < n_steps_max:
                                og.sim.step_physics()
                                i += 1
                            # Render at the end
                            og.sim.render()

                            # og.sim.step_physics()
                            # contact_list = obj.contact_list()
                            # if len(contact_list) != 0:
                            #     continue
                            # for _ in range(100):
                            #     og.sim.step_physics()
                            # self.env.step([])
                            # obj.keep_still()

                            cur_q = obj.get_position_orientation()[1]
                            if tf.qua_diff(cur_q, qua) > 30./180.*np.pi:
                                continue

                            # check ontop
                            success = obj.states[object_states.OnTop].get_value(obj2)
                            if success: break                        
                        if not success: 
                            print('ontop failed')
                            return False

                    # inside
                    elif base_keystate['spatial_relationship'].startswith('Inside'):
                        relationship = base_keystate['spatial_relationship'].split(',')
                        rela_type = relationship[0].strip()
                        obj2_name = relationship[1].strip().replace(' ', '_').replace('-', '_')
                        obj2 = self.env.scene.object_registry("name", obj2_name)
                        ori_obj2_link_name = relationship[2].strip() if len(relationship) == 4 else None
                        ori_obj2_jnt_pos = relationship[3].strip() if len(relationship) == 4 else None
                        ori_obj2_jnt_pos = float(ori_obj2_jnt_pos) if ori_obj2_jnt_pos != None and ori_obj2_jnt_pos != 'none' else None
                        obj2_link_name = ori_obj2_link_name
                        if ori_obj2_link_name is not None:
                            ori_obj2_link_semantic = None
                            ori_obj2_link_semantics = get_object_link_semantics(base_data['objs'][obj2_name]['uid'])
                            if ori_obj2_link_name in ori_obj2_link_semantics:
                                ori_obj2_link_semantic = ori_obj2_link_semantics[ori_obj2_link_name]

                            if ori_obj2_link_semantic is not None:
                                obj2_links_semantic = get_object_link_semantics(self.objects_uids_dict[obj2_name])
                                for ln, sem in obj2_links_semantic.items():
                                    if sem == ori_obj2_link_semantic:
                                        jnt = self.get_jointPrim_w_link(
                                            self.env.scene.object_registry("name", obj2_name), ln)
                                        if ori_obj2_jnt_pos is not None and \
                                            abs(float(jnt.get_state(normalized=True)[0])-ori_obj2_jnt_pos) < 0.3:
                                            obj2_link_name = ln
                                            break
                                        elif ori_obj2_jnt_pos is None:
                                            obj2_link_name = ln
                                            break

                        success = self.place_object(
                            obj_name, 
                            rela_type, 
                            obj2_name, 
                            obj2_link_name, 
                            rescale_num=0,
                            orientation=obj.get_position_orientation()[1]
                        )
                        if not success: 
                            print('inside failed')
                            return False

                    else:
                        raise NotImplementedError("NotImplemented:", base_keystate['spatial_relationship'])

                    self.env.step([])
                    self.env.step([])
                    self.env.step([])

            if link_name != 'none':
                shooting_pos = link.get_position_orientation()[0].cpu().numpy()
            else:
                shooting_pos_act = obj.get_position_orientation()[0].cpu().numpy()
                shooting_pos_anc = obj_anc.get_position_orientation()[0].cpu().numpy()
                shooting_pos = (shooting_pos_act + shooting_pos_anc) / 2.0
                
            camera_distance = cs.camera_distance
            #===================================
            camerav_rgb = [] # rgb
            camerav_depth = [] # linear depth
            camerav_seg = [] # seg_instance
            camerav_seg_id = [] # seg_instance id
            camerav_pose = []
            #===================================
            
            while camera_distance >= cs.min_camera_distance:
                camera_poses = vsl.generate_camera_poses_hemispherical(
                    num_views=cs.camera_num_views, radius=camera_distance, pos=shooting_pos)
                max_size = 0
                for cpose in camera_poses:
                    self.viewer_camera.set_position_orientation(
                        position=cpose[0], orientation=cpose[1])
                    for _ in range(3):
                        self.env.step([])
                    obs, info = self.viewer_camera.get_obs()
                    if link_name != 'none':   # move joint
                        seg_instance_id = obs['seg_instance_id'].cpu().numpy()
                        for i,n in info['seg_instance_id'].items():
                            if 'scene_0' in n:
                                if n.split('/')[3] == obj_name and n.split('/')[4] == link_name:
                                    vis_size = vsl.get_pixels_num_with_value(seg_instance_id, i)
                                    if vis_size > max_size:
                                        max_size = vis_size
                                        selected_pose = cpose
                        cam_pose = tf.PosQua_to_TransMat(*self.viewer_camera.get_position_orientation())
                        diff = tf.PosEuler_to_TransMat((0, 0, 0), (np.pi, 0, 0))
                        cam_pose = cam_pose @ diff
                        camerav_rgb.append(obs['rgb'].cpu().numpy()[..., :3])
                        camerav_depth.append(obs['depth_linear'].cpu().numpy())
                        camerav_seg.append(obs['seg_instance'].cpu().numpy())
                        camerav_seg_id.append(obs['seg_instance_id'].cpu().numpy())
                        camerav_pose.append(cam_pose)

                    else:   # move obj
                        seg_instance = obs['seg_instance'].cpu().numpy()
                        for i,n in info['seg_instance'].items():
                            if n == obj_name:
                                vis_size = vsl.get_pixels_num_with_value(seg_instance, i)
                                if vis_size > max_size:
                                    max_size = vis_size
                                    selected_pose = cpose
                        
                        cam_pose = tf.PosQua_to_TransMat(*self.viewer_camera.get_position_orientation())
                        diff = tf.PosEuler_to_TransMat((0, 0, 0), (np.pi, 0, 0))
                        cam_pose = cam_pose @ diff
                        camerav_rgb.append(obs['rgb'].cpu().numpy()[..., :3])
                        camerav_depth.append(obs['depth_linear'].cpu().numpy())
                        camerav_seg.append(obs['seg_instance'].cpu().numpy())
                        camerav_seg_id.append(obs['seg_instance_id'].cpu().numpy())
                        camerav_pose.append(cam_pose)
                        
                if max_size < cs.object_min_size_camera:
                    camera_distance -= cs.camera_distance_interval
                    print('max_size:', max_size, ', set camera idstance to', camera_distance)
                else:
                    break
                
            
            if max_size <= 0:
                print('error: max_size == 0')
                return False
            

            # set camera to selected_pose
            self.viewer_camera.set_position_orientation(
                position=selected_pose[0], orientation=selected_pose[1])
            
            self.env.step([])
            self.env.step([])
            self.env.step([])
            
            # camera data
            camera_data = dict()
            check_mask_names_copy = copy.deepcopy(check_mask_names)
            cam_idx = 1
            for cam_name, cam in self.cameras.items():
                if cam_name == 'camerav':
                    continue
                new_name = f"camera{cam_idx:02d}"
                cam_pose = tf.PosQua_to_TransMat(*cam.get_position_orientation())
                diff = tf.PosEuler_to_TransMat((0, 0, 0), (np.pi, 0, 0))
                cam_pose = cam_pose @ diff
                obs, info = cam.get_obs()
                camera_data[new_name] = dict(
                    rgb=obs['rgb'].cpu().numpy()[..., :3],
                    depth=obs['depth_linear'].cpu().numpy(),
                    instance=obs['seg_instance'].cpu().numpy(),
                    instanceid=obs['seg_instance_id'].cpu().numpy(),
                    pose=cam_pose
                )
                cam_idx += 1
            
                # record maskid
                for i, n in info['seg_instance'].items():
                    if n in self.objects_uids_dict and 'maskid' not in objs_data[n]:
                        objs_data[n]['maskid'] = i
                    # check mask
                    if n in check_mask_names_copy:
                        check_mask_names_copy.remove(n)

                # update links_name_maskid
                if link_name != 'none':
                    allLinksname_id_dict = {}
                    for i,n in info['seg_instance_id'].items():
                        if 'scene_0' in n:
                            allLinksname_id_dict[n.split('/')[3] + '-' + n.split('/')[4]] = i
                    for link_name in links_name:
                        obj_link_name = obj_name + '-' + link_name
                        if link_name not in data_dict['links_name_maskid'] and obj_link_name in allLinksname_id_dict:
                            data_dict['links_name_maskid'][link_name] = allLinksname_id_dict[obj_link_name]
            
            #===============================
            for i in range(len(camerav_rgb)): 
                new_name = f"camera{cam_idx:02d}"
                camera_data[new_name] = dict(
                    rgb=camerav_rgb[i],
                    depth=camerav_depth[i],
                    instance=camerav_seg[i],
                    instanceid=camerav_seg_id[i],
                    pose=camerav_pose[i]
                )
                cam_idx += 1
            #===============================
            
            # check data
            if len(check_mask_names_copy) != 0:
                print('!! Error check_mask_names_copy =', check_mask_names_copy)
                return False
            if link_name != 'none' and 'links_name_maskid' not in data_dict:
                print('!! Error links_name_maskid')
                return False

            key_state = dict(camera_data=camera_data)

            # record obj/link pose
            if link_name == 'none':
                # record obj pose
                opos, oqua = obj.get_position_orientation()
                key_state['obj_pose'] = opos.tolist() + oqua.tolist()

                # record objs pose
                objs_pose = dict()
                for oname, odata in objs_data.items():
                    if 'relevance' in odata and odata['relevance'] is True:
                        opos, oqua = self.env.scene.object_registry("name", oname).get_position_orientation()
                        objs_pose[oname] = opos.tolist() + oqua.tolist()
                key_state['objs_pose'] = objs_pose
            else:
                # record links_pose
                links_pose = dict()
                for link_name in links_name:
                    lpos, lqua = obj.links[link_name].get_position_orientation()
                    links_pose[link_name] = lpos.tolist() + lqua.tolist()
                key_state['links_pose'] = links_pose
                key_state['joint'] = float(joint.get_state(normalized=True)[0])
            data_dict['key_state_{:03d}'.format(idx)] = key_state

        data_dict['objs'] = objs_data
        data_dict['skill_name'] = self.correct_skill_name
        
        # save dataset
        if save_dataset:
            scene_num = os.path.basename(self.scene_yaml_path).split('_')[0]
            augment_num = "{:04d}".format(augment_number)
            if save_path is None:
                save_path = os.path.dirname(base_dataset_path) + f'/{scene_num}-{augment_num}_dataset.pkl'
            # save yaml
            with open(save_path, 'wb') as f:
                pickle.dump(data_dict, f)

            print(f"====== Saved dataset to {save_path} ======")

        return True
    

    def get_state(self, base_dataset_path:str, cameras=['camera1', 'camera2', 'camerav']):
        """
        get state for network input
        """
        # load base dataset
        with open(base_dataset_path, 'rb') as f:
            base_data = pickle.load(f)

        obj_name = base_data['obj_name']
        obj = self.env.scene.object_registry("name", obj_name)
        obj.sleep_threshold = 10000.0
        link_name = self.tele_obj_cfg['operated_link_name']
        if link_name != 'none':
            joint = self.get_jointPrim_w_link(obj, link_name)
            link = obj.links[link_name]

        data_dict = dict()
        data_dict['obj_name'] = obj_name
        data_dict['length'] = base_data['length'] + 1
        objs_data = {n: dict(uid=v) for n, v in self.objects_uids_dict.items()}
        check_mask_names = []
        for n,v in objs_data.items():
            if 'relevance' in base_data['objs'][n] and base_data['objs'][n]['relevance']:
                objs_data[n]['relevance'] = True
                check_mask_names.append(n)

        # get camera intrinsic_matrix
        intrinsic_matrix = torch.tensor([
            [415.3663,   0.0000, 256.0000],
            [  0.0000, 415.3663, 256.0000],
            [  0.0000,   0.0000,   1.0000]
        ])
        camera_intrinsic = dict()
        for cam_name in cameras:
            # cam = self.cameras[cam_name]
            # camera_intrinsic[cam_name] = cam.intrinsic_matrix
            camera_intrinsic[cam_name] = intrinsic_matrix
        data_dict['camera_intrinsic'] = camera_intrinsic

        print('='*100)
        print('Task name: ', self.task_name)
        print('Skill name: ', self.correct_skill_name)
        print('='*100)

        if link_name != 'none':
            print('>>> object: {}, link: {}, joint: {}'.format(obj_name, link_name, joint.joint_name))
            if joint.control_type != ControlType.POSITION:
                joint.set_control_type(ControlType.POSITION, kp=cs.joint_position_kp)
            links_name = self.links_from_joint(obj, joint)
            data_dict['operated_link_name'] = link_name
            data_dict['links_name_maskid'] = dict()
        else:
            print('>>> Tele-object: {}'.format(obj_name))
            data_dict['operated_link_name'] = None
            data_dict['links_name_maskid'] = None

        idx = 0

        """
        if 'camerav' in cameras:
            if link_name != 'none':
                shooting_pos = link.get_position_orientation()[0].cpu().numpy()
            else:
                shooting_pos = obj.get_position_orientation()[0].cpu().numpy()

            camera_distance = cs.camera_distance
            while camera_distance >= cs.min_camera_distance:
                camera_poses = vsl.generate_camera_poses_hemispherical(
                    num_views=cs.camera_num_views, radius=camera_distance, pos=shooting_pos)
                max_size = 0
                selected_pose = None
                for cpose in camera_poses:
                    self.viewer_camera.set_position_orientation(
                        position=cpose[0], orientation=cpose[1])
                    for _ in range(3):
                        self.env.step([])
                    obs, info = self.viewer_camera.get_obs()
                    if link_name != 'none':   # move joint
                        seg_instance_id = obs['seg_instance_id'].cpu().numpy()
                        for i,n in info['seg_instance_id'].items():
                            if 'scene_0' in n:
                                if n.split('/')[3] == obj_name and n.split('/')[4] == link_name:
                                    vis_size = vsl.get_pixels_num_with_value(seg_instance_id, i)
                                    if vis_size > max_size:
                                        max_size = vis_size
                                        selected_pose = cpose

                    else:   # move obj
                        seg_instance = obs['seg_instance'].cpu().numpy()
                        for i,n in info['seg_instance'].items():
                            if n == obj_name:
                                vis_size = vsl.get_pixels_num_with_value(seg_instance, i)
                                if vis_size > max_size:
                                    max_size = vis_size
                                    selected_pose = cpose
                    
                if max_size < cs.object_min_size_camera:
                    camera_distance -= cs.camera_distance_interval
                    print('max_size:', max_size, ', set camera idstance to', camera_distance)
                else:
                    break
            
            if max_size <= 0:
                print('error: max_size == 0')
                return False

            # set camera to selected_pose
            self.viewer_camera.set_position_orientation(
                position=selected_pose[0], orientation=selected_pose[1])
            
            self.env.step([])
            self.env.step([])
            self.env.step([])
        """

        # camera data
        camera_data = dict()
        check_mask_names_copy = copy.deepcopy(check_mask_names)
        for cam_name in cameras:
            cam = self.cameras[cam_name]
            cam_pose = tf.PosQua_to_TransMat(*cam.get_position_orientation())
            diff = tf.PosEuler_to_TransMat((0, 0, 0), (np.pi, 0, 0))
            cam_pose = cam_pose @ diff
            obs, info = cam.get_obs()
            camera_data[cam_name] = dict(
                rgb=obs['rgb'].cpu().numpy()[..., :3],
                depth=obs['depth_linear'].cpu().numpy(),
                instance=obs['seg_instance'].cpu().numpy(),
                instanceid=obs['seg_instance_id'].cpu().numpy(),
                pose=cam_pose
            )
            
            # record maskid
            for i, n in info['seg_instance'].items():
                if n in self.objects_uids_dict and 'maskid' not in objs_data[n]:
                    objs_data[n]['maskid'] = i
                # check mask
                if n in check_mask_names_copy:
                    check_mask_names_copy.remove(n)

            # update links_name_maskid
            if link_name != 'none':
                allLinksname_id_dict = {}
                for i,n in info['seg_instance_id'].items():
                    if 'scene_0' in n:
                        allLinksname_id_dict[n.split('/')[3] + '-' + n.split('/')[4]] = i
                for link_name in links_name:
                    obj_link_name = obj_name + '-' + link_name
                    if link_name not in data_dict['links_name_maskid'] and obj_link_name in allLinksname_id_dict:
                        data_dict['links_name_maskid'][link_name] = allLinksname_id_dict[obj_link_name]
        
        # check data
        if len(check_mask_names_copy) != 0:
            print('!! Error check_mask_names_copy =', check_mask_names_copy)
            return False
        if link_name != 'none' and 'links_name_maskid' not in data_dict:
            print('!! Error links_name_maskid')
            return False

        key_state = dict(camera_data=camera_data)

        # record obj/link pose
        if link_name == 'none':
            # record obj pose
            opos, oqua = obj.get_position_orientation()
            key_state['obj_pose'] = opos.tolist() + oqua.tolist()

            # record objs pose
            objs_pose = dict()
            for oname, odata in objs_data.items():
                if 'relevance' in odata and odata['relevance'] is True:
                    opos, oqua = self.env.scene.object_registry("name", oname).get_position_orientation()
                    objs_pose[oname] = opos.tolist() + oqua.tolist()
            key_state['objs_pose'] = objs_pose
        else:
            # record links_pose
            links_pose = dict()
            for link_name in links_name:
                lpos, lqua = obj.links[link_name].get_position_orientation()
                links_pose[link_name] = lpos.tolist() + lqua.tolist()
            key_state['links_pose'] = links_pose
            key_state['joint'] = float(joint.get_state(normalized=True)[0])
        data_dict['key_state_{:03d}'.format(idx)] = key_state

        data_dict['objs'] = objs_data
        data_dict['skill_name'] = self.correct_skill_name

        return data_dict


    def set_and_check(
            self, 
            base_dataset_path:str, 
            transformation: np.ndarray, 
            cameras=['camera1', 'camera2', 'camerav'],
            return_imgs=False,
            show=False):
        """
        set object to predicted location and check success.
        """
        # load base dataset
        with open(base_dataset_path, 'rb') as f:
            base_data = pickle.load(f)

        obj_name = base_data['obj_name']
        obj = self.env.scene.object_registry("name", obj_name)
        obj_pose = tf.PosQua_to_TransMat(*obj.get_position_orientation())
        obj_pose = transformation @ obj_pose
        pos_ori, qua_ori = tf.TransMat_to_PosQua(obj_pose)

        collision = True
        ps = vsl.sample_points_in_sphere(pos_ori, radius=0.03, num_points=3000)
        qs = vsl.sample_quaternions_around(qua_ori, max_angle_deg=30, num_samples=3000)
        for p,q in zip(ps, qs):
            obj.set_position_orientation(position=p, orientation=q)
            obj.keep_still()
            if show:
                self.env.step([])
            else:
                og.sim.step_physics()
            colli = check_collision(obj)
            if not colli:
                collision = False
                break
        
        imgs = None
        if collision:
            print('collision!!')
            return False, imgs

        obj.set_position_orientation(position=pos_ori, orientation=qua_ori)
        obj.keep_still()
        if not show:
            for _ in range(50):
                og.sim.step_physics()
        else:
            for _ in range(50):
                self.env.step([])

        if return_imgs:
            imgs = []
            for cam_name in cameras:
                cam = self.cameras[cam_name]
                obs, info = cam.get_obs()
                imgs.append(obs['rgb'].cpu().numpy()[..., :3])

        idx = 0
        base_keystate: dict = base_data['key_state_{:03d}'.format(idx)]

        # inside
        if base_keystate['spatial_relationship'].startswith('Inside'):
            relationship = base_keystate['spatial_relationship'].split(',')
            rela_type = relationship[0].strip()
            obj2_name = relationship[1].strip().replace(' ', '_').replace('-', '_')
            obj2 = self.env.scene.object_registry("name", obj2_name)
            ori_obj2_link_name = relationship[2].strip() if len(relationship) == 4 else None
            ori_obj2_jnt_pos = relationship[3].strip() if len(relationship) == 4 else None
            ori_obj2_jnt_pos = float(ori_obj2_jnt_pos) if ori_obj2_jnt_pos != None and ori_obj2_jnt_pos != 'none' else None
            obj2_link_name = ori_obj2_link_name
            if ori_obj2_link_name is not None:
                ori_obj2_link_semantic = None
                ori_obj2_link_semantics = get_object_link_semantics(base_data['objs'][obj2_name]['uid'])
                if ori_obj2_link_name in ori_obj2_link_semantics:
                    ori_obj2_link_semantic = ori_obj2_link_semantics[ori_obj2_link_name]

                if ori_obj2_link_semantic is not None:
                    obj2_links_semantic = get_object_link_semantics(self.objects_uids_dict[obj2_name])
                    for ln, sem in obj2_links_semantic.items():
                        if sem == ori_obj2_link_semantic:
                            jnt = self.get_jointPrim_w_link(
                                self.env.scene.object_registry("name", obj2_name), ln)
                            if ori_obj2_jnt_pos is not None and \
                                abs(float(jnt.get_state(normalized=True)[0])-ori_obj2_jnt_pos) < 0.3:
                                obj2_link_name = ln
                                break
                            elif ori_obj2_jnt_pos is None:
                                obj2_link_name = ln
                                break
            
            success = self.check_object_relation(obj_name, rela_type, obj2_name, obj2_link_name)

        else:
            raise NotImplementedError("NotImplemented:", base_keystate['spatial_relationship'])
        
        return success, imgs


    @staticmethod
    def contact_transfer(
        pcd1:np.ndarray, 
        contact_indices:np.ndarray, 
        pcd2:np.ndarray, 
        w_xyz=1., 
        w_feat=0.01
    ):
        """
        args:
            - pcd1: {points, features}
            - pcd2: {points, features}

        """
        if contact_indices.ndim == 0:
            contact_indices = np.expand_dims(contact_indices, axis=0)
        # source = np.concatenate([pcd1['points']*w_xyz, pcd1['features']*w_feat], axis=1)[contact_indices]
        # target = np.concatenate([pcd2['points']*w_xyz, pcd2['features']*w_feat], axis=1)
        source = pcd1['points'][contact_indices]
        target = pcd2['points']
        nn = NearestNeighbors(n_neighbors=1).fit(target)
        dists, indices = nn.kneighbors(source)

        indices = indices.squeeze()
        if indices.ndim == 0:
            indices = np.expand_dims(indices, axis=0)
        return indices


    def get_loaded_obj_pcd(self, obj_name):
        """
        {points, normals, colors, features}
        """
        cfg = self.get_object_cfg(obj_name)
        # load pointcloud
        uid = self.objects_uids_dict[obj_name]
        usd_path, _ = get_model_path(uid)
        pcd_path = usd_path.replace('.usd', '_pcd.pkl')
        with open(pcd_path, 'rb') as f:
            pcd = pickle.load(f)
        pcd['points'] *= cfg['scale'][0]
        return pcd
    

    @staticmethod
    def get_obj_pcd(uid):
        usd_path, _ = get_model_path(uid)
        pcd_path = usd_path.replace('.usd', '_pcd.pkl')
        with open(pcd_path, 'rb') as f:
            pcd = pickle.load(f)
        return pcd


    @staticmethod
    def get_obj_points(obj, visual=False)->np.ndarray:
        points_in_world = []

        for link_name, link in obj._links.items():
            if visual:
                hull_points = link.visual_boundary_points_world
            else:
                hull_points = link.collision_boundary_points_world

            if hull_points is not None:
                points_in_world.extend(hull_points.tolist())
        
        return np.array(points_in_world, dtype=np.float32)


    @staticmethod
    def get_jointPrim_w_link(obj, link_name):
        for joint in obj.joints.values():
            if joint.body1.split('/')[-1] == link_name:
                return joint
        return None


    @staticmethod
    def check_inside_link(obj, other, iou_thresh=0.75):
        # First check that the inner object's position is inside the outer's AABB.
        # Since we usually check for a small set of outer objects, this is cheap
        aabb_extent = obj.aabb_extent
        aabb_center = obj.aabb_center
        aabb_lower = aabb_center - aabb_extent/2.
        aabb_upper = aabb_center + aabb_extent/2.
        inner_object_pos = (aabb_lower + aabb_upper) / 2.0

        other_aabb_extent = other.aabb_extent
        other_aabb_center = other.aabb_center
        other_aabb_lo = other_aabb_center - other_aabb_extent/2.
        other_aabb_hi = other_aabb_center + other_aabb_extent/2.

        if not (
            torch.le(other_aabb_lo, inner_object_pos).all() and torch.le(inner_object_pos, other_aabb_hi).all()
        ):
            return False

        intersect_aabb_upper = []
        intersect_aabb_lower = []
        for i in range(3):
            u = min(aabb_upper[i], other_aabb_hi[i])
            l = max(aabb_lower[i], other_aabb_lo[i])
            intersect_aabb_upper.append(u)
            intersect_aabb_lower.append(l)
        
        intersect_aabb_upper = np.array(intersect_aabb_upper)
        intersect_aabb_lower = np.array(intersect_aabb_lower)

        if np.max(intersect_aabb_lower-intersect_aabb_upper) >= 0:
            return False
        
        intersect_size = np.prod(intersect_aabb_upper-intersect_aabb_lower)
        assert intersect_size > 0

        obj_size = np.prod(aabb_extent.cpu().numpy())
        iou = intersect_size/obj_size

        return iou > iou_thresh


    def set_objects_contact_id(self):
        obj_names = [n for n,_ in self.objects_uids_dict.items()]
        obj_names.append('ground')
        obj_names.sort()
        objs_contact_id = {}
        for i, n in enumerate(obj_names):
            objs_contact_id[n] = i+1
        return objs_contact_id
    

    def get_contact_state(self, obj, shooting_offset=0.02, visual=False):
        """
        
        return: 
            - contact_data (dict): 
                    contact_id: int
                    uid:
                    pose: (7,) xyz/xyzw
                    contact_points: (n, 3)
                    contact_distances: (n,)

        """
        # load pointcloud
        obj_pcd = self.get_loaded_obj_pcd(obj.name)
        points = obj_pcd['points']
        normals = obj_pcd['normals']
        
        # ray detect
        T_w_o = T.pose2mat(obj.get_position_orientation()).numpy()
        rays_start = tf.transPts_T(points, T_w_o)
        rays_direction = tf.transVecs_T(normals, T_w_o)
        rays_end = rays_start + rays_direction * shooting_offset
        ray_results = raytest_batch(rays_start, rays_end, 
                                    only_closest=True,
                                    ignore_bodies=obj.link_prim_paths, 
                                    ignore_collisions=obj.link_prim_paths)

        contact_data = dict()
        contact_ids = np.zeros((rays_start.shape[0],), dtype=np.int8)
        contact_objs = []
        contact_points_idx = []
        for i, result in enumerate(ray_results):
            if result['hit']:
                hit_obj_name = result["rigidBody"].split("/")[-2]
                if hit_obj_name not in self.objects_contact_id_dict:
                    continue
                contact_id = self.objects_contact_id_dict[hit_obj_name]
                contact_ids[i] = contact_id
                contact_points_idx.append(i)
                contact_obj = self.env.scene.object_registry("name", hit_obj_name)
                if contact_obj not in contact_objs:
                    contact_objs.append(contact_obj)

                if hit_obj_name in contact_data:
                    contact_data[hit_obj_name]['contact_points'].append(result['position'])
                    contact_data[hit_obj_name]['contact_distances'].append(result['distance'])
                    contact_data[hit_obj_name]['contact_direction'].append(rays_direction[i])
                else:
                    pos, qua = contact_obj.get_position_orientation()
                    pose = pos.tolist() + qua.tolist()
                    contact_data[hit_obj_name] = dict(
                        contact_id=contact_id,
                        uid=self.objects_uids_dict[hit_obj_name],
                        pose=pose,
                        contact_points=[result['position'], ],
                        contact_distances=[result['distance'], ],
                        contact_direction=[rays_direction[i], ]
                    )
        contact_points_idx = np.array(contact_points_idx)

        for hit_obj_name, data in contact_data.items():
            hit_obj_pcd = self.get_loaded_obj_pcd(hit_obj_name)
            contact_obj = self.env.scene.object_registry("name", hit_obj_name)
            T_w_ho = T.pose2mat(contact_obj.get_position_orientation()).numpy()
            hit_obj_pts = tf.transPts_T(hit_obj_pcd['points'], T_w_ho)
            nn = NearestNeighbors(n_neighbors=1).fit(hit_obj_pts)
            dists, indices = nn.kneighbors(np.array(data['contact_points']))
            data['contact_idx'] = indices.squeeze()
            data['contact_direction'] = tf.transVecs_T(
                np.array(data['contact_direction']), 
                np.linalg.inv(T_w_ho)
            )
            data['contact_distances'] = np.array(data['contact_distances'])

            if visual:
                # objA
                colorsA = obj_pcd['colors']
                colorsA[contact_points_idx] = np.array([1, 0, 0])   # set contact point to red
                # objB
                contact_points = np.array(data['contact_points'])   # (n,3)
                pointsB = np.concatenate((hit_obj_pts, contact_points), axis=0)
                colorsB = np.expand_dims(np.array([1, 0, 0]), axis=0).repeat(contact_points.shape[0], axis=0)
                colorsB = np.concatenate((hit_obj_pcd['colors'], colorsB), axis=0)

                plotter = pv.Plotter()
                # all line
                lines_start = np.arange(rays_start.shape[0]).reshape((rays_start.shape[0], 1))  # (n,1)
                lines_end = lines_start+rays_start.shape[0] # (n,1)
                lines_end = lines_end.reshape((rays_start.shape[0], 1))
                lines = np.concatenate((lines_start, lines_end), axis=1)    # (n, 2)
                rays_points = np.concatenate((rays_start, rays_end), axis=0)
                lines_extra = np.ones((lines.shape[0], 1), dtype=np.uint8)*2
                lines = np.concatenate((lines_extra, lines), axis=1).ravel()
                poly = pv.PolyData()
                poly.points = rays_points
                poly.lines = lines
                plotter.add_mesh(poly, color=[0.5, 0.5, 0.5], line_width=1)
                # contact line
                c_lines_start = np.arange(contact_points_idx.shape[0]).reshape((contact_points_idx.shape[0], 1))  # (n,1)
                c_lines_end = c_lines_start + contact_points_idx.shape[0] # (n,1)
                c_lines_end = c_lines_end.reshape((contact_points_idx.shape[0], 1))
                c_lines = np.concatenate((c_lines_start, c_lines_end), axis=1)    # (n, 2)
                c_rays_points = np.concatenate((rays_start[contact_points_idx], contact_points), axis=0)
                c_lines_extra = np.ones((c_lines.shape[0], 1), dtype=np.uint8)*2
                c_lines = np.concatenate((c_lines_extra, c_lines), axis=1).ravel()
                c_poly = pv.PolyData()
                c_poly.points = c_rays_points
                c_poly.lines = c_lines
                plotter.add_mesh(c_poly, color="red", line_width=5)
                # objA/B pcd
                plotter.add_points(rays_start, scalars=colorsA, rgb=True, point_size=10)
                plotter.add_points(pointsB, scalars=colorsB, rgb=True, point_size=10)
                plotter.show()

            data.pop('contact_points', None)


        contact_data[obj.name] = contact_ids


        return points, contact_data, contact_objs


    def get_native_bbox(self)->dict:
        """
        return: 
        """
        nativeBbox_dict = {}
        nativeBbox_offset_dict = {}
        for obj_name, _ in self.objects_uids_dict.items():
            obj = self.env.scene.object_registry("name", obj_name)
            pos, _ = obj.get_position_orientation()
            aabb_extent = obj.aabb_extent.cpu().numpy() # (3,)
            aabb_center =  obj.aabb_center.cpu().numpy() - pos.cpu().numpy()    # (3,)
            p_min = aabb_center - aabb_extent/2
            p_max = aabb_center + aabb_extent/2
            nativeBbox_dict[obj_name] = np.r_[p_min, p_max]
            nativeBbox_offset_dict[obj_name] = aabb_center
        return nativeBbox_dict, nativeBbox_offset_dict


    def links_from_joint(self, obj, joint):
        """
        return: links_name, list[link_name]
        """
        lname = joint.body1.split('/')[-1]
        links_name = [lname,]
        link = obj.links[lname]
        children = list(link.prim.GetChildren())

        while children:
            prim = children.pop()
            prim_path = str(prim.GetPath())
            if prim_path.split('/')[-1].startswith('j_'):
                lname = prim_path.split('/')[-1].replace('j_','')
                links_name.append(lname)
                link = obj.links[lname]
                children.extend(list(link.prim.GetChildren()))

        return links_name
    

    def test(self):
        # tele_obj, tele_jnt = self.choose_tele_obj_joint()

        # print('tele_obj.name =', tele_obj.name)
        
        # self.env.step([])
        # self.visual_camera_obs()
        
        # run
        self.esc = False
        while not self.esc:
            #* ================= read action =================
            # if torch.max(torch.abs(self.obj_actions)) > 0:
            #     clear_debug_drawing()
                # reset obj pose
                # pos, qua = self.step_obj(tele_obj, self.obj_actions)
                # tele_obj.set_position_orientation(position=pos, orientation=qua)
                # tele_obj.keep_still()
                # self.obj_actions *= 0

            #* ================= step =================
            self.env.step([])

            self.visual_camera_obs()


    def visual_camera_obs(self):
        """
        ["rgb", "depth_linear", "seg_instance", "seg_instance_id"]
        """

        """
        info = {
            1: '/World/ground_plane/geom', 
            2: '/World/scene_0/ground/base_link/visuals', 
            3: '/World/scene_0/cutting_board/base_link/visuals', 
            4: '/World/scene_0/fridge/link_0/visuals', 
            5: '/World/scene_0/fruit_bowl/base_link/visuals', 
            6: '/World/scene_0/oven_mitt/baseLink/visuals', 
            7: '/World/scene_0/fridge/base_link/visuals', 
            8: '/World/scene_0/table/base_link/visuals'}

        info = {
            2: 'groundPlane', 
            3: 'ground', 
            4: 'fruit_bowl', 
            5: 'cutting_board', 
            6: 'fridge', 
            7: 'oven_mitt', 
            8: 'table'
        }
        """

    
        # viewer camera
        obs, _ = self.viewer_camera.get_obs()
        viewer_rgb = obs['rgb'].cpu().numpy()[..., (2,1,0)]
        viewer_dep = obs['depth_linear'].cpu().numpy()
        viewer_dep = vsl.depth2RGB(viewer_dep)
        viewer_seg_instance = obs['seg_instance'].cpu().numpy()
        viewer_seg_instance_id = obs['seg_instance_id'].cpu().numpy()
        viewer_seg_instance = vsl.segmentation_to_rgb(viewer_seg_instance, random_colors=True)
        viewer_seg_instance_id = vsl.segmentation_to_rgb(viewer_seg_instance_id, random_colors=True)

        cv2.imshow('viewer_rgb', viewer_rgb)
        cv2.imshow('viewer_dep', viewer_dep)
        cv2.imshow('viewer_seg_instance', viewer_seg_instance)
        cv2.imshow('viewer_seg_instance_id', viewer_seg_instance_id)

        cv2.waitKey(10)


    def draw_object_axes(self, obj, length=0.3, size=3, reference_frame='world'):
        """

        args:
        - obj: prim
        - reference_frame: ['world', 'object']
        """
        pos, qua = obj.get_position_orientation()

        if reference_frame == 'object':
            T_matrix = torch.zeros((4,4))
            T_matrix[3,3] = 0
            T_matrix[:3,:3] = T.quat2mat(qua)
            T_matrix[:3, 3] = pos

            pts = torch.tensor([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ], dtype=torch.float32) * length

            end_pts = T.transform_points(pts, T_matrix)
        
        elif reference_frame == 'world':
            end_pts = torch.tensor([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ], dtype=torch.float32) + pos

        draw_line(pos.tolist(), end_pts[0].tolist(), color=(1.0, 0.0, 0.0, 1.0), size=size)
        draw_line(pos.tolist(), end_pts[1].tolist(), color=(0.0, 1.0, 0.0, 1.0), size=size)
        draw_line(pos.tolist(), end_pts[2].tolist(), color=(0.0, 0.0, 1.0, 1.0), size=size)
        down_pt = torch.tensor([0, 0, -1], dtype=torch.float32) + pos
        draw_line(pos.tolist(), down_pt.tolist(), color=(1.0, 0.7, 0.3, 1.0), size=size)


    def step_obj(self, tele_obj, action: torch.Tensor, reference_frame:str="world"):
        """

        args:

        return: 
            - pos, qua
        """
        t_w_o, q_w_o = tele_obj.get_position_orientation()
        if reference_frame == 'object':
            pos, qua = T.pose_transform(t_w_o, q_w_o, action[:3], T.euler2quat(action[3:]))
        elif reference_frame == 'world':
            pos = t_w_o + action[:3]
            qua = T.quat_multiply(T.euler2quat(action[3:]), q_w_o)
        else:
            raise KeyError("reference_frame must be one of [world, object]")

        return pos, qua


    def step_jnt(self, joint, delta):
        pos = joint.get_state(normalized=True)[0]
        self.set_jnt_position(joint, pos+delta)


    def set_jnt_position(self, joint, target_position, abs_delta=0.1, tolerance=0.01):
        """
        set joint' position to normalized target_position
        """
        assert joint.control_type == ControlType.POSITION
        for _ in range(100):
            pos = joint.get_state(normalized=True)[0]
            error = target_position-pos
            if abs(error) < tolerance:
                break
            if abs(error) < abs_delta:
                delta = error
            elif error > 0:
                delta = abs_delta
            elif error < 0:
                delta = -abs_delta
            else:
                print('error =', error)
            # joint.set_pos(pos=pos+delta, normalized=True, drive=True)
            joint.set_pos(pos=pos+delta, normalized=True, drive=True)
            og.sim.step_physics()
            og.sim.step_physics()
            og.sim.step_physics()
            self.env.step([])
        joint.keep_still()

        pos = joint.get_state(normalized=True)[0]
        error = target_position-pos
        return abs(error) < tolerance



    def choose_tele_obj_joint(self):
        """
        return: 
            - tele_obj: prim
            - tele_joint: prim
        """
        
        tele_obj_list = [cfg['name'] for cfg in self.objects_cfg]
        tele_obj_name = choose_from_options(options=tele_obj_list, name="teleop object")
        tele_obj = self.env.scene.object_registry("name", tele_obj_name)

        if tele_obj.n_dof == 0:
            return tele_obj, None

        joint_semantics = get_object_joint_semantics(self.objects_uids_dict[tele_obj_name])
        tele_joint_name = choose_from_options(options=joint_semantics, name="teleop object")
        tele_joint = tele_obj.joints[tele_joint_name]
        return tele_obj, tele_joint


    def init_teleop_spacemouse(self):
        self.spacemouse_thresh = {'pos': 0.6 * cs.obj_control_ratio, 'rot': 0.6 * cs.obj_control_ratio}
        self.spacemouse_thread = threading.Thread(target=self.read_spacemouse, daemon=True)

        KeyboardEventHandler.initialize()
        # key state
        KeyboardEventHandler.add_keyboard_callback(key=lazy.carb.input.KeyboardInput.KEY_1, callback_fn=self.key_state_cb)
        # collect
        # KeyboardEventHandler.add_keyboard_callback(key=lazy.carb.input.KeyboardInput.KEY_2, callback_fn=self.state_cb)
        # esc
        KeyboardEventHandler.add_keyboard_callback(key=lazy.carb.input.KeyboardInput.ESCAPE, callback_fn=self.escape)
        # wake/sleep
        KeyboardEventHandler.add_keyboard_callback(key=lazy.carb.input.KeyboardInput.C, callback_fn=self.switch_wake)
        KeyboardEventHandler.add_keyboard_callback(key=lazy.carb.input.KeyboardInput.V, callback_fn=self.switch_sleep)


    def read_spacemouse(self):
        """
        read spacemouse input, and set self.obj_actions/self.jnt_action

        
        self.obj_actions = torch.zeros((6,), dtype=torch.float32)
        self.jnt_action = 0.
        """
        success = pyspacemouse.open(DeviceNumber=0)
        assert success

        dof_keys = {
            'x': 'x',
            'y': 'y',
            'z': 'z',
            'r': '-pitch',
            'p': 'roll',
            'y': '-yaw'
        }
        while True:
            state = pyspacemouse.read() # t,x,y,z,pitch,yaw,roll,buttons
            # set obj pos delta
            delta_pos = torch.Tensor([state.x, state.y, state.z])
            valid = (torch.abs(delta_pos) > self.spacemouse_thresh['pos']).type(torch.float)
            delta_pos = delta_pos * valid * self.obj_move_delta

            delta_rot = torch.Tensor([-state.pitch, state.roll, -state.yaw])
            valid = (torch.abs(delta_rot) > self.spacemouse_thresh['rot']).type(torch.float)
            delta_rot = delta_rot * valid * self.obj_rota_delta

            self.obj_actions = torch.cat((delta_pos, delta_rot))
            
            # set joint delta
            if state.buttons[0] > 0:
                self.jnt_action = self.jnt_delta
            elif state.buttons[-1] > 0:
                self.jnt_action = -self.jnt_delta
            else:
                self.jnt_action = 0
            
            time.sleep(0.01)


    def init_teleop_keyboard(self):
        """
        wake/sleep：c/v
        """
        KeyboardEventHandler.initialize()
        # obj move
        KeyboardEventHandler.add_keyboard_callback(key=lazy.carb.input.KeyboardInput.W, callback_fn=self.obj_moveX)
        KeyboardEventHandler.add_keyboard_callback(key=lazy.carb.input.KeyboardInput.S, callback_fn=self.obj_moveX_)
        KeyboardEventHandler.add_keyboard_callback(key=lazy.carb.input.KeyboardInput.A, callback_fn=self.obj_moveY)
        KeyboardEventHandler.add_keyboard_callback(key=lazy.carb.input.KeyboardInput.D, callback_fn=self.obj_moveY_)
        KeyboardEventHandler.add_keyboard_callback(key=lazy.carb.input.KeyboardInput.Q, callback_fn=self.obj_moveZ)
        KeyboardEventHandler.add_keyboard_callback(key=lazy.carb.input.KeyboardInput.E, callback_fn=self.obj_moveZ_)
        # obj rotation
        KeyboardEventHandler.add_keyboard_callback(key=lazy.carb.input.KeyboardInput.I, callback_fn=self.obj_rotaX)
        KeyboardEventHandler.add_keyboard_callback(key=lazy.carb.input.KeyboardInput.K, callback_fn=self.obj_rotaX_)
        KeyboardEventHandler.add_keyboard_callback(key=lazy.carb.input.KeyboardInput.J, callback_fn=self.obj_rotaY)
        KeyboardEventHandler.add_keyboard_callback(key=lazy.carb.input.KeyboardInput.L, callback_fn=self.obj_rotaY_)
        KeyboardEventHandler.add_keyboard_callback(key=lazy.carb.input.KeyboardInput.U, callback_fn=self.obj_rotaZ)
        KeyboardEventHandler.add_keyboard_callback(key=lazy.carb.input.KeyboardInput.O, callback_fn=self.obj_rotaZ_)
        # joint move
        KeyboardEventHandler.add_keyboard_callback(key=lazy.carb.input.KeyboardInput.Z, callback_fn=self.jnt_move)
        KeyboardEventHandler.add_keyboard_callback(key=lazy.carb.input.KeyboardInput.X, callback_fn=self.jnt_move_)
        # key state
        KeyboardEventHandler.add_keyboard_callback(key=lazy.carb.input.KeyboardInput.KEY_1, callback_fn=self.key_state_cb)
        # esc
        KeyboardEventHandler.add_keyboard_callback(key=lazy.carb.input.KeyboardInput.ESCAPE, callback_fn=self.escape)
        # wake/sleep
        KeyboardEventHandler.add_keyboard_callback(key=lazy.carb.input.KeyboardInput.C, callback_fn=self.switch_wake)
        KeyboardEventHandler.add_keyboard_callback(key=lazy.carb.input.KeyboardInput.V, callback_fn=self.switch_sleep)


    def switch_wake(self):
        self.obj_sleep = False
        print('set obj to wake')

    def switch_sleep(self):
        self.obj_sleep = True
        print('set obj to sleep')
        

    def key_state_cb(self):
        self.iskeystate = True
    
    # def state_cb(self):
    #     self.issave = True

    def escape(self):
        self.esc = True

    def obj_moveX(self):
        self.obj_actions *= 0
        self.obj_actions[0] += self.obj_move_delta
    
    def obj_moveX_(self):
        self.obj_actions *= 0
        self.obj_actions[0] -= self.obj_move_delta
    
    def obj_moveY(self):
        self.obj_actions *= 0
        self.obj_actions[1] += self.obj_move_delta
    
    def obj_moveY_(self):
        self.obj_actions *= 0
        self.obj_actions[1] -= self.obj_move_delta
    
    def obj_moveZ(self):
        self.obj_actions *= 0
        self.obj_actions[2] += self.obj_move_delta
    
    def obj_moveZ_(self):
        self.obj_actions *= 0
        self.obj_actions[2] -= self.obj_move_delta
    
    def obj_rotaX(self):
        self.obj_actions *= 0
        self.obj_actions[3] += self.obj_rota_delta
    
    def obj_rotaX_(self):
        self.obj_actions *= 0
        self.obj_actions[3] -= self.obj_rota_delta
    
    def obj_rotaY(self):
        self.obj_actions *= 0
        self.obj_actions[4] += self.obj_rota_delta
    
    def obj_rotaY_(self):
        self.obj_actions *= 0
        self.obj_actions[4] -= self.obj_rota_delta
    
    def obj_rotaZ(self):
        self.obj_actions *= 0
        self.obj_actions[5] += self.obj_rota_delta
    
    def obj_rotaZ_(self):
        self.obj_actions *= 0
        self.obj_actions[5] -= self.obj_rota_delta
    
    def jnt_move(self):
        self.jnt_action = self.jnt_delta
    
    def jnt_move_(self):
        self.jnt_action = -self.jnt_delta


    def check_object_relation(
            self, 
            obj1_name:str, 
            type:str, 
            obj2_name:str, 
            obj2_link_name:str=None,
        ):
        """

        args:
            - type: OnTop / Inside
            - obj2_link_name: link of obj2
        """
        assert type in ['OnTop', 'Inside']

        obj1 = self.env.scene.object_registry("name", obj1_name)
        obj2 = self.env.scene.object_registry("name", obj2_name)

        # if check_collision(obj1):
        #     print('collision！！')
        #     return False

        # obj1, OnTop, obj2
        if type == 'OnTop':
            return obj1.states[object_states.OnTop].get_value(obj2)
        
        # obj1, Inside, obj2
        if obj2_link_name is None or obj2_link_name not in obj2.links:
            return obj1.states[object_states.Inside].get_value(obj2)

        # obj1, Inside, obj2, obj2_link
        obj2_link = obj2.links[obj2_link_name]
        success = False
        if self.check_inside_link(obj1, obj2_link) and obj1.states[object_states.Inside].get_value(obj2):
            success = True
        return success
    

    def place_object(
            self, 
            obj1_name:str, 
            type:str, 
            obj2_name:str, 
            obj2_link_name:str=None, 
            rescale_num=3, 
            rescale_factor=0.5, 
            orientation:Union[tuple, torch.Tensor]=None, 
            aabb_offset=None
        ):
        """

        args:
            - type: OnTop / Inside
            - obj2_link_name: link of obj2
        """
        assert type in ['OnTop', 'Inside']

        obj1 = self.env.scene.object_registry("name", obj1_name)
        obj2 = self.env.scene.object_registry("name", obj2_name)

        obj1.enable_gravity()
        obj2.enable_gravity()

        # obj1, OnTop, obj2
        if type == 'OnTop':
            success = obj1.states[object_states.OnTop].set_value(
                obj2, True, aabb_offset=aabb_offset)
            for _ in range(10):
                og.sim.step_physics()
            # success = obj1.states[object_states.OnTop].set_value(
                # obj2, True)
            return success
        
        # obj1, Inside, obj2
        if obj2_link_name is None or obj2_link_name not in obj2.links:
            for rs in range(rescale_num+1):
                success = obj1.states[object_states.Inside].set_value(obj2, True)
                if not success and rs < rescale_num:
                    print(f'resize {obj1_name}')
                    self.env.scene.remove_object(obj1)
                    obj1_cfg = self.get_object_cfg(obj1_name, rescale_factor)  # dict
                    obj1_type = obj1_cfg.pop('type')
                    obj_cls = globals()[obj1_type]
                    obj1 = obj_cls(**obj1_cfg)
                    self.env.scene.add_object(obj1)
                    self.env.step([])
                    obj1.keep_still()
                else:
                    break
            
            if not success:
                try_num = 500
                target_aabb_min = np.array(obj2.aabb_center - obj2.aabb_extent/2)
                target_aabb_max = np.array(obj2.aabb_center + obj2.aabb_extent/2)
                for i in range(try_num):
                    pos = (target_aabb_max+target_aabb_min)/2.
                    pos += (target_aabb_max-target_aabb_min)*(random.random()*0.4-0.2)
                    # = target_aabb_min + (target_aabb_max-target_aabb_min)*random.random()
                    pos[2] = target_aabb_min[2] + (target_aabb_max[2]-target_aabb_min[2])*(random.random()*0.7+0.3)
                    orientation = tf.rotate_about_world_xy_axis(orientation, 90)
                    obj1.set_position_orientation(position=pos, orientation=orientation)
                    obj1.keep_still()
                    og.sim.step_physics()
                    
                    if len(obj1.contact_list()) != 0:
                        continue
                    
                    # for _ in range(50):
                    for _ in range(50):
                        og.sim.step_physics()
                        # self.env.step([])   # for debugging

                    if obj1.states[object_states.Inside].get_value(obj2):
                        success = True
                        break

            return success
        

        # obj1, Inside, obj2, obj2_link
        obj2_link = obj2.links[obj2_link_name]
        target_aabb_min = np.array(obj2_link.aabb_center - obj2_link.aabb_extent/2)
        target_aabb_max = np.array(obj2_link.aabb_center + obj2_link.aabb_extent/2)
        success = False
        for rs in range(rescale_num+1):
            try_num = 500 
            for i in range(try_num):
                pos = (target_aabb_max+target_aabb_min)/2.
                pos += (target_aabb_max-target_aabb_min)*(random.random()*0.4-0.2)
                # = target_aabb_min + (target_aabb_max-target_aabb_min)*random.random()
                # pos[2] = target_aabb_min[2] + (target_aabb_max[2]-target_aabb_min[2])*(random.random()*0.7+0.3)
                pos[2] = target_aabb_min[2] + (target_aabb_max[2]-target_aabb_min[2])*(random.random()*0.7+0.5)
                if orientation is None:
                    orientation = T.random_quaternion()[0]
                elif i > 200:
                    orientation = tf.rotate_about_world_xy_axis(orientation, 90)
                obj1.set_position_orientation(position=pos, orientation=orientation)
                obj1.keep_still()
                og.sim.step_physics()
                
                if len(obj1.contact_list()) != 0:
                    continue

                for _ in range(50):
                    og.sim.step_physics()
                    # self.env.step([])   # for debugging

                inside = obj1.states[object_states.Inside].get_value(obj2)
                check_link = self.check_inside_link(obj1, obj2_link)
                if check_link and inside:
                    success = True
                    break
            
            if not success and rs < rescale_num:
                print(f'resize {obj1_name}')
                self.env.scene.remove_object(obj1)
                obj1_cfg = self.get_object_cfg(obj1_name, rescale_factor)  # dict
                obj1_type = obj1_cfg.pop('type')
                obj_cls = globals()[obj1_type]
                obj1 = obj_cls(**obj1_cfg)
                self.env.scene.add_object(obj1)
                self.env.step([])
                obj1.keep_still()
            else:
                break

        return success
    

    def get_object_cfg(self, obj_name, rescale_factor=1)->dict:
        """

        return: obj_cfg (dict)
        """
        for obj_cfg in self.objects_cfg:
            if obj_cfg['name'] == obj_name:
                obj_cfg['scale'] = [s*rescale_factor for s in obj_cfg['scale']]
                return copy.deepcopy(obj_cfg)
        return None


    def parse_scene_yaml(self, scene_yaml=None, ifaugment=False, fixed_base=True, fixed_base_operate=False):
        """
        return:
            - task_name: str
            - skill_name: str
            - objects_params: list[dict]
            - objects_relationships: list[list(obj1_name, type, obj2_name, obj2_link_name)]
        """
        objects_params = []
        objects_relationships = []
        objects_uids_dict = dict()
        objects_jointValues_dict = dict()
        tele_obj_cfg = None
        skill_name = None
        correct_skill_name = None
        for yaml_item in scene_yaml:
            if 'task_name' in yaml_item:
                task_name = yaml_item['task_name']

            if 'skill_name' in yaml_item:
                skill_name = yaml_item['skill_name']
            
            if 'correct_skill_name' in yaml_item:
                correct_skill_name = yaml_item['correct_skill_name']

            if 'name' not in yaml_item:
                continue

            object_params = dict()
            if not yaml_item['uid'].isdigit() and len(yaml_item['uid']) == 6:   # behaviour
                object_params['type'] = 'DatasetObject'
                object_params['name'] = yaml_item['name'].replace(' ', '_').replace('-', '_')
                object_params['category'] = uid_cate_dict[yaml_item['uid']]
                object_params['model'] = yaml_item['uid']

            else:   # partnet/objaverse
                object_params['type'] = 'USDObject'
                object_params['name'] = yaml_item['name'].replace(' ', '_').replace('-', '_')
                object_params['usd_path'] = yaml_item['model']

            scale = yaml_item['scale']
            if yaml_item['uid'] in uid_aabb_dict:
                obj_size = np.array(uid_aabb_dict[yaml_item['uid']])*yaml_item['scale']
                if yaml_item['size'] != 'none' and np.max(obj_size) < yaml_item['size']:
                    scale = float(yaml_item['scale'] * yaml_item['size'] / np.max(obj_size))
            
            object_params['scale'] = [scale]*3
            objects_uids_dict[object_params['name']] = yaml_item['uid']
            objects_jointValues_dict[object_params['name']] = yaml_item['joint_values']

            # set fixed_base
            if fixed_base:
                object_params['fixed_base'] = True
            else:
                object_params['fixed_base'] = True if ifaugment else not yaml_item['movable']

            # record operated_link_name
            if 'operated_link_name' in yaml_item:
                assert tele_obj_cfg is None
                tele_obj_cfg = {'obj_name': object_params['name'], 'operated_link_name': yaml_item['operated_link_name']}
                if yaml_item['operated_link_name'] != 'none':
                    object_params['fixed_base'] = True
                else:
                    object_params['fixed_base'] = fixed_base_operate

            if self.istest:
                object_params['fixed_base'] = True

            objects_params.append(object_params)

            if (yaml_item['task_relevance'] == 'none' and random.random() > 0.5) or yaml_item['spatial_relationship'] == 'none':
                relationship = ['OnTop', 'ground']
            else:
                relationship = yaml_item['spatial_relationship'].split(',')
            relationship_type = relationship[0].strip()
            relationship_obj2_name = relationship[1].strip().replace(' ', '_').replace('-', '_')
            relationship_obj2_link_name = relationship[2].strip() if len(relationship) == 3 else None
            assert relationship_type.lower() in ['ontop', 'inside']
            relationship_type = 'OnTop' if relationship_type.lower() == 'ontop' else 'Inside'
            objects_relationships.append(
                [object_params['name'], relationship_type, relationship_obj2_name, relationship_obj2_link_name])

        self.objects_uids_dict = objects_uids_dict
        self.objects_jointValues_dict = objects_jointValues_dict
        self.correct_skill_name = correct_skill_name

        return task_name, skill_name, tele_obj_cfg, objects_params, objects_relationships
    