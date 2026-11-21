import os

joint_position_kp = 5e3
contact_shooting_offset = 0.05

image_height = 512
image_width = 512 
focal_length = 17

camera_num_views = 32
camera_distance = 2.7
min_camera_distance = 0.4
camera_distance_interval = 0.2
object_min_size_camera = 200
ground_range = 2.5

augment_scene_count = 20
augment_state_count = 20
augment_object_probability = 1
test_scene_count = 20
test_state_count = 10

obj_control_ratio = 0.5


task_meta_path = "Kettle/Kettle-lift_kettle_and_pour_water/dataset/"
if not os.path.exists(task_meta_path):
    os.makedirs(task_meta_path)

dataset_path = "Kettle/Kettle-Lift_the_kettle_above_the_cup_and_tilt_it_to_pour_water_h5/train"
