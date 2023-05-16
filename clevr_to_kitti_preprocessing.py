import numpy as np
import math
import json
import argparse
import os

def create_label_and_calib_files(scene, split):
    filename = scene["image_filename"]
    filename, extension = filename.split('.')

    objects = scene["objects"]
    camera_extrinsics = np.array(scene["camera_extrinsics"])
    rot = scene['directions']['right']

    if split == "train":
        label = 'label_2'
    else:
        label = 'label_2_gt'
    f1 = open(f'CLEVR_random/{split}ing/{label}/{filename}.txt', "w")
    for object in objects:
        s = generate_object_label(object, camera_extrinsics, rot)
        f1.write(s)
    f1.close()

    s = generate_calib_file(scene)
    s_str = " ".join(s)
    f2 = open(f'CLEVR_random/{split}ing/calib/{filename}.txt',"w")
    f2.write("P0: 0 0 0 0 0 0 0 0 0 0 0 0\n")
    f2.write("P1: 0 0 0 0 0 0 0 0 0 0 0 0\n")
    f2.write("P2: " + s_str + '\n')
    f2.write("P3: 0 0 0 0 0 0 0 0 0 0 0 0\n")
    f2.write("R0_rect: 0 0 0 0 0 0 0 0 0\n")
    f2.write("Tr_velo_to_cam: 0 0 0 0 0 0 0 0 0 0 0 0\n")
    f2.write("Tr_imu_to_velo: 0 0 0 0 0 0 0 0 0 0 0 0\n")
    f2.close()


def generate_calib_file(scene):
    proj_matrix = np.array(scene["camera_projection"])
    proj_matrix = np.reshape(proj_matrix, (12,))
    proj_matrix = proj_matrix.tolist()
    proj_matrix = list(map(str, proj_matrix))
    return proj_matrix


def generate_object_label(object, camera_extrinsics, rot):
    type = object["shape"]
    truncated = "0"
    occluded = "3"

    world_coords = object['3d_coords']
    world_coords = world_coords - np.array([0,0,world_coords[2]])
    location = convert_to_camera_coords(camera_extrinsics, world_coords)

    size = object["size"]
    whl = get_dimensions(type, size)
    rotation = object["rotation"]
    rotation_y = convert_to_radians(rotation)
    alpha = calculate_alpha(location, rotation_y)
    xmin, ymin, xmax, ymax = calculate_2d_bb(object, rot)

    object_label = [type, truncated, occluded, alpha, xmin, ymin, xmax, ymax, whl, whl, whl, location[0], location[1], location[2], rotation_y, '\n']
    object_label = list(map(str, object_label))
    return " ".join(object_label)

def convert_to_camera_coords(camera_extrinsics, world_coords):
    [x, y, z] = world_coords
    homo_world_coords = np.array([x, y, z, 1])
    camera_coords = camera_extrinsics @ homo_world_coords
    return camera_coords

def get_dimensions(type, size):
    if size == "large":
        r = 0.7
    else:
        r = 0.35
    # if type == "cube":
    #     # TODO: check cube sizing
    #     r /= math.sqrt(2)
    return r * 2

def convert_to_radians(theta):
    if theta > 180:
        theta -= 360
    return math.radians(theta)

def calculate_alpha(cam_to_obj, obj_orientation):
    obj_rotation_vector = np.array([math.cos(obj_orientation), math.sin(obj_orientation), 0.0])
    obj_rotation_vector /= np.linalg.norm(obj_rotation_vector)
    cam_to_obj_normalized = cam_to_obj / np.linalg.norm(cam_to_obj)
    cosine = np.dot(obj_rotation_vector, cam_to_obj_normalized)
    return np.arccos(cosine)

#modified from https://github.com/larchen/clevr-vqa/blob/master/bounding_box.py
def calculate_2d_bb(obj, rotation):
    [x, y, z] = obj['pixel_coords']

    [x1, y1, z1] = obj['3d_coords']

    cos_theta, sin_theta, _ = rotation

    x1 = x1 * cos_theta + y1 * sin_theta
    y1 = x1 * -sin_theta + y1 * cos_theta


    height_d = 6.9 * z1 * (15 - y1) / 2.0
    height_u = height_d
    width_l = height_d
    width_r = height_d

    if obj['shape'] == 'cylinder':
        d = 9.4 + y1
        h = 6.4
        s = z1

        height_u *= (s*(h/d + 1)) / ((s*(h/d + 1)) - (s*(h-s)/d))
        height_d = height_u * (h-s+d)/ (h + s + d)

        width_l *= 11/(10 + y1)
        width_r = width_l

    if obj['shape'] == 'cube':
        height_u *= 1.3 * 10 / (10 + y1)
        height_d = height_u
        width_l = height_u
        width_r = height_u

    ymin = (y - height_d)
    ymax = (y + height_u)
    xmin = (x - width_l)
    xmax = (x + width_r)

    return xmin, ymin, xmax, ymax

def create_imageset(split):
    f = open(f'CLEVR_random/ImageSets/{split}.txt', "w")
    for filename in os.listdir(f'CLEVR_random/{split}ing/image_2'):
        name, extension = filename.split('.')
        f.write(name + '\n')
    f.close()

if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-f", "--name", help="scene annotation json file")
    argParser.add_argument("-s", "--split", help="data split (training or testing)")
    args = argParser.parse_args()

    f = open(args.name)
    data = json.load(f)
    scenes = data["scenes"]
    for scene in scenes:
        create_label_and_calib_files(scene, args.split)
    
    create_imageset(args.split)






