import json
import os 
import numpy as np

def read_dataset(path):
    
    data_path = os.path.join(path, 'dataset.json')
    with open(data_path, 'r') as f:
        dataset = json.load(f)
        
    count = dataset['count']
    ids = dataset['ids']
    train_ids = dataset['train_ids']
    val_ids = dataset['val_ids']
    
    train_dicts = []
    val_dicts = []
    
    for id in train_ids:
        train_dict = {}
        train_dict ['img'] = os.path.join(path, 'rgb', '2x', f'{id}')
        train_dict ['cam'] = os.path.join(path, 'camera', f'{id}.json')
        # id is 0_id, we only want the id and convert to int
        train_dict ['pos_id'] =  int(id.split('_')[0])
        train_dict ['time_id'] = int(id.split('_')[1])
        train_dicts.append(train_dict)
    for id in val_ids:
        val_dict = {}
        val_dict ['img'] = os.path.join(path, 'rgb', '2x', f'{id}')
        val_dict ['cam'] = os.path.join(path, 'camera', f'{id}.json')
        val_dict ['pos_id'] =  int(id.split('_')[0])
        val_dict ['time_id'] = int(id.split('_')[1])
        val_dicts.append(val_dict)
        
    assert len(train_dicts) + len(val_dicts) == count
    
    return train_dicts, val_dicts


def read_camera(path):
    with open(path, 'r') as f:
        camera = json.load(f)
        
    focal_length = camera['focal_length']
    image_size = camera['image_size']
    orientation = camera['orientation']
    position = camera['position']
    principal_point = camera['principal_point']
    
    return focal_length, image_size, orientation, position, principal_point


def read_iphone_scene(dir, scene_name):
    scene_dir = os.path.join(dir, scene_name)
    train_dicts, val_dicts = read_dataset(scene_dir)

    dict_to_save = {}
    
    cam0 =  train_dicts[0]['cam']
    focal_length, image_size, orientation, position, principal_point = read_camera(cam0)
    
    dict_to_save["w"] = image_size[0] / 2
    dict_to_save["h"] = image_size[1] / 2
    dict_to_save["fl_x"] = focal_length / 2
    dict_to_save["fl_y"] = focal_length / 2
    dict_to_save["cx"] = principal_point[0] / 2
    dict_to_save["cy"] = principal_point[1] / 2
    
    frame_train = []
    frame_val = []
    len_time = len(train_dicts)
    for train_dict in train_dicts:
        frame_dict = {}
        frame_dict["file_path"] = train_dict['img']
        
        focal_length, image_size, orientation, position, principal_point = read_camera(train_dict['cam'])
        c2w = np.eye(4)
        
        R = np.asarray(orientation) # list [3,3]
        T = np.asarray(position) # list [3]
        
        c2w[:3,:3] = R
        c2w[:3,3] = T
        
        frame_dict["transform_matrix"] = c2w.tolist()

        frame_dict["time"] = train_dict['time_id'] / (len_time - 1)
        frame_dict["pos_id"] = train_dict['pos_id']
        
        frame_train.append(frame_dict)
    
    for val_dict in val_dicts:
        frame_dict = {}
        frame_dict["file_path"] = val_dict['img']
        
        focal_length, image_size, orientation, position, principal_point = read_camera(val_dict['cam'])
        c2w = np.eye(4)
        
        R = np.asarray(orientation) # list [3,3]
        T = np.asarray(position) # list [3]
        
        c2w[:3,:3] = R
        c2w[:3,3] = T
        
        frame_dict["transform_matrix"] = c2w.tolist()
        
        frame_dict["time"] = val_dict['time_id'] / (len_time - 1)
        frame_dict["pos_id"] = val_dict['pos_id']
        
        frame_val.append(frame_dict)
        
    dict_to_save["frames"] = frame_train 
    with open(os.path.join(scene_dir, 'transforms_train.json'), 'w') as f:
        json.dump(dict_to_save, f, indent=4)
        
    dict_to_save["frames"] = frame_val
    with open(os.path.join(scene_dir, 'transforms_test.json'), 'w') as f:
        json.dump(dict_to_save, f, indent=4)
        
        
    
# data_dir = "/data/guest_storage/zhanpengluo/Dataset/dynamic_reconstruction/iphone"
# scene_name ="apple"
# read_iphone_scene(data_dir, scene_name)
    
    
npy_file = "/data/guest_storage/zhanpengluo/Dataset/dynamic_reconstruction/iphone/apple/points-before.npy"
points = np.load(npy_file)
print(points.shape)
    
    





