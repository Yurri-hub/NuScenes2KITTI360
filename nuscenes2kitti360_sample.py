# convertion from NuScenes dataset to KITTI format
# inspired by https://github.com/poodarchu/nuscenes_to_kitti_format
# converting only camera captions (JPG files)
# converting all samples in every sequence data

# regardless of attributes indexed 2(if blocked) in KITTI label
# however, object minimum visibility level is adjustable

start_index = 0000
data_root = '/home/public/NPBGpp/NuScenes/v1.0-mini/'
img_output_root = './samples/data_2d_raw/'
label_output_root = './samples/label/'
pose_output_root = './samples/data_poses/'
calibration_output_root = './samples/calibration/'
pc_output_root = './samples/data_3d_semantics/'

min_visibility_level = '2'
delete_dontcare_objects = True

from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import BoxVisibility, view_points
import numpy as np
import cv2
import os
import shutil
from scipy.spatial.transform import Rotation as R
from plyfile import PlyData,PlyElement


category_reflection = \
{
    'human.pedestrian.adult': 'Pedestrian',
    'human.pedestrian.child': 'Pedestrian',
    'human.pedestrian.wheelchair': 'DontCare',
    'human.pedestrian.stroller': 'DontCare',
    'human.pedestrian.personal_mobility': 'DontCare',
    'human.pedestrian.police_officer': 'Pedestrian',
    'human.pedestrian.construction_worker': 'Pedestrian',
    'animal': 'DontCare',
    'vehicle.car': 'Car',
    'vehicle.motorcycle': 'Cyclist',
    'vehicle.bicycle': 'Cyclist',
    'vehicle.bus.bendy': 'Tram',
    'vehicle.bus.rigid': 'Tram',
    'vehicle.truck': 'Truck',
    'vehicle.construction': 'DontCare',
    'vehicle.emergency.ambulance': 'DontCare',
    'vehicle.emergency.police': 'DontCare',
    'vehicle.trailer': 'Tram',
    'movable_object.barrier': 'DontCare',
    'movable_object.trafficcone': 'DontCare',
    'movable_object.pushable_pullable': 'DontCare',
    'movable_object.debris': 'DontCare',
    'static_object.bicycle_rack': 'DontCare', 
}


if __name__ == '__main__':

    nusc = NuScenes(version='v1.0-mini', dataroot=data_root, verbose=True)
    sensor_list = ['CAM_FRONT', 'CAM_BACK', 'CAM_FRONT_LEFT', 'CAM_BACK_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT']

    frame_counter = start_index

    # if os.path.isdir(img_output_root) == True:
    #     print('previous image output found. deleting...')
    #     shutil.rmtree(img_output_root)
    # os.makedirs(img_output_root)
    # # if os.path.isdir(label_output_root) == True:
    # #     print('previous label output found. deleting...')
    # #     shutil.rmtree(label_output_root)
    # # os.makedirs(label_output_root)
    if os.path.isdir(pose_output_root) == True:
        print('previous pose output found. deleting...')
        shutil.rmtree(pose_output_root)
    os.makedirs(pose_output_root)
    if os.path.isdir(calibration_output_root) == True:
        print('previous intrinsic output found. deleting...')
        shutil.rmtree(calibration_output_root)
    os.makedirs(calibration_output_root)
    if os.path.isdir(pc_output_root) == True:
        print('previous pointcloud output found. deleting...')
        shutil.rmtree(pc_output_root)
    os.makedirs(pc_output_root)
    
    for present_scene in nusc.scene:
        # 这里先准备各个scene的image
        # os.makedirs(img_output_root + present_scene['name'])
        os.makedirs(pose_output_root + present_scene['name'])
        os.makedirs(calibration_output_root + present_scene['name'])
        os.makedirs(pc_output_root + present_scene['name'])

    # 这个样子是将几种机位按类型分开来
    for present_sample in tqdm(nusc.sample):
        if present_sample['prev'] == '':
            cali_flag = True
            frame_counter = start_index
            points_perscene = []

        # point cloud 激光获取的信息是带有标注的
        lidar_data = nusc.get('sample_data', present_sample['data']['LIDAR_TOP'])
        lidar_file = data_root + lidar_data['filename']
        points = np.fromfile(lidar_file, dtype=np.float32).reshape([-1, 5])[:, :3]
        # 每次跟ego_pose处理
        sensor_data = nusc.get('sample_data', present_sample['data']['LIDAR_TOP'])
        sensor_pose = nusc.get('calibrated_sensor', sensor_data['calibrated_sensor_token'])
        per_pose = nusc.get('ego_pose', sensor_data['ego_pose_token'])
        rota = [sensor_pose['rotation'][1], sensor_pose['rotation'][2], \
                sensor_pose['rotation'][3], sensor_pose['rotation'][0]]
        rota = R.from_quat(rota).as_matrix()
        per_rota = [per_pose['rotation'][1], per_pose['rotation'][2], \
                    per_pose['rotation'][3], per_pose['rotation'][0]]
        per_rota = R.from_quat(per_rota).as_matrix()
        trans = np.array(sensor_pose['translation'])
        per_trans = np.array(per_pose['translation'])
        rota = np.matmul(per_rota, rota)
        trans = np.matmul(per_rota, trans) + per_trans
        for i in range(points.shape[0]):
            points[i, :] = np.matmul(rota, points[i, :]) + trans
        # 所有时间叠加
        if present_sample['prev'] == '':
            points_perscene = points
        else:
            points_perscene = np.concatenate((points_perscene, points), axis=0)
        # 保存
        # if True:
        if present_sample['next'] == '':
            lidar_save = pc_output_root + nusc.get('scene', present_sample['scene_token'])['name'] \
                    + '/mvs.ply'
            points = [(points_perscene[i,0], points_perscene[i,1], points_perscene[i,2]) \
                    for i in range(points_perscene.shape[0])]
            vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
            el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
            PlyData([el]).write(lidar_save)

        for present_sensor in sensor_list:
            data_path, box_list, cam_intrinsic = nusc.get_sample_data(present_sample['data'][present_sensor], BoxVisibility.ALL)

            # # image
            # img_root = '/home/public/NPBGpp/NuScenes/v1.0-mini/' + \
            #         nusc.get('sample_data', present_sample['data'][present_sensor])['filename']
            # img_transfer = img_output_root + nusc.get('scene', present_sample['scene_token'])['name'] + '/' + present_sensor + '/'
            # if not os.path.exists(img_transfer):
            #     os.makedirs(img_transfer)
            # seqname = str(frame_counter).zfill(10)
            # cmd = 'cp ' + img_root + ' ' + img_transfer + seqname + '.jpg'
            # os.system(cmd)

            # pose
            sensor_data = nusc.get('sample_data', present_sample['data'][present_sensor])
            sensor_pose = nusc.get('calibrated_sensor', sensor_data['calibrated_sensor_token'])
            per_pose = nusc.get('ego_pose', sensor_data['ego_pose_token'])
            rota = [sensor_pose['rotation'][1], sensor_pose['rotation'][2], \
                    sensor_pose['rotation'][3], sensor_pose['rotation'][0]]
            rota = R.from_quat(rota).as_matrix()
            per_rota = [per_pose['rotation'][1], per_pose['rotation'][2], \
                        per_pose['rotation'][3], per_pose['rotation'][0]]
            per_rota = R.from_quat(per_rota).as_matrix()
            trans = np.array(sensor_pose['translation'])
            per_trans = np.array(per_pose['translation'])
            rota = np.matmul(per_rota, rota)
            trans = np.matmul(per_rota, trans) + per_trans
            for i in range(3):
                for j in range(3):
                    if i == 0 and j == 0:
                        RT_str = str(rota[i][j])
                    else: RT_str += ' ' + str(rota[i][j])
                RT_str += ' ' + str(trans[i])
            RT_str += ' 0.000000 0 0.000000 1'
            pose_file = pose_output_root + nusc.get('scene', present_sample['scene_token'])['name'] \
                    + '/' + present_sensor + '.txt'
            with open(pose_file, 'a') as output_f:
                output_f.write(str(frame_counter) + ' ' + RT_str + '\n')

            # calibration
            if cali_flag:
                for i in range(3):
                    for j in range(3):
                        if i == 0 and j == 0:
                            cali_str = str(cam_intrinsic[i][j])
                        else: cali_str += ' ' + str(cam_intrinsic[i][j])
                    cali_str += ' 0.0'
                cali_file = calibration_output_root + nusc.get('scene', present_sample['scene_token'])['name'] \
                        + '/' + 'perspective.txt'
                with open(cali_file, 'a') as output_f:
                    output_f.write(present_sensor + ': ' + cali_str + '\n')
        
        cali_flag = False
        frame_counter += 1
