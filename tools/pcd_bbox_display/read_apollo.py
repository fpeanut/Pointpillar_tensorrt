import os
import time

import open3d as o3d

import numpy as np
import utils_kitti, open3d_vis as o3dvs

def read_txt(pcd_path,bbox_path):
    with open(pcd_path) as f1:
       lines1 = f1.readlines()
       pcd_list = []
       for line in lines1:
           scope = line.strip().split(',')
           pcd_list.append((scope[0], scope[1], scope[2], scope[3])	)
    with open(bbox_path) as f2:
       lines2 = f2.readlines()
       bbox_list = []
       for line in lines2:
           scope = line.strip().split(',')
           bbox_list.append((scope[0], scope[1], scope[2], scope[3],scope[4],scope[5],scope[6])	)
    pcd = np.array(pcd_list, dtype=np.float32)
    bbox = np.array(bbox_list, dtype=np.float32)
    return pcd,bbox

def dataloader(cloud_path , boxes_path, load_dim):
    data = np.fromfile(cloud_path, dtype=np.float32, count=-1).reshape([-1, load_dim])
    # data=data
    result = np.loadtxt(boxes_path).reshape(-1, 8)#修改你检测结果的特征维度
    return result, data

def main():

    # 显示
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    #####批量显示######
    # 这里可以修改为接收apollo的点云数据
    pcd_txt = "./waymo_pcd/"
    bbox_txt = "./bbox/"
    for file in os.listdir(pcd_txt):
        # pcd, bbox = read_txt(pcd_txt + file, bbox_txt + file)
        bbox, pcd = dataloader(pcd_txt + file, bbox_txt+file.rsplit('.bin', 1)[0]+".txt", 6)
        # print(pcd.shape,bbox.shape)
        pcd, points_color = o3dvs._draw_points(pcd, vis, 2, (0.5, 0.5, 0.5), 'xyz')
        o3dvs._draw_bboxes(bbox, vis, points_color, pcd)

        vis.poll_events()
        vis.update_renderer()
        vis.clear_geometries()

    vis.run()

if __name__ == "__main__":
    main()





