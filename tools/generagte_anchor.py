import torch
from  mmdet3d.core.anchor import build_prior_generator

def test_anchor_3d_range_generator():
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    anchor_generator_cfg = dict(
        type='AlignedAnchor3DRangeGenerator',
        ranges=[[-74.88, -74.88, -0.0345, 74.88, 74.88, -0.0345],
                [-74.88, -74.88, -0.1188, 74.88, 74.88, -0.1188],
                [-74.88, -74.88, 0, 74.88, 74.88, 0]],
        sizes=[
            [4.73, 2.08, 1.77],  # car
            [1.81, 0.84, 1.77],  # cyclist
            [0.91, 0.84, 1.74]  # pedestrian
        ],
        rotations=[0, 1.57],
        reshape_out=False)

    anchor_generator = build_prior_generator(anchor_generator_cfg)
    featmap_size = (468, 468)
    mr_anchors = anchor_generator.single_level_grid_anchors(
        featmap_size, 1, device=device)
    # 打开txt文件
    # with open('waymod_v2/anchors.txt', 'w') as f:
    #     for i in range(mr_anchors.shape[1]):
    #         for j in range(mr_anchors.shape[2]):
    #             for k in range(mr_anchors.shape[3]):
    #                 for l in  range(mr_anchors.shape[4]):
    #                     for m in range(mr_anchors.shape[5]):
    #                         f.write('{:.6f} '.format(mr_anchors[0][i][j][k][l][m].item()))
    #                     f.write('\n')
    # print("over！")
    print(mr_anchors[0][0][0])
