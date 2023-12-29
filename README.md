# Pointpillar_tensorrt
pointpillar mmdeteion3d model Tensorrt deployment, improved and faster

This repository contains sources and model for [PointPillars](https://arxiv.org/abs/1812.05784) inference using TensorRT.
The model is created with [mmdetection3d](https://github.com/open-mmlab/mmdetection3d/tree/v1.0.0rc0).

Overall inference has five phases:

- Convert points cloud into 4-channle voxels
- Extend 4-channel voxels to 10-channel voxel features
- Run pfe TensorRT engine to get 64-channel voxel features
- Run rpn backbone TensorRT engine to get 3D-detection raw data
- Parse bounding box, class type and direction

## Model && Data

The demo use the waymo data from Waymo Open Dataset.
The onnx file can be converted by [onnx_tools](/Pointpillar_tensorrt/tools/trans_backbonev.py&&trans_vfe.py)
If you want use my onnx transform code,you need to git clone mmdetection3d v1.0.0rc0(https://github.com/open-mmlab/mmdetection3d/tree/v1.0.0rc0).

### Prerequisites

To build the pointpillars inference, **TensorRT** and **CUDA** are needed.

## Environments

- NVIDIA RTX 3060 && RTX 3070ti
- in TensorRT 8.2.3 && TensorRT 8.6 && TensorRT 7.2.1

### Compile && Run

```shell
$ mkdir build && cd build
$ cmake .. && make 
$ cd ..
$ ./ApolloPP
```

### Visualization

You should install `open3d` in python environment.

```shell
$ cp -r ./Pointpillar_tensorrt/data/waymo_pcd ./Pointpillar_tensorrt/tools/pcd_bbox_display/waymo_pcd
$ cp -r ./Pointpillar_tensorrt/data/bbox ./Pointpillar_tensorrt/tools/pcd_bbox_display/bbox
$ cd tools/pcd_bbox_display
$ python read_apollo.py
```

| trt fp16 | pytorch |
| -------- | ------- |
| ![trt fp16](https://tvax2.sinaimg.cn/large/0080fUsgly1h534cyivy0j31e70qrh62.jpg) | ![pytorch](https://tva3.sinaimg.cn/large/0080fUsgly1h534bzakywj31eg0qn1bm.jpg) |



#### Performance in RTX3070ti of FP16

```
| Function(unit:ms) | NVIDIA RTX 3070ti Laptop GPU |
| ----------------- | --------------------------- |
| Preprocess        | 0.611175 ms                 |
| AnchorMask        | 0.30804  ms                 |
| Pfe               | 5.21747  ms                 |
| Scatter           | 0.089236 ms                 |
| Backbone          | 8.74011  ms                 |
| Postprocess       | 0.324388 ms                 |
| Summary           | 15.2936  ms                 |
```

## Note

- The waymo pretrained model in this project is trained only using 4-channel (x, y, z, i), which is different from the mmdetection3d pretrained_model.
- The demo will cache the onnx file to improve performance. If a new onnx will be used, please remove the cache file in "./model".
- The improvement point of this article is to perform the anchor operation as post-processing, and only export the backbone to the convolution output as an onnx model.
- Deployment environment：
    docker run --gpus all -it --name env_pyt_1.12 -v $(pwd):/app nvcr.io/nvidia/pytorch:22.03-py3
    opencv install:apt install libopencv-dev
    yaml install:apt-get update&& apt-get install libyaml-cpp-dev

## References

- [PointPillars: Fast Encoders for Object Detection from Point Clouds](https://arxiv.org/abs/1812.05784)
- [mmdetection3d](https://github.com/open-mmlab/mmdetection3d/tree/v1.0.0rc0)
- [MMDet3d-PointPillars](https://github.com/Tartisan/mmdetection3d)
