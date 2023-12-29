/******************************************************************************
 * Copyright 2020 The Apollo Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *****************************************************************************/

/*
 * Copyright 2018-2019 Autoware Foundation. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @author Kosuke Murakami
 * @date 2019/02/26
 */

/**
 * @author Yan haixu
 * Contact: just github.com/hova88
 * @date 2021/04/30
 */

/**
 * @author Ye xiubo
 * Contact:github.com/speshowBUAA
 * @date 2022/01/05
 */
#include "pointpillars.h"

#include <chrono>
#include <cstring>
#include <iostream>

using std::chrono::duration;
using std::chrono::high_resolution_clock;

#define ANCHOR_NUM 1314144
// #define KNUMBER_FORSCAN 1024
// static const int kNumIndsForScan_=1024;
void PointPillars::InitParams() {
  YAML::Node params = YAML::LoadFile(pp_config_);
  kPillarXSize =
      params["DATA_CONFIG"]["DATA_PROCESSOR"][2]["VOXEL_SIZE"][0].as<float>();
  kPillarYSize =
      params["DATA_CONFIG"]["DATA_PROCESSOR"][2]["VOXEL_SIZE"][1].as<float>();
  kPillarZSize =
      params["DATA_CONFIG"]["DATA_PROCESSOR"][2]["VOXEL_SIZE"][2].as<float>();
  kMinXRange = params["DATA_CONFIG"]["POINT_CLOUD_RANGE"][0].as<float>();
  kMinYRange = params["DATA_CONFIG"]["POINT_CLOUD_RANGE"][1].as<float>();
  kMinZRange = params["DATA_CONFIG"]["POINT_CLOUD_RANGE"][2].as<float>();
  kMaxXRange = params["DATA_CONFIG"]["POINT_CLOUD_RANGE"][3].as<float>();
  kMaxYRange = params["DATA_CONFIG"]["POINT_CLOUD_RANGE"][4].as<float>();
  kMaxZRange = params["DATA_CONFIG"]["POINT_CLOUD_RANGE"][5].as<float>();
  kNumClass = params["CLASS_NAMES"].size();
  kMaxNumPillars =
      params["DATA_CONFIG"]["DATA_PROCESSOR"][2]["MAX_NUMBER_OF_VOXELS"]["test"]
          .as<int>();
  kMaxNumPointsPerPillar =
      params["DATA_CONFIG"]["DATA_PROCESSOR"][2]["MAX_POINTS_PER_VOXEL"]
          .as<int>();
  kNumInputBoxFeature = 7;
  kNumOutputBoxFeature = params["MODEL"]["DENSE_HEAD"]["TARGET_ASSIGNER_CONFIG"]
                               ["BOX_CODER_CONFIG"]["code_size"]
                                   .as<int>();
  kBatchSize = 1;
  kNmsPreMaxsize =
      params["MODEL"]["POST_PROCESSING"]["NMS_CONFIG"]["NMS_PRE_MAXSIZE"]
          .as<int>();
  kNmsPostMaxsize =
      params["MODEL"]["POST_PROCESSING"]["NMS_CONFIG"]["NMS_POST_MAXSIZE"]
          .as<int>();
  kVfeChannels = 64;
  score_threshold_ =
      params["MODEL"]["POST_PROCESSING"]["NMS_CONFIG"]["SCORE_THRESH"]
          .as<float>();
  nms_overlap_threshold_ =
      params["MODEL"]["POST_PROCESSING"]["NMS_CONFIG"]["NMS_THRESH"]
          .as<float>();

  // Generate secondary parameters based on above.
  kGridXSize =
      static_cast<int>((kMaxXRange - kMinXRange) / kPillarXSize);  // 468
  kGridYSize =
      static_cast<int>((kMaxYRange - kMinYRange) / kPillarYSize);  // 468
  kGridZSize = static_cast<int>((kMaxZRange - kMinZRange) / kPillarZSize);  // 1
  assert(1 == kGridZSize);
  kRpnInputSize = kVfeChannels * kGridYSize * kGridXSize;

  for (const auto &anchor_size :
       params["MODEL"]["DENSE_HEAD"]["ANCHOR_SIZES"]) {
    anchor_sizes_.push_back(anchor_size.as<float>());
  }
  for (const auto &anchor_bottom_height :
       params["MODEL"]["DENSE_HEAD"]["ANCHOR_BOTTOM_HEIGHTS"]) {
    anchor_bottom_heights_.push_back(anchor_bottom_height.as<float>());
  }
  for (const auto &anchor_rotation :
       params["MODEL"]["DENSE_HEAD"]["ANCHOR_ROTATIONS"]) {
    anchor_rotations_.push_back(anchor_rotation.as<float>());
  }
}

void PointPillars::InitAnchors(){
  anchors_px_ = new float[ANCHOR_NUM]();
  anchors_py_ = new float[ANCHOR_NUM]();
  anchors_pz_ = new float[ANCHOR_NUM]();
  anchors_dx_ = new float[ANCHOR_NUM]();
  anchors_dy_ = new float[ANCHOR_NUM]();
  anchors_dz_ = new float[ANCHOR_NUM]();
  anchors_ro_ = new float[ANCHOR_NUM]();
  box_anchors_min_x_ = new float[ANCHOR_NUM]();
  box_anchors_min_y_ = new float[ANCHOR_NUM]();
  box_anchors_max_x_ = new float[ANCHOR_NUM]();
  box_anchors_max_y_ = new float[ANCHOR_NUM]();

  GenerateAnchors();

  ConvertAnchors2BoxAnchors(anchors_px_, anchors_py_, box_anchors_min_x_,
                            box_anchors_min_y_, box_anchors_max_x_,
                            box_anchors_max_y_);
  PutAnchorsInDeviceMemory();
}

void PointPillars::GenerateAnchors() {
  for (int i = 0; i < ANCHOR_NUM; ++i) {
    anchors_px_[i] = 0;
    anchors_py_[i] = 0;
    anchors_pz_[i] = 0;
    anchors_dx_[i] = 0;
    anchors_dy_[i] = 0;
    anchors_dz_[i] = 0;
    anchors_ro_[i] = 0;
    box_anchors_min_x_[i] = 0;
    box_anchors_min_y_[i] = 0;
    box_anchors_max_x_[i] = 0;
    box_anchors_max_y_[i] = 0;
  }
  int ind = 0; 
  for (int i = 0; i < kGridYSize; ++i) {
    float y = kMinYRange + (i + 0.5) * kPillarYSize;
    for (int j = 0; j < kGridXSize; ++j) {
      float x = kMinXRange + (j + 0.5) * kPillarXSize;
      for (int k = 0; k < kNumClass; ++k) {
        float z = anchor_bottom_heights_[k];
        float l = anchor_sizes_[k * kNumClass + 0];
        float w = anchor_sizes_[k * kNumClass + 1];
        float h = anchor_sizes_[k * kNumClass + 2];
        for (float ro : anchor_rotations_) {

          anchors_px_[ind] = x;
          anchors_py_[ind] = y;
          anchors_pz_[ind] = z;
          anchors_dx_[ind] = l;
          anchors_dy_[ind] = w;
          anchors_dz_[ind] = h;
          anchors_ro_[ind] = ro;
          ind++;
        }
      }
    }
  }
}

void PointPillars::ConvertAnchors2BoxAnchors(float* anchors_px,
                                             float* anchors_py,
                                             float* box_anchors_min_x_,
                                             float* box_anchors_min_y_,
                                             float* box_anchors_max_x_,
                                             float* box_anchors_max_y_){
  float* flipped_anchors_dx = new float[ANCHOR_NUM]();
  float* flipped_anchors_dy = new float[ANCHOR_NUM]();    
  int ind = 0;  
  int base_ind = ind;
  int ro_count = 0; 
  for (int i = 0; i < kGridYSize; ++i) {
    for (int j = 0; j < kGridXSize; ++j) {
      ro_count=0;
      for (int k = 0; k < kNumClass; ++k) {
        float l = anchor_sizes_[k * kNumClass + 0];
        float w = anchor_sizes_[k * kNumClass + 1];
        for (float ro : anchor_rotations_) {
          if(ro<=0.78){
            flipped_anchors_dx[base_ind]=l;
            flipped_anchors_dy[base_ind]=w;
          }
          else{
            flipped_anchors_dx[base_ind]=w;
            flipped_anchors_dy[base_ind]=l;
          }
          ro_count++;
          base_ind++;
        }
      }
    }
  }   

  for (int i = 0; i < kGridYSize; ++i) {
    for (int j = 0; j < kGridXSize; ++j) {
      for (int k = 0; k < kNumClass*2; ++k) {
        box_anchors_min_x_[ind] =
            anchors_px[ind] - flipped_anchors_dx[ind] / 2.0f;
        box_anchors_min_y_[ind] =
            anchors_py[ind] - flipped_anchors_dy[ind] / 2.0f;
        box_anchors_max_x_[ind] =
            anchors_px[ind] + flipped_anchors_dx[ind] / 2.0f;
        box_anchors_max_y_[ind] =
            anchors_py[ind] + flipped_anchors_dy[ind] / 2.0f;
        ind++;        
      }
    }
  }     
  delete[] flipped_anchors_dx;
  delete[] flipped_anchors_dy;                            
}

void PointPillars::PutAnchorsInDeviceMemory(){
  GPU_CHECK(cudaMemcpy(dev_box_anchors_min_x_, box_anchors_min_x_,
                       ANCHOR_NUM * sizeof(float), cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(dev_box_anchors_min_y_, box_anchors_min_y_,
                       ANCHOR_NUM * sizeof(float), cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(dev_box_anchors_max_x_, box_anchors_max_x_,
                       ANCHOR_NUM * sizeof(float), cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(dev_box_anchors_max_y_, box_anchors_max_y_,
                       ANCHOR_NUM * sizeof(float), cudaMemcpyHostToDevice));

  GPU_CHECK(cudaMemcpy(dev_anchors_px_, anchors_px_, ANCHOR_NUM * sizeof(float),
                       cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(dev_anchors_py_, anchors_py_, ANCHOR_NUM * sizeof(float),
                       cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(dev_anchors_pz_, anchors_pz_, ANCHOR_NUM * sizeof(float),
                       cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(dev_anchors_dx_, anchors_dx_, ANCHOR_NUM * sizeof(float),
                       cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(dev_anchors_dy_, anchors_dy_, ANCHOR_NUM * sizeof(float),
                       cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(dev_anchors_dz_, anchors_dz_, ANCHOR_NUM * sizeof(float),
                       cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(dev_anchors_ro_, anchors_ro_, ANCHOR_NUM * sizeof(float),
                       cudaMemcpyHostToDevice));
}

PointPillars::PointPillars(const bool use_onnx, const std::string pfe_file,
                           const std::string backbone_file,
                           const std::string pp_config)
    : use_onnx_(use_onnx),
      pfe_file_(pfe_file),
      backbone_file_(backbone_file),
      pp_config_(pp_config) {
  InitParams();
  InitTRT(use_onnx_);
  // GenerateAnchors();

  preprocess_points_cuda_ptr_.reset(new PreprocessPointsCuda(
      kNumThreads, kMaxNumPillars, kMaxNumPointsPerPillar, kNumPointFeature,
      kGridXSize, kGridYSize, kGridZSize, kPillarXSize, kPillarYSize,
      kPillarZSize, kMinXRange, kMinYRange, kMinZRange, kMaxXRange, kMaxYRange,
      kMaxZRange,kNumIndsForScan));

  anchor_mask_cuda_ptr_.reset(new AnchorMaskCuda(
      kNumThreads, kNumIndsForScan, ANCHOR_NUM, kMinXRange, kMinYRange,
      kPillarXSize, kPillarYSize, kGridXSize, kGridYSize));
  scatter_cuda_ptr_.reset(
      new ScatterCuda(kVfeChannels, kGridXSize, kGridYSize));

  const float float_min = std::numeric_limits<float>::lowest();
  const float float_max = std::numeric_limits<float>::max();
  postprocess_cuda_ptr_.reset(new PostprocessCudaV2(
      float_min, float_max, ANCHOR_NUM, kNumClass,
      score_threshold_, kNumThreads,nms_overlap_threshold_,
      kNumBoxCorners, kNumOutputBoxFeature));
  DeviceMemoryMalloc();
  InitAnchors();
}

void PointPillars::DeviceMemoryMalloc() {
  // for pillars
  GPU_CHECK(cudaMalloc(reinterpret_cast<void **>(&dev_num_points_per_pillar_),
                       kMaxNumPillars * sizeof(float)));  // M
  GPU_CHECK(cudaMalloc(reinterpret_cast<void **>(&dev_pillar_point_feature_),
                       kMaxNumPillars * kMaxNumPointsPerPillar *
                           kNumPointFeature * sizeof(float)));  // [M , m , 4]
  GPU_CHECK(cudaMalloc(reinterpret_cast<void **>(&dev_pillar_coors_),
                       kMaxNumPillars * 4 * sizeof(int)));  // [M , 4]

  GPU_CHECK(cudaMalloc(reinterpret_cast<void **>(&dev_pfe_gather_feature_),
                       kMaxNumPillars * kMaxNumPointsPerPillar *
                           kNumGatherPointFeature * sizeof(float)));
  // for trt inference
  // create GPU buffers and a stream
  GPU_CHECK(
      cudaMalloc(&pfe_buffers_[0], kMaxNumPillars * kMaxNumPointsPerPillar *
                                       kNumGatherPointFeature * sizeof(float)));
  GPU_CHECK(cudaMalloc(&pfe_buffers_[1],
                       kMaxNumPillars * kVfeChannels * sizeof(float)));
  GPU_CHECK(cudaMalloc(&rpn_buffers_[0],kRpnInputSize * sizeof(float)));
  GPU_CHECK(cudaMalloc(&rpn_buffers_[1],1*kGridYSize * kGridXSize*18*sizeof(float)))
  GPU_CHECK(cudaMalloc(&rpn_buffers_[2],1*kGridYSize * kGridXSize*42*sizeof(float)))
  GPU_CHECK(cudaMalloc(&rpn_buffers_[3],1*kGridYSize * kGridXSize*12*sizeof(float)))

  // for scatter kernel
  GPU_CHECK(cudaMalloc(reinterpret_cast<void **>(&dev_scattered_feature_),
                       kNumThreads * kGridYSize * kGridXSize * sizeof(float)));

  // for filter
  GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_anchors_px_),
                       ANCHOR_NUM * sizeof(float)));
  GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_anchors_py_),
                       ANCHOR_NUM * sizeof(float)));
  GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_anchors_pz_),
                       ANCHOR_NUM * sizeof(float)));
  GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_anchors_dx_),
                       ANCHOR_NUM * sizeof(float)));
  GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_anchors_dy_),
                       ANCHOR_NUM * sizeof(float)));
  GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_anchors_dz_),
                       ANCHOR_NUM * sizeof(float)));
  GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_anchors_ro_),
                       ANCHOR_NUM * sizeof(float)));
  GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_filtered_box_),
                       ANCHOR_NUM * kNumOutputBoxFeature * sizeof(float)));
  GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_filtered_score_),
                       ANCHOR_NUM * sizeof(float)));
  GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_filtered_label_),
                       ANCHOR_NUM * sizeof(int)));
  GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_filtered_dir_),
                       ANCHOR_NUM * sizeof(int)));
  GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_box_for_nms_),
                       ANCHOR_NUM * kNumBoxCorners * sizeof(float)));
  GPU_CHECK(
      cudaMalloc(reinterpret_cast<void**>(&dev_filter_count_), sizeof(int)));

  //for postprocess
  GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_sparse_pillar_map_),
                      kNumIndsForScan * kNumIndsForScan * sizeof(int)));
  GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_cumsum_along_x_),
                       kNumIndsForScan * kNumIndsForScan * sizeof(int)));
  GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_cumsum_along_y_),
                       kNumIndsForScan * kNumIndsForScan * sizeof(int)));
  // for make anchor mask kernel
  GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_box_anchors_min_x_),
                       ANCHOR_NUM * sizeof(float)));
  GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_box_anchors_min_y_),
                       ANCHOR_NUM * sizeof(float)));
  GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_box_anchors_max_x_),
                       ANCHOR_NUM * sizeof(float)));
  GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_box_anchors_max_y_),
                       ANCHOR_NUM * sizeof(float)));
  GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_anchor_mask_),
                      ANCHOR_NUM * sizeof(int)));
}

PointPillars::~PointPillars() {
  // GPU_CHECK(cudaFree(dev_anchors_));
  GPU_CHECK(cudaFree(dev_cumsum_along_x_));
  GPU_CHECK(cudaFree(dev_cumsum_along_y_));
  GPU_CHECK(cudaFree(dev_anchor_mask_));
  // for pillars
  GPU_CHECK(cudaFree(dev_num_points_per_pillar_));
  GPU_CHECK(cudaFree(dev_pillar_point_feature_));
  GPU_CHECK(cudaFree(dev_pillar_coors_));
  // for pfe forward
  GPU_CHECK(cudaFree(dev_pfe_gather_feature_));

  GPU_CHECK(cudaFree(pfe_buffers_[0]));
  GPU_CHECK(cudaFree(pfe_buffers_[1]));

  GPU_CHECK(cudaFree(rpn_buffers_[0]));
  GPU_CHECK(cudaFree(rpn_buffers_[1]));
  GPU_CHECK(cudaFree(rpn_buffers_[2]));
  GPU_CHECK(cudaFree(rpn_buffers_[3]));

  GPU_CHECK(cudaFree(dev_box_anchors_min_x_));
  GPU_CHECK(cudaFree(dev_box_anchors_min_y_));
  GPU_CHECK(cudaFree(dev_box_anchors_max_x_));
  GPU_CHECK(cudaFree(dev_box_anchors_max_y_));

  GPU_CHECK(cudaFree(dev_anchors_px_));
  GPU_CHECK(cudaFree(dev_anchors_py_));
  GPU_CHECK(cudaFree(dev_anchors_pz_));
  GPU_CHECK(cudaFree(dev_anchors_dx_));
  GPU_CHECK(cudaFree(dev_anchors_dy_));
  GPU_CHECK(cudaFree(dev_anchors_dz_));
  GPU_CHECK(cudaFree(dev_anchors_ro_));
  GPU_CHECK(cudaFree(dev_filtered_box_));
  GPU_CHECK(cudaFree(dev_filtered_score_));
  GPU_CHECK(cudaFree(dev_filtered_label_));
  GPU_CHECK(cudaFree(dev_filtered_dir_));
  GPU_CHECK(cudaFree(dev_box_for_nms_));
  GPU_CHECK(cudaFree(dev_filter_count_));

  pfe_context_->destroy();
  backbone_context_->destroy();
  pfe_engine_->destroy();
  backbone_engine_->destroy();
  // for post process
  GPU_CHECK(cudaFree(dev_scattered_feature_));
  GPU_CHECK(cudaFree(dev_sparse_pillar_map_));
  // delete[] host_box_;
  // delete[] host_score_;
  // delete[] host_filtered_count_;

  delete[] anchors_px_;
  delete[] anchors_py_;
  delete[] anchors_pz_;
  delete[] anchors_dx_;
  delete[] anchors_dy_;
  delete[] anchors_dz_;
  delete[] anchors_ro_;
  delete[] box_anchors_min_x_;
  delete[] box_anchors_min_y_;
  delete[] box_anchors_max_x_;
  delete[] box_anchors_max_y_;
}

void PointPillars::SetDeviceMemoryToZero() {
  voxel_num_ = 0;

  GPU_CHECK(cudaMemset(dev_num_points_per_pillar_, 0,
                       kMaxNumPillars * sizeof(float)));
  GPU_CHECK(cudaMemset(dev_pillar_point_feature_, 0,
                       kMaxNumPillars * kMaxNumPointsPerPillar *
                           kNumPointFeature * sizeof(float)));
  GPU_CHECK(cudaMemset(dev_pillar_coors_, 0, kMaxNumPillars * 4 * sizeof(int)));

  GPU_CHECK(cudaMemset(dev_pfe_gather_feature_, 0,
                       kMaxNumPillars * kMaxNumPointsPerPillar *
                           kNumGatherPointFeature * sizeof(float)));
  GPU_CHECK(cudaMemset(pfe_buffers_[0], 0,
                       kMaxNumPillars * kMaxNumPointsPerPillar *
                           kNumGatherPointFeature * sizeof(float)));
  GPU_CHECK(cudaMemset(pfe_buffers_[1], 0,
                       kMaxNumPillars * kVfeChannels * sizeof(float)));
  GPU_CHECK(cudaMemset(rpn_buffers_[0], 0,kRpnInputSize * sizeof(float)));
  GPU_CHECK(cudaMemset(rpn_buffers_[1], 0,1*kGridYSize * kGridXSize*18*sizeof(float)));
  GPU_CHECK(cudaMemset(rpn_buffers_[2], 0,1*kGridYSize * kGridXSize*42*sizeof(float)));
  GPU_CHECK(cudaMemset(rpn_buffers_[3], 0,1*kGridYSize * kGridXSize*12*sizeof(float)));
  GPU_CHECK(cudaMemset(dev_scattered_feature_, 0,
                       kNumThreads * kGridYSize * kGridXSize * sizeof(float)));
  GPU_CHECK(cudaMemset(dev_sparse_pillar_map_, 0,
                      kNumIndsForScan * kNumIndsForScan * sizeof(int)));
  GPU_CHECK(cudaMemset(dev_anchor_mask_, 0, ANCHOR_NUM * sizeof(int)));
}

void PointPillars::InitTRT(const bool use_onnx) {
  if (use_onnx_) {
    // create a TensorRT model from the onnx model and load it into an engine
    OnnxToTRTModel(pfe_file_, &pfe_engine_);
    OnnxToTRTModel(backbone_file_, &backbone_engine_);
  } else {
    EngineToTRTModel(pfe_file_, &pfe_engine_);
    EngineToTRTModel(backbone_file_, &backbone_engine_);
  }
  if (pfe_engine_ == nullptr || backbone_engine_ == nullptr) {
    std::cerr << "Failed to load ONNX file.";
  }

  // create execution context from the engine
  pfe_context_ = pfe_engine_->createExecutionContext();
  backbone_context_ = backbone_engine_->createExecutionContext();
  if (pfe_context_ == nullptr || backbone_context_ == nullptr) {
    std::cerr << "Failed to create TensorRT Execution Context.";
  }
}

void PointPillars::OnnxToTRTModel(
    const std::string &model_file,  // name of the onnx model
    nvinfer1::ICudaEngine **engine_ptr) {
  std::string model_cache = model_file + ".cache";
  std::fstream trt_cache(model_cache, std::ifstream::in);
  if (!trt_cache.is_open()) {
    std::cout << "Building TRT engine." << std::endl;
    // create the builder
    const auto explicit_batch =
        static_cast<uint32_t>(kBatchSize) << static_cast<uint32_t>(
            nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(g_logger_);
    nvinfer1::INetworkDefinition *network =
        builder->createNetworkV2(explicit_batch);

    // parse onnx model
    int verbosity = static_cast<int>(nvinfer1::ILogger::Severity::kWARNING);
    auto parser = nvonnxparser::createParser(*network, g_logger_);
    if (!parser->parseFromFile(model_file.c_str(), verbosity)) {
      std::string msg("failed to parse onnx file");
      g_logger_.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
      exit(EXIT_FAILURE);
    }

    // Build the engine
    builder->setMaxBatchSize(kBatchSize);
    // builder->setHalf2Mode(true);
    nvinfer1::IBuilderConfig *config = builder->createBuilderConfig();
    config->setMaxWorkspaceSize(1 << 30);
    bool has_fast_fp16 = builder->platformHasFastFp16();
    if (has_fast_fp16) {
      std::cout << "the platform supports Fp16, use Fp16." << std::endl;
      config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    nvinfer1::ICudaEngine *engine =
        builder->buildEngineWithConfig(*network, *config);
    if (engine == nullptr) {
      std::cerr << ": engine init null!" << std::endl;
      exit(-1);
    }

    // serialize the engine, then close everything down
    auto model_stream = (engine->serialize());
    std::fstream trt_out(model_cache, std::ifstream::out);
    if (!trt_out.is_open()) {
      std::cout << "Can't store trt cache.\n";
      exit(-1);
    }
    trt_out.write((char *)model_stream->data(), model_stream->size());
    trt_out.close();
    model_stream->destroy();

    *engine_ptr = engine;
    parser->destroy();
    network->destroy();
    config->destroy();
    builder->destroy();
  } else {
    std::cout << "Load TRT cache." << std::endl;
    char *data;
    unsigned int length;

    // get length of file:
    trt_cache.seekg(0, trt_cache.end);
    length = trt_cache.tellg();
    trt_cache.seekg(0, trt_cache.beg);

    data = (char *)malloc(length);
    if (data == NULL) {
      std::cout << "Can't malloc data.\n";
      exit(-1);
    }

    trt_cache.read(data, length);
    // create context
    auto runtime = nvinfer1::createInferRuntime(g_logger_);
    if (runtime == nullptr) {
      std::cerr << ": runtime null!" << std::endl;
      exit(-1);
    }
    // plugin_ = nvonnxparser::createPluginFactory(g_logger_);
    nvinfer1::ICudaEngine *engine =
        (runtime->deserializeCudaEngine(data, length, 0));
    if (engine == nullptr) {
      std::cerr << ": engine null!" << std::endl;
      exit(-1);
    }
    *engine_ptr = engine;
    free(data);
    trt_cache.close();
  }
}

void PointPillars::EngineToTRTModel(const std::string &engine_file,
                                    nvinfer1::ICudaEngine **engine_ptr) {
  int verbosity = static_cast<int>(nvinfer1::ILogger::Severity::kWARNING);
  std::stringstream gieModelStream;
  gieModelStream.seekg(0, gieModelStream.beg);

  std::ifstream cache(engine_file);
  gieModelStream << cache.rdbuf();
  cache.close();
  nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(g_logger_);

  if (runtime == nullptr) {
    std::string msg("failed to build runtime parser");
    g_logger_.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
    exit(EXIT_FAILURE);
  }
  gieModelStream.seekg(0, std::ios::end);
  const int modelSize = gieModelStream.tellg();

  gieModelStream.seekg(0, std::ios::beg);
  void *modelMem = malloc(modelSize);
  gieModelStream.read((char *)modelMem, modelSize);

  nvinfer1::ICudaEngine *engine =
      runtime->deserializeCudaEngine(modelMem, modelSize, NULL);
  if (engine == nullptr) {
    std::string msg("failed to build engine parser");
    g_logger_.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
    exit(EXIT_FAILURE);
  }
  *engine_ptr = engine;

  for (int bi = 0; bi < engine->getNbBindings(); bi++) {
    if (engine->bindingIsInput(bi) == true)
      printf("Binding %d (%s): Input. \n", bi, engine->getBindingName(bi));
    else
      printf("Binding %d (%s): Output. \n", bi, engine->getBindingName(bi));
  }
}

void PointPillars::DoInference(const float *in_points_array,
                               const int in_num_points,
                               std::vector<float> *out_detections,
                               std::vector<int> *out_labels,
                               std::vector<float> *out_scores) {
  SetDeviceMemoryToZero();
  cudaDeviceSynchronize();
  // [STEP 1] : load pointcloud
  auto load_start = high_resolution_clock::now();
  float *dev_points;
  GPU_CHECK(cudaMalloc(reinterpret_cast<void **>(&dev_points),
                       in_num_points * kNumPointFeature * sizeof(float)));
  GPU_CHECK(cudaMemcpy(dev_points, in_points_array,
                       in_num_points * kNumPointFeature * sizeof(float),
                       cudaMemcpyHostToDevice));
  auto load_end = high_resolution_clock::now();

  // [STEP 2] : preprocess
  auto preprocess_start = high_resolution_clock::now();
  host_pillar_count_[0] = 0;
  preprocess_points_cuda_ptr_->DoPreprocessPointsCuda(
      dev_points, in_num_points, dev_num_points_per_pillar_,
      dev_pillar_point_feature_, dev_pillar_coors_, host_pillar_count_,
      dev_pfe_gather_feature_,dev_sparse_pillar_map_);
  cudaDeviceSynchronize();
  auto preprocess_end = high_resolution_clock::now();
  // [STEP 3]:anchor mask
  auto anchorMask_start = high_resolution_clock::now();
  anchor_mask_cuda_ptr_->DoAnchorMaskCuda(
    dev_sparse_pillar_map_, dev_cumsum_along_x_, dev_cumsum_along_y_,
    dev_box_anchors_min_x_, dev_box_anchors_min_y_, dev_box_anchors_max_x_,
    dev_box_anchors_max_y_, dev_anchor_mask_);
  cudaDeviceSynchronize();
  auto anchorMask_end = high_resolution_clock::now();

  // [STEP 4] : pfe forward
  cudaStream_t stream;
  GPU_CHECK(cudaStreamCreate(&stream));
  auto pfe_start = high_resolution_clock::now();
  GPU_CHECK(cudaMemcpyAsync(pfe_buffers_[0], dev_pfe_gather_feature_,
                            kMaxNumPillars * kMaxNumPointsPerPillar *
                                kNumGatherPointFeature *
                                sizeof(float),  /// kNumGatherPointFeature
                            cudaMemcpyDeviceToDevice, stream));
  pfe_context_->enqueueV2(pfe_buffers_, stream, nullptr);
  cudaDeviceSynchronize();
  auto pfe_end = high_resolution_clock::now();

  // [STEP 5] : scatter pillar feature
  auto scatter_start = high_resolution_clock::now();
  scatter_cuda_ptr_->DoScatterCuda(host_pillar_count_[0], dev_pillar_coors_,
                                   reinterpret_cast<float *>(pfe_buffers_[1]),
                                   dev_scattered_feature_);
  cudaDeviceSynchronize();
  auto scatter_end = high_resolution_clock::now();

  // [STEP 6] : backbone forward
  auto backbone_start = high_resolution_clock::now();
  GPU_CHECK(cudaMemcpyAsync(rpn_buffers_[0], dev_scattered_feature_,
                            kBatchSize * kRpnInputSize * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream));
  backbone_context_->enqueueV2(rpn_buffers_, stream, nullptr);
  cudaDeviceSynchronize();
  auto backbone_end = high_resolution_clock::now();

  // [STEP 7]: postprocess (multihead)
  auto postprocess_start = high_resolution_clock::now();
  GPU_CHECK(cudaMemset(dev_filter_count_, 0, sizeof(int)));
  postprocess_cuda_ptr_->DoPostprocessCuda(
      reinterpret_cast<float *>(rpn_buffers_[2]),  // [bboxes]
      reinterpret_cast<float *>(rpn_buffers_[1]),  // [scores]
      reinterpret_cast<float *>(rpn_buffers_[3]), //dir_cls
      dev_anchor_mask_, dev_anchors_px_, dev_anchors_py_, dev_anchors_pz_,
      dev_anchors_dx_, dev_anchors_dy_, dev_anchors_dz_, dev_anchors_ro_,
      dev_filtered_box_, dev_filtered_score_, dev_filtered_label_,
      dev_filtered_dir_, dev_box_for_nms_, dev_filter_count_,out_detections,
      out_labels);
  // cudaDeviceSynchronize();
  auto postprocess_end = high_resolution_clock::now();

  // release the stream and the buffers
  duration<double> preprocess_cost = preprocess_end - preprocess_start;
  duration<double> anchorMask_cost=anchorMask_end-anchorMask_start;
  duration<double> pfe_cost = pfe_end - pfe_start;
  duration<double> scatter_cost = scatter_end - scatter_start;
  duration<double> backbone_cost = backbone_end - backbone_start;
  duration<double> postprocess_cost = postprocess_end - postprocess_start;
  duration<double> pointpillars_cost = postprocess_end - preprocess_start;
  std::cout << "------------------------------------" << std::endl;
  std::cout << setiosflags(ios::left) << setw(14) << "Module" << setw(12)
            << "Time" << resetiosflags(ios::left) << std::endl;
  std::cout << "------------------------------------" << std::endl;
  std::string Modules[] = {"Preprocess", "AnchorMask","Pfe",         "Scatter",
                           "Backbone",   "Postprocess", "Summary"};
  double Times[] = {preprocess_cost.count(), anchorMask_cost.count(), pfe_cost.count(),
                    scatter_cost.count(),     backbone_cost.count(),
                    postprocess_cost.count(), pointpillars_cost.count()};

  for (int i = 0; i < sizeof(Times)/sizeof(Times[0]); ++i) {
    std::cout << setiosflags(ios::left) << setw(14) << Modules[i] << setw(8)
              << Times[i] * 1000 << " ms" << resetiosflags(ios::left)
              << std::endl;
  }
  std::cout << "------------------------------------" << std::endl;

  cudaStreamDestroy(stream);
  GPU_CHECK(cudaFree(dev_points));
}
