/*
用于测试代码生成的anchor是否正确，可以跟tools/generagte_anchor.py 生成的结果进行对比，验证是否正确
*/

#include<iostream>
#include<vector>
#include"params.h"
#include "cuda_runtime_api.h"
using namespace std;
const float kPillarXSize = ParamsV2::kPillarXSize;
const float kPillarYSize = ParamsV2::kPillarYSize;
const float kPillarZSize = ParamsV2::kPillarZSize;
const float kMinXRange = ParamsV2::kMinXRange;
const float kMinYRange = ParamsV2::kMinYRange;
const float kMinZRange = ParamsV2::kMinZRange;
const float kMaxXRange = ParamsV2::kMaxXRange;
const float kMaxYRange = ParamsV2::kMaxYRange;
const float kMaxZRange = ParamsV2::kMaxZRange;
const int kNumClass = ParamsV2::kNumClass;
const int kMaxNumPillars = ParamsV2::kMaxNumPillars;
const int kMaxNumPointsPerPillar = ParamsV2::kMaxNumPointsPerPillar;
const int kNumPointFeature = ParamsV2::kNumPointFeature;
const int kGridXSize =
    static_cast<int>((kMaxXRange - kMinXRange) / kPillarXSize);
const int kGridYSize =
    static_cast<int>((kMaxYRange - kMinYRange) / kPillarYSize);
const int kGridZSize =
    static_cast<int>((kMaxZRange - kMinZRange) / kPillarZSize);
const int kRpnInputSize = 64 * kGridXSize * kGridYSize;
const int kNumAnchor = ParamsV2::kNumAnchor;
const int kNumOutputBoxFeature = ParamsV2::kNumOutputBoxFeature;
const int kRpnBoxOutputSize = kNumAnchor * kNumOutputBoxFeature;
const int kRpnClsOutputSize = kNumAnchor * kNumClass;
const int kRpnDirOutputSize = kNumAnchor * 2;
const int kBatchSize = ParamsV2::kBatchSize;
const int kNumIndsForScan = ParamsV2::kNumIndsForScan;
const int kNumThreads = ParamsV2::kNumThreads;
// if you change kNumThreads, need to modify NUM_THREADS_MACRO in
// common.h
const int kNumBoxCorners = ParamsV2::kNumBoxCorners;
const std::vector<int> kAnchorStrides = ParamsV2::AnchorStrides();// must be power of 2, feat downsample scale
const std::vector<int> kAnchorRanges{
    0,
    kGridXSize,
    0,
    kGridYSize,
    static_cast<int>(kGridXSize * 0.25),
    static_cast<int>(kGridXSize * 0.75),
    static_cast<int>(kGridYSize * 0.25),
    static_cast<int>(kGridYSize * 0.75)};
const std::vector<int> kNumAnchorSets = ParamsV2::NumAnchorSets();
const std::vector<std::vector<float>> kAnchorDxSizes =
    ParamsV2::AnchorDxSizes();
const std::vector<std::vector<float>> kAnchorDySizes =
    ParamsV2::AnchorDySizes();
const std::vector<std::vector<float>> kAnchorDzSizes =
    ParamsV2::AnchorDzSizes();
const std::vector<std::vector<float>> kAnchorZCoors =
    ParamsV2::AnchorZCoors();
const std::vector<std::vector<int>> kNumAnchorRo =
    ParamsV2::NumAnchorRo();
const std::vector<std::vector<float>> kAnchorRo =
    ParamsV2::AnchorRo();

float* anchors_px_;
float* anchors_py_;
float* anchors_pz_;
float* anchors_dx_;
float* anchors_dy_;
float* anchors_dz_;
float* anchors_ro_;

float* box_anchors_min_x_;
float* box_anchors_min_y_;
float* box_anchors_max_x_;
float* box_anchors_max_y_;

float* dev_box_anchors_min_x_;
float* dev_box_anchors_min_y_;
float* dev_box_anchors_max_x_;
float* dev_box_anchors_max_y_;

float* dev_anchors_px_;
float* dev_anchors_py_;
float* dev_anchors_pz_;
float* dev_anchors_dx_;
float* dev_anchors_dy_;
float* dev_anchors_dz_;
float* dev_anchors_ro_;

void GenerateAnchors(float* anchors_px_, float* anchors_py_,
                    float* anchors_pz_, float* anchors_dx_,
                    float* anchors_dy_, float* anchors_dz_,
                    float* anchors_ro_);
void ConvertAnchors2BoxAnchors(float* anchors_px_, float* anchors_py_,
                                float* box_anchors_min_x_,
                                float* box_anchors_min_y_,
                                float* box_anchors_max_x_,
                                float* box_anchors_max_y_);


void InitAnchors(){
    anchors_px_ = new float[kNumAnchor]();
    anchors_py_ = new float[kNumAnchor]();
    anchors_pz_ = new float[kNumAnchor]();
    anchors_dx_ = new float[kNumAnchor]();
    anchors_dy_ = new float[kNumAnchor]();
    anchors_dz_ = new float[kNumAnchor]();
    anchors_ro_ = new float[kNumAnchor]();
    box_anchors_min_x_ = new float[kNumAnchor]();
    box_anchors_min_y_ = new float[kNumAnchor]();
    box_anchors_max_x_ = new float[kNumAnchor]();
    box_anchors_max_y_ = new float[kNumAnchor]();

    GenerateAnchors(anchors_px_, anchors_py_, anchors_pz_, anchors_dx_,
                anchors_dy_, anchors_dz_, anchors_ro_);
    
    ConvertAnchors2BoxAnchors(anchors_px_, anchors_py_, box_anchors_min_x_,
                            box_anchors_min_y_, box_anchors_max_x_,
                            box_anchors_max_y_);
}
void GenerateAnchors(float* anchors_px_, float* anchors_py_,
                    float* anchors_pz_, float* anchors_dx_,
                    float* anchors_dy_, float* anchors_dz_,
                    float* anchors_ro_){
  for (int i = 0; i < kNumAnchor; ++i) {
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
  int index=0;
  for (size_t head = 0; head < kNumAnchorSets.size(); ++head) {
    float x_stride = kPillarXSize * kAnchorStrides[head];// kPillarXSize=0.16
    float y_stride = kPillarYSize * kAnchorStrides[head];// kPillarYSize=0.16
    int x_ind_start = kAnchorRanges[head * 4 + 0] / kAnchorStrides[head];
    int x_ind_end = kAnchorRanges[head * 4 + 1] / kAnchorStrides[head];
    int y_ind_start = kAnchorRanges[head * 4 + 2] / kAnchorStrides[head];
    int y_ind_end = kAnchorRanges[head * 4 + 3] / kAnchorStrides[head];
    // coors of first anchor's center
    float x_offset = kMinXRange + x_stride / 2.0;// kMinXRange=0
    float y_offset = kMinYRange + y_stride / 2.0;// kMinYRange=-39.68

    std::vector<float> anchor_x_count, anchor_y_count;
    for (int i = x_ind_start; i < x_ind_end; ++i) {
      float anchor_coor_x = static_cast<float>(i) * x_stride + x_offset;
      anchor_x_count.push_back(anchor_coor_x);
    }
    for (int i = y_ind_start; i < y_ind_end; ++i) {
      float anchor_coor_y = static_cast<float>(i) * y_stride + y_offset;
      anchor_y_count.push_back(anchor_coor_y);
    }

    for (int y = 0; y < y_ind_end - y_ind_start; ++y) {
      for (int x = 0; x < x_ind_end - x_ind_start; ++x) {
        int ro_count = 0;
        for (size_t c = 0; c < kNumAnchorRo[head].size(); ++c) {//3
          for (int i = 0; i < kNumAnchorRo[head][c]; ++i) {//2
            anchors_px_[ind] = anchor_x_count[x];
            anchors_py_[ind] = anchor_y_count[y];
            anchors_ro_[ind] = kAnchorRo[head][ro_count];
            anchors_pz_[ind] = kAnchorZCoors[head][c];
            anchors_dx_[ind] = kAnchorDxSizes[head][c];
            anchors_dy_[ind] = kAnchorDySizes[head][c];
            anchors_dz_[ind] = kAnchorDzSizes[head][c];
            ro_count++;
            ind++;
          }
        }
      }
    }
  }
  index++;
  std::cout<<"ind number:"<<index<<endl;
}



void ConvertAnchors2BoxAnchors(float* anchors_px,
                                             float* anchors_py,
                                             float* box_anchors_min_x_,
                                             float* box_anchors_min_y_,
                                             float* box_anchors_max_x_,
                                             float* box_anchors_max_y_) {
  // flipping box's dimension
  float* flipped_anchors_dx = new float[kNumAnchor]();
  float* flipped_anchors_dy = new float[kNumAnchor]();

  int ind = 0;
  for (size_t head = 0; head < kNumAnchorSets.size(); ++head) {

    int num_x_inds =
        (kAnchorRanges[head * 4 + 1] - kAnchorRanges[head * 4 + 0]) /
        kAnchorStrides[head];
    int num_y_inds =
        (kAnchorRanges[head * 4 + 3] - kAnchorRanges[head * 4 + 2]) /
        kAnchorStrides[head];
    int base_ind = ind;
    int ro_count = 0;
    std::cout<<num_x_inds<<";"<<num_y_inds<<std::endl;
    for (int x = 0; x < num_x_inds; ++x) {
      for (int y = 0; y < num_y_inds; ++y) {
        ro_count = 0;
        for (size_t c = 0; c < kNumAnchorRo[head].size(); ++c) {
          for (int i = 0; i < kNumAnchorRo[head][c]; ++i) {
            if (kAnchorRo[head][ro_count] <= 0.78) {
              flipped_anchors_dx[base_ind] = kAnchorDxSizes[head][c];
              flipped_anchors_dy[base_ind] = kAnchorDySizes[head][c];
            } else {
              flipped_anchors_dx[base_ind] = kAnchorDySizes[head][c];
              flipped_anchors_dy[base_ind] = kAnchorDxSizes[head][c];
            }
            if(base_ind>=1314140){
                std::cout<<"kAnchorDxSizes[head][c]:"<<kAnchorDxSizes[head][c]<<";kAnchorDySizes[head][c]:"<<kAnchorDySizes[head][c]<<std::endl;
            }
            ro_count++;
            base_ind++;
          }
        }
      }
    }
    std::cout<<"base_ind:"<<base_ind<<std::endl;
    std::cout<<"ind:"<<kAnchorRo[head].size()<<std::endl;
    for (int x = 0; x < num_x_inds; ++x) {
      for (int y = 0; y < num_y_inds; ++y) {
        for (size_t i = 0; i < kAnchorRo[head].size(); ++i) {
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
    std::cout<<"ind:"<<ind<<std::endl;
  }

  delete[] flipped_anchors_dx;
  delete[] flipped_anchors_dy;
}

int main(void){
    InitAnchors();
    for(int i=13100;i<13140;i++){
        // std::cout<<anchors_px_[i]<<";"<<anchors_py_[i]<<";"<<anchors_pz_[i]<<";"
        // <<anchors_dx_[i]<<";"<<anchors_dy_[i]<<";"<<anchors_dz_[i]<<";"<<anchors_ro_[i]<<endl;

        std::cout<<box_anchors_min_x_[i]<<";"<<box_anchors_min_y_[i]<<";"<<box_anchors_max_x_[i]<<";"
        <<box_anchors_max_y_[i]<<endl;
    }

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
    return 0;
}
