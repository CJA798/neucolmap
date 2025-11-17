// Copyright (c) 2024, COLMAP contributors.
// All rights reserved.

#pragma once

#ifdef COLMAP_ONNX_ENABLED

#include "colmap/feature/extractor.h"
#include "colmap/feature/matcher.h"
#include "colmap/sensor/bitmap.h"

#include <onnxruntime_cxx_api.h>
#include <memory>
#include <string>

namespace colmap {

// SuperPoint feature extractor using ONNX Runtime
class SuperPointExtractor {
 public:
  struct Options {
    // Path to SuperPoint ONNX model
    std::string model_path = "/usr/local/share/colmap/models/superpoint_only.onnx";
    
    // Maximum number of keypoints
    int max_num_keypoints = 1024;
    
    // Number of threads for inference
    int num_threads = 4;
    
    // Whether to use GPU (if available)
    bool use_gpu = true;
    
    // GPU device index
    int gpu_index = 0;
  };

  explicit SuperPointExtractor(const Options& options);
  ~SuperPointExtractor();

  // Extract features from a single image
  bool Extract(const Bitmap& bitmap,
               FeatureKeypoints* keypoints,
               FeatureDescriptors* descriptors);

 private:
  Options options_;
  std::unique_ptr<Ort::Env> env_;
  std::unique_ptr<Ort::Session> session_;
  std::unique_ptr<Ort::SessionOptions> session_options_;
  
  // Preprocess image for ONNX model
  std::vector<float> PreprocessImage(const Bitmap& bitmap, int& out_height, int& out_width);
};

// Simple descriptor matcher using mutual nearest neighbors
class DescriptorMatcher {
 public:
  // Match descriptors using mutual nearest neighbors
  static void MatchDescriptors(const FeatureDescriptors& descriptors1,
                                const FeatureDescriptors& descriptors2,
                                FeatureMatches* matches,
                                float ratio_test_threshold = 0.8);
};

}  // namespace colmap

#endif  // COLMAP_ONNX_ENABLED