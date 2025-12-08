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

// SuperPoint + LightGlue feature extractor and matcher using ONNX Runtime
class SuperPointLightGlue {
 public:
  struct Options {
    // Path to the ONNX model file
    std::string model_path = "/usr/local/share/colmap/models/superpoint_lightglue_pipeline.onnx";
    
    // Image resize dimensions (model expects consistent size)
    int image_size = 1024;
    
    // Maximum number of keypoints
    int max_num_keypoints = 1024;
    
    // Number of threads for inference
    int num_threads = 4;
    
    // Whether to use GPU (if available)
    bool use_gpu = true;
    
    // GPU device index
    int gpu_index = 0;
  };

  explicit SuperPointLightGlue(const Options& options);
  ~SuperPointLightGlue();

  // Extract features from two images and match them
  bool ExtractAndMatch(const Bitmap& bitmap1,
                       const Bitmap& bitmap2,
                       FeatureKeypoints* keypoints1,
                       FeatureKeypoints* keypoints2,
                       FeatureMatches* matches);

 private:
  Options options_;
  std::unique_ptr<Ort::Env> env_;
  std::unique_ptr<Ort::Session> session_;
  std::unique_ptr<Ort::SessionOptions> session_options_;
  
  // Preprocess image for ONNX model
  std::vector<float> PreprocessImage(const Bitmap& bitmap);
};

}  // namespace colmap

#endif  // COLMAP_ONNX_ENABLED