// NEUCOLMAP

#pragma once

#include "colmap/feature/extractor.h"
#include "colmap/sensor/bitmap.h"
#include "colmap/util/logging.h"

#include <memory>
#include <string>

namespace colmap {

struct SuperPointExtractionOptions {
  // Path to SuperGlue installation
  std::string superglue_path = "";
  
  // Maximum number of keypoints
  int max_keypoints = 2048;
  
  bool Check() const;
};

class SuperPointFeatureExtractor : public FeatureExtractor {
 public:
  explicit SuperPointFeatureExtractor(const SuperPointExtractionOptions& options);
  
  static std::unique_ptr<FeatureExtractor> Create(
      const FeatureExtractionOptions& options);
  
  bool Extract(const Bitmap& bitmap,
               FeatureKeypoints* keypoints,
               FeatureDescriptors* descriptors) override;

 private:
  SuperPointExtractionOptions options_;
  std::string python_path_;
  std::string script_path_;
};

}  // namespace colmap