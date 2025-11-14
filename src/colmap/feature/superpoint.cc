// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
// (standard copyright header...)

#include "colmap/feature/superpoint.h"

#include "colmap/util/logging.h"
#include "colmap/util/misc.h"

#include <cstdlib>
#include <fstream>
#include <sstream>

#include "colmap/util/file.h"

namespace colmap {

bool SuperPointExtractionOptions::Check() const {
  CHECK_OPTION_GT(max_keypoints, 0);
  CHECK_OPTION(!superglue_path.empty());
  return true;
}

SuperPointFeatureExtractor::SuperPointFeatureExtractor(
    const SuperPointExtractionOptions& options)
    : options_(options) {
  // Setup paths
  python_path_ = options_.superglue_path + "/venv/bin/python";
  script_path_ = options_.superglue_path + "/extract_features.py";
  
  // Verify paths exist
  if (!ExistsFile(python_path_)) {
    LOG(WARNING) << "Python not found at: " << python_path_;
  }
  if (!ExistsFile(script_path_)) {
    LOG(WARNING) << "Script not found at: " << script_path_;
  }
}

std::unique_ptr<FeatureExtractor> SuperPointFeatureExtractor::Create(
    const FeatureExtractionOptions& options) {
  SuperPointExtractionOptions superpoint_options;
  
  // Get SuperGlue path from environment or use default
  const char* env_path = std::getenv("SUPERGLUE_PATH");
  if (env_path != nullptr) {
    superpoint_options.superglue_path = env_path;
  } else {
    // Default path - we'll set this from the user's home directory
    const char* home = std::getenv("HOME");
    if (home != nullptr) {
      superpoint_options.superglue_path = std::string(home) + "/SuperGluePretrainedNetwork";
    }
  }
  
  superpoint_options.max_keypoints = 2048;  // Default
  
  return std::make_unique<SuperPointFeatureExtractor>(superpoint_options);
}

bool SuperPointFeatureExtractor::Extract(const Bitmap& bitmap,
                                         FeatureKeypoints* keypoints,
                                         FeatureDescriptors* descriptors) {
  // Save image to temp file
  const std::string temp_image_path = "/tmp/colmap_superpoint_input.png";
  const std::string temp_features_path = "/tmp/colmap_superpoint_output.txt";
  
  if (!bitmap.Write(temp_image_path)) {
    LOG(ERROR) << "Failed to write temporary image";
    return false;
  }
  
  // Build command
  std::stringstream cmd;
  cmd << python_path_ << " " << script_path_ 
      << " " << temp_image_path
      << " " << temp_features_path
      << " --max_keypoints " << options_.max_keypoints
      << " 2>&1";  // Redirect stderr to stdout
  
  LOG(INFO) << "Running: " << cmd.str();
  
  // Execute Python script
  FILE* pipe = popen(cmd.str().c_str(), "r");
  if (!pipe) {
    LOG(ERROR) << "Failed to run SuperPoint extraction";
    return false;
  }
  
  // Read output
  char buffer[256];
  std::string result;
  while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
    result += buffer;
  }
  
  int return_code = pclose(pipe);
  if (return_code != 0) {
    LOG(ERROR) << "SuperPoint extraction failed with code: " << return_code;
    LOG(ERROR) << "Output: " << result;
    return false;
  }
  
  LOG(INFO) << "SuperPoint output: " << result;
  
  // Parse results
  std::ifstream file(temp_features_path);
  if (!file.is_open()) {
    LOG(ERROR) << "Failed to open features file";
    return false;
  }
  
  size_t num_features;
  file >> num_features;
  
  keypoints->resize(num_features);
  descriptors->resize(num_features, 256);
  
  for (size_t i = 0; i < num_features; ++i) {
    float x, y, score;
    file >> x >> y >> score;
    
    (*keypoints)[i].x = x;
    (*keypoints)[i].y = y;
    
    // Read 256-dimensional descriptor
    for (int j = 0; j < 256; ++j) {
      file >> (*descriptors)(i, j);
    }
  }
  
  file.close();
  
  LOG(INFO) << "Extracted " << num_features << " SuperPoint features";
  
  return true;
}

}  // namespace colmap