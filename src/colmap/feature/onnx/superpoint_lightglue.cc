// Copyright (c) 2024, COLMAP contributors.
// All rights reserved.

#include "colmap/feature/onnx/superpoint_lightglue.h"

#ifdef COLMAP_ONNX_ENABLED

#include "colmap/util/logging.h"
#include "colmap/util/misc.h"
#include "colmap/util/file.h"

#include <opencv2/opencv.hpp>

namespace colmap {

SuperPointLightGlue::SuperPointLightGlue(const Options& options)
    : options_(options) {
  // Initialize ONNX Runtime environment
  env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "SuperPointLightGlue");
  
  // Configure session options
  session_options_ = std::make_unique<Ort::SessionOptions>();
  session_options_->SetIntraOpNumThreads(options_.num_threads);
  session_options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  
  // Enable GPU if requested
  if (options_.use_gpu) {
    try {
      OrtCUDAProviderOptions cuda_options;
      cuda_options.device_id = options_.gpu_index;
      session_options_->AppendExecutionProvider_CUDA(cuda_options);
      LOG(INFO) << "SuperPointLightGlue: Using GPU device " << options_.gpu_index;
    } catch (const Ort::Exception& e) {
      LOG(WARNING) << "Failed to enable CUDA: " << e.what();
      LOG(WARNING) << "Falling back to CPU";
    }
  }
  
  // Load the model
  THROW_CHECK(ExistsFile(options_.model_path))
      << "Model file not found: " << options_.model_path;
  
  try {
    session_ = std::make_unique<Ort::Session>(*env_, options_.model_path.c_str(), *session_options_);
    LOG(INFO) << "SuperPointLightGlue: Model loaded from " << options_.model_path;
  } catch (const Ort::Exception& e) {
    LOG(FATAL) << "Failed to load ONNX model: " << e.what();
  }
}

SuperPointLightGlue::~SuperPointLightGlue() = default;

std::vector<float> SuperPointLightGlue::PreprocessImage(const Bitmap& bitmap) {
  // Convert to OpenCV Mat
  cv::Mat img(bitmap.Height(), bitmap.Width(), CV_8UC3);
  for (int y = 0; y < bitmap.Height(); ++y) {
    for (int x = 0; x < bitmap.Width(); ++x) {
      BitmapColor<uint8_t> color;
      THROW_CHECK(bitmap.GetPixel(x, y, &color));
      img.at<cv::Vec3b>(y, x) = cv::Vec3b(color.b, color.g, color.r);
    }
  }
  
  // Convert to grayscale
  cv::Mat gray;
  cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
  
  // Resize to target size
  cv::Mat resized;
  cv::resize(gray, resized, cv::Size(options_.image_size, options_.image_size));
  
  // Normalize to [0, 1]
  cv::Mat normalized;
  resized.convertTo(normalized, CV_32F, 1.0 / 255.0);
  
  // Convert to vector
  std::vector<float> input_data(options_.image_size * options_.image_size);
  for (int y = 0; y < options_.image_size; ++y) {
    for (int x = 0; x < options_.image_size; ++x) {
      input_data[y * options_.image_size + x] = normalized.at<float>(y, x);
    }
  }
  
  return input_data;
}

bool SuperPointLightGlue::ExtractAndMatch(
    const Bitmap& bitmap1,
    const Bitmap& bitmap2,
    FeatureKeypoints* keypoints1,
    FeatureKeypoints* keypoints2,
    FeatureMatches* matches) {
  
  THROW_CHECK_NOTNULL(keypoints1);
  THROW_CHECK_NOTNULL(keypoints2);
  THROW_CHECK_NOTNULL(matches);
  
  try {
    // Preprocess both images
    auto image1_data = PreprocessImage(bitmap1);
    auto image2_data = PreprocessImage(bitmap2);
    
    // Stack images into batch [2, 1, H, W]
    const int batch_size = 2;
    const int channels = 1;
    const int height = options_.image_size;
    const int width = options_.image_size;
    
    std::vector<float> input_tensor_data(batch_size * channels * height * width);
    
    // Copy image 1
    std::copy(image1_data.begin(), image1_data.end(), input_tensor_data.begin());
    
    // Copy image 2
    std::copy(image2_data.begin(), image2_data.end(), 
              input_tensor_data.begin() + channels * height * width);
    
    // Create input tensor
    std::vector<int64_t> input_shape = {batch_size, channels, height, width};
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_tensor_data.data(),
        input_tensor_data.size(),
        input_shape.data(),
        input_shape.size());
    
    // Get input/output names
    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name = session_->GetInputNameAllocated(0, allocator);
    std::vector<const char*> input_names = {input_name.get()};
    std::vector<const char*> output_names = {"keypoints", "matches", "mscores"};
    
    // Run inference
    auto output_tensors = session_->Run(
        Ort::RunOptions{nullptr},
        input_names.data(),
        &input_tensor,
        1,
        output_names.data(),
        3);
    
    // Extract keypoints (they are INT64, not float!)
    int64_t* keypoints_data = output_tensors[0].GetTensorMutableData<int64_t>();
    auto keypoints_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    
    const int num_keypoints = keypoints_shape[1];  // Should be 1024
    
    // Scale factor to convert from resized coordinates to original coordinates
    const float scale_x1 = static_cast<float>(bitmap1.Width()) / options_.image_size;
    const float scale_y1 = static_cast<float>(bitmap1.Height()) / options_.image_size;
    const float scale_x2 = static_cast<float>(bitmap2.Width()) / options_.image_size;
    const float scale_y2 = static_cast<float>(bitmap2.Height()) / options_.image_size;
    
    // DEBUG: Print raw model output
    LOG(INFO) << "  DEBUG RAW MODEL OUTPUT:";
    LOG(INFO) << StringPrintf("    Keypoint shape: [%lld, %lld, %lld]", 
                             keypoints_shape[0], keypoints_shape[1], keypoints_shape[2]);
    
    
                             LOG(INFO) << "    First 5 raw keypoints from model:";
    for (int i = 0; i < 5; ++i) {
      LOG(INFO) << StringPrintf("      KP %d: (%lld, %lld)", 
                               i, keypoints_data[i*2], keypoints_data[i*2+1]);
    }
    LOG(INFO) << StringPrintf("    Scale factors: x1=%.2f, y1=%.2f, x2=%.2f, y2=%.2f",
                             scale_x1, scale_y1, scale_x2, scale_y2);


    // Extract keypoints for image 1
    keypoints1->resize(num_keypoints);
    for (int i = 0; i < num_keypoints; ++i) {
      (*keypoints1)[i].x = static_cast<float>(keypoints_data[i * 2 + 0]) * scale_x1;
      (*keypoints1)[i].y = static_cast<float>(keypoints_data[i * 2 + 1]) * scale_y1;
    }
    
    // Extract keypoints for image 2
    keypoints2->resize(num_keypoints);
    int offset = num_keypoints * 2;
    for (int i = 0; i < num_keypoints; ++i) {
      (*keypoints2)[i].x = static_cast<float>(keypoints_data[offset + i * 2 + 0]) * scale_x2;
      (*keypoints2)[i].y = static_cast<float>(keypoints_data[offset + i * 2 + 1]) * scale_y2;
    }
    
    // Extract matches
    int64_t* matches_data = output_tensors[1].GetTensorMutableData<int64_t>();
    float* scores_data = output_tensors[2].GetTensorMutableData<float>();
    auto matches_shape = output_tensors[1].GetTensorTypeAndShapeInfo().GetShape();
    
    const int num_matches = matches_shape[0];
    
    // DEBUG: Print raw match data
    LOG(INFO) << StringPrintf("  DEBUG MATCHES: shape=[%lld, %lld], first 5:",
                             matches_shape[0], matches_shape[1]);
    for (int i = 0; i < std::min(5, num_matches); ++i) {
      LOG(INFO) << StringPrintf("    Match %d: [%lld, %lld, %lld]",
                               i, matches_data[i*3], matches_data[i*3+1], matches_data[i*3+2]);
    }
    
    matches->resize(num_matches);
    
    for (int i = 0; i < num_matches; ++i) {
      // Format is [batch_idx, img1_kpt_idx, img2_kpt_idx]
      (*matches)[i].point2D_idx1 = static_cast<point2D_t>(matches_data[i * 3 + 1]);  // Column 1!
      (*matches)[i].point2D_idx2 = static_cast<point2D_t>(matches_data[i * 3 + 2]);  // Column 2!
    }
    
    LOG(INFO) << StringPrintf(
        "SuperPointLightGlue: Extracted %d keypoints per image, found %d matches",
        num_keypoints, num_matches);
    
    return num_matches > 0;
    
  } catch (const Ort::Exception& e) {
    LOG(ERROR) << "ONNX Runtime error: " << e.what();
    return false;
  }
}

}  // namespace colmap

#endif  // COLMAP_ONNX_ENABLED