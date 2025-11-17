// Copyright (c) 2024, COLMAP contributors.
// All rights reserved.

#include "colmap/feature/onnx/superpoint_lightglue.h"

#ifdef COLMAP_ONNX_ENABLED

#include "colmap/util/logging.h"
#include "colmap/util/misc.h"
#include "colmap/util/file.h"

#include <opencv2/opencv.hpp>

namespace colmap {

SuperPointExtractor::SuperPointExtractor(const Options& options)
    : options_(options) {
  // Initialize ONNX Runtime environment
  env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "SuperPoint");
  
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
      LOG(INFO) << "SuperPoint: Using GPU device " << options_.gpu_index;
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
    LOG(INFO) << "SuperPoint: Model loaded from " << options_.model_path;
  } catch (const Ort::Exception& e) {
    LOG(FATAL) << "Failed to load ONNX model: " << e.what();
  }
}

SuperPointExtractor::~SuperPointExtractor() = default;

std::vector<float> SuperPointExtractor::PreprocessImage(const Bitmap& bitmap, int& out_height, int& out_width) {
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
  
  // Use original dimensions
  out_height = gray.rows;
  out_width = gray.cols;
  
  // Normalize to [0, 1]
  cv::Mat normalized;
  gray.convertTo(normalized, CV_32F, 1.0 / 255.0);
  
  // Convert to vector
  std::vector<float> input_data(out_height * out_width);
  for (int y = 0; y < out_height; ++y) {
    for (int x = 0; x < out_width; ++x) {
      input_data[y * out_width + x] = normalized.at<float>(y, x);
    }
  }
  
  return input_data;
}

bool SuperPointExtractor::Extract(
    const Bitmap& bitmap,
    FeatureKeypoints* keypoints,
    FeatureDescriptors* descriptors) {
  
  THROW_CHECK_NOTNULL(keypoints);
  THROW_CHECK_NOTNULL(descriptors);
  
  try {
    // Preprocess image
    int height, width;
    auto image_data = PreprocessImage(bitmap, height, width);
    
    // Create input tensor [1, 1, H, W] - BATCH SIZE = 1
    std::vector<int64_t> input_shape = {1, 1, height, width};
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        image_data.data(),
        image_data.size(),
        input_shape.data(),
        input_shape.size());
    
    // Get input/output names
    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name = session_->GetInputNameAllocated(0, allocator);
    std::vector<const char*> input_names = {input_name.get()};
    std::vector<const char*> output_names = {"keypoints", "scores", "descriptors"};
    
    // Run inference
    auto output_tensors = session_->Run(
        Ort::RunOptions{nullptr},
        input_names.data(),
        &input_tensor,
        1,
        output_names.data(),
        3);
    
    // Extract outputs - KEYPOINTS ARE INT64!
    int64_t* kpts_data = output_tensors[0].GetTensorMutableData<int64_t>();  // Changed from float!
    auto kpts_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    
    // CRITICAL DEBUG: Check raw tensor data
    LOG(INFO) << StringPrintf("  Keypoint tensor shape: [%lld, %lld, %lld]",
                             kpts_shape[0], kpts_shape[1], kpts_shape[2]);
    LOG(INFO) << "  Raw tensor data (first 10 keypoints):";
    for (int i = 0; i < 10; ++i) {
      LOG(INFO) << StringPrintf("    KP %d: [%lld, %lld]", i, kpts_data[i*2], kpts_data[i*2+1]);
    }
    
    float* scores_data = output_tensors[1].GetTensorMutableData<float>();
    float* desc_data = output_tensors[2].GetTensorMutableData<float>();
    auto desc_shape = output_tensors[2].GetTensorTypeAndShapeInfo().GetShape();
    
    const int num_keypoints = kpts_shape[1];
    const int descriptor_dim = desc_shape[2];
    
    LOG(INFO) << StringPrintf("  SuperPoint: Extracted %d keypoints (dim=%d) from [%dx%d] image", 
                             num_keypoints, descriptor_dim, width, height);
    LOG(INFO) << StringPrintf("  First keypoint: [%lld, %lld]",
                             kpts_data[0], kpts_data[1]);
    
    // Copy keypoints - convert from int64 to float
    keypoints->resize(num_keypoints);
    for (int i = 0; i < num_keypoints; ++i) {
      (*keypoints)[i].x = static_cast<float>(kpts_data[i * 2 + 0]);
      (*keypoints)[i].y = static_cast<float>(kpts_data[i * 2 + 1]);
    }
    
    // Copy descriptors - convert float32 to uint8
    descriptors->resize(num_keypoints, descriptor_dim);
    for (int i = 0; i < num_keypoints; ++i) {
      for (int j = 0; j < descriptor_dim; ++j) {
        float val = desc_data[i * descriptor_dim + j];
        // Descriptors are L2-normalized, typically in [-1, 1] range
        // Scale to [0, 255]
        (*descriptors)(i, j) = static_cast<uint8_t>(
            std::clamp((val + 1.0f) * 127.5f, 0.0f, 255.0f));
      }
    }
    
    return num_keypoints > 0;
    
  } catch (const Ort::Exception& e) {
    LOG(ERROR) << "ONNX Runtime error: " << e.what();
    return false;
  }
}

void DescriptorMatcher::MatchDescriptors(
    const FeatureDescriptors& descriptors1,
    const FeatureDescriptors& descriptors2,
    FeatureMatches* matches,
    float ratio_test_threshold) {
  
  matches->clear();
  
  // Convert uint8 descriptors back to normalized float
  Eigen::MatrixXf desc1_float = (descriptors1.cast<float>().array() / 127.5f - 1.0f).matrix();
  Eigen::MatrixXf desc2_float = (descriptors2.cast<float>().array() / 127.5f - 1.0f).matrix();
  
  const int num1 = desc1_float.rows();
  const int num2 = desc2_float.rows();
  
  // Forward matching with ratio test
  std::vector<int> forward_matches(num1, -1);
  for (int i = 0; i < num1; ++i) {
    float best_dist = std::numeric_limits<float>::max();
    float second_best_dist = std::numeric_limits<float>::max();
    int best_idx = -1;
    
    for (int j = 0; j < num2; ++j) {
      float dist = (desc1_float.row(i) - desc2_float.row(j)).squaredNorm();
      if (dist < best_dist) {
        second_best_dist = best_dist;
        best_dist = dist;
        best_idx = j;
      } else if (dist < second_best_dist) {
        second_best_dist = dist;
      }
    }
    
    // Ratio test
    if (best_idx >= 0 && best_dist < ratio_test_threshold * second_best_dist) {
      forward_matches[i] = best_idx;
    }
  }
  
  // Backward matching
  std::vector<int> backward_matches(num2, -1);
  for (int j = 0; j < num2; ++j) {
    float best_dist = std::numeric_limits<float>::max();
    int best_idx = -1;
    
    for (int i = 0; i < num1; ++i) {
      float dist = (desc2_float.row(j) - desc1_float.row(i)).squaredNorm();
      if (dist < best_dist) {
        best_dist = dist;
        best_idx = i;
      }
    }
    
    backward_matches[j] = best_idx;
  }
  
  // Mutual nearest neighbors
  for (int i = 0; i < num1; ++i) {
    if (forward_matches[i] >= 0 && backward_matches[forward_matches[i]] == i) {
      FeatureMatch match;
      match.point2D_idx1 = i;
      match.point2D_idx2 = forward_matches[i];
      matches->push_back(match);
    }
  }
}

}  // namespace colmap

#endif  // COLMAP_ONNX_ENABLED