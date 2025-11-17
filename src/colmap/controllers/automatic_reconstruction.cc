#include "colmap/controllers/automatic_reconstruction.h"

#include "colmap/controllers/feature_extraction.h"
#include "colmap/controllers/feature_matching.h"
#include "colmap/controllers/incremental_pipeline.h"
#include "colmap/controllers/option_manager.h"
#include "colmap/image/undistortion.h"
#include "colmap/mvs/fusion.h"
#include "colmap/mvs/meshing.h"
#include "colmap/mvs/patch_match.h"
#include "colmap/scene/database.h"
#include "colmap/util/logging.h"

#ifdef COLMAP_ONNX_ENABLED
#include "colmap/feature/onnx/superpoint_lightglue.h"
#endif

#include "colmap/util/string.h"
#include "colmap/util/file.h"
#include "colmap/estimators/two_view_geometry.h"

namespace colmap {

AutomaticReconstructionController::AutomaticReconstructionController(
    const Options& options,
    std::shared_ptr<ReconstructionManager> reconstruction_manager)
    : options_(options),
      reconstruction_manager_(std::move(reconstruction_manager)),
      active_thread_(nullptr) {
  THROW_CHECK_DIR_EXISTS(options_.workspace_path);
  THROW_CHECK_DIR_EXISTS(options_.image_path);
  THROW_CHECK_NOTNULL(reconstruction_manager_);

  option_manager_.AddAllOptions();

  *option_manager_.image_path = options_.image_path;
  option_manager_.image_reader->image_names = options_.image_names;
  option_manager_.mapper->image_names = {options_.image_names.begin(),
                                         options_.image_names.end()};
  *option_manager_.database_path =
      JoinPaths(options_.workspace_path, "database.db");

  if (options_.data_type == DataType::VIDEO) {
    option_manager_.ModifyForVideoData();
  } else if (options_.data_type == DataType::INDIVIDUAL) {
    option_manager_.ModifyForIndividualData();
  } else if (options_.data_type == DataType::INTERNET) {
    option_manager_.ModifyForInternetData();
  } else {
    LOG(FATAL_THROW) << "Data type not supported";
  }

  THROW_CHECK(ExistsCameraModelWithName(options_.camera_model));

  if (options_.quality == Quality::LOW) {
    option_manager_.ModifyForLowQuality();
  } else if (options_.quality == Quality::MEDIUM) {
    option_manager_.ModifyForMediumQuality();
  } else if (options_.quality == Quality::HIGH) {
    option_manager_.ModifyForHighQuality();
  } else if (options_.quality == Quality::EXTREME) {
    option_manager_.ModifyForExtremeQuality();
  }



  // Log selected matching approach
  switch (options_.matching_approach) {
    case Options::MatchingApproach::DEFAULT_SIFT:
      LOG(INFO) << "Using Default SIFT pipeline";
      break;
      
    case Options::MatchingApproach::SUPERPOINT_LIGHTGLUE:
      LOG(INFO) << "Using SuperPoint + LightGlue pipeline";
      break;
      
    case Options::MatchingApproach::XFEAT:
      LOG(INFO) << "Using XFeat pipeline";
      LOG(WARNING) << "XFeat not yet implemented, falling back to SIFT";
      break;
      
    case Options::MatchingApproach::DISK:
      LOG(INFO) << "Using DISK pipeline";
      LOG(WARNING) << "DISK not yet implemented, falling back to SIFT";
      break;
  }


  option_manager_.feature_extraction->num_threads = options_.num_threads;
  option_manager_.feature_matching->num_threads = options_.num_threads;
  option_manager_.sequential_pairing->num_threads = options_.num_threads;
  option_manager_.vocab_tree_pairing->num_threads = options_.num_threads;
  option_manager_.mapper->num_threads = options_.num_threads;
  option_manager_.poisson_meshing->num_threads = options_.num_threads;

  option_manager_.two_view_geometry->ransac_options.random_seed =
      options_.random_seed;
  option_manager_.mapper->random_seed = options_.random_seed;

  ImageReaderOptions& reader_options = *option_manager_.image_reader;
  reader_options.image_path = *option_manager_.image_path;
  reader_options.as_rgb = option_manager_.feature_extraction->RequiresRGB();
  if (!options_.mask_path.empty()) {
    reader_options.mask_path = options_.mask_path;
    option_manager_.image_reader->mask_path = options_.mask_path;
    option_manager_.stereo_fusion->mask_path = options_.mask_path;
  }
  reader_options.single_camera = options_.single_camera;
  reader_options.single_camera_per_folder = options_.single_camera_per_folder;
  reader_options.camera_model = options_.camera_model;
  reader_options.camera_params = options_.camera_params;

  option_manager_.feature_extraction->use_gpu = options_.use_gpu;
  option_manager_.feature_matching->use_gpu = options_.use_gpu;
  option_manager_.mapper->ba_use_gpu = options_.use_gpu;
  option_manager_.bundle_adjustment->use_gpu = options_.use_gpu;

  option_manager_.feature_extraction->gpu_index = options_.gpu_index;
  option_manager_.feature_matching->gpu_index = options_.gpu_index;
  option_manager_.patch_match_stereo->gpu_index = options_.gpu_index;
  option_manager_.mapper->ba_gpu_index = options_.gpu_index;
  option_manager_.bundle_adjustment->gpu_index = options_.gpu_index;

  if (options_.extraction) {
    feature_extractor_ =
        CreateFeatureExtractorController(*option_manager_.database_path,
                                         reader_options,
                                         *option_manager_.feature_extraction);
  }

  if (options_.matching) {
    exhaustive_matcher_ =
        CreateExhaustiveFeatureMatcher(*option_manager_.exhaustive_pairing,
                                       *option_manager_.feature_matching,
                                       *option_manager_.two_view_geometry,
                                       *option_manager_.database_path);

    if (!options_.vocab_tree_path.empty()) {
      option_manager_.sequential_pairing->loop_detection = true;
      option_manager_.sequential_pairing->vocab_tree_path =
          options_.vocab_tree_path;
    }

    sequential_matcher_ =
        CreateSequentialFeatureMatcher(*option_manager_.sequential_pairing,
                                       *option_manager_.feature_matching,
                                       *option_manager_.two_view_geometry,
                                       *option_manager_.database_path);

    if (!options_.vocab_tree_path.empty()) {
      option_manager_.vocab_tree_pairing->vocab_tree_path =
          options_.vocab_tree_path;
      vocab_tree_matcher_ =
          CreateVocabTreeFeatureMatcher(*option_manager_.vocab_tree_pairing,
                                        *option_manager_.feature_matching,
                                        *option_manager_.two_view_geometry,
                                        *option_manager_.database_path);
    }
  }
}

void AutomaticReconstructionController::Stop() {
  if (active_thread_ != nullptr) {
    active_thread_->Stop();
  }
  Thread::Stop();
}


#ifdef COLMAP_ONNX_ENABLED
void AutomaticReconstructionController::RunMLFeatureExtractionAndMatching() {
  LOG(INFO) << "Starting ML-based feature extraction and matching";
  
  // Initialize SuperPoint extractor
  SuperPointExtractor::Options ml_options;
  ml_options.use_gpu = options_.use_gpu;
  ml_options.num_threads = options_.num_threads;
  
  SuperPointExtractor extractor(ml_options);
  
  // Open database
  auto database = Database::Open(*option_manager_.database_path);
  
  // Get image paths
  std::vector<std::string> image_paths = GetRecursiveFileList(options_.image_path);
  std::sort(image_paths.begin(), image_paths.end());
  
  // Filter to only image files
  std::vector<std::string> valid_image_paths;
  for (const auto& path : image_paths) {
    if (HasFileExtension(path, ".jpg") || HasFileExtension(path, ".jpeg") || 
        HasFileExtension(path, ".png") || HasFileExtension(path, ".JPG")) {
      valid_image_paths.push_back(path);
    }
  }
  
  if (valid_image_paths.empty()) {
    LOG(ERROR) << "No valid images found in: " << options_.image_path;
    return;
  }
  
  LOG(INFO) << "Found " << valid_image_paths.size() << " images";
  
  // Load images and register in database
  std::vector<image_t> image_ids;
  std::vector<Bitmap> bitmaps;
  std::vector<FeatureKeypoints> all_keypoints;
  std::vector<FeatureDescriptors> all_descriptors;
  
  camera_t shared_camera_id = kInvalidCameraId;
  
  for (size_t i = 0; i < valid_image_paths.size(); ++i) {
    const std::string& path = valid_image_paths[i];
    const std::string name = GetPathBaseName(path);
    
    LOG(INFO) << StringPrintf("Processing image [%d/%d]: %s", 
                              i + 1, valid_image_paths.size(), name.c_str());
    
    Bitmap bitmap;
    if (!bitmap.Read(path)) {
      LOG(WARNING) << "Failed to load image: " << path;
      continue;
    }
    
    // Create or get camera
    camera_t camera_id;
    if (options_.single_camera && shared_camera_id != kInvalidCameraId) {
      camera_id = shared_camera_id;
    } else {
      Camera camera;
      camera.model_id = CameraModelNameToId(options_.camera_model);
      camera.width = bitmap.Width();
      camera.height = bitmap.Height();
      
      // Initialize camera parameters
      const double focal_length = std::max(bitmap.Width(), bitmap.Height()) * 1.2;
      const double cx = bitmap.Width() / 2.0;
      const double cy = bitmap.Height() / 2.0;
      
      if (camera.ModelName() == "SIMPLE_RADIAL") {
        camera.params = {focal_length, cx, cy, 0.0};
      } else if (camera.ModelName() == "PINHOLE") {
        camera.params = {focal_length, focal_length, cx, cy};
      } else if (camera.ModelName() == "SIMPLE_PINHOLE") {
        camera.params = {focal_length, cx, cy};
      } else {
        LOG(FATAL) << "Unsupported camera model: " << camera.ModelName();
      }
      
      camera_id = database->WriteCamera(camera);
      
      if (options_.single_camera && shared_camera_id == kInvalidCameraId) {
        shared_camera_id = camera_id;
      }
    }
    
    // Register image in database
    Image image;
    image.SetName(name);
    image.SetCameraId(camera_id);
    image_t image_id = database->WriteImage(image);
    
    // Extract features with SuperPoint
    FeatureKeypoints keypoints;
    FeatureDescriptors descriptors;
    
    if (!extractor.Extract(bitmap, &keypoints, &descriptors)) {
      LOG(WARNING) << "Failed to extract features";
      continue;
    }
    
    // Write to database
    database->WriteKeypoints(image_id, keypoints);
    database->WriteDescriptors(image_id, descriptors);
    
    LOG(INFO) << StringPrintf("  Extracted %d keypoints", keypoints.size());
    
    // Store for matching
    image_ids.push_back(image_id);
    all_keypoints.push_back(keypoints);
    all_descriptors.push_back(descriptors);
  }
  
  const size_t num_images = image_ids.size();
  LOG(INFO) << "Registered " << num_images << " images in database";
  
  // Match all pairs
  LOG(INFO) << "Matching image pairs with geometric verification...";
  size_t num_total_matches = 0;
  size_t num_verified_pairs = 0;
  
  for (size_t i = 0; i < num_images; ++i) {
    for (size_t j = i + 1; j < num_images; ++j) {
      LOG(INFO) << StringPrintf("Matching images [%d-%d]", i, j);
      
      // Match descriptors
      FeatureMatches matches;
      DescriptorMatcher::MatchDescriptors(all_descriptors[i], all_descriptors[j], 
                                         &matches, 0.8);
      
      if (matches.empty()) {
        LOG(WARNING) << "No matches found";
        continue;
      }
      
      LOG(INFO) << StringPrintf("  Found %d descriptor matches", matches.size());
      
      // Write raw matches
      database->WriteMatches(image_ids[i], image_ids[j], matches);
      
      // Convert keypoints to Eigen::Vector2d for geometric verification
      std::vector<Eigen::Vector2d> points1, points2;
      points1.reserve(all_keypoints[i].size());
      points2.reserve(all_keypoints[j].size());
      
      for (const auto& kp : all_keypoints[i]) {
        points1.emplace_back(kp.x, kp.y);
      }
      for (const auto& kp : all_keypoints[j]) {
        points2.emplace_back(kp.x, kp.y);
      }
      
      // Perform geometric verification
      const Camera& camera1 = database->ReadCamera(
          database->ReadImage(image_ids[i]).CameraId());
      const Camera& camera2 = database->ReadCamera(
          database->ReadImage(image_ids[j]).CameraId());
      
      TwoViewGeometryOptions two_view_options;
      two_view_options.ransac_options = option_manager_.two_view_geometry->ransac_options;
      
      TwoViewGeometry two_view_geometry = EstimateTwoViewGeometry(
          camera1, points1,
          camera2, points2,
          matches,
          two_view_options);
      
      // Write verified geometry
      database->WriteTwoViewGeometry(image_ids[i], image_ids[j], two_view_geometry);
      
      num_total_matches += matches.size();
      
      if (two_view_geometry.inlier_matches.size() >= 
          static_cast<size_t>(two_view_options.min_num_inliers)) {
        num_verified_pairs++;
        LOG(INFO) << StringPrintf("  ✅ Verified: %d inliers (config: %d)", 
                                 two_view_geometry.inlier_matches.size(),
                                 static_cast<int>(two_view_geometry.config));
      } else {
        LOG(INFO) << StringPrintf("  ❌ Failed: %d inliers (config: %d)", 
                                 two_view_geometry.inlier_matches.size(),
                                 static_cast<int>(two_view_geometry.config));
      }
    }
  }
  
  LOG(INFO) << StringPrintf("ML feature extraction complete: %d images, %d matches, %d verified pairs", 
                           num_images, num_total_matches, num_verified_pairs);
}
#endif


void AutomaticReconstructionController::Run() {
  if (IsStopped()) {
    return;
  }

  #ifdef COLMAP_ONNX_ENABLED
    // Check if using ML-based features
    if (options_.matching_approach != Options::MatchingApproach::DEFAULT_SIFT) {
      if (options_.extraction && options_.matching) {
        RunMLFeatureExtractionAndMatching();
      }
      
      if (IsStopped()) {
        return;
      }
      
      if (options_.sparse) {
        RunSparseMapper();
      }
      
      if (IsStopped()) {
        return;
      }
      
      if (options_.dense) {
        RunDenseMapper();
      }
      
      return;
    }
  #endif


  if (options_.extraction) {
    RunFeatureExtraction();
  }

  if (IsStopped()) {
    return;
  }

  if (options_.matching) {
    RunFeatureMatching();
  }

  if (IsStopped()) {
    return;
  }

  if (options_.sparse) {
    RunSparseMapper();
  }

  if (IsStopped()) {
    return;
  }

  if (options_.dense) {
    RunDenseMapper();
  }
}

void AutomaticReconstructionController::RunFeatureExtraction() {
  THROW_CHECK_NOTNULL(feature_extractor_);
  active_thread_ = feature_extractor_.get();
  feature_extractor_->Start();
  feature_extractor_->Wait();
  feature_extractor_.reset();
  active_thread_ = nullptr;
}

void AutomaticReconstructionController::RunFeatureMatching() {
  Thread* matcher = nullptr;
  if (options_.data_type == DataType::VIDEO) {
    matcher = sequential_matcher_.get();
  } else if (options_.data_type == DataType::INDIVIDUAL ||
             options_.data_type == DataType::INTERNET) {
    auto database = Database::Open(*option_manager_.database_path);
    const size_t num_images = database->NumImages();
    if (options_.vocab_tree_path.empty() || num_images < 200) {
      matcher = exhaustive_matcher_.get();
    } else {
      matcher = vocab_tree_matcher_.get();
    }
  }

  THROW_CHECK_NOTNULL(matcher);
  active_thread_ = matcher;
  matcher->Start();
  matcher->Wait();
  exhaustive_matcher_.reset();
  sequential_matcher_.reset();
  vocab_tree_matcher_.reset();
  active_thread_ = nullptr;
}

void AutomaticReconstructionController::RunSparseMapper() {
  const auto sparse_path = JoinPaths(options_.workspace_path, "sparse");
  if (ExistsDir(sparse_path)) {
    auto dir_list = GetDirList(sparse_path);
    std::sort(dir_list.begin(), dir_list.end());
    if (dir_list.size() > 0) {
      LOG(WARNING)
          << "Skipping sparse reconstruction because it is already computed";
      for (const auto& dir : dir_list) {
        reconstruction_manager_->Read(dir);
      }
      return;
    }
  }

  IncrementalPipeline mapper(option_manager_.mapper,
                             *option_manager_.image_path,
                             *option_manager_.database_path,
                             reconstruction_manager_);
  mapper.SetCheckIfStoppedFunc([&]() { return IsStopped(); });
  mapper.Run();

  CreateDirIfNotExists(sparse_path);
  reconstruction_manager_->Write(sparse_path);
  option_manager_.Write(JoinPaths(sparse_path, "project.ini"));
}

void AutomaticReconstructionController::RunDenseMapper() {
  CreateDirIfNotExists(JoinPaths(options_.workspace_path, "dense"));

  for (size_t i = 0; i < reconstruction_manager_->Size(); ++i) {
    if (IsStopped()) {
      return;
    }

    const std::string dense_path =
        JoinPaths(options_.workspace_path, "dense", std::to_string(i));
    const std::string fused_path = JoinPaths(dense_path, "fused.ply");

    std::string meshing_path;
    if (options_.mesher == Mesher::POISSON) {
      meshing_path = JoinPaths(dense_path, "meshed-poisson.ply");
    } else if (options_.mesher == Mesher::DELAUNAY) {
      meshing_path = JoinPaths(dense_path, "meshed-delaunay.ply");
    }

    if (ExistsFile(fused_path) && ExistsFile(meshing_path)) {
      continue;
    }

    // Image undistortion.

    if (!ExistsDir(dense_path)) {
      CreateDirIfNotExists(dense_path);

      UndistortCameraOptions undistortion_options;
      undistortion_options.max_image_size =
          option_manager_.patch_match_stereo->max_image_size;
      COLMAPUndistorter undistorter(undistortion_options,
                                    *reconstruction_manager_->Get(i),
                                    *option_manager_.image_path,
                                    dense_path);
      undistorter.SetCheckIfStoppedFunc([&]() { return IsStopped(); });
      undistorter.Run();
    }

    if (IsStopped()) {
      return;
    }

    // Patch match stereo.

#if defined(COLMAP_CUDA_ENABLED)
    {
      mvs::PatchMatchController patch_match_controller(
          *option_manager_.patch_match_stereo, dense_path, "COLMAP", "");
      patch_match_controller.SetCheckIfStoppedFunc(
          [&]() { return IsStopped(); });
      patch_match_controller.Run();
    }
#else   // COLMAP_CUDA_ENABLED
    LOG(WARNING) << "Skipping patch match stereo because CUDA is not available";
    return;
#endif  // COLMAP_CUDA_ENABLED

    if (IsStopped()) {
      return;
    }

    // Stereo fusion.

    if (!ExistsFile(fused_path)) {
      auto fusion_options = *option_manager_.stereo_fusion;
      const int num_reg_images =
          reconstruction_manager_->Get(i)->NumRegImages();
      fusion_options.min_num_pixels =
          std::min(num_reg_images + 1, fusion_options.min_num_pixels);
      mvs::StereoFusion fuser(
          fusion_options,
          dense_path,
          "COLMAP",
          "",
          option_manager_.patch_match_stereo->geom_consistency ? "geometric"
                                                               : "photometric");
      fuser.SetCheckIfStoppedFunc([&]() { return IsStopped(); });
      fuser.Run();

      LOG(INFO) << "Writing output: " << fused_path;
      WriteBinaryPlyPoints(fused_path, fuser.GetFusedPoints());
      mvs::WritePointsVisibility(fused_path + ".vis",
                                 fuser.GetFusedPointsVisibility());
    }

    if (IsStopped()) {
      return;
    }

    // Surface meshing.

    if (!ExistsFile(meshing_path)) {
      if (options_.mesher == Mesher::POISSON) {
        mvs::PoissonMeshing(
            *option_manager_.poisson_meshing, fused_path, meshing_path);
      } else if (options_.mesher == Mesher::DELAUNAY) {
#if defined(COLMAP_CGAL_ENABLED)
        mvs::DenseDelaunayMeshing(
            *option_manager_.delaunay_meshing, dense_path, meshing_path);
#else  // COLMAP_CGAL_ENABLED
        LOG(WARNING)
            << "Skipping Delaunay meshing because CGAL is not available";
        return;

#endif  // COLMAP_CGAL_ENABLED
      }
    }
  }
}

}  // namespace colmap
