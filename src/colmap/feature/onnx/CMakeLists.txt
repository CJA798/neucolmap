if(ONNXRUNTIME_FOUND)
    set(FOLDER_NAME "feature/onnx")

    # Find OpenCV
    find_package(OpenCV REQUIRED COMPONENTS core imgproc)

    COLMAP_ADD_LIBRARY(
        NAME colmap_feature_onnx
        SRCS
            superpoint_lightglue.h
            superpoint_lightglue.cc
            disk_lightglue.h
            disk_lightglue.cc
        PUBLIC_LINK_LIBS
            colmap_feature
            colmap_util
            colmap_sensor
            ONNXRuntime::ONNXRuntime
            ${OpenCV_LIBS}
        PRIVATE_LINK_LIBS
    )

    target_include_directories(colmap_feature_onnx PRIVATE ${OpenCV_INCLUDE_DIRS})

    COLMAP_ADD_TEST(
        NAME feature_onnx_test
        SRCS superpoint_lightglue_test.cc
        LINK_LIBS colmap_feature_onnx
    )
endif()