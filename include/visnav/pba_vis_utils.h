#pragma once

#include <iostream>
#include <opencv4/opencv2/core/mat.hpp>
#include <opencv4/opencv2/imgproc.hpp>

void depth_color_code(const double& depth, double rgb[3]) {
    double min_inv_depth = 0.1;
    double max_inv_depth = 9.;

    // clamp inv depth
    double clamped = depth;
    if(clamped < min_inv_depth) clamped = min_inv_depth;
    if(clamped > max_inv_depth) clamped = max_inv_depth;

    // Normalize inv_depth, range [0.02,10]
    int normed_depth = (clamped - min_inv_depth) / (max_inv_depth - min_inv_depth) * 255;
    cv::Mat depth_(1, 1, CV_8UC1);
    depth_.at<int>(0,0) = normed_depth;
    cv::Mat color;
    cv::applyColorMap(depth_, color, cv::COLORMAP_JET);

    rgb[0] = color.at<cv::Vec3b>(0,0)[0] / 255.;
    rgb[1] = color.at<cv::Vec3b>(0,0)[1] / 255.;
    rgb[2] = color.at<cv::Vec3b>(0,0)[2] / 255.;
}

