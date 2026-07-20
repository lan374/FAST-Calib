/* 
Developer: Chunran Zheng <zhengcr@connect.hku.hk>

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#include "qr_detect.hpp"
#include "lidar_detect.hpp"
#include "data_preprocess.hpp"
#include <limits>

int main(int argc, char **argv) 
{
    ros::init(argc, argv, "mono_qr_pattern");
    ros::NodeHandle nh;

    // 读取参数
    Params params = loadParameters(nh);

    // 初始化 QR 检测和 LiDAR 检测
    QRDetectPtr qrDetectPtr;
    qrDetectPtr.reset(new QRDetect(nh, params));

    LidarDetectPtr lidarDetectPtr;
    lidarDetectPtr.reset(new LidarDetect(nh, params));

    DataPreprocessPtr dataPreprocessPtr;
    dataPreprocessPtr.reset(new DataPreprocess(params));

    // 读取图像和点云
    cv::Mat img_input = dataPreprocessPtr->img_input_;
    pcl::PointCloud<Common::Point>::Ptr cloud_input = dataPreprocessPtr->cloud_input_;
    
    // 检测 QR 码
    PointCloud<PointXYZ>::Ptr qr_center_cloud(new PointCloud<PointXYZ>);
    qr_center_cloud->reserve(4);
    qrDetectPtr->detect_qr(img_input, qr_center_cloud);

    // 检测 LiDAR 数据
    PointCloud<PointXYZ>::Ptr lidar_center_cloud(new PointCloud<PointXYZ>);
    lidar_center_cloud->reserve(4);
    
    switch (dataPreprocessPtr->lidar_type_)
    {
        case LiDARType::Solid:
            lidarDetectPtr->detect_solid_lidar(cloud_input, lidar_center_cloud);
            break;

        case LiDARType::Mech:
            lidarDetectPtr->detect_mech_lidar(cloud_input, lidar_center_cloud);
            break;

        default:
            std::cerr << BOLDYELLOW 
                    << "[Main] Unknown LiDAR type." 
                    << RESET << std::endl;
            break;
    }

    // 对 QR 和 LiDAR 检测到的圆心进行排序
    PointCloud<PointXYZ>::Ptr qr_centers_sorted(new PointCloud<PointXYZ>);
    PointCloud<PointXYZ>::Ptr lidar_centers(new PointCloud<PointXYZ>);
    sortPatternCenters(qr_center_cloud, qr_centers_sorted, "camera");
    sortPatternCenters(lidar_center_cloud, lidar_centers, "lidar");

    // 相机滚转安装时极角排序会循环错位；穷举 4 种移位，取 RMSE 最小的配对
    PointCloud<PointXYZ>::Ptr qr_centers(new PointCloud<PointXYZ>);
    Eigen::Matrix4f transformation = Eigen::Matrix4f::Identity();
    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_lidar_centers(new pcl::PointCloud<pcl::PointXYZ>);
    double rmse = -1.0;

    if (qr_centers_sorted->size() == 4 && lidar_centers->size() == 4)
    {
      pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ> svd;
      int best_shift = 0;
      double best_rmse = std::numeric_limits<double>::max();
      Eigen::Matrix4f best_T = Eigen::Matrix4f::Identity();
      PointCloud<PointXYZ>::Ptr best_qr(new PointCloud<PointXYZ>);

      std::cout << BOLDYELLOW << "[Pairing] Trying 4 cyclic shifts of camera centers:" << RESET << std::endl;
      for (int shift = 0; shift < 4; ++shift)
      {
        PointCloud<PointXYZ>::Ptr qr_shifted(new PointCloud<PointXYZ>);
        qr_shifted->resize(4);
        for (int i = 0; i < 4; ++i)
          (*qr_shifted)[i] = (*qr_centers_sorted)[(i + shift) % 4];

        Eigen::Matrix4f T;
        svd.estimateRigidTransformation(*lidar_centers, *qr_shifted, T);

        pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZ>);
        alignPointCloud(lidar_centers, aligned, T);
        double r = computeRMSE(qr_shifted, aligned);

        std::cout << BOLDYELLOW << "  shift=" << shift << " RMSE=" << BOLDRED
                  << std::fixed << std::setprecision(4) << r << " m" << RESET << std::endl;

        if (r >= 0.0 && r < best_rmse)
        {
          best_rmse = r;
          best_shift = shift;
          best_T = T;
          *best_qr = *qr_shifted;
        }
      }

      *qr_centers = *best_qr;
      transformation = best_T;
      rmse = best_rmse;
      std::cout << BOLDGREEN << "[Pairing] Selected shift=" << best_shift
                << " with RMSE=" << std::fixed << std::setprecision(4) << rmse
                << " m" << RESET << std::endl;
    }
    else
    {
      *qr_centers = *qr_centers_sorted;
      pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ> svd;
      svd.estimateRigidTransformation(*lidar_centers, *qr_centers, transformation);
      alignPointCloud(lidar_centers, aligned_lidar_centers, transformation);
      rmse = computeRMSE(qr_centers, aligned_lidar_centers);
    }

    // 保存修正配对后的圆心（供多场景联合标定使用）
    saveTargetHoleCenters(lidar_centers, qr_centers, params);

    aligned_lidar_centers->clear();
    aligned_lidar_centers->reserve(lidar_centers->size());
    alignPointCloud(lidar_centers, aligned_lidar_centers, transformation);

    if (rmse > 0)
    {
      std::cout << BOLDYELLOW << "[Result] RMSE: " << BOLDRED << std::fixed << std::setprecision(4)
      << rmse << " m" << RESET << std::endl;
    }

    std::cout << BOLDYELLOW << "[Result] Single-scene calibration: extrinsic parameters T_cam_lidar = " << RESET << std::endl;
    std::cout << BOLDCYAN << std::fixed << std::setprecision(6) << transformation << RESET << std::endl;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    projectPointCloudToImage(cloud_input, transformation, qrDetectPtr->cameraMatrix_, qrDetectPtr->distCoeffs_, img_input, colored_cloud);

    saveCalibrationResults(params, transformation, colored_cloud, qrDetectPtr->imageCopy_);

    ros::Publisher colored_cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("colored_cloud", 1);
    ros::Publisher aligned_lidar_centers_pub = nh.advertise<sensor_msgs::PointCloud2>("aligned_lidar_centers", 1);

    // 主循环
    ros::Rate rate(1);
    while (ros::ok()) 
    {
      if (DEBUG) 
      {
        // 发布 QR 检测结果
        sensor_msgs::PointCloud2 qr_centers_msg;
        pcl::toROSMsg(*qr_centers, qr_centers_msg);
        qr_centers_msg.header.stamp = ros::Time::now();
        qr_centers_msg.header.frame_id = "map";
        qrDetectPtr->qr_pub_.publish(qr_centers_msg);

        // 发布 LiDAR 检测结果
        sensor_msgs::PointCloud2 lidar_centers_msg;
        pcl::toROSMsg(*lidar_centers, lidar_centers_msg);
        lidar_centers_msg.header = qr_centers_msg.header;
        lidarDetectPtr->center_pub_.publish(lidar_centers_msg);

        // 发布中间结果
        sensor_msgs::PointCloud2 filtered_cloud_msg;
        pcl::toROSMsg(*lidarDetectPtr->getFilteredCloud(), filtered_cloud_msg);
        filtered_cloud_msg.header = qr_centers_msg.header;
        lidarDetectPtr->filtered_pub_.publish(filtered_cloud_msg);

        sensor_msgs::PointCloud2 plane_cloud_msg;
        pcl::toROSMsg(*lidarDetectPtr->getPlaneCloud(), plane_cloud_msg);
        plane_cloud_msg.header = qr_centers_msg.header;
        lidarDetectPtr->plane_pub_.publish(plane_cloud_msg);

        sensor_msgs::PointCloud2 aligned_cloud_msg;
        pcl::toROSMsg(*lidarDetectPtr->getAlignedCloud(), aligned_cloud_msg);
        aligned_cloud_msg.header = qr_centers_msg.header;
        lidarDetectPtr->aligned_pub_.publish(aligned_cloud_msg);

        sensor_msgs::PointCloud2 edge_cloud_msg;
        pcl::toROSMsg(*lidarDetectPtr->getEdgeCloud(), edge_cloud_msg);
        edge_cloud_msg.header = qr_centers_msg.header;
        lidarDetectPtr->edge_pub_.publish(edge_cloud_msg);

        sensor_msgs::PointCloud2 lidar_centers_z0_msg;
        pcl::toROSMsg(*lidarDetectPtr->getCenterZ0Cloud(), lidar_centers_z0_msg);
        lidar_centers_z0_msg.header = qr_centers_msg.header;
        lidarDetectPtr->center_z0_pub_.publish(lidar_centers_z0_msg);

        // 发布外参变换后的LiDAR点云
        sensor_msgs::PointCloud2 aligned_lidar_centers_msg;
        pcl::toROSMsg(*aligned_lidar_centers, aligned_lidar_centers_msg);
        aligned_lidar_centers_msg.header = qr_centers_msg.header;
        aligned_lidar_centers_pub.publish(aligned_lidar_centers_msg);

        // 发布彩色点云
        sensor_msgs::PointCloud2 colored_cloud_msg;
        pcl::toROSMsg(*colored_cloud, colored_cloud_msg);
        colored_cloud_msg.header = qr_centers_msg.header;
        colored_cloud_pub.publish(colored_cloud_msg);

        // cv::imshow("result", qrDetectPtr->imageCopy_);
      }
      // cv::waitKey(1);
      ros::spinOnce();
      rate.sleep();
    }

    return 0;
}