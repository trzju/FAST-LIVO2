/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#ifndef IMU_PROCESSING_H
#define IMU_PROCESSING_H

#include <Eigen/Eigen>
#include "common_lib.h"
#include <condition_variable>
#include <nav_msgs/Odometry.h>
#include <utils/so3_math.h>
#include <fstream>
const bool time_list(PointType &x, PointType &y) { return (x.curvature < y.curvature); }

/// *************IMU Process and undistortion
class ImuProcess
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ImuProcess();
  ~ImuProcess();

  void Reset();
  void Reset(double start_timestamp, const sensor_msgs::ImuConstPtr &lastimu);
  void set_extrinsic(const V3D &transl, const M3D &rot);
  void set_extrinsic(const V3D &transl);
  void set_extrinsic(const MD(4, 4) & T);
  void set_gyr_cov_scale(const V3D &scaler);
  void set_acc_cov_scale(const V3D &scaler);
  void set_gyr_bias_cov(const V3D &b_g);
  void set_acc_bias_cov(const V3D &b_a);
  void set_inv_expo_cov(const double &inv_expo);
  void set_imu_init_frame_num(const int &num);
  /// Gravity in the IMU's acceleration unit (1 for a Livox IMU, which reports g). <= 0: take the mean acceleration norm
  /// measured during the initialization (upstream behavior; wrong by the walking accelerations if the device moves).
  void set_acc_norm(double norm) { acc_norm_ = norm; }
  /// Minimum length [s] of the initialization. > 0 also averages the accelerations in one frame (gyro-integrated), so
  /// the initial gravity direction survives a device that is already walking. 0: upstream (imu_int_frame samples).
  void set_init_time(double t) { init_time_ = t; }
  /// Initial standard deviation of the gyro bias [deg/s] and accelerometer bias [m/s^2] once the initialization ends;
  /// the biases start at zero. <= 0: upstream (variance 1e-5, i.e. 0.18 deg/s), which lets a real gyro bias of 1-2 deg/s
  /// converge only over minutes, so the first minutes of the trajectory turn with the wrong rate.
  void set_init_bias_std(double bg_deg_s, double ba) { init_bg_std_ = bg_deg_s * M_PI / 180.0; init_ba_std_ = ba; }
  void disable_imu();
  void disable_gravity_est();
  void disable_bias_est();
  void disable_exposure_est();
  void Process2(LidarMeasureGroup &lidar_meas, StatesGroup &stat, PointCloudXYZI::Ptr cur_pcl_un_);
  void UndistortPcl(LidarMeasureGroup &lidar_meas, StatesGroup &state_inout, PointCloudXYZI &pcl_out);
  /// Bias-corrected angular velocity (IMU frame) of the last propagated IMU sample.
  V3D getLastAngVel() const { return angvel_last; }

  ofstream fout_imu;
  double IMU_mean_acc_norm;
  double acc_unit = 1.0; // gravity in the IMU's acceleration unit, used to scale to m/s^2 (acc_norm or measured at init)
  V3D unbiased_gyr;

  V3D cov_acc;
  V3D cov_gyr;
  V3D cov_bias_gyr;
  V3D cov_bias_acc;
  double cov_inv_expo;
  int extra_expo_num = 0; // inverse exposure states of cameras 1.. (state index 19..) that receive process noise
  double first_lidar_time;
  bool imu_time_init = false;
  bool imu_need_init = true;
  M3D Eye3d;
  V3D Zero3d;
  int lidar_type;

private:
  void IMU_init(const MeasureGroup &meas, StatesGroup &state, int &N);
  void Forward_without_imu(LidarMeasureGroup &meas, StatesGroup &state_inout, PointCloudXYZI &pcl_out);
  PointCloudXYZI pcl_wait_proc;
  sensor_msgs::ImuConstPtr last_imu;
  PointCloudXYZI::Ptr cur_pcl_un_;
  vector<Pose6D> IMUpose;
  M3D Lid_rot_to_IMU;
  V3D Lid_offset_to_IMU;
  V3D mean_acc;
  V3D mean_gyr;
  V3D angvel_last;
  V3D acc_s_last;
  double last_prop_end_time;
  double time_last_scan;
  int init_iter_num = 1, MAX_INI_COUNT = 20;
  double acc_norm_ = 0.0, init_time_ = 0.0;
  double init_bg_std_ = 0.0, init_ba_std_ = 0.0; // rad/s, m/s^2
  double init_t0_ = 0.0, init_t_last_ = 0.0; // stamps of the first and last IMU sample of the initialization
  M3D init_R_;                                // rotation of the last sample relative to the first one (gyro)
  V3D init_f_sum_;                            // sum of the specific forces rotated into the first sample's frame
  bool b_first_frame = true;
  bool imu_en = true;
  bool gravity_est_en = true;
  bool ba_bg_est_en = true;
  bool exposure_estimate_en = true;
};
typedef std::shared_ptr<ImuProcess> ImuProcessPtr;
#endif