/*
Loop closure for the map output: keyframes along the odometry, loop detection by registering a keyframe against the
submap of an earlier keyframe nearby (with a search over the vertical drift), and a keyframe pose graph optimized with
Levenberg-Marquardt. The estimator itself keeps running on its own (drifting) map; the corrections are applied to the
saved trajectory and point clouds at the end.
*/

#ifndef LOOP_CLOSURE_H_
#define LOOP_CLOSURE_H_

#include "common_lib.h"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <sophus/se3.h>
#include <string>
#include <vector>

struct LoopClosureConfig
{
  double keyframe_dist = 1.0;                   // m of motion between keyframes
  double keyframe_angle = 15.0;                 // deg of rotation between keyframes
  double keyframe_voxel = 0.15;                 // m, voxel size of keyframe clouds and submaps
  double search_radius = 10.0;                  // m, horizontal (odometry) distance of a loop candidate
  double min_path_gap = 30.0;                   // m of trajectory between the two keyframes of a loop
  int try_every = 3;                            // look for a loop from every n-th keyframe
  int target_half = 10;                         // keyframes on each side of the earlier keyframe in its submap
  int source_half = 2;                          // keyframes on each side of the later keyframe in its cloud
  double z_search_ratio = 0.01;                 // vertical search range: this share of the path between the keyframes,
  double z_search_min = 1.0, z_search_max = 6.0; //   clamped to [min, max] m
  double yaw_search = 0.0, yaw_step = 1.0;      // deg: coarse search of the heading of keyframe i (each side), 0 = off
  double xy_search = 0.0, xy_step = 0.5;        // m: coarse horizontal search after the heading one, 0 = off
  bool dump_keyframes = false;                  // write keyframes.bin (keyframe poses and clouds) for loop_offline
  double inlier_dist = 0.2;                     // m, a source point closer than this to the target is an inlier
  double min_inlier_ratio = 0.6;                // accept a loop with at least this share of inliers
  double max_rmse = 0.10;                       // m, and at most this inlier RMSE
  double odom_sigma_t = 0.02, odom_sigma_r = 0.2; // m, deg: odometry between consecutive keyframes
  double loop_sigma_t = 0.05, loop_sigma_r = 0.5; // m, deg: loop closure
  double loop_max_residual_t = 0.3, loop_max_residual_r = 2.0; // m, deg: loops further off the optimized graph are dropped
};

class LoopClosure
{
public:
  explicit LoopClosure(const LoopClosureConfig &cfg) : cfg_(cfg) {}

  /// Final pose (IMU in world) of one update and its LiDAR points in the IMU frame.
  void addFrame(double time, const M3D &R, const V3D &p, const PointCloudXYZI &scan_imu);
  /// Detect loops among the keyframes and optimize the keyframe pose graph. Writes keyframes.txt and loops.txt to
  /// log_dir. Returns the number of loop closures used (0: corrections stay identity).
  int optimize(const std::string &log_dir);
  /// Keyframes (time, path, odometry pose, cloud) in a binary file, to re-run the loop closure offline (loop_offline).
  void saveKeyframes(const std::string &path) const;
  bool loadKeyframes(const std::string &path);

  size_t numFrames() const { return frames_.size(); }
  double frameTime(size_t i) const { return frames_[i].time; }
  /// World correction of frame i: corrected pose = correction * odometry pose. Identity before optimize().
  const Sophus::SE3 &correction(size_t i) const { return corrections_[i]; }
  Sophus::SE3 correctedPose(size_t i) const { return corrections_[i] * frames_[i].pose; }

private:
  typedef pcl::PointCloud<pcl::PointXYZ> Cloud;
  struct Frame
  {
    double time;
    Sophus::SE3 pose;
  };
  struct Keyframe
  {
    size_t frame;     // index of its frame
    double time, path; // s, m of trajectory up to it
    Sophus::SE3 pose;  // odometry
    Cloud::Ptr cloud;  // points of the frames since the previous keyframe, in this keyframe's frame
  };
  struct Edge
  {
    int a, b;       // keyframes, measurement z = T_a^-1 T_b
    Sophus::SE3 z;
    bool loop;
    int result;     // loop: index into the registration results
  };
  struct LoopResult
  {
    int i, j;
    double dz, coarse_ratio, ratio, rmse;
    double yaw, dx, dy; // deg, m: heading and horizontal part of the coarse correction (dz: vertical part)
    bool accepted;
    bool used;                  // kept in the final pose graph
    double residual_t, residual_r; // m, deg: disagreement with the optimized graph
    Sophus::SE3 z; // T_j^-1 T_i from the registration
  };

  Cloud::Ptr submap(int center, int half) const;
  LoopResult registerLoop(int i, int j) const;
  void optimizeGraph(const std::vector<Edge> &edges, std::vector<Sophus::SE3> &poses) const;

  LoopClosureConfig cfg_;
  std::vector<Frame> frames_;
  std::vector<Sophus::SE3> corrections_;
  std::vector<Keyframe> keyframes_;
  std::vector<std::pair<Sophus::SE3, Cloud::Ptr>> pending_; // frames since the last keyframe (pose, IMU-frame points)
  double path_ = 0.0;
};

#endif // LOOP_CLOSURE_H_
