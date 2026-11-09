/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#ifndef VIO_H_
#define VIO_H_

#include "voxel_map.h"
#include "feature.h"
#include <opencv2/imgproc/imgproc_c.h>
#include <pcl/filters/voxel_grid.h>
#include <set>
#include <vikit/math_utils.h>
#include <vikit/robust_cost.h>
#include <vikit/vision.h>
#include <vikit/pinhole_camera.h>

struct SubSparseMap
{
  vector<float> propa_errors;
  vector<float> errors;
  vector<vector<float>> warp_patch;
  vector<int> search_levels;
  vector<VisualPoint *> voxel_points;
  vector<double> inv_expo_list;
  vector<int> cam_ids;
  vector<pointWithVar> add_from_voxel_map;

  SubSparseMap()
  {
    propa_errors.reserve(SIZE_LARGE);
    errors.reserve(SIZE_LARGE);
    warp_patch.reserve(SIZE_LARGE);
    search_levels.reserve(SIZE_LARGE);
    voxel_points.reserve(SIZE_LARGE);
    inv_expo_list.reserve(SIZE_LARGE);
    cam_ids.reserve(SIZE_LARGE);
    add_from_voxel_map.reserve(SIZE_SMALL);
  };

  void reset()
  {
    propa_errors.clear();
    errors.clear();
    warp_patch.clear();
    search_levels.clear();
    voxel_points.clear();
    inv_expo_list.clear();
    cam_ids.clear();
    add_from_voxel_map.clear();
  }
};

class Warp
{
public:
  Matrix2d A_cur_ref;
  int search_level;
  Warp(int level, Matrix2d warp_matrix) : search_level(level), A_cur_ref(warp_matrix) {}
  ~Warp() {}
};

class VOXEL_POINTS
{
public:
  std::vector<VisualPoint *> voxel_points;
  int count;
  VOXEL_POINTS(int num) : count(num) {}
  ~VOXEL_POINTS() 
  { 
    for (VisualPoint* vp : voxel_points) 
    {
      if (vp != nullptr) { delete vp; vp = nullptr; }
    }
  }
};

/// One camera of the rig: model, extrinsics, current frame and per-frame grid buffers.
/// All cameras must have the same image size.
struct CameraContext
{
  int id = 0;
  vk::AbstractCamera *cam = nullptr;
  bool is_pinhole = true;
  M3D Rcl, Rci_calib, Rci, Rcw, Jdphi_dR, Jdp_dR, Jdp_dt; // Rci/Pci include the time offset compensation of the current frame
  V3D Pcl, Pci_calib, Pci, Pcw;
  double time_offset = 0.0;       // capture time of this camera minus its image stamp used as EKF time (s)
  double cos_max_incidence = 0.0; // view cone: points with z < cos * |p| in the camera frame are not used
  cv::Mat mask;                   // optional CV_8UC1 mask at the processing resolution, 0 = ignore
  vector<double> vignetting;      // optional log V(theta) = c0 theta^2 + c1 theta^4 + c2 theta^6, images are divided by V
  cv::Mat vignetting_gain;        // CV_32FC3 1/V per pixel (built in initializeVIO)
  FramePtr frame;                 // current frame, null when this camera has no image at the current time
  cv::Mat img, img_rgb, img_cp;   // current gray / color / debug-drawing images
  vector<int> grid_num, border_flag;
  vector<float> map_dist, scan_value;
  vector<VisualPoint *> retrieve_voxel_points;
  vector<pointWithVar> append_voxel_points;
  std::vector<std::vector<V3D>> rays_with_sample_points;
};

class VIOManager
{
public:
  int grid_size;
  vk::AbstractCamera *cam; // camera 0 (single-camera configs set only this one)
  vk::PinholeCamera *pinhole_cam;
  StatesGroup *state;
  StatesGroup *state_propagat;
  M3D Rli, Rci, Rcl, Rcw, Jdphi_dR, Jdp_dt, Jdp_dR;
  V3D Pli, Pci, Pcl, Pcw;
  std::vector<CameraContext> cams;
  bool exposure_per_camera = true; // one inverse exposure state per camera instead of a shared one
  bool exposure_regauge = false;   // keep the inverse exposure times near 1 by rescaling them all together
  bool vignetting_color_only = false; // apply the vignetting correction to the coloring images only, not to the VIO
  V3D omega_imu = V3D::Zero(), vel_world = V3D::Zero(); // motion at the frame time, for camera time offsets
  vector<int> map_index;
  vector<int> update_flag;
  vector<float> patch_buffer;
  bool normal_en, inverse_composition_en, exposure_estimate_en, raycast_en, has_ref_patch_cache;
  bool ncc_en = false, colmap_output_en = false;

  int width, height, grid_n_width, grid_n_height, length;
  double image_resize_factor;
  double fx, fy, cx, cy;
  int patch_pyrimid_level, patch_size, patch_size_total, patch_size_half, border, warp_len;
  int max_iterations, total_points;

  double img_point_cov, outlier_threshold, ncc_thre;
  
  SubSparseMap *visual_submap;

  double compute_jacobian_time, update_ekf_time;
  double ave_total = 0;
  // double ave_build_residual_time = 0;
  // double ave_ekf_time = 0;

  int frame_count = 0;
  bool plot_flag;

  Matrix<double, DIM_STATE, DIM_STATE> G, H_T_H;
  MatrixXd K, H_sub_inv;

  ofstream fout_camera, fout_colmap;
  unordered_map<VOXEL_LOCATION, VOXEL_POINTS *> feat_map;
  unordered_map<VOXEL_LOCATION, int> sub_feat_map; 
  unordered_map<int, Warp *> warp_map;
  cv::Mat img_test;

  enum CellType
  {
    TYPE_MAP = 1,
    TYPE_POINTCLOUD,
    TYPE_UNKNOWN
  };

  VIOManager();
  ~VIOManager();
  void updateStateInverse(int level);
  void updateState(int level);
  /// Single-camera entry point (camera 0).
  void processFrame(cv::Mat &img, vector<pointWithVar> &pg, const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &feat_map, double img_time);
  /// Multi-camera entry point: imgs[i] belongs to camera i, an empty image skips that camera.
  void processFrame(std::vector<cv::Mat> &imgs, vector<pointWithVar> &pg, const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &feat_map, double img_time);
  void retrieveFromVisualSparseMap(CameraContext &cc, vector<pointWithVar> &pg, const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map);
  void generateVisualMapPoints(CameraContext &cc, vector<pointWithVar> &pg, vector<char> &pg_used);
  void setImuToLidarExtrinsic(const V3D &transl, const M3D &rot);
  void setLidarToCameraExtrinsic(vector<double> &R, vector<double> &P);
  /// Add a camera of the rig (call before initializeVIO). Camera 0 is also stored in `cam`.
  void addCamera(vk::AbstractCamera *camera, const vector<double> &R_cl, const vector<double> &P_cl, double time_offset, double max_incidence_deg,
                 const cv::Mat &mask = cv::Mat(), const vector<double> &vignetting = vector<double>());
  void initializeVIO();
  void getImagePatch(cv::Mat img, V2D pc, float *patch_tmp, int level);
  void computeProjectionJacobian(const CameraContext &cc, const V3D &p, MD(2, 3) & J) const;
  void computeJacobianAndUpdateEKF();
  void resetGrid(CameraContext &cc);
  void updateVisualMapPoints();
  void getWarpMatrixAffine(const vk::AbstractCamera &cam_ref, const vk::AbstractCamera &cam_cur, const Vector2d &px_ref, const Vector3d &f_ref,
                           const double depth_ref, const SE3 &T_cur_ref, const int level_ref, const int pyramid_level, const int halfpatch_size,
                           Matrix2d &A_cur_ref);
  void getWarpMatrixAffineHomography(const vk::AbstractCamera &cam_ref, const vk::AbstractCamera &cam_cur, const V2D &px_ref,
                                     const V3D &xyz_ref, const V3D &normal_ref, const SE3 &T_cur_ref, const int level_ref, Matrix2d &A_cur_ref);
  void warpAffine(const Matrix2d &A_cur_ref, const cv::Mat &img_ref, const Vector2d &px_ref, const int level_ref, const int search_level,
                  const int pyramid_level, const int halfpatch_size, float *patch);
  void insertPointIntoVoxelMap(VisualPoint *pt_new);
  void plotTrackedPoints();
  void updateFrameState(StatesGroup state);
  void updateCameraExtrinsics();
  void regaugeExposure();
  void projectPatchFromRefToCur(const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map);
  void updateReferencePatch(const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map);
  void precomputeReferencePatches(int level);
  void dumpDataForColmap();
  double calculateNCC(float *ref_patch, float *cur_patch, int patch_size);
  int getBestSearchLevel(const Matrix2d &A_cur_ref, const int max_level);
  V3F getInterpolatedPixel(cv::Mat img, V2D pc);
  /// Project a world point into camera cc (current frame pose). False if behind / outside the view cone,
  /// the image border or the mask.
  bool projectToCamera(const CameraContext &cc, const V3D &pt_w, V2D &px, int boundary, V3D *pt_c_out = nullptr) const;
  int expoGroup(int cam_id) const { return exposure_per_camera ? cam_id : 0; }
  int numExpoGroups() const { return exposure_per_camera ? static_cast<int>(cams.size()) : 1; }
  /// The camera with a valid projection whose optical axis is closest to the point (for coloring), or -1.
  int bestCameraForPoint(const V3D &pt_w, V2D &px, int boundary) const;
  
  // void resetRvizDisplay();
  // deque<VisualPoint *> map_cur_frame;
  // deque<VisualPoint *> sub_map_ray;
  // deque<VisualPoint *> sub_map_ray_fov;
  // deque<VisualPoint *> visual_sub_map_cur;
  // deque<VisualPoint *> visual_converged_point;
  // std::vector<std::vector<V3D>> sample_points;

  // PointCloudXYZI::Ptr pg_down;
  // pcl::VoxelGrid<PointType> downSizeFilter;
};
typedef std::shared_ptr<VIOManager> VIOManagerPtr;

#endif // VIO_H_