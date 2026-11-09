/*
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#include "vio.h"

VIOManager::VIOManager()
{
  // downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
}

VIOManager::~VIOManager()
{
  delete visual_submap;
  for (auto& pair : warp_map) delete pair.second;
  warp_map.clear();
  for (auto& pair : feat_map) delete pair.second;
  feat_map.clear();
}

void VIOManager::setImuToLidarExtrinsic(const V3D &transl, const M3D &rot)
{
  Pli = -rot.transpose() * transl;
  Rli = rot.transpose();
}

void VIOManager::setLidarToCameraExtrinsic(vector<double> &R, vector<double> &P)
{
  Rcl << MAT_FROM_ARRAY(R);
  Pcl << VEC_FROM_ARRAY(P);
}

void VIOManager::addCamera(vk::AbstractCamera *camera, const vector<double> &R_cl, const vector<double> &P_cl, double time_offset,
                           double max_incidence_deg, const cv::Mat &mask, const vector<double> &vignetting)
{
  if (cams.size() >= MAX_CAMS) throw std::runtime_error("Too many cameras, increase MAX_CAMS in common_lib.h");
  CameraContext cc;
  cc.id = static_cast<int>(cams.size());
  cc.cam = camera;
  cc.is_pinhole = dynamic_cast<vk::PinholeCamera *>(camera) != nullptr;
  cc.Rcl << MAT_FROM_ARRAY(R_cl);
  cc.Pcl << VEC_FROM_ARRAY(P_cl);
  cc.time_offset = time_offset;
  cc.cos_max_incidence = std::cos(max_incidence_deg / 180.0 * M_PI);
  cc.mask = mask;
  cc.vignetting = vignetting;
  if (cc.id == 0)
  {
    cam = camera;
    Rcl = cc.Rcl;
    Pcl = cc.Pcl;
  }
  cams.push_back(std::move(cc));
}

void VIOManager::initializeVIO()
{
  visual_submap = new SubSparseMap;

  if (cams.empty())
  {
    // Single-camera configuration: camera 0 from `cam`, Rcl and Pcl.
    vector<double> R_cl(9), P_cl(3);
    for (int r = 0; r < 3; r++)
    {
      P_cl[r] = Pcl(r);
      for (int c = 0; c < 3; c++) R_cl[r * 3 + c] = Rcl(r, c);
    }
    addCamera(cam, R_cl, P_cl, 0.0, 90.0);
  }

  fx = cam->fx();
  fy = cam->fy();
  cx = cam->cx();
  cy = cam->cy();
  image_resize_factor = cam->scale();

  printf("intrinsic: %.6lf, %.6lf, %.6lf, %.6lf\n", fx, fy, cx, cy);

  width = cam->width();
  height = cam->height();

  printf("width: %d, height: %d, scale: %f\n", width, height, image_resize_factor);

  for (auto &cc : cams)
  {
    if (cc.cam->width() != width || cc.cam->height() != height) throw std::runtime_error("All cameras must have the same image size");
    if (!cc.mask.empty() && (cc.mask.cols != width || cc.mask.rows != height || cc.mask.type() != CV_8UC1))
      throw std::runtime_error("Camera mask must be CV_8UC1 with the camera image size");
    cc.Rci_calib = cc.Rcl * Rli;
    cc.Pci_calib = cc.Rcl * Pli + cc.Pcl;
    if (!cc.vignetting.empty())
    {
      // gain 1/V(theta) per pixel, theta = angle to the optical axis; capped so that the dark border is not amplified
      cc.vignetting_gain.create(height, width, CV_32FC3);
      for (int v = 0; v < height; v++)
        for (int u = 0; u < width; u++)
        {
          const V3D f = cc.cam->cam2world(u, v);
          const double theta = std::acos(std::max(-1.0, std::min(1.0, f[2] / f.norm())));
          double log_v = 0.0, t2 = theta * theta, tp = t2;
          for (double c : cc.vignetting) { log_v += c * tp; tp *= t2; }
          const float g = static_cast<float>(std::min(3.0, std::exp(-log_v)));
          cc.vignetting_gain.at<cv::Vec3f>(v, u) = cv::Vec3f(g, g, g);
        }
    }
    printf("[ VIO ] camera %d: %s model, time offset %.4f s, max incidence %.1f deg%s\n", cc.id, cc.is_pinhole ? "pinhole" : "non-pinhole",
           cc.time_offset, std::acos(cc.cos_max_incidence) * 180.0 / M_PI, cc.mask.empty() ? "" : ", with mask");
    if (!cc.vignetting.empty()) printf("[ VIO ] camera %d: vignetting correction, gain at 60 deg %.3f\n", cc.id,
                                       std::exp(-(cc.vignetting[0] * std::pow(M_PI / 3, 2) + (cc.vignetting.size() > 1 ? cc.vignetting[1] * std::pow(M_PI / 3, 4) : 0.0) +
                                                  (cc.vignetting.size() > 2 ? cc.vignetting[2] * std::pow(M_PI / 3, 6) : 0.0))));
  }
  if (inverse_composition_en && (cams.size() > 1 || !cams[0].is_pinhole))
  {
    printf("[ VIO ] inverse composition supports a single pinhole camera only, using forward composition.\n");
    inverse_composition_en = false;
  }
  if (colmap_output_en && (cams.size() > 1 || !cams[0].is_pinhole))
  {
    printf("[ VIO ] COLMAP output supports a single pinhole camera only, disabled.\n");
    colmap_output_en = false;
  }
  updateCameraExtrinsics();
  Rci = cams[0].Rci;
  Pci = cams[0].Pci;
  Jdphi_dR = cams[0].Jdphi_dR;
  Jdp_dR = cams[0].Jdp_dR;

  if (grid_size > 10)
  {
    grid_n_width = (width + grid_size - 1) / grid_size;
    grid_n_height = (height + grid_size - 1) / grid_size;
  }
  else
  {
    grid_size = static_cast<int>(height / grid_n_height);
    grid_n_height = (height + grid_size - 1) / grid_size;
    grid_n_width = (width + grid_size - 1) / grid_size;
  }
  length = grid_n_width * grid_n_height;

  for (auto &cc : cams)
  {
    cc.grid_num.resize(length);
    cc.map_dist.resize(length);
    cc.scan_value.resize(length);
    cc.retrieve_voxel_points.reserve(length);
    cc.append_voxel_points.reserve(length);
  }

  if(raycast_en)
  {
    printf("grid_size: %d, grid_n_height: %d, grid_n_width: %d, length: %d\n", grid_size, grid_n_height, grid_n_width, length);

    float d_min = 0.1;
    float d_max = 3.0;
    float step = 0.2;
    for (auto &cc : cams)
    {
      cc.border_flag.assign(length, 0);
      std::vector<std::vector<V3D>>().swap(cc.rays_with_sample_points);
      cc.rays_with_sample_points.reserve(length);
      for (int grid_row = 1; grid_row <= grid_n_height; grid_row++)
      {
        for (int grid_col = 1; grid_col <= grid_n_width; grid_col++)
        {
          std::vector<V3D> SamplePointsEachGrid;
          int index = (grid_row - 1) * grid_n_width + grid_col - 1;

          if (grid_row == 1 || grid_col == 1 || grid_row == grid_n_height || grid_col == grid_n_width) cc.border_flag[index] = 1;

          int u = grid_size / 2 + (grid_col - 1) * grid_size;
          int v = grid_size / 2 + (grid_row - 1) * grid_size;
          V3D ray = cc.cam->cam2world(u, v);
          // rays outside the view cone of this camera are never sampled
          if (ray[2] <= cc.cos_max_incidence * ray.norm()) cc.border_flag[index] = 1;
          for (float d_temp = d_min; d_temp <= d_max; d_temp += step)
          {
            V3D xyz = ray;
            xyz *= d_temp / xyz[2];
            SamplePointsEachGrid.push_back(xyz);
          }
          cc.rays_with_sample_points.push_back(SamplePointsEachGrid);
        }
      }
    }
  }

  if(colmap_output_en)
  {
    pinhole_cam = dynamic_cast<vk::PinholeCamera*>(cam);
    fout_colmap.open(DEBUG_FILE_DIR("Colmap/sparse/0/images.txt"), ios::out);
    fout_colmap << "# Image list with two lines of data per image:\n";
    fout_colmap << "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n";
    fout_colmap << "#   POINTS2D[] as (X, Y, POINT3D_ID)\n";
    fout_camera.open(DEBUG_FILE_DIR("Colmap/sparse/0/cameras.txt"), ios::out);
    fout_camera << "# Camera list with one line of data per camera:\n";
    fout_camera << "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n";
    fout_camera << "1 PINHOLE " << width << " " << height << " "
        << std::fixed << std::setprecision(6)  // 控制浮点数精度为10位
        << fx << " " << fy << " "
        << cx << " " << cy << std::endl;
    fout_camera.close();
  }
  map_index.resize(length);

  patch_size_total = patch_size * patch_size;
  patch_size_half = static_cast<int>(patch_size / 2);
  patch_buffer.resize(patch_size_total);
  warp_len = patch_size_total * patch_pyrimid_level;
  border = (patch_size_half + 1) * (1 << patch_pyrimid_level);

  sub_feat_map.clear();
}

void VIOManager::updateCameraExtrinsics()
{
  for (auto &cc : cams)
  {
    cc.Rci = cc.Rci_calib;
    cc.Pci = cc.Pci_calib;
    if (cc.time_offset != 0.0)
    {
      // The image was captured at t + time_offset while the EKF state is at t. Fold the IMU motion over
      // time_offset into the extrinsic: R_c(t+dt)w = Rci dR^T Rwi^T, dR / dp = rotation / displacement of the IMU
      // over dt expressed in the IMU frame at t.
      const double dt = cc.time_offset;
      M3D dR = Exp(omega_imu, dt);
      V3D dp = state->rot_end.transpose() * vel_world * dt;
      cc.Rci = cc.Rci_calib * dR.transpose();
      cc.Pci = cc.Pci_calib - cc.Rci * dp;
    }
    V3D Pic = -cc.Rci.transpose() * cc.Pci;
    M3D tmp;
    tmp << SKEW_SYM_MATRX(Pic);
    cc.Jdphi_dR = cc.Rci;
    cc.Jdp_dR = -cc.Rci * tmp;
  }
}

void VIOManager::regaugeExposure()
{
  // The photometric residuals only constrain ratios of inverse exposure times; their common scale is a gauge that
  // drifts (the estimate is biased low) and the visual information shrinks with its square. Rescaling every stored
  // and estimated value by the same factor leaves all ratios unchanged.
  const double e0 = state->invExpo(0);
  if (e0 > 0.8 && e0 < 1.25) return;
  const double s = 1.0 / e0;
  for (auto &v : feat_map)
    for (VisualPoint *pt : v.second->voxel_points)
    {
      if (pt == nullptr) continue;
      for (Feature *f : pt->obs_) f->inv_expo_time_ *= s;
    }
  for (int g = 0; g < numExpoGroups(); g++)
  {
    state->invExpo(g) *= s;
    const int idx = StatesGroup::expoIndex(g);
    state->cov.row(idx) *= s;
    state->cov.col(idx) *= s;
  }
  printf("[ VIO ] exposure gauge rescaled by %.3f\n", s);
}

void VIOManager::resetGrid(CameraContext &cc)
{
  fill(cc.grid_num.begin(), cc.grid_num.end(), TYPE_UNKNOWN);
  fill(cc.map_dist.begin(), cc.map_dist.end(), 10000.0f);
  fill(cc.scan_value.begin(), cc.scan_value.end(), 0.0f);

  cc.retrieve_voxel_points.clear();
  cc.retrieve_voxel_points.resize(length);

  cc.append_voxel_points.clear();
  cc.append_voxel_points.resize(length);
}

// void VIOManager::resetRvizDisplay()
// {
  // sub_map_ray.clear();
  // sub_map_ray_fov.clear();
  // visual_sub_map_cur.clear();
  // visual_converged_point.clear();
  // map_cur_frame.clear();
  // sample_points.clear();
// }

void VIOManager::computeProjectionJacobian(const CameraContext &cc, const V3D &p, MD(2, 3) & J) const
{
  if (cc.is_pinhole)
  {
    const double x = p[0];
    const double y = p[1];
    const double z_inv = 1. / p[2];
    const double z_inv_2 = z_inv * z_inv;
    J(0, 0) = cc.cam->fx() * z_inv;
    J(0, 1) = 0.0;
    J(0, 2) = -cc.cam->fx() * x * z_inv_2;
    J(1, 0) = 0.0;
    J(1, 1) = cc.cam->fy() * z_inv;
    J(1, 2) = -cc.cam->fy() * y * z_inv_2;
    return;
  }
  // Other models (fisheye): central differences of the projection, the pinhole Jacobian is far off at large angles.
  const double h = 1e-5 * std::max(1.0, p.norm());
  for (int k = 0; k < 3; k++)
  {
    V3D dp = V3D::Zero();
    dp[k] = h;
    J.col(k) = (cc.cam->world2cam(V3D(p + dp)) - cc.cam->world2cam(V3D(p - dp))) / (2.0 * h);
  }
}

bool VIOManager::projectToCamera(const CameraContext &cc, const V3D &pt_w, V2D &px, int boundary, V3D *pt_c_out) const
{
  const V3D pt_c = cc.frame->w2f(pt_w);
  if (pt_c_out != nullptr) *pt_c_out = pt_c;
  if (pt_c[2] <= cc.cos_max_incidence * pt_c.norm()) return false;
  px = cc.cam->world2cam(pt_c);
  if (!cc.cam->isInFrame(px.cast<int>(), boundary)) return false;
  if (!cc.mask.empty() && cc.mask.at<uchar>(static_cast<int>(px[1]), static_cast<int>(px[0])) == 0) return false;
  return true;
}

int VIOManager::bestCameraForPoint(const V3D &pt_w, V2D &px, int boundary) const
{
  int best = -1;
  double best_cos = -2.0;
  for (const auto &cc : cams)
  {
    if (!cc.frame) continue;
    V2D pc;
    V3D pt_c;
    if (!projectToCamera(cc, pt_w, pc, boundary, &pt_c)) continue;
    const double cos_axis = pt_c[2] / pt_c.norm();
    if (cos_axis > best_cos)
    {
      best_cos = cos_axis;
      best = cc.id;
      px = pc;
    }
  }
  return best;
}

void VIOManager::getImagePatch(cv::Mat img, V2D pc, float *patch_tmp, int level)
{
  const float u_ref = pc[0];
  const float v_ref = pc[1];
  const int scale = (1 << level);
  const int u_ref_i = floorf(pc[0] / scale) * scale;
  const int v_ref_i = floorf(pc[1] / scale) * scale;
  const float subpix_u_ref = (u_ref - u_ref_i) / scale;
  const float subpix_v_ref = (v_ref - v_ref_i) / scale;
  const float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
  const float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
  const float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
  const float w_ref_br = subpix_u_ref * subpix_v_ref;
  for (int x = 0; x < patch_size; x++)
  {
    uint8_t *img_ptr = (uint8_t *)img.data + (v_ref_i - patch_size_half * scale + x * scale) * width + (u_ref_i - patch_size_half * scale);
    for (int y = 0; y < patch_size; y++, img_ptr += scale)
    {
      patch_tmp[patch_size_total * level + x * patch_size + y] =
          w_ref_tl * img_ptr[0] + w_ref_tr * img_ptr[scale] + w_ref_bl * img_ptr[scale * width] + w_ref_br * img_ptr[scale * width + scale];
    }
  }
}

void VIOManager::insertPointIntoVoxelMap(VisualPoint *pt_new)
{
  V3D pt_w(pt_new->pos_[0], pt_new->pos_[1], pt_new->pos_[2]);
  double voxel_size = 0.5;
  float loc_xyz[3];
  for (int j = 0; j < 3; j++)
  {
    loc_xyz[j] = pt_w[j] / voxel_size;
    if (loc_xyz[j] < 0) { loc_xyz[j] -= 1.0; }
  }
  VOXEL_LOCATION position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
  auto iter = feat_map.find(position);
  if (iter != feat_map.end())
  {
    iter->second->voxel_points.push_back(pt_new);
    iter->second->count++;
  }
  else
  {
    VOXEL_POINTS *ot = new VOXEL_POINTS(0);
    ot->voxel_points.push_back(pt_new);
    feat_map[position] = ot;
  }
}

void VIOManager::getWarpMatrixAffineHomography(const vk::AbstractCamera &cam_ref, const vk::AbstractCamera &cam_cur, const V2D &px_ref,
                                               const V3D &xyz_ref, const V3D &normal_ref, const SE3 &T_cur_ref, const int level_ref,
                                               Matrix2d &A_cur_ref)
{
  // create homography matrix
  const V3D t = T_cur_ref.inverse().translation();
  Eigen::Matrix3d H_cur_ref =
      T_cur_ref.rotation_matrix() * (normal_ref.dot(xyz_ref) * Eigen::Matrix3d::Identity() - t * normal_ref.transpose());
  // H maps rays to points in the current frame up to the sign of n.X; keep that scale positive so that camera
  // models that distinguish z < 0 (fisheye) project the mapped rays to the right side.
  if (normal_ref.dot(xyz_ref) < 0) H_cur_ref = -H_cur_ref;
  // Compute affine warp matrix A_ref_cur using homography projection
  const int kHalfPatchSize = 4;
  V3D f_du_ref(cam_ref.cam2world(px_ref + Eigen::Vector2d(kHalfPatchSize, 0) * (1 << level_ref)));
  V3D f_dv_ref(cam_ref.cam2world(px_ref + Eigen::Vector2d(0, kHalfPatchSize) * (1 << level_ref)));
  //   f_du_ref = f_du_ref/f_du_ref[2];
  //   f_dv_ref = f_dv_ref/f_dv_ref[2];
  const V3D f_cur(H_cur_ref * xyz_ref);
  const V3D f_du_cur = H_cur_ref * f_du_ref;
  const V3D f_dv_cur = H_cur_ref * f_dv_ref;
  V2D px_cur(cam_cur.world2cam(f_cur));
  V2D px_du_cur(cam_cur.world2cam(f_du_cur));
  V2D px_dv_cur(cam_cur.world2cam(f_dv_cur));
  A_cur_ref.col(0) = (px_du_cur - px_cur) / kHalfPatchSize;
  A_cur_ref.col(1) = (px_dv_cur - px_cur) / kHalfPatchSize;
}

void VIOManager::getWarpMatrixAffine(const vk::AbstractCamera &cam_ref, const vk::AbstractCamera &cam_cur, const Vector2d &px_ref,
                                     const Vector3d &f_ref, const double depth_ref, const SE3 &T_cur_ref, const int level_ref,
                                     const int pyramid_level, const int halfpatch_size, Matrix2d &A_cur_ref)
{
  // Compute affine warp matrix A_ref_cur
  const Vector3d xyz_ref(f_ref * depth_ref);
  Vector3d xyz_du_ref(cam_ref.cam2world(px_ref + Vector2d(halfpatch_size, 0) * (1 << level_ref) * (1 << pyramid_level)));
  Vector3d xyz_dv_ref(cam_ref.cam2world(px_ref + Vector2d(0, halfpatch_size) * (1 << level_ref) * (1 << pyramid_level)));
  xyz_du_ref *= xyz_ref[2] / xyz_du_ref[2];
  xyz_dv_ref *= xyz_ref[2] / xyz_dv_ref[2];
  const Vector2d px_cur(cam_cur.world2cam(T_cur_ref * (xyz_ref)));
  const Vector2d px_du(cam_cur.world2cam(T_cur_ref * (xyz_du_ref)));
  const Vector2d px_dv(cam_cur.world2cam(T_cur_ref * (xyz_dv_ref)));
  A_cur_ref.col(0) = (px_du - px_cur) / halfpatch_size;
  A_cur_ref.col(1) = (px_dv - px_cur) / halfpatch_size;
}

void VIOManager::warpAffine(const Matrix2d &A_cur_ref, const cv::Mat &img_ref, const Vector2d &px_ref, const int level_ref, const int search_level,
                               const int pyramid_level, const int halfpatch_size, float *patch)
{
  const int patch_size = halfpatch_size * 2;
  const Matrix2f A_ref_cur = A_cur_ref.inverse().cast<float>();
  if (isnan(A_ref_cur(0, 0)))
  {
    printf("Affine warp is NaN, probably camera has no translation\n"); // TODO
    return;
  }

  float *patch_ptr = patch;
  for (int y = 0; y < patch_size; ++y)
  {
    for (int x = 0; x < patch_size; ++x) //, ++patch_ptr)
    {
      Vector2f px_patch(x - halfpatch_size, y - halfpatch_size);
      px_patch *= (1 << search_level);
      px_patch *= (1 << pyramid_level);
      const Vector2f px(A_ref_cur * px_patch + px_ref.cast<float>());
      if (px[0] < 0 || px[1] < 0 || px[0] >= img_ref.cols - 1 || px[1] >= img_ref.rows - 1)
        patch_ptr[patch_size_total * pyramid_level + y * patch_size + x] = 0;
      else
        patch_ptr[patch_size_total * pyramid_level + y * patch_size + x] = (float)vk::interpolateMat_8u(img_ref, px[0], px[1]);
    }
  }
}

int VIOManager::getBestSearchLevel(const Matrix2d &A_cur_ref, const int max_level)
{
  // Compute patch level in other image
  int search_level = 0;
  double D = A_cur_ref.determinant();
  while (D > 3.0 && search_level < max_level)
  {
    search_level += 1;
    D *= 0.25;
  }
  return search_level;
}

double VIOManager::calculateNCC(float *ref_patch, float *cur_patch, int patch_size)
{
  double sum_ref = std::accumulate(ref_patch, ref_patch + patch_size, 0.0);
  double mean_ref = sum_ref / patch_size;

  double sum_cur = std::accumulate(cur_patch, cur_patch + patch_size, 0.0);
  double mean_curr = sum_cur / patch_size;

  double numerator = 0, demoniator1 = 0, demoniator2 = 0;
  for (int i = 0; i < patch_size; i++)
  {
    double n = (ref_patch[i] - mean_ref) * (cur_patch[i] - mean_curr);
    numerator += n;
    demoniator1 += (ref_patch[i] - mean_ref) * (ref_patch[i] - mean_ref);
    demoniator2 += (cur_patch[i] - mean_curr) * (cur_patch[i] - mean_curr);
  }
  return numerator / sqrt(demoniator1 * demoniator2 + 1e-10);
}

void VIOManager::retrieveFromVisualSparseMap(CameraContext &cc, vector<pointWithVar> &pg, const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map)
{
  if (feat_map.size() <= 0) return;
  double ts0 = omp_get_wtime();
  const int n_before = visual_submap->voxel_points.size();

  // Controls whether to include the visual submap from the previous frame.
  sub_feat_map.clear();

  float voxel_size = 0.5;

  if (!normal_en) warp_map.clear();

  // Depth image of the current scan: z for pinhole cameras, range for wide-angle models (z shrinks towards the border).
  cv::Mat depth_img = cv::Mat::zeros(height, width, CV_32FC1);
  float *it = (float *)depth_img.data;

  int loc_xyz[3];

  for (int i = 0; i < pg.size(); i++)
  {
    V3D pt_w = pg[i].point_w;

    for (int j = 0; j < 3; j++)
    {
      loc_xyz[j] = floor(pt_w[j] / voxel_size);
      if (loc_xyz[j] < 0) { loc_xyz[j] -= 1.0; }
    }
    VOXEL_LOCATION position(loc_xyz[0], loc_xyz[1], loc_xyz[2]);

    auto iter = sub_feat_map.find(position);
    if (iter == sub_feat_map.end()) { sub_feat_map[position] = 0; }
    else { iter->second = 0; }

    V2D px;
    V3D pt_c;
    if (projectToCamera(cc, pt_w, px, border, &pt_c))
    {
      float depth = cc.is_pinhole ? pt_c[2] : pt_c.norm();
      int col = int(px[0]);
      int row = int(px[1]);
      it[width * row + col] = depth;
    }
  }

  vector<VOXEL_LOCATION> DeleteKeyList;

  for (auto &iter : sub_feat_map)
  {
    VOXEL_LOCATION position = iter.first;

    auto corre_voxel = feat_map.find(position);

    if (corre_voxel != feat_map.end())
    {
      bool voxel_in_fov = false;
      std::vector<VisualPoint *> &voxel_points = corre_voxel->second->voxel_points;
      int voxel_num = voxel_points.size();

      for (int i = 0; i < voxel_num; i++)
      {
        VisualPoint *pt = voxel_points[i];
        if (pt == nullptr) continue;
        if (pt->obs_.size() == 0) continue;

        V2D pc;
        if (projectToCamera(cc, pt->pos_, pc, border))
        {
          voxel_in_fov = true;
          int index = static_cast<int>(pc[1] / grid_size) * grid_n_width + static_cast<int>(pc[0] / grid_size);
          cc.grid_num[index] = TYPE_MAP;
          Vector3d obs_vec(cc.frame->pos() - pt->pos_);
          float cur_dist = obs_vec.norm();
          if (cur_dist <= cc.map_dist[index])
          {
            cc.map_dist[index] = cur_dist;
            cc.retrieve_voxel_points[index] = pt;
          }
        }
      }
      if (!voxel_in_fov) { DeleteKeyList.push_back(position); }
    }
  }

  // RayCasting Module
  if (raycast_en)
  {
    for (int i = 0; i < length; i++)
    {
      if (cc.grid_num[i] == TYPE_MAP || cc.border_flag[i] == 1) continue;

      for (const auto &it : cc.rays_with_sample_points[i])
      {
        V3D sample_point_w = cc.frame->f2w(it);

        for (int j = 0; j < 3; j++)
        {
          loc_xyz[j] = floor(sample_point_w[j] / voxel_size);
          if (loc_xyz[j] < 0) { loc_xyz[j] -= 1.0; }
        }

        VOXEL_LOCATION sample_pos(loc_xyz[0], loc_xyz[1], loc_xyz[2]);

        auto corre_sub_feat_map = sub_feat_map.find(sample_pos);
        if (corre_sub_feat_map != sub_feat_map.end()) break;

        auto corre_feat_map = feat_map.find(sample_pos);
        if (corre_feat_map != feat_map.end())
        {
          bool voxel_in_fov = false;

          std::vector<VisualPoint *> &voxel_points = corre_feat_map->second->voxel_points;
          int voxel_num = voxel_points.size();
          if (voxel_num == 0) continue;

          for (int j = 0; j < voxel_num; j++)
          {
            VisualPoint *pt = voxel_points[j];

            if (pt == nullptr) continue;
            if (pt->obs_.size() == 0) continue;

            V2D pc;
            if (projectToCamera(cc, pt->pos_, pc, border))
            {
              voxel_in_fov = true;
              int index = static_cast<int>(pc[1] / grid_size) * grid_n_width + static_cast<int>(pc[0] / grid_size);
              cc.grid_num[index] = TYPE_MAP;
              Vector3d obs_vec(cc.frame->pos() - pt->pos_);

              float cur_dist = obs_vec.norm();

              if (cur_dist <= cc.map_dist[index])
              {
                cc.map_dist[index] = cur_dist;
                cc.retrieve_voxel_points[index] = pt;
              }
            }
          }

          if (voxel_in_fov) sub_feat_map[sample_pos] = 0;
          break;
        }
        else
        {
          VOXEL_LOCATION sample_pos(loc_xyz[0], loc_xyz[1], loc_xyz[2]);
          auto iter = plane_map.find(sample_pos);
          if (iter != plane_map.end())
          {
            VoxelOctoTree *current_octo;
            current_octo = iter->second->find_correspond(sample_point_w);
            if (current_octo->plane_ptr_->is_plane_)
            {
              pointWithVar plane_center;
              VoxelPlane &plane = *current_octo->plane_ptr_;
              plane_center.point_w = plane.center_;
              plane_center.normal = plane.normal_;
              visual_submap->add_from_voxel_map.push_back(plane_center);
              break;
            }
          }
        }
      }
    }
  }

  for (auto &key : DeleteKeyList)
  {
    sub_feat_map.erase(key);
  }

  const double inv_expo_cur = state->invExpo(expoGroup(cc.id));

  for (int i = 0; i < length; i++)
  {
    if (cc.grid_num[i] == TYPE_MAP)
    {
      VisualPoint *pt = cc.retrieve_voxel_points[i];

      V2D pc(cc.frame->w2c(pt->pos_));

      V3D pt_cam(cc.frame->w2f(pt->pos_));
      const float pt_depth = cc.is_pinhole ? pt_cam[2] : pt_cam.norm();
      bool depth_continous = false;
      for (int u = -patch_size_half; u <= patch_size_half; u++)
      {
        for (int v = -patch_size_half; v <= patch_size_half; v++)
        {
          if (u == 0 && v == 0) continue;

          float depth = it[width * (v + int(pc[1])) + u + int(pc[0])];

          if (depth == 0.) continue;

          double delta_dist = abs(pt_depth - depth);

          if (delta_dist > 0.5)
          {
            depth_continous = true;
            break;
          }
        }
        if (depth_continous) break;
      }
      if (depth_continous) continue;

      Feature *ref_ftr;
      std::vector<float> patch_wrap(warp_len);

      int search_level;
      Matrix2d A_cur_ref_zero;

      if (!pt->is_normal_initialized_) continue;

      if (normal_en)
      {
        float phtometric_errors_min = std::numeric_limits<float>::max();

        if (pt->obs_.size() == 1)
        {
          ref_ftr = *pt->obs_.begin();
          pt->ref_patch = ref_ftr;
          pt->has_ref_patch_ = true;
        }
        else if (!pt->has_ref_patch_)
        {
          for (auto it = pt->obs_.begin(), ite = pt->obs_.end(); it != ite; ++it)
          {
            Feature *ref_patch_temp = *it;
            float *patch_temp = ref_patch_temp->patch_;
            float phtometric_errors = 0.0;
            int count = 0;
            for (auto itm = pt->obs_.begin(), itme = pt->obs_.end(); itm != itme; ++itm)
            {
              if ((*itm)->id_ == ref_patch_temp->id_) continue;
              float *patch_cache = (*itm)->patch_;

              for (int ind = 0; ind < patch_size_total; ind++)
              {
                phtometric_errors += (patch_temp[ind] - patch_cache[ind]) * (patch_temp[ind] - patch_cache[ind]);
              }
              count++;
            }
            phtometric_errors = phtometric_errors / count;
            if (phtometric_errors < phtometric_errors_min)
            {
              phtometric_errors_min = phtometric_errors;
              ref_ftr = ref_patch_temp;
            }
          }
          pt->ref_patch = ref_ftr;
          pt->has_ref_patch_ = true;
        }
        else { ref_ftr = pt->ref_patch; }
      }
      else
      {
        if (!pt->getCloseViewObs(cc.frame->pos(), ref_ftr, pc)) continue;
      }

      const CameraContext &ref_cc = cams[ref_ftr->cam_id_];

      if (normal_en)
      {
        V3D norm_vec = (ref_ftr->T_f_w_.rotation_matrix() * pt->normal_).normalized();

        V3D pf(ref_ftr->T_f_w_ * pt->pos_);

        SE3 T_cur_ref = cc.frame->T_f_w_ * ref_ftr->T_f_w_.inverse();

        getWarpMatrixAffineHomography(*ref_cc.cam, *cc.cam, ref_ftr->px_, pf, norm_vec, T_cur_ref, 0, A_cur_ref_zero);

        search_level = getBestSearchLevel(A_cur_ref_zero, 2);
      }
      else
      {
        auto iter_warp = warp_map.find(ref_ftr->id_);
        if (iter_warp != warp_map.end())
        {
          search_level = iter_warp->second->search_level;
          A_cur_ref_zero = iter_warp->second->A_cur_ref;
        }
        else
        {
          getWarpMatrixAffine(*ref_cc.cam, *cc.cam, ref_ftr->px_, ref_ftr->f_, (ref_ftr->pos() - pt->pos_).norm(),
                              cc.frame->T_f_w_ * ref_ftr->T_f_w_.inverse(), ref_ftr->level_, 0, patch_size_half, A_cur_ref_zero);

          search_level = getBestSearchLevel(A_cur_ref_zero, 2);

          Warp *ot = new Warp(search_level, A_cur_ref_zero);
          warp_map[ref_ftr->id_] = ot;
        }
      }

      for (int pyramid_level = 0; pyramid_level <= patch_pyrimid_level - 1; pyramid_level++)
      {
        warpAffine(A_cur_ref_zero, ref_ftr->img_, ref_ftr->px_, ref_ftr->level_, search_level, pyramid_level, patch_size_half, patch_wrap.data());
      }

      getImagePatch(cc.img, pc, patch_buffer.data(), 0);

      float error = 0.0;
      for (int ind = 0; ind < patch_size_total; ind++)
      {
        error += (ref_ftr->inv_expo_time_ * patch_wrap[ind] - inv_expo_cur * patch_buffer[ind]) *
                 (ref_ftr->inv_expo_time_ * patch_wrap[ind] - inv_expo_cur * patch_buffer[ind]);
      }

      if (ncc_en)
      {
        double ncc = calculateNCC(patch_wrap.data(), patch_buffer.data(), patch_size_total);
        if (ncc < ncc_thre)
        {
          // grid_num[i] = TYPE_UNKNOWN;
          continue;
        }
      }

      if (error > outlier_threshold * patch_size_total) continue;

      visual_submap->voxel_points.push_back(pt);
      visual_submap->propa_errors.push_back(error);
      visual_submap->search_levels.push_back(search_level);
      visual_submap->errors.push_back(error);
      visual_submap->warp_patch.push_back(patch_wrap);
      visual_submap->inv_expo_list.push_back(ref_ftr->inv_expo_time_);
      visual_submap->cam_ids.push_back(cc.id);
    }
  }

  printf("[ VIO ] Camera %d: retrieve %d points from visual sparse map (%.4f s)\n", cc.id,
         static_cast<int>(visual_submap->voxel_points.size()) - n_before, omp_get_wtime() - ts0);
}

void VIOManager::computeJacobianAndUpdateEKF()
{
  if (total_points == 0) return;

  compute_jacobian_time = update_ekf_time = 0.0;

  for (int level = patch_pyrimid_level - 1; level >= 0; level--)
  {
    if (inverse_composition_en)
    {
      has_ref_patch_cache = false;
      updateStateInverse(level);
    }
    else
      updateState(level);
  }
  state->cov -= G * state->cov;
  updateFrameState(*state);
}

void VIOManager::generateVisualMapPoints(CameraContext &cc, vector<pointWithVar> &pg, vector<char> &pg_used)
{
  if (pg.size() <= 10) return;

  // map_index: index into pg of the candidate of each cell, -1 for candidates from the voxel map
  fill(map_index.begin(), map_index.end(), -1);

  for (int i = 0; i < pg.size(); i++)
  {
    if (pg_used[i]) continue; // already a new visual map point from another camera of this frame
    if (pg[i].normal == V3D(0, 0, 0)) continue;

    V3D pt = pg[i].point_w;
    V2D pc;

    if (projectToCamera(cc, pt, pc, border)) // 20px is the patch size in the matcher
    {
      int index = static_cast<int>(pc[1] / grid_size) * grid_n_width + static_cast<int>(pc[0] / grid_size);

      if (cc.grid_num[index] != TYPE_MAP)
      {
        float cur_value = vk::shiTomasiScore(cc.img, pc[0], pc[1]);
        // if (cur_value < 5) continue;
        if (cur_value > cc.scan_value[index])
        {
          cc.scan_value[index] = cur_value;
          cc.append_voxel_points[index] = pg[i];
          cc.grid_num[index] = TYPE_POINTCLOUD;
          map_index[index] = i;
        }
      }
    }
  }

  for (int j = 0; j < visual_submap->add_from_voxel_map.size(); j++)
  {
    V3D pt = visual_submap->add_from_voxel_map[j].point_w;
    V2D pc;

    if (projectToCamera(cc, pt, pc, border)) // 20px is the patch size in the matcher
    {
      int index = static_cast<int>(pc[1] / grid_size) * grid_n_width + static_cast<int>(pc[0] / grid_size);

      if (cc.grid_num[index] != TYPE_MAP)
      {
        float cur_value = vk::shiTomasiScore(cc.img, pc[0], pc[1]);
        if (cur_value > cc.scan_value[index])
        {
          cc.scan_value[index] = cur_value;
          cc.append_voxel_points[index] = visual_submap->add_from_voxel_map[j];
          cc.grid_num[index] = TYPE_POINTCLOUD;
          map_index[index] = -1;
        }
      }
    }
  }

  const double inv_expo_cur = state->invExpo(expoGroup(cc.id));
  int add = 0;
  for (int i = 0; i < length; i++)
  {
    if (cc.grid_num[i] == TYPE_POINTCLOUD) // && (scan_value[i]>=50))
    {
      pointWithVar pt_var = cc.append_voxel_points[i];
      V3D pt = pt_var.point_w;

      V3D norm_vec(cc.frame->T_f_w_.rotation_matrix() * pt_var.normal);
      V3D dir(cc.frame->T_f_w_ * pt);
      dir.normalize();
      double cos_theta = dir.dot(norm_vec);
      // if(std::fabs(cos_theta)<0.34) continue; // 70 degree
      V2D pc(cc.frame->w2c(pt));

      float *patch = new float[patch_size_total];
      getImagePatch(cc.img, pc, patch, 0);

      VisualPoint *pt_new = new VisualPoint(pt);

      Vector3d f = cc.cam->cam2world(pc);
      Feature *ftr_new = new Feature(pt_new, patch, pc, f, cc.frame->T_f_w_, 0);
      ftr_new->img_ = cc.img;
      ftr_new->id_ = cc.frame->id_;
      ftr_new->cam_id_ = cc.id;
      ftr_new->inv_expo_time_ = inv_expo_cur;

      pt_new->addFrameRef(ftr_new);
      pt_new->covariance_ = pt_var.var;
      pt_new->is_normal_initialized_ = true;

      if (cos_theta < 0) { pt_new->normal_ = -pt_var.normal; }
      else { pt_new->normal_ = pt_var.normal; }

      pt_new->previous_normal_ = pt_new->normal_;

      insertPointIntoVoxelMap(pt_new);
      if (map_index[i] >= 0) pg_used[map_index[i]] = 1;
      add += 1;
    }
  }

  printf("[ VIO ] Camera %d: append %d new visual map points\n", cc.id, add);
}

void VIOManager::updateVisualMapPoints()
{
  if (total_points == 0) return;

  update_flag.assign(total_points, 0);
  int update_num = 0;
  for (int i = 0; i < total_points; i++)
  {
    VisualPoint *pt = visual_submap->voxel_points[i];
    if (pt == nullptr) continue;
    if (pt->is_converged_)
    {
      pt->deleteNonRefPatchFeatures();
      continue;
    }

    CameraContext &cc = cams[visual_submap->cam_ids[i]];
    SE3 pose_cur = cc.frame->T_f_w_;
    V2D pc(cc.frame->w2c(pt->pos_));
    bool add_flag = false;

    // TODO: condition: distance and view_angle
    // Step 1: time
    // Compare with the oldest observation of this point by the same camera (obs_ is ordered newest first).
    Feature *last_feature = nullptr;
    for (auto it = pt->obs_.rbegin(); it != pt->obs_.rend(); ++it)
    {
      if ((*it)->cam_id_ == cc.id)
      {
        last_feature = *it;
        break;
      }
    }
    // if(new_frame_->id_ >= last_feature->id_ + 10) add_flag = true; // 10

    if (last_feature == nullptr) { add_flag = true; }
    else
    {
      // Step 2: delta_pose
      SE3 pose_ref = last_feature->T_f_w_;
      SE3 delta_pose = pose_ref * pose_cur.inverse();
      double delta_p = delta_pose.translation().norm();
      double delta_theta = (delta_pose.rotation_matrix().trace() > 3.0 - 1e-6) ? 0.0 : std::acos(0.5 * (delta_pose.rotation_matrix().trace() - 1));
      if (delta_p > 0.5 || delta_theta > 0.3) add_flag = true; // 0.5 || 0.3

      // Step 3: pixel distance
      Vector2d last_px = last_feature->px_;
      double pixel_dist = (pc - last_px).norm();
      if (pixel_dist > 40) add_flag = true;
    }

    // Maintain the size of 3D point observation features.
    if (pt->obs_.size() >= 30)
    {
      Feature *ref_ftr;
      pt->findMinScoreFeature(cc.frame->pos(), ref_ftr);
      pt->deleteFeatureRef(ref_ftr);
      // cout<<"pt->obs_.size() exceed 20 !!!!!!"<<endl;
    }
    if (add_flag)
    {
      float *patch_temp = new float[patch_size_total];
      getImagePatch(cc.img, pc, patch_temp, 0);
      update_num += 1;
      update_flag[i] = 1;
      Vector3d f = cc.cam->cam2world(pc);
      Feature *ftr_new = new Feature(pt, patch_temp, pc, f, cc.frame->T_f_w_, visual_submap->search_levels[i]);
      ftr_new->img_ = cc.img;
      ftr_new->id_ = cc.frame->id_;
      ftr_new->cam_id_ = cc.id;
      ftr_new->inv_expo_time_ = state->invExpo(expoGroup(cc.id));
      pt->addFrameRef(ftr_new);
    }
  }
  printf("[ VIO ] Update %d points in visual submap\n", update_num);
}

void VIOManager::updateReferencePatch(const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map)
{
  if (total_points == 0) return;

  for (int i = 0; i < visual_submap->voxel_points.size(); i++)
  {
    VisualPoint *pt = visual_submap->voxel_points[i];

    if (!pt->is_normal_initialized_) continue;
    if (pt->is_converged_) continue;
    if (pt->obs_.size() <= 5) continue;
    if (update_flag[i] == 0) continue;

    const V3D &p_w = pt->pos_;
    float loc_xyz[3];
    for (int j = 0; j < 3; j++)
    {
      loc_xyz[j] = p_w[j] / 0.5;
      if (loc_xyz[j] < 0) { loc_xyz[j] -= 1.0; }
    }
    VOXEL_LOCATION position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
    auto iter = plane_map.find(position);
    if (iter != plane_map.end())
    {
      VoxelOctoTree *current_octo;
      current_octo = iter->second->find_correspond(p_w);
      if (current_octo->plane_ptr_->is_plane_)
      {
        VoxelPlane &plane = *current_octo->plane_ptr_;
        float dis_to_plane = plane.normal_(0) * p_w(0) + plane.normal_(1) * p_w(1) + plane.normal_(2) * p_w(2) + plane.d_;
        float dis_to_plane_abs = fabs(dis_to_plane);
        float dis_to_center = (plane.center_(0) - p_w(0)) * (plane.center_(0) - p_w(0)) +
                              (plane.center_(1) - p_w(1)) * (plane.center_(1) - p_w(1)) + (plane.center_(2) - p_w(2)) * (plane.center_(2) - p_w(2));
        float range_dis = sqrt(dis_to_center - dis_to_plane * dis_to_plane);
        if (range_dis <= 3 * plane.radius_)
        {
          Eigen::Matrix<double, 1, 6> J_nq;
          J_nq.block<1, 3>(0, 0) = p_w - plane.center_;
          J_nq.block<1, 3>(0, 3) = -plane.normal_;
          double sigma_l = J_nq * plane.plane_var_ * J_nq.transpose();
          sigma_l += plane.normal_.transpose() * pt->covariance_ * plane.normal_;

          if (dis_to_plane_abs < 3 * sqrt(sigma_l))
          {
            if (pt->previous_normal_.dot(plane.normal_) < 0) { pt->normal_ = -plane.normal_; }
            else { pt->normal_ = plane.normal_; }

            double normal_update = (pt->normal_ - pt->previous_normal_).norm();

            pt->previous_normal_ = pt->normal_;

            if (normal_update < 0.0001 && pt->obs_.size() > 10)
            {
              pt->is_converged_ = true;
              // visual_converged_point.push_back(pt);
            }
          }
        }
      }
    }

    float score_max = -1000.;
    for (auto it = pt->obs_.begin(), ite = pt->obs_.end(); it != ite; ++it)
    {
      Feature *ref_patch_temp = *it;
      float *patch_temp = ref_patch_temp->patch_;
      float NCC = 0.0;
      float score = 0.0;
      int count = 0;

      V3D pf = ref_patch_temp->T_f_w_ * pt->pos_;
      V3D norm_vec = ref_patch_temp->T_f_w_.rotation_matrix() * pt->normal_;
      pf.normalize();
      double cos_angle = pf.dot(norm_vec);
      // if(fabs(cos_angle) < 0.86) continue; // 20 degree

      if (abs(ref_patch_temp->mean_) < 1e-6)
      {
        float ref_sum = std::accumulate(patch_temp, patch_temp + patch_size_total, 0.0);
        ref_patch_temp->mean_ = ref_sum / patch_size_total;
      }
      const float ref_mean = ref_patch_temp->mean_;

      for (auto itm = pt->obs_.begin(), itme = pt->obs_.end(); itm != itme; ++itm)
      {
        if ((*itm)->id_ == ref_patch_temp->id_) continue;
        float *patch_cache = (*itm)->patch_;

        if (abs((*itm)->mean_) < 1e-6)
        {
          float other_sum = std::accumulate(patch_cache, patch_cache + patch_size_total, 0.0);
          (*itm)->mean_ = other_sum / patch_size_total;
        }
        const float other_mean = (*itm)->mean_;

        float NCC_up = 0.0;
        float NCC_down1 = 0.0;
        float NCC_down2 = 0.0;
        for (int ind = 0; ind < patch_size_total; ind++)
        {
          NCC_up += (patch_temp[ind] - ref_mean) * (patch_cache[ind] - other_mean);
          NCC_down1 += (patch_temp[ind] - ref_mean) * (patch_temp[ind] - ref_mean);
          NCC_down2 += (patch_cache[ind] - other_mean) * (patch_cache[ind] - other_mean);
        }
        NCC += fabs(NCC_up / sqrt(NCC_down1 * NCC_down2 + 1e-10));
        count++;
      }

      NCC = NCC / count;

      score = NCC + cos_angle;

      ref_patch_temp->score_ = score;

      if (score > score_max)
      {
        score_max = score;
        pt->ref_patch = ref_patch_temp;
        pt->has_ref_patch_ = true;
      }
    }

  }
}

void VIOManager::projectPatchFromRefToCur(const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map)
{
  if (total_points == 0) return;
  // if(new_frame_->id_ != 2) return; //124

  // Debug output for the single-camera case only (camera 0).
  const FramePtr &new_frame_ = cams[0].frame;
  const vk::AbstractCamera &cam0 = *cams[0].cam;
  int patch_size = 25;
  string dir = string(ROOT_DIR) + "Log/ref_cur_combine/";

  cv::Mat result = cv::Mat::zeros(height, width, CV_8UC1);
  cv::Mat result_normal = cv::Mat::zeros(height, width, CV_8UC1);
  cv::Mat result_dense = cv::Mat::zeros(height, width, CV_8UC1);

  cv::Mat img_photometric_error = new_frame_->img_.clone();

  uchar *it = (uchar *)result.data;
  uchar *it_normal = (uchar *)result_normal.data;
  uchar *it_dense = (uchar *)result_dense.data;

  struct pixel_member
  {
    Vector2f pixel_pos;
    uint8_t pixel_value;
  };

  int num = 0;
  for (int i = 0; i < visual_submap->voxel_points.size(); i++)
  {
    VisualPoint *pt = visual_submap->voxel_points[i];

    if (pt->is_normal_initialized_)
    {
      Feature *ref_ftr;
      ref_ftr = pt->ref_patch;
      // Feature* ref_ftr;
      V2D pc(new_frame_->w2c(pt->pos_));
      V2D pc_prior(new_frame_->w2c_prior(pt->pos_));

      V3D norm_vec(ref_ftr->T_f_w_.rotation_matrix() * pt->normal_);
      V3D pf(ref_ftr->T_f_w_ * pt->pos_);

      if (pf.dot(norm_vec) < 0) norm_vec = -norm_vec;

      // norm_vec << norm_vec(1), norm_vec(0), norm_vec(2);
      cv::Mat img_cur = new_frame_->img_;
      cv::Mat img_ref = ref_ftr->img_;

      SE3 T_cur_ref = new_frame_->T_f_w_ * ref_ftr->T_f_w_.inverse();
      Matrix2d A_cur_ref;
      getWarpMatrixAffineHomography(cam0, cam0, ref_ftr->px_, pf, norm_vec, T_cur_ref, 0, A_cur_ref);

      // const Matrix2f A_ref_cur = A_cur_ref.inverse().cast<float>();
      int search_level = getBestSearchLevel(A_cur_ref.inverse(), 2);

      double D = A_cur_ref.determinant();
      if (D > 3) continue;

      num++;

      cv::Mat ref_cur_combine_temp;
      int radius = 20;
      cv::hconcat(img_cur, img_ref, ref_cur_combine_temp);
      cv::cvtColor(ref_cur_combine_temp, ref_cur_combine_temp, CV_GRAY2BGR);

      getImagePatch(img_cur, pc, patch_buffer.data(), 0);

      float error_est = 0.0;
      float error_gt = 0.0;

      for (int ind = 0; ind < patch_size_total; ind++)
      {
        error_est += (ref_ftr->inv_expo_time_ * visual_submap->warp_patch[i][ind] - state->inv_expo_time * patch_buffer[ind]) *
                     (ref_ftr->inv_expo_time_ * visual_submap->warp_patch[i][ind] - state->inv_expo_time * patch_buffer[ind]);
      }
      std::string ref_est = "ref_est " + std::to_string(1.0 / ref_ftr->inv_expo_time_);
      std::string cur_est = "cur_est " + std::to_string(1.0 / state->inv_expo_time);
      std::string cur_propa = "cur_gt " + std::to_string(error_gt);
      std::string cur_optimize = "cur_est " + std::to_string(error_est);

      cv::putText(ref_cur_combine_temp, ref_est, cv::Point2f(ref_ftr->px_[0] + img_cur.cols - 40, ref_ftr->px_[1] + 40), cv::FONT_HERSHEY_COMPLEX, 0.4,
                  cv::Scalar(0, 255, 0), 1, 8, 0);

      cv::putText(ref_cur_combine_temp, cur_est, cv::Point2f(pc[0] - 40, pc[1] + 40), cv::FONT_HERSHEY_COMPLEX, 0.4, cv::Scalar(0, 255, 0), 1, 8, 0);
      cv::putText(ref_cur_combine_temp, cur_propa, cv::Point2f(pc[0] - 40, pc[1] + 60), cv::FONT_HERSHEY_COMPLEX, 0.4, cv::Scalar(0, 0, 255), 1, 8,
                  0);
      cv::putText(ref_cur_combine_temp, cur_optimize, cv::Point2f(pc[0] - 40, pc[1] + 80), cv::FONT_HERSHEY_COMPLEX, 0.4, cv::Scalar(0, 255, 0), 1, 8,
                  0);

      cv::rectangle(ref_cur_combine_temp, cv::Point2f(ref_ftr->px_[0] + img_cur.cols - radius, ref_ftr->px_[1] - radius),
                    cv::Point2f(ref_ftr->px_[0] + img_cur.cols + radius, ref_ftr->px_[1] + radius), cv::Scalar(0, 0, 255), 1);
      cv::rectangle(ref_cur_combine_temp, cv::Point2f(pc[0] - radius, pc[1] - radius), cv::Point2f(pc[0] + radius, pc[1] + radius),
                    cv::Scalar(0, 255, 0), 1);
      cv::rectangle(ref_cur_combine_temp, cv::Point2f(pc_prior[0] - radius, pc_prior[1] - radius),
                    cv::Point2f(pc_prior[0] + radius, pc_prior[1] + radius), cv::Scalar(255, 255, 255), 1);
      cv::circle(ref_cur_combine_temp, cv::Point2f(ref_ftr->px_[0] + img_cur.cols, ref_ftr->px_[1]), 1, cv::Scalar(0, 0, 255), -1, 8);
      cv::circle(ref_cur_combine_temp, cv::Point2f(pc[0], pc[1]), 1, cv::Scalar(0, 255, 0), -1, 8);
      cv::circle(ref_cur_combine_temp, cv::Point2f(pc_prior[0], pc_prior[1]), 1, cv::Scalar(255, 255, 255), -1, 8);
      cv::imwrite(dir + std::to_string(new_frame_->id_) + "_" + std::to_string(ref_ftr->id_) + "_" + std::to_string(num) + ".png",
                  ref_cur_combine_temp);

      std::vector<std::vector<pixel_member>> pixel_warp_matrix;

      for (int y = 0; y < patch_size; ++y)
      {
        vector<pixel_member> pixel_warp_vec;
        for (int x = 0; x < patch_size; ++x) //, ++patch_ptr)
        {
          Vector2f px_patch(x - patch_size / 2, y - patch_size / 2);
          px_patch *= (1 << search_level);
          const Vector2f px_ref(px_patch + ref_ftr->px_.cast<float>());
          uint8_t pixel_value = (uint8_t)vk::interpolateMat_8u(img_ref, px_ref[0], px_ref[1]);

          const Vector2f px(A_cur_ref.cast<float>() * px_patch + pc.cast<float>());
          if (px[0] < 0 || px[1] < 0 || px[0] >= img_cur.cols - 1 || px[1] >= img_cur.rows - 1)
            continue;
          else
          {
            pixel_member pixel_warp;
            pixel_warp.pixel_pos << px[0], px[1];
            pixel_warp.pixel_value = pixel_value;
            pixel_warp_vec.push_back(pixel_warp);
          }
        }
        pixel_warp_matrix.push_back(pixel_warp_vec);
      }

      float x_min = 1000;
      float y_min = 1000;
      float x_max = 0;
      float y_max = 0;

      for (int i = 0; i < pixel_warp_matrix.size(); i++)
      {
        vector<pixel_member> pixel_warp_row = pixel_warp_matrix[i];
        for (int j = 0; j < pixel_warp_row.size(); j++)
        {
          float x_temp = pixel_warp_row[j].pixel_pos[0];
          float y_temp = pixel_warp_row[j].pixel_pos[1];
          if (x_temp < x_min) x_min = x_temp;
          if (y_temp < y_min) y_min = y_temp;
          if (x_temp > x_max) x_max = x_temp;
          if (y_temp > y_max) y_max = y_temp;
        }
      }
      int x_min_i = floor(x_min);
      int y_min_i = floor(y_min);
      int x_max_i = ceil(x_max);
      int y_max_i = ceil(y_max);
      Matrix2f A_cur_ref_Inv = A_cur_ref.inverse().cast<float>();
      for (int i = x_min_i; i < x_max_i; i++)
      {
        for (int j = y_min_i; j < y_max_i; j++)
        {
          Eigen::Vector2f pc_temp(i, j);
          Vector2f px_patch = A_cur_ref_Inv * (pc_temp - pc.cast<float>());
          if (px_patch[0] > (-patch_size / 2 * (1 << search_level)) && px_patch[0] < (patch_size / 2 * (1 << search_level)) &&
              px_patch[1] > (-patch_size / 2 * (1 << search_level)) && px_patch[1] < (patch_size / 2 * (1 << search_level)))
          {
            const Vector2f px_ref(px_patch + ref_ftr->px_.cast<float>());
            uint8_t pixel_value = (uint8_t)vk::interpolateMat_8u(img_ref, px_ref[0], px_ref[1]);
            it_normal[width * j + i] = pixel_value;
          }
        }
      }
    }
  }
  for (int i = 0; i < visual_submap->voxel_points.size(); i++)
  {
    VisualPoint *pt = visual_submap->voxel_points[i];

    if (!pt->is_normal_initialized_) continue;

    Feature *ref_ftr;
    V2D pc(new_frame_->w2c(pt->pos_));
    ref_ftr = pt->ref_patch;

    Matrix2d A_cur_ref;
    getWarpMatrixAffine(cam0, cam0, ref_ftr->px_, ref_ftr->f_, (ref_ftr->pos() - pt->pos_).norm(), new_frame_->T_f_w_ * ref_ftr->T_f_w_.inverse(),
                        0, 0, patch_size_half, A_cur_ref);
    int search_level = getBestSearchLevel(A_cur_ref.inverse(), 2);
    double D = A_cur_ref.determinant();
    if (D > 3) continue;

    cv::Mat img_cur = new_frame_->img_;
    cv::Mat img_ref = ref_ftr->img_;
    for (int y = 0; y < patch_size; ++y)
    {
      for (int x = 0; x < patch_size; ++x) //, ++patch_ptr)
      {
        Vector2f px_patch(x - patch_size / 2, y - patch_size / 2);
        px_patch *= (1 << search_level);
        const Vector2f px_ref(px_patch + ref_ftr->px_.cast<float>());
        uint8_t pixel_value = (uint8_t)vk::interpolateMat_8u(img_ref, px_ref[0], px_ref[1]);

        const Vector2f px(A_cur_ref.cast<float>() * px_patch + pc.cast<float>());
        if (px[0] < 0 || px[1] < 0 || px[0] >= img_cur.cols - 1 || px[1] >= img_cur.rows - 1)
          continue;
        else
        {
          int col = int(px[0]);
          int row = int(px[1]);
          it[width * row + col] = pixel_value;
        }
      }
    }
  }
  cv::Mat ref_cur_combine;
  cv::Mat ref_cur_combine_normal;
  cv::Mat ref_cur_combine_error;

  cv::hconcat(result, new_frame_->img_, ref_cur_combine);
  cv::hconcat(result_normal, new_frame_->img_, ref_cur_combine_normal);

  cv::cvtColor(ref_cur_combine, ref_cur_combine, CV_GRAY2BGR);
  cv::cvtColor(ref_cur_combine_normal, ref_cur_combine_normal, CV_GRAY2BGR);
  cv::absdiff(img_photometric_error, result_normal, img_photometric_error);
  cv::hconcat(img_photometric_error, new_frame_->img_, ref_cur_combine_error);

  cv::imwrite(dir + std::to_string(new_frame_->id_) + "_0_" + ".png", ref_cur_combine);
  cv::imwrite(dir + std::to_string(new_frame_->id_) + +"_0_" +
                  "photometric"
                  ".png",
              ref_cur_combine_error);
  cv::imwrite(dir + std::to_string(new_frame_->id_) + "_0_" + "normal" + ".png", ref_cur_combine_normal);
}

void VIOManager::precomputeReferencePatches(int level)
{
  double t1 = omp_get_wtime();
  if (total_points == 0) return;
  MD(1, 2) Jimg;
  MD(2, 3) Jdpi;
  MD(1, 3) Jdphi, Jdp, JdR, Jdt;

  const int H_DIM = total_points * patch_size_total;

  H_sub_inv.resize(H_DIM, 6);
  H_sub_inv.setZero();
  M3D p_w_hat;

  for (int i = 0; i < total_points; i++)
  {
    const int scale = (1 << level);

    VisualPoint *pt = visual_submap->voxel_points[i];

    if (pt == nullptr) continue;

    cv::Mat img = pt->ref_patch->img_;

    double depth((pt->pos_ - pt->ref_patch->pos()).norm());
    V3D pf = pt->ref_patch->f_ * depth;
    V2D pc = pt->ref_patch->px_;
    M3D R_ref_w = pt->ref_patch->T_f_w_.rotation_matrix();

    computeProjectionJacobian(cams[pt->ref_patch->cam_id_], pf, Jdpi);
    p_w_hat << SKEW_SYM_MATRX(pt->pos_);

    const float u_ref = pc[0];
    const float v_ref = pc[1];
    const int u_ref_i = floorf(pc[0] / scale) * scale;
    const int v_ref_i = floorf(pc[1] / scale) * scale;
    const float subpix_u_ref = (u_ref - u_ref_i) / scale;
    const float subpix_v_ref = (v_ref - v_ref_i) / scale;
    const float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
    const float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
    const float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
    const float w_ref_br = subpix_u_ref * subpix_v_ref;

    for (int x = 0; x < patch_size; x++)
    {
      uint8_t *img_ptr = (uint8_t *)img.data + (v_ref_i + x * scale - patch_size_half * scale) * width + u_ref_i - patch_size_half * scale;
      for (int y = 0; y < patch_size; ++y, img_ptr += scale)
      {
        float du =
            0.5f *
            ((w_ref_tl * img_ptr[scale] + w_ref_tr * img_ptr[scale * 2] + w_ref_bl * img_ptr[scale * width + scale] +
              w_ref_br * img_ptr[scale * width + scale * 2]) -
             (w_ref_tl * img_ptr[-scale] + w_ref_tr * img_ptr[0] + w_ref_bl * img_ptr[scale * width - scale] + w_ref_br * img_ptr[scale * width]));
        float dv =
            0.5f *
            ((w_ref_tl * img_ptr[scale * width] + w_ref_tr * img_ptr[scale + scale * width] + w_ref_bl * img_ptr[width * scale * 2] +
              w_ref_br * img_ptr[width * scale * 2 + scale]) -
             (w_ref_tl * img_ptr[-scale * width] + w_ref_tr * img_ptr[-scale * width + scale] + w_ref_bl * img_ptr[0] + w_ref_br * img_ptr[scale]));

        Jimg << du, dv;
        Jimg = Jimg * (1.0 / scale);

        JdR = Jimg * Jdpi * R_ref_w * p_w_hat;
        Jdt = -Jimg * Jdpi * R_ref_w;

        H_sub_inv.block<1, 6>(i * patch_size_total + x * patch_size + y, 0) << JdR, Jdt;
      }
    }
  }
  has_ref_patch_cache = true;
}

void VIOManager::updateStateInverse(int level)
{
  if (total_points == 0) return;
  StatesGroup old_state = (*state);
  V2D pc;
  MD(1, 2) Jimg;
  MD(2, 3) Jdpi;
  MD(1, 3) Jdphi, Jdp, JdR, Jdt;
  VectorXd z;
  MatrixXd H_sub;
  bool EKF_end = false;
  float last_error = std::numeric_limits<float>::max();
  compute_jacobian_time = update_ekf_time = 0.0;
  M3D P_wi_hat;
  bool z_init = true;
  const int H_DIM = total_points * patch_size_total;

  z.resize(H_DIM);
  z.setZero();

  H_sub.resize(H_DIM, 6);
  H_sub.setZero();

  for (int iteration = 0; iteration < max_iterations; iteration++)
  {
    double t1 = omp_get_wtime();
    double count_outlier = 0;
    if (has_ref_patch_cache == false) precomputeReferencePatches(level);
    int n_meas = 0;
    float error = 0.0;
    M3D Rwi(state->rot_end);
    V3D Pwi(state->pos_end);
    P_wi_hat << SKEW_SYM_MATRX(Pwi);
    for (auto &cc : cams)
    {
      cc.Rcw = cc.Rci * Rwi.transpose();
      cc.Pcw = -cc.Rci * Rwi.transpose() * Pwi + cc.Pci;
    }

    M3D p_hat;

    for (int i = 0; i < total_points; i++)
    {
      float patch_error = 0.0;

      const int scale = (1 << level);

      VisualPoint *pt = visual_submap->voxel_points[i];

      if (pt == nullptr) continue;

      const CameraContext &cc = cams[visual_submap->cam_ids[i]];
      const cv::Mat &img = cc.img;
      V3D pf = cc.Rcw * pt->pos_ + cc.Pcw;
      pc = cc.cam->world2cam(pf);

      const float u_ref = pc[0];
      const float v_ref = pc[1];
      const int u_ref_i = floorf(pc[0] / scale) * scale;
      const int v_ref_i = floorf(pc[1] / scale) * scale;
      const float subpix_u_ref = (u_ref - u_ref_i) / scale;
      const float subpix_v_ref = (v_ref - v_ref_i) / scale;
      const float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
      const float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
      const float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
      const float w_ref_br = subpix_u_ref * subpix_v_ref;

      const vector<float> &P = visual_submap->warp_patch[i];
      for (int x = 0; x < patch_size; x++)
      {
        uint8_t *img_ptr = (uint8_t *)img.data + (v_ref_i + x * scale - patch_size_half * scale) * width + u_ref_i - patch_size_half * scale;
        for (int y = 0; y < patch_size; ++y, img_ptr += scale)
        {
          double res = w_ref_tl * img_ptr[0] + w_ref_tr * img_ptr[scale] + w_ref_bl * img_ptr[scale * width] +
                       w_ref_br * img_ptr[scale * width + scale] - P[patch_size_total * level + x * patch_size + y];
          z(i * patch_size_total + x * patch_size + y) = res;
          patch_error += res * res;
          MD(1, 3) J_dR = H_sub_inv.block<1, 3>(i * patch_size_total + x * patch_size + y, 0);
          MD(1, 3) J_dt = H_sub_inv.block<1, 3>(i * patch_size_total + x * patch_size + y, 3);
          JdR = J_dR * Rwi + J_dt * P_wi_hat * Rwi;
          Jdt = J_dt * Rwi;
          H_sub.block<1, 6>(i * patch_size_total + x * patch_size + y, 0) << JdR, Jdt;
          n_meas++;
        }
      }
      visual_submap->errors[i] = patch_error;
      error += patch_error;
    }

    error = error / n_meas;

    compute_jacobian_time += omp_get_wtime() - t1;

    double t3 = omp_get_wtime();

    if (error <= last_error)
    {
      old_state = (*state);
      last_error = error;

      auto &&H_sub_T = H_sub.transpose();
      H_T_H.setZero();
      G.setZero();
      H_T_H.block<6, 6>(0, 0) = H_sub_T * H_sub;
      MD(DIM_STATE, DIM_STATE) &&K_1 = (H_T_H + (state->cov / img_point_cov).inverse()).inverse();
      auto &&HTz = H_sub_T * z;
      auto vec = (*state_propagat) - (*state);
      G.block<DIM_STATE, 6>(0, 0) = K_1.block<DIM_STATE, 6>(0, 0) * H_T_H.block<6, 6>(0, 0);
      auto solution = -K_1.block<DIM_STATE, 6>(0, 0) * HTz + vec - G.block<DIM_STATE, 6>(0, 0) * vec.block<6, 1>(0, 0);
      (*state) += solution;
      auto &&rot_add = solution.block<3, 1>(0, 0);
      auto &&t_add = solution.block<3, 1>(3, 0);

      if ((rot_add.norm() * 57.3f < 0.001f) && (t_add.norm() * 100.0f < 0.001f)) { EKF_end = true; }
    }
    else
    {
      (*state) = old_state;
      EKF_end = true;
    }

    update_ekf_time += omp_get_wtime() - t3;

    if (iteration == max_iterations || EKF_end) break;
  }
}

void VIOManager::updateState(int level)
{
  if (total_points == 0) return;
  StatesGroup old_state = (*state);

  VectorXd z;
  MatrixXd H_sub;
  bool EKF_end = false;
  float last_error = std::numeric_limits<float>::max();

  // Measured states: rotation and position (columns 0-5) and, with exposure estimation, one inverse exposure
  // time per exposure group (columns 6..), stored at StatesGroup::expoIndex() in the full state.
  const int n_expo = exposure_estimate_en ? numExpoGroups() : 0;
  const int n_col = 6 + n_expo;
  std::vector<int> col_to_state(n_col);
  for (int k = 0; k < 6; k++) col_to_state[k] = k;
  for (int g = 0; g < n_expo; g++) col_to_state[6 + g] = StatesGroup::expoIndex(g);

  const int H_DIM = total_points * patch_size_total;
  z.resize(H_DIM);
  z.setZero();
  H_sub.resize(H_DIM, n_col);
  H_sub.setZero();

  for (int iteration = 0; iteration < max_iterations; iteration++)
  {
    double t1 = omp_get_wtime();

    M3D Rwi(state->rot_end);
    V3D Pwi(state->pos_end);
    for (auto &cc : cams)
    {
      cc.Rcw = cc.Rci * Rwi.transpose();
      cc.Pcw = -cc.Rci * Rwi.transpose() * Pwi + cc.Pci;
      cc.Jdp_dt = cc.Rci * Rwi.transpose();
    }

    float error = 0.0;
    int n_meas = 0;
    // int max_threads = omp_get_max_threads();
    // int desired_threads = std::min(max_threads, total_points);
    // omp_set_num_threads(desired_threads);

    #ifdef MP_EN
      omp_set_num_threads(MP_PROC_NUM);
      #pragma omp parallel for reduction(+:error, n_meas)
    #endif
    for (int i = 0; i < total_points; i++)
    {
      // printf("thread is %d, i=%d, i address is %p\n", omp_get_thread_num(), i, &i);
      MD(1, 2) Jimg;
      MD(2, 3) Jdpi;
      MD(1, 3) Jdphi, Jdp, JdR, Jdt;

      float patch_error = 0.0;
      int search_level = visual_submap->search_levels[i];
      int pyramid_level = level + search_level;
      int scale = (1 << pyramid_level);
      float inv_scale = 1.0f / scale;

      VisualPoint *pt = visual_submap->voxel_points[i];

      if (pt == nullptr) continue;

      const CameraContext &cc = cams[visual_submap->cam_ids[i]];
      const cv::Mat &img = cc.img;
      const int expo_group = expoGroup(cc.id);
      const double inv_expo_cur = state->invExpo(expo_group);

      V3D pf = cc.Rcw * pt->pos_ + cc.Pcw;
      V2D pc = cc.cam->world2cam(pf);

      computeProjectionJacobian(cc, pf, Jdpi);
      M3D p_hat;
      p_hat << SKEW_SYM_MATRX(pf);

      float u_ref = pc[0];
      float v_ref = pc[1];
      int u_ref_i = floorf(pc[0] / scale) * scale;
      int v_ref_i = floorf(pc[1] / scale) * scale;
      float subpix_u_ref = (u_ref - u_ref_i) / scale;
      float subpix_v_ref = (v_ref - v_ref_i) / scale;
      float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
      float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
      float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
      float w_ref_br = subpix_u_ref * subpix_v_ref;

      const vector<float> &P = visual_submap->warp_patch[i];
      double inv_ref_expo = visual_submap->inv_expo_list[i];
      // ROS_ERROR("inv_ref_expo: %.3lf, state->inv_expo_time: %.3lf\n", inv_ref_expo, state->inv_expo_time);

      for (int x = 0; x < patch_size; x++)
      {
        uint8_t *img_ptr = (uint8_t *)img.data + (v_ref_i + x * scale - patch_size_half * scale) * width + u_ref_i - patch_size_half * scale;
        for (int y = 0; y < patch_size; ++y, img_ptr += scale)
        {
          float du =
              0.5f *
              ((w_ref_tl * img_ptr[scale] + w_ref_tr * img_ptr[scale * 2] + w_ref_bl * img_ptr[scale * width + scale] +
                w_ref_br * img_ptr[scale * width + scale * 2]) -
               (w_ref_tl * img_ptr[-scale] + w_ref_tr * img_ptr[0] + w_ref_bl * img_ptr[scale * width - scale] + w_ref_br * img_ptr[scale * width]));
          float dv =
              0.5f *
              ((w_ref_tl * img_ptr[scale * width] + w_ref_tr * img_ptr[scale + scale * width] + w_ref_bl * img_ptr[width * scale * 2] +
                w_ref_br * img_ptr[width * scale * 2 + scale]) -
               (w_ref_tl * img_ptr[-scale * width] + w_ref_tr * img_ptr[-scale * width + scale] + w_ref_bl * img_ptr[0] + w_ref_br * img_ptr[scale]));

          Jimg << du, dv;
          Jimg = Jimg * inv_expo_cur;
          Jimg = Jimg * inv_scale;
          Jdphi = Jimg * Jdpi * p_hat;
          Jdp = -Jimg * Jdpi;
          JdR = Jdphi * cc.Jdphi_dR + Jdp * cc.Jdp_dR;
          Jdt = Jdp * cc.Jdp_dt;

          double cur_value =
              w_ref_tl * img_ptr[0] + w_ref_tr * img_ptr[scale] + w_ref_bl * img_ptr[scale * width] + w_ref_br * img_ptr[scale * width + scale];
          double res = inv_expo_cur * cur_value - inv_ref_expo * P[patch_size_total * level + x * patch_size + y];

          const int row = i * patch_size_total + x * patch_size + y;
          z(row) = res;

          patch_error += res * res;
          n_meas += 1;

          H_sub.block<1, 6>(row, 0) << JdR, Jdt;
          if (exposure_estimate_en) H_sub(row, 6 + expo_group) = cur_value;
        }
      }
      visual_submap->errors[i] = patch_error;
      error += patch_error;
    }

    error = error / n_meas;

    compute_jacobian_time += omp_get_wtime() - t1;

    // printf("\nPYRAMID LEVEL %i\n---------------\n", level);
    // std::cout << "It. " << iteration
    //           << "\t last_error = " << last_error
    //           << "\t new_error = " << error
    //           << std::endl;

    double t3 = omp_get_wtime();

    if (error <= last_error)
    {
      old_state = (*state);
      last_error = error;

      // K = (H.transpose() / img_point_cov * H + state->cov.inverse()).inverse() * H.transpose() / img_point_cov; auto
      // vec = (*state_propagat) - (*state); G = K*H;
      // (*state) += (-K*z + vec - G*vec);

      const MatrixXd HTH_c = H_sub.transpose() * H_sub;
      const VectorXd HTz_c = H_sub.transpose() * z;
      H_T_H.setZero();
      VD(DIM_STATE) HTz = VD(DIM_STATE)::Zero();
      for (int a = 0; a < n_col; a++)
      {
        HTz(col_to_state[a]) = HTz_c(a);
        for (int b = 0; b < n_col; b++) H_T_H(col_to_state[a], col_to_state[b]) = HTH_c(a, b);
      }
      MD(DIM_STATE, DIM_STATE) &&K_1 = (H_T_H + (state->cov / img_point_cov).inverse()).inverse();
      auto vec = (*state_propagat) - (*state);
      G = K_1 * H_T_H;
      MD(DIM_STATE, 1)
      solution = -K_1 * HTz + vec - G * vec;

      (*state) += solution;
      auto &&rot_add = solution.block<3, 1>(0, 0);
      auto &&t_add = solution.block<3, 1>(3, 0);

      // if ((rot_add.norm() * 57.3f < 0.001f) && (t_add.norm() * 100.0f < 0.001f) && (expo_add.norm() < 0.001f)) EKF_end = true;
      if ((rot_add.norm() * 57.3f < 0.001f) && (t_add.norm() * 100.0f < 0.001f))  EKF_end = true;
    }
    else
    {
      (*state) = old_state;
      EKF_end = true;
    }

    update_ekf_time += omp_get_wtime() - t3;

    if (iteration == max_iterations || EKF_end) break;
  }
  // if (state->inv_expo_time < 0.0)  {ROS_ERROR("reset expo time!!!!!!!!!!\n"); state->inv_expo_time = 0.0;}
}

void VIOManager::updateFrameState(StatesGroup state)
{
  M3D Rwi(state.rot_end);
  V3D Pwi(state.pos_end);
  for (auto &cc : cams)
  {
    cc.Rcw = cc.Rci * Rwi.transpose();
    cc.Pcw = -cc.Rci * Rwi.transpose() * Pwi + cc.Pci;
    if (cc.frame) cc.frame->T_f_w_ = SE3(cc.Rcw, cc.Pcw);
  }
  Rcw = cams[0].Rcw;
  Pcw = cams[0].Pcw;
}

void VIOManager::plotTrackedPoints()
{
  int total_points = visual_submap->voxel_points.size();
  if (total_points == 0) return;
  for (int i = 0; i < total_points; i++)
  {
    VisualPoint *pt = visual_submap->voxel_points[i];
    CameraContext &cc = cams[visual_submap->cam_ids[i]];
    V2D pc(cc.frame->w2c(pt->pos_));

    if (visual_submap->errors[i] <= visual_submap->propa_errors[i])
    {
      // inlier_count++;
      cv::circle(cc.img_cp, cv::Point2f(pc[0], pc[1]), 7, cv::Scalar(0, 255, 0), -1, 8); // Green Sparse Align tracked
    }
    else
    {
      cv::circle(cc.img_cp, cv::Point2f(pc[0], pc[1]), 7, cv::Scalar(255, 0, 0), -1, 8); // Blue Sparse Align tracked
    }
  }
}

V3F VIOManager::getInterpolatedPixel(cv::Mat img, V2D pc)
{
  const float u_ref = pc[0];
  const float v_ref = pc[1];
  const int u_ref_i = floorf(pc[0]);
  const int v_ref_i = floorf(pc[1]);
  const float subpix_u_ref = (u_ref - u_ref_i);
  const float subpix_v_ref = (v_ref - v_ref_i);
  const float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
  const float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
  const float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
  const float w_ref_br = subpix_u_ref * subpix_v_ref;
  uint8_t *img_ptr = (uint8_t *)img.data + ((v_ref_i)*width + (u_ref_i)) * 3;
  float B = w_ref_tl * img_ptr[0] + w_ref_tr * img_ptr[0 + 3] + w_ref_bl * img_ptr[width * 3] + w_ref_br * img_ptr[width * 3 + 0 + 3];
  float G = w_ref_tl * img_ptr[1] + w_ref_tr * img_ptr[1 + 3] + w_ref_bl * img_ptr[1 + width * 3] + w_ref_br * img_ptr[width * 3 + 1 + 3];
  float R = w_ref_tl * img_ptr[2] + w_ref_tr * img_ptr[2 + 3] + w_ref_bl * img_ptr[2 + width * 3] + w_ref_br * img_ptr[width * 3 + 2 + 3];
  V3F pixel(B, G, R);
  return pixel;
}

void VIOManager::dumpDataForColmap()
{
  static int cnt = 1;
  std::ostringstream ss;
  ss << std::setw(5) << std::setfill('0') << cnt;
  std::string cnt_str = ss.str();
  std::string image_path = std::string(ROOT_DIR) + "Log/Colmap/images/" + cnt_str + ".png";

  cv::Mat img_rgb_undistort;
  pinhole_cam->undistortImage(cams[0].img_rgb, img_rgb_undistort);
  cv::imwrite(image_path, img_rgb_undistort);

  Eigen::Quaterniond q(cams[0].frame->T_f_w_.rotation_matrix());
  Eigen::Vector3d t = cams[0].frame->T_f_w_.translation();
  fout_colmap << cnt << " "
            << std::fixed << std::setprecision(6)  // 保证浮点数精度为6位
            << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << " "
            << t.x() << " " << t.y() << " " << t.z() << " "
            << 1 << " "  // CAMERA_ID (假设相机ID为1)
            << cnt_str << ".png" << std::endl;
  fout_colmap << "0.0 0.0 -1" << std::endl;
  cnt++;
}

void VIOManager::processFrame(cv::Mat &img, vector<pointWithVar> &pg, const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &feat_map, double img_time)
{
  std::vector<cv::Mat> imgs{img};
  processFrame(imgs, pg, feat_map, img_time);
  img = imgs[0];
}

void VIOManager::processFrame(std::vector<cv::Mat> &imgs, vector<pointWithVar> &pg, const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &feat_map,
                              double img_time)
{
  visual_submap->reset();
  total_points = 0;
  updateCameraExtrinsics();

  int n_frames = 0;
  for (auto &cc : cams)
  {
    cc.frame.reset();
    if (cc.id >= static_cast<int>(imgs.size()) || imgs[cc.id].empty()) continue;
    cv::Mat &img = imgs[cc.id];
    if (width != img.cols || height != img.rows)
    {
      cv::resize(img, img, cv::Size(img.cols * image_resize_factor, img.rows * image_resize_factor), 0, 0, CV_INTER_LINEAR);
    }
    cv::Mat img_vig;
    if (!cc.vignetting_gain.empty() && img.channels() == 3)
    {
      cv::Mat f;
      img.convertTo(f, CV_32FC3);
      cv::multiply(f, cc.vignetting_gain, f);
      f.convertTo(img_vig, CV_8UC3); // saturating
      if (!vignetting_color_only) img = img_vig;
    }
    cc.img_rgb = img_vig.empty() ? img.clone() : img_vig;
    cc.img_cp = img.clone();
    // Always a fresh buffer: features of earlier frames keep references to the previous gray image.
    cv::Mat gray;
    if (img.channels() == 3) cv::cvtColor(img, gray, CV_BGR2GRAY);
    else gray = img.clone();
    cc.img = gray;
    cc.frame.reset(new Frame(cc.cam, cc.img));
    n_frames++;
  }
  if (n_frames == 0)
  {
    printf("[ VIO ] Empty Image!\n");
    return;
  }

  updateFrameState(*state);

  double t1 = omp_get_wtime();

  for (auto &cc : cams)
  {
    if (!cc.frame) continue;
    resetGrid(cc);
    retrieveFromVisualSparseMap(cc, pg, feat_map);
  }
  total_points = visual_submap->voxel_points.size();
  printf("[ VIO ] Retrieve %d points from visual sparse map (%d cameras)\n", total_points, n_frames);

  double t2 = omp_get_wtime();

  computeJacobianAndUpdateEKF();

  // Per-camera photometric RMS (exposure-compensated intensity units) of the tracked patches before / after the
  // update, and the median NCC between warped reference patch and current patch after the update (independent of
  // the exposure estimate, i.e. a check of the geometric alignment).
  for (const auto &cc : cams)
  {
    if (!cc.frame) continue;
    double e0 = 0.0, e1 = 0.0;
    std::vector<double> nccs;
    for (int i = 0; i < total_points; i++)
    {
      if (visual_submap->cam_ids[i] != cc.id) continue;
      e0 += visual_submap->propa_errors[i];
      e1 += visual_submap->errors[i];
      V2D pc;
      if (projectToCamera(cc, visual_submap->voxel_points[i]->pos_, pc, border))
      {
        getImagePatch(cc.img, pc, patch_buffer.data(), 0);
        nccs.push_back(calculateNCC(visual_submap->warp_patch[i].data(), patch_buffer.data(), patch_size_total));
      }
    }
    const int n = nccs.size();
    if (n > 0)
    {
      std::nth_element(nccs.begin(), nccs.begin() + n / 2, nccs.end());
      printf("[ VIO ] Camera %d: %d patches, photometric RMS %.2f -> %.2f, NCC %.3f, inv expo %.4f\n", cc.id, n, std::sqrt(e0 / (n * patch_size_total)),
             std::sqrt(e1 / (n * patch_size_total)), nccs[n / 2], state->invExpo(expoGroup(cc.id)));
    }
  }

  double t3 = omp_get_wtime();

  vector<char> pg_used(pg.size(), 0);
  for (auto &cc : cams)
  {
    if (cc.frame) generateVisualMapPoints(cc, pg, pg_used);
  }

  double t4 = omp_get_wtime();

  plotTrackedPoints();

  if (plot_flag && cams.size() == 1) projectPatchFromRefToCur(feat_map);

  double t5 = omp_get_wtime();

  updateVisualMapPoints();

  double t6 = omp_get_wtime();

  updateReferencePatch(feat_map);

  double t7 = omp_get_wtime();

  if(colmap_output_en)  dumpDataForColmap();

  if (exposure_estimate_en && exposure_regauge) regaugeExposure();

  frame_count++;
  ave_total = ave_total * (frame_count - 1) / frame_count + (t7 - t1 - (t5 - t4)) / frame_count;

  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;34m|                         VIO Time                            |\033[0m\n");
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;34m| %-29s | %-27zu |\033[0m\n", "Sparse Map Size", feat_map.size());
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;34m| %-29s | %-27s |\033[0m\n", "Algorithm Stage", "Time (secs)");
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "retrieveFromVisualSparseMap", t2 - t1);
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "computeJacobianAndUpdateEKF", t3 - t2);
  printf("\033[1;32m| %-27s   | %-27lf |\033[0m\n", "-> computeJacobian", compute_jacobian_time);
  printf("\033[1;32m| %-27s   | %-27lf |\033[0m\n", "-> updateEKF", update_ekf_time);
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "generateVisualMapPoints", t4 - t3);
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "updateVisualMapPoints", t6 - t5);
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "updateReferencePatch", t7 - t6);
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "Current Total Time", t7 - t1 - (t5 - t4));
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "Average Total Time", ave_total);
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
}
