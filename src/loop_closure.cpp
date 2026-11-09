#include "loop_closure.h"

#include <Eigen/Sparse>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/registration/gicp.h>
#include <unordered_map>

namespace
{
typedef pcl::PointCloud<pcl::PointXYZ> Cloud;
typedef Eigen::Matrix<double, 6, 1> V6;
typedef Eigen::Matrix<double, 6, 6> M6;

Cloud::Ptr voxelDown(const Cloud &in, double voxel)
{
  struct Acc
  {
    double x = 0, y = 0, z = 0;
    int n = 0;
  };
  std::unordered_map<int64_t, Acc> grid;
  grid.reserve(in.size() / 2 + 1);
  const double inv = 1.0 / voxel;
  for (const auto &p : in.points)
  {
    const int64_t kx = static_cast<int64_t>(std::floor(p.x * inv)) + (1 << 20), ky = static_cast<int64_t>(std::floor(p.y * inv)) + (1 << 20),
                  kz = static_cast<int64_t>(std::floor(p.z * inv)) + (1 << 20);
    Acc &a = grid[(kx << 42) | (ky << 21) | kz];
    a.x += p.x;
    a.y += p.y;
    a.z += p.z;
    a.n++;
  }
  Cloud::Ptr out(new Cloud);
  out->reserve(grid.size());
  for (const auto &kv : grid)
    out->push_back(pcl::PointXYZ(kv.second.x / kv.second.n, kv.second.y / kv.second.n, kv.second.z / kv.second.n));
  return out;
}

void appendTransformed(const Cloud &in, const Sophus::SE3 &T, Cloud &out)
{
  const Eigen::Matrix3f R = T.rotation_matrix().cast<float>();
  const Eigen::Vector3f t = T.translation().cast<float>();
  out.reserve(out.size() + in.size());
  for (const auto &p : in.points)
  {
    const Eigen::Vector3f q = R * p.getVector3fMap() + t;
    out.push_back(pcl::PointXYZ(q[0], q[1], q[2]));
  }
}

Eigen::Matrix4f toMatrix(const Sophus::SE3 &T)
{
  Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
  m.block<3, 3>(0, 0) = T.rotation_matrix().cast<float>();
  m.block<3, 1>(0, 3) = T.translation().cast<float>();
  return m;
}

Sophus::SE3 fromMatrix(const Eigen::Matrix4f &m)
{
  Eigen::Quaterniond q(m.block<3, 3>(0, 0).cast<double>());
  q.normalize();
  return Sophus::SE3(q, m.block<3, 1>(0, 3).cast<double>());
}

/// Share of points within max_dist of the target (after transforming them with T) and the RMSE of those.
void inlierStats(const Cloud &src, const Sophus::SE3 &T, const pcl::KdTreeFLANN<pcl::PointXYZ> &tree, double max_dist, double &ratio, double &rmse)
{
  const Eigen::Matrix3f R = T.rotation_matrix().cast<float>();
  const Eigen::Vector3f t = T.translation().cast<float>();
  std::vector<int> idx(1);
  std::vector<float> d2(1);
  int n = 0;
  double sum = 0.0;
  for (const auto &p : src.points)
  {
    const Eigen::Vector3f q = R * p.getVector3fMap() + t;
    if (tree.nearestKSearch(pcl::PointXYZ(q[0], q[1], q[2]), 1, idx, d2) > 0 && d2[0] < max_dist * max_dist)
    {
      n++;
      sum += d2[0];
    }
  }
  ratio = src.empty() ? 0.0 : double(n) / src.size();
  rmse = n > 0 ? std::sqrt(sum / n) : 0.0;
}
} // namespace

void LoopClosure::addFrame(double time, const M3D &R, const V3D &p, const PointCloudXYZI &scan_imu)
{
  const Sophus::SE3 pose(R, p);
  if (!frames_.empty()) path_ += (p - frames_.back().pose.translation()).norm();
  frames_.push_back({time, pose});
  corrections_.push_back(Sophus::SE3());

  Cloud::Ptr scan(new Cloud);
  scan->reserve(scan_imu.size());
  for (const auto &q : scan_imu.points) scan->push_back(pcl::PointXYZ(q.x, q.y, q.z));
  pending_.emplace_back(pose, scan);

  bool new_keyframe = keyframes_.empty();
  if (!new_keyframe)
  {
    const Sophus::SE3 d = keyframes_.back().pose.inverse() * pose;
    const double angle = Eigen::AngleAxisd(d.rotation_matrix()).angle() * 180.0 / M_PI;
    new_keyframe = d.translation().norm() >= cfg_.keyframe_dist || angle >= cfg_.keyframe_angle;
  }
  if (!new_keyframe) return;

  Keyframe kf;
  kf.frame = frames_.size() - 1;
  kf.time = time;
  kf.path = path_;
  kf.pose = pose;
  Cloud merged;
  const Sophus::SE3 inv = pose.inverse();
  for (const auto &f : pending_) appendTransformed(*f.second, inv * f.first, merged);
  kf.cloud = voxelDown(merged, cfg_.keyframe_voxel);
  keyframes_.push_back(kf);
  pending_.clear();
}

LoopClosure::Cloud::Ptr LoopClosure::submap(int center, int half) const
{
  Cloud merged;
  const Sophus::SE3 inv = keyframes_[center].pose.inverse();
  const int lo = std::max(0, center - half), hi = std::min(int(keyframes_.size()) - 1, center + half);
  for (int k = lo; k <= hi; k++) appendTransformed(*keyframes_[k].cloud, inv * keyframes_[k].pose, merged);
  return voxelDown(merged, cfg_.keyframe_voxel);
}

LoopClosure::LoopResult LoopClosure::registerLoop(int i, int j) const
{
  LoopResult r{i, j, 0.0, 0.0, 0.0, 0.0, false, false, 0.0, 0.0, Sophus::SE3()};
  const Cloud::Ptr target = submap(j, cfg_.target_half), source = submap(i, cfg_.source_half);
  if (target->size() < 100 || source->size() < 100) return r;
  pcl::KdTreeFLANN<pcl::PointXYZ> tree;
  tree.setInputCloud(target);
  const Sophus::SE3 &Ti = keyframes_[i].pose, &Tj = keyframes_[j].pose;

  // Coarse: the odometry drifts mostly in height, so search the world z offset of keyframe i first.
  Cloud sub;
  const size_t step = std::max<size_t>(1, source->size() / 1500);
  for (size_t k = 0; k < source->size(); k += step) sub.push_back(source->points[k]);
  const double z_range = std::min(cfg_.z_search_max, std::max(cfg_.z_search_min, cfg_.z_search_ratio * (keyframes_[i].path - keyframes_[j].path)));
  const int nz = static_cast<int>(std::ceil(z_range / 0.2));
  double best_ratio = -1.0, rmse;
  for (int k = -nz; k <= nz; k++)
  {
    const double dz = 0.2 * k;
    double ratio;
    inlierStats(sub, Tj.inverse() * Sophus::SE3(M3D::Identity(), V3D(0, 0, dz)) * Ti, tree, 0.3, ratio, rmse);
    if (ratio > best_ratio + 1e-9 || (ratio > best_ratio - 1e-9 && std::fabs(dz) < std::fabs(r.dz)))
    {
      best_ratio = ratio;
      r.dz = dz;
    }
  }
  r.coarse_ratio = best_ratio;
  if (best_ratio < 0.1) return r;

  // Fine: GICP from the best height, first with wide then with narrow correspondences.
  pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> gicp;
  gicp.setInputSource(source);
  gicp.setInputTarget(target);
  gicp.setMaximumIterations(50);
  gicp.setTransformationEpsilon(1e-6);
  Cloud aligned;
  gicp.setMaxCorrespondenceDistance(1.0);
  gicp.align(aligned, toMatrix(Tj.inverse() * Sophus::SE3(M3D::Identity(), V3D(0, 0, r.dz)) * Ti));
  if (!gicp.hasConverged()) return r;
  gicp.setMaxCorrespondenceDistance(0.3);
  gicp.align(aligned, gicp.getFinalTransformation());
  if (!gicp.hasConverged()) return r;

  r.z = fromMatrix(gicp.getFinalTransformation());
  inlierStats(*source, r.z, tree, cfg_.inlier_dist, r.ratio, r.rmse);
  r.accepted = r.ratio >= cfg_.min_inlier_ratio && r.rmse <= cfg_.max_rmse;
  return r;
}

void LoopClosure::optimizeGraph(const std::vector<Edge> &edges, std::vector<Sophus::SE3> &T) const
{
  const int K = T.size(), n = 6 * (K - 1); // keyframe 0 is fixed
  auto information = [](double sigma_t, double sigma_r_deg) {
    const double sr = sigma_r_deg * M_PI / 180.0;
    V6 d;
    d << V3D::Constant(1.0 / (sigma_t * sigma_t)), V3D::Constant(1.0 / (sr * sr));
    return M6(d.asDiagonal());
  };
  const M6 info_odom = information(cfg_.odom_sigma_t, cfg_.odom_sigma_r), info_loop = information(cfg_.loop_sigma_t, cfg_.loop_sigma_r);
  const double c2 = 9.0; // Cauchy kernel on the loop edges (scale 3 in whitened units)
  auto residual = [](const Sophus::SE3 &z, const Sophus::SE3 &a, const Sophus::SE3 &b) { return V6((z.inverse() * a.inverse() * b).log()); };
  auto totalCost = [&](const std::vector<Sophus::SE3> &P) {
    double c = 0.0;
    for (const Edge &e : edges)
    {
      const V6 r = residual(e.z, P[e.a], P[e.b]);
      const double chi2 = r.dot((e.loop ? info_loop : info_odom) * r);
      c += e.loop ? c2 * std::log1p(chi2 / c2) : chi2;
    }
    return c;
  };

  double cost = totalCost(T), lambda = 1e-4;
  const double cost0 = cost;
  for (int it = 0; it < 100; it++)
  {
    std::vector<Eigen::Triplet<double>> trip;
    Eigen::VectorXd g = Eigen::VectorXd::Zero(n);
    auto addBlock = [&trip](int r0, int c0, const M6 &B) {
      for (int a = 0; a < 6; a++)
        for (int b = 0; b < 6; b++) trip.emplace_back(r0 + a, c0 + b, B(a, b));
    };
    for (const Edge &e : edges)
    {
      const M6 &I = e.loop ? info_loop : info_odom;
      const V6 r = residual(e.z, T[e.a], T[e.b]);
      const double w = e.loop ? 1.0 / (1.0 + r.dot(I * r) / c2) : 1.0; // iteratively reweighted
      M6 Ja, Jb;
      const double h = 1e-6;
      for (int k = 0; k < 6; k++)
      {
        V6 d = V6::Zero();
        d[k] = h;
        Ja.col(k) = (residual(e.z, T[e.a] * Sophus::SE3::exp(d), T[e.b]) - residual(e.z, T[e.a] * Sophus::SE3::exp(-d), T[e.b])) / (2 * h);
        Jb.col(k) = (residual(e.z, T[e.a], T[e.b] * Sophus::SE3::exp(d)) - residual(e.z, T[e.a], T[e.b] * Sophus::SE3::exp(-d))) / (2 * h);
      }
      const M6 WI = w * I;
      const int ia = 6 * (e.a - 1), ib = 6 * (e.b - 1);
      if (e.a > 0)
      {
        addBlock(ia, ia, Ja.transpose() * WI * Ja);
        g.segment<6>(ia) += Ja.transpose() * WI * r;
      }
      if (e.b > 0)
      {
        addBlock(ib, ib, Jb.transpose() * WI * Jb);
        g.segment<6>(ib) += Jb.transpose() * WI * r;
      }
      if (e.a > 0 && e.b > 0)
      {
        addBlock(ia, ib, Ja.transpose() * WI * Jb);
        addBlock(ib, ia, Jb.transpose() * WI * Ja);
      }
    }
    Eigen::SparseMatrix<double> H(n, n);
    H.setFromTriplets(trip.begin(), trip.end());
    const Eigen::VectorXd diag = H.diagonal();

    bool improved = false, converged = false;
    for (int tries = 0; tries < 10 && !improved; tries++)
    {
      Eigen::SparseMatrix<double> A = H;
      for (int k = 0; k < n; k++) A.coeffRef(k, k) += lambda * std::max(diag[k], 1e-9);
      Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver(A);
      if (solver.info() != Eigen::Success)
      {
        lambda *= 10;
        continue;
      }
      const Eigen::VectorXd dx = solver.solve(-g);
      std::vector<Sophus::SE3> Tn = T;
      for (int k = 1; k < K; k++) Tn[k] = T[k] * Sophus::SE3::exp(V6(dx.segment<6>(6 * (k - 1))));
      const double cn = totalCost(Tn);
      if (cn < cost)
      {
        improved = true;
        converged = cost - cn < 1e-9 * cost;
        T.swap(Tn);
        cost = cn;
        lambda = std::max(lambda / 3.0, 1e-9);
      }
      else
        lambda *= 5.0;
    }
    if (!improved || converged) break;
  }
  std::cout << "[ Loop ] pose graph: " << K << " keyframes, " << edges.size() << " edges, cost " << cost0 << " -> " << cost << std::endl;
}

int LoopClosure::optimize(const std::string &log_dir)
{
  const int K = keyframes_.size();
  if (K < 2) return 0;

  // One candidate per tried keyframe: the horizontally closest earlier keyframe far enough back along the path.
  std::vector<std::pair<int, int>> candidates;
  for (int i = 0; i < K; i += std::max(1, cfg_.try_every))
  {
    int best = -1;
    double best_d = cfg_.search_radius;
    for (int j = 0; j < i && keyframes_[i].path - keyframes_[j].path >= cfg_.min_path_gap; j++)
    {
      const double d = (keyframes_[i].pose.translation() - keyframes_[j].pose.translation()).head<2>().norm();
      if (d < best_d)
      {
        best_d = d;
        best = j;
      }
    }
    if (best >= 0) candidates.emplace_back(i, best);
  }
  std::vector<LoopResult> results(candidates.size());
#pragma omp parallel for schedule(dynamic)
  for (int c = 0; c < int(candidates.size()); c++) results[c] = registerLoop(candidates[c].first, candidates[c].second);

  std::vector<Edge> edges;
  for (int k = 0; k + 1 < K; k++) edges.push_back({k, k + 1, keyframes_[k].pose.inverse() * keyframes_[k + 1].pose, false, -1});
  for (size_t c = 0; c < results.size(); c++)
    if (results[c].accepted) edges.push_back({results[c].j, results[c].i, results[c].z, true, int(c)});
  std::vector<Sophus::SE3> poses(K);
  for (int k = 0; k < K; k++) poses[k] = keyframes_[k].pose;
  // Optimize; loops that still disagree with the result are dropped and the graph is solved again without them.
  auto loopResiduals = [&](bool drop) {
    std::vector<Edge> kept;
    for (const Edge &e : edges)
    {
      if (e.loop)
      {
        LoopResult &r = results[e.result];
        const V6 res = (e.z.inverse() * poses[e.a].inverse() * poses[e.b]).log();
        r.residual_t = res.head<3>().norm();
        r.residual_r = res.tail<3>().norm() * 180.0 / M_PI;
        r.used = !drop || (r.residual_t <= cfg_.loop_max_residual_t && r.residual_r <= cfg_.loop_max_residual_r);
        if (!r.used) continue;
      }
      kept.push_back(e);
    }
    return kept;
  };
  if (edges.size() > size_t(K - 1))
  {
    optimizeGraph(edges, poses);
    std::vector<Edge> kept = loopResiduals(true);
    if (kept.size() < edges.size())
    {
      std::cout << "[ Loop ] dropping " << edges.size() - kept.size() << " loops that disagree with the optimized graph" << std::endl;
      edges.swap(kept);
      for (int k = 0; k < K; k++) poses[k] = keyframes_[k].pose;
      if (edges.size() > size_t(K - 1)) optimizeGraph(edges, poses);
      loopResiduals(false);
    }
  }
  const int n_loops = int(edges.size()) - (K - 1);

  // Corrections of the keyframes, interpolated in time for the frames between them.
  std::vector<Sophus::SE3> dk(K);
  double max_shift = 0.0;
  for (int k = 0; k < K; k++)
  {
    dk[k] = poses[k] * keyframes_[k].pose.inverse();
    max_shift = std::max(max_shift, (poses[k].translation() - keyframes_[k].pose.translation()).norm());
  }
  int k = 0;
  for (size_t f = 0; f < frames_.size(); f++)
  {
    while (k + 1 < K && keyframes_[k + 1].frame <= f) k++;
    if (f <= keyframes_[0].frame || k + 1 >= K)
      corrections_[f] = dk[k];
    else
    {
      const double a = std::min(1.0, std::max(0.0, (frames_[f].time - keyframes_[k].time) / (keyframes_[k + 1].time - keyframes_[k].time)));
      corrections_[f] = dk[k] * Sophus::SE3::exp(a * (dk[k].inverse() * dk[k + 1]).log());
    }
  }

  std::ofstream fk(log_dir + "keyframes.txt");
  fk << "# time path_m | odometry: x y z qx qy qz qw | optimized: x y z qx qy qz qw\n" << std::fixed << std::setprecision(6);
  for (int k = 0; k < K; k++)
  {
    fk << keyframes_[k].time << " " << keyframes_[k].path;
    for (const Sophus::SE3 *T : {&keyframes_[k].pose, &poses[k]})
    {
      const Eigen::Quaterniond q(T->rotation_matrix());
      fk << " " << T->translation().transpose() << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w();
    }
    fk << "\n";
  }
  std::ofstream fl(log_dir + "loops.txt");
  fl << "# keyframe_i keyframe_j time_i time_j path_gap_m z_offset_m coarse_inlier_ratio inlier_ratio rmse_m accepted used "
        "residual_m residual_deg\n" << std::fixed
     << std::setprecision(4);
  for (const LoopResult &r : results)
    fl << r.i << " " << r.j << " " << keyframes_[r.i].time << " " << keyframes_[r.j].time << " " << keyframes_[r.i].path - keyframes_[r.j].path << " "
       << r.dz << " " << r.coarse_ratio << " " << r.ratio << " " << r.rmse << " " << int(r.accepted) << " " << int(r.used) << " " << r.residual_t << " "
       << r.residual_r << "\n";
  std::cout << "[ Loop ] " << K << " keyframes, " << candidates.size() << " loop candidates, " << n_loops << " accepted, largest keyframe shift "
            << max_shift << " m" << std::endl;
  return n_loops;
}
