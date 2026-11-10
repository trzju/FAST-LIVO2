/*
Re-run the loop closure on the keyframes a mapping run saved (loop/dump_keyframes: true -> Log/loop/keyframes.bin),
with other parameters, without replaying the recording. Writes keyframes.txt and loops.txt to out_dir like the node.

  loop_offline <keyframes.bin> <out_dir> [name=value ...]     e.g. yaw_search=6 xy_search=2 min_inlier_ratio=0.55
*/
#include "loop_closure.h"

#include <cstdlib>
#include <iostream>
#include <map>
#include <string>

int main(int argc, char **argv)
{
  if (argc < 3)
  {
    std::cerr << "usage: loop_offline <keyframes.bin> <out_dir> [name=value ...]" << std::endl;
    return 2;
  }
  LoopClosureConfig cfg;
  const std::map<std::string, double *> reals = {
      {"search_radius", &cfg.search_radius}, {"min_path_gap", &cfg.min_path_gap}, {"z_search_ratio", &cfg.z_search_ratio},
      {"z_search_min", &cfg.z_search_min}, {"z_search_max", &cfg.z_search_max}, {"yaw_search", &cfg.yaw_search},
      {"yaw_step", &cfg.yaw_step}, {"xy_search", &cfg.xy_search}, {"xy_step", &cfg.xy_step}, {"inlier_dist", &cfg.inlier_dist},
      {"min_inlier_ratio", &cfg.min_inlier_ratio}, {"max_rmse", &cfg.max_rmse}, {"odom_sigma_t", &cfg.odom_sigma_t},
      {"odom_sigma_r", &cfg.odom_sigma_r}, {"loop_sigma_t", &cfg.loop_sigma_t}, {"loop_sigma_r", &cfg.loop_sigma_r},
      {"loop_max_residual_t", &cfg.loop_max_residual_t}, {"loop_max_residual_r", &cfg.loop_max_residual_r},
      {"keyframe_voxel", &cfg.keyframe_voxel}};
  const std::map<std::string, int *> ints = {{"try_every", &cfg.try_every}, {"target_half", &cfg.target_half}, {"source_half", &cfg.source_half}};
  for (int a = 3; a < argc; a++)
  {
    const std::string arg = argv[a];
    const size_t eq = arg.find('=');
    const std::string key = arg.substr(0, eq);
    if (eq == std::string::npos || (!reals.count(key) && !ints.count(key)))
    {
      std::cerr << "unknown parameter: " << arg << std::endl;
      return 2;
    }
    const std::string val = arg.substr(eq + 1);
    if (reals.count(key)) *reals.at(key) = std::atof(val.c_str());
    else *ints.at(key) = std::atoi(val.c_str());
  }
  LoopClosure loop(cfg);
  if (!loop.loadKeyframes(argv[1]))
  {
    std::cerr << "cannot read " << argv[1] << std::endl;
    return 1;
  }
  std::string out = argv[2];
  if (out.back() != '/') out += '/';
  loop.optimize(out);
  return 0;
}
