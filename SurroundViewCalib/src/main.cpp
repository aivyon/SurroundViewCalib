#include "dynamic_calib_pipeline.hpp"
#include <iostream>
#include <csignal>
#include <thread>

int main() {
  std::cout << "Starting Dynamic Calibration Pipeline (multithreaded demo)\n";

  // --- Init context & config ---
  Context ctx;
  ctx.cfg = Config{};
  ctx.cfg.ref_cam_id = 0;          // front camera
  ctx.cfg.step.alpha = 1.0f;       // full step fraction
  ctx.cfg.step.max_rot_rad = 0.02f;
  ctx.cfg.step.max_n_rad   = 0.02f;
  ctx.cfg.step.max_log_d   = 0.02f;
  ctx.cfg.step.max_t_meters= 0.01f;

  // example intrinsics (identity for simplicity)
  for(int c=0;c<4;c++){
    ctx.cams[c].K = Mat3::I();
    ctx.cams[c].R = Mat3::I();
    ctx.cams[c].t = {0,0,0};
    ctx.cams[c].n = {0,1,0};
    ctx.cams[c].d = 1.0f;
    ctx.cams[c].H_ground = compose_H_from_pose_plane(ctx.cams[c].K, ctx.cams[c].R,
                                                     ctx.cams[c].t, ctx.cams[c].n, ctx.cams[c].d);
  }

  // --- Start threads ---
  start_all(ctx);

  // --- Run for a while (simulate main loop / signal handling) ---
  std::signal(SIGINT, [](int){});
  using namespace std::chrono_literals;
  for(int i=0;i<10; ++i){
    std::this_thread::sleep_for(1s);
    std::cout << ".";
    std::cout.flush();
  }

  // --- Stop threads ---
  stop_all(ctx);

  std::cout << "\nAll threads stopped.\n";
  std::cout << "Queue stats: q1_max=" << ctx.tm.q1_max
            << " q2_max=" << ctx.tm.q2_max
            << " q3_max=" << ctx.tm.q3_max
            << " q4_max=" << ctx.tm.q4_max << "\n";
  std::cout << "Drops: " << ctx.tm.drops1 << "," << ctx.tm.drops2
            << "," << ctx.tm.drops3 << "," << ctx.tm.drops4 << "\n";
  std::cout << "Mode=" << toString(ctx.mode.load())
            << " frames_in_state=" << ctx.frames_in_state.load() << "\n";
  std::cout << "Done.\n";
  return 0;
}
