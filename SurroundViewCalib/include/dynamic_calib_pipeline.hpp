#pragma once
/**********************
 * Dynamic Surround-View Calibration (Multithreaded, Vision-only baseline)
 * Stages: Capture -> Features -> GroundEst (CHAIN/VO/PIECEWISE) -> Update/H -> Render
 * Rings:  SPSC between each stage (bounded, back-pressure)
 * Stepper: Geometric step in (R,t,n,d) for stable H updates
 **********************/

// ============================ Common & Types ============================
#include <cstdint>
#include <cmath>
#include <atomic>
#include <thread>
#include <vector>
#include <array>
#include <optional>
#include <string>
#include <algorithm>
#include <chrono>
#include <cassert>

// ---------- Modes ----------
enum class Mode { CHAIN, VO, PIECEWISE };
inline const char* toString(Mode m){
  switch(m){ case Mode::CHAIN: return "CHAIN"; case Mode::VO: return "VO"; case Mode::PIECEWISE: return "PIECEWISE"; }
  return "UNKNOWN";
}

// ---------- Time ----------
struct Timestamp { int64_t mono_ns{0}; int64_t frame_id{0}; };

// ============================ Config ============================
struct Rates {
  int FEATURES  = 2;   // run “heavy” feature extraction every N frames
  int CHAIN     = 3;   // pairwise chain rate
  int VO        = 6;   // VO/plane rate
  int PIECEWISE = 10;  // piecewise plane rate
  int REBUILD   = 50;  // LUT rebuild period (or based on drift)
};

struct Thresholds {
  float SLOW_SPEED      = 8.0f;  // m/s
  float TEX_DENS_MIN    = 0.002f;
  float TEX_SPREAD_MIN  = 0.35f;
  float SEAM_CONF_MIN   = 0.25f;
  float PLANAR_BAD      = 0.60f; // normalized residual → PIECEWISE
  float CONF_GATE       = 0.45f; // min confidence to apply updates
  float MAX_DH_ROT_RAD  = 0.02f; // per-tick rotation bound
};

struct FSMParams {
  float enter_hys = 0.05f; // small hysteresis
  float exit_hys  = 0.05f;
  int   hold_frames = 15;  // frames to hold a mode before switching
};

struct StepLimits {
  float max_rot_rad   = 0.02f; // R step
  float max_n_rad     = 0.02f; // normal angular step
  float max_log_d     = 0.02f; // |Δ log d|
  float max_t_meters  = 0.01f; // optional translation step
  float alpha         = 1.0f;  // global scale 0..1
};

struct Config {
  Rates rates;
  Thresholds th;
  FSMParams fsm;
  StepLimits step;
  int ref_cam_id = 0;   // chain mode reference camera (usually front = 0)
};

// ============================ Status / Expected ============================
struct Status{
  bool ok{true}; std::string msg;
  static Status OK(){ return {}; }
  static Status Fail(std::string m){ return {false, std::move(m)}; }
};

template<class T>
struct Expected{
  T value{};
  Status st = Status::OK();
  bool ok() const { return st.ok; }
};

// ============================ DTOs (runtime messages) ============================
struct FrameView {                // views (pointers) to 4 camera buffers
  uint8_t* img[4]{};
  int w[4]{}, h[4]{}, stride[4]{};
  Timestamp ts{};
};

struct Keypoint { float x,y,score; };
struct Track    { Keypoint p_prev, p_cur; bool ok; };

struct FeatQuality { float tex_density{0}, tex_spread{0}; };
struct SeamQuality { float inliers_ratio{0}, residual{1}, conf{0}; };
struct VOQuality   { int inliers{0}; float residual{1}, conf{0}; };
struct PlaneQuality{ float inliers_ratio{0}, residual{1}; };

struct FeaturesMsg {
  FrameView fv;
  std::vector<Keypoint> kps[4];
  std::vector<Track>    tracks[4];
  FeatQuality featq;
  SeamQuality seamq;
  VOQuality   voq;
  // add fields as needed (descriptors, E-matrices, etc.)
};

struct PlaneEstimate { // n·P + d = 0
  float n[3]{0,1,0}; float d{1};
  PlaneQuality q;
};

struct GpeMsg {
  FeaturesMsg base;
  Mode mode{Mode::CHAIN};
  PlaneEstimate plane;
  float conf{0}; // fused confidence 0..1
};

struct CalibUpdateMsg {
  GpeMsg gpe;
  float dH[4][9]{}; // per-camera ΔH target (or absolute target; your choice)
  float conf{0};
  bool  lut_rebuild{false};
};

struct BevFrame {
  CalibUpdateMsg cu;
  // optional: stitched output or handles to GPU resources
};

// ============================ SPSC Ring (bounded, lock-free) ============================
template<class T, size_t N>
class SpscRing {
  static_assert((N & (N-1))==0, "N must be power of two");
public:
  bool push(T&& v) noexcept {
    auto h = head_.load(std::memory_order_relaxed);
    auto n = (h + 1) & mask_;
    if(n == tail_.load(std::memory_order_acquire)) return false; // full
    buf_[h] = std::move(v);
    head_.store(n, std::memory_order_release);
    return true;
  }
  std::optional<T> pop() noexcept {
    auto t = tail_.load(std::memory_order_relaxed);
    if(t == head_.load(std::memory_order_acquire)) return std::nullopt; // empty
    auto v = std::move(buf_[t]);
    tail_.store((t + 1) & mask_, std::memory_order_release);
    return v;
  }
  size_t size() const noexcept {
    auto h = head_.load(std::memory_order_acquire);
    auto t = tail_.load(std::memory_order_acquire);
    return (h + N - t) & mask_;
  }
private:
  static constexpr size_t mask_ = N-1;
  std::array<T,N> buf_{};
  std::atomic<size_t> head_{0}, tail_{0};
};

// ============================ Timers / Rate control ============================
struct Timers {
  std::atomic<int> frame_id{0};
  std::array<int,6> last {{-1,-1,-1,-1,-1,-1}}; // FEATURES, CHAIN, VO, PIECE, REBUILD, LOG
  bool shouldRun(int idx, int period){
    int f = frame_id.load(std::memory_order_relaxed);
    if(f - last[idx] >= period){ last[idx]=f; return true; }
    return false;
  }
};

// ============================ Confidence Fusion ============================
inline float clamp01(float x){ return std::max(0.f, std::min(1.f, x)); }

struct ConfidenceFusion {
  float fuse(const FeatQuality& fq, const SeamQuality& sq,
             const VOQuality& voq, const PlaneQuality& pq) const {
    float seam_part = sq.inliers_ratio * (1.f/(1.f + sq.residual));
    float vo_part   = clamp01(voq.inliers * (1.f/(1.f + voq.residual)) * 0.01f);
    float base = clamp01(0.5f*fq.tex_spread + 0.5f*std::max(seam_part, vo_part));
    float rans = clamp01(pq.inliers_ratio * (1.f/(1.f + pq.residual)));
    return clamp01(0.6f*base + 0.4f*rans);
  }
};

// ============================ Mode FSM (deterministic, with hold) ============================
struct ModeDecision { Mode target{Mode::CHAIN}; float conf{0}; };

struct ModeFSM {
  explicit ModeFSM(const FSMParams& p):p_(p){}
  ModeDecision decide(const FeaturesMsg& f, float planarity_norm, float speed_mps){
    bool tex_ok  = (f.featq.tex_spread >= 0.35f);
    bool seam_ok = (f.seamq.conf       >= 0.25f);
    if(speed_mps <= 8.0f - p_.enter_hys && tex_ok && seam_ok)
      return {Mode::CHAIN, f.seamq.conf};
    if(planarity_norm >= 0.60f + p_.enter_hys)
      return {Mode::PIECEWISE, 0.6f};
    return {Mode::VO, f.voq.conf};
  }
  bool allowSwitch(Mode cur, Mode nxt, int frames_in_state) const {
    if(cur==nxt) return true;
    return frames_in_state >= p_.hold_frames;
  }
private:
  FSMParams p_;
};

// ============================ Geometric Homography Stepper (plane/pose) ============================
// Minimal SO(3) helpers (small-angle safe)
struct Vec3 { float x,y,z; };
struct Mat3 {
  float m[9]{}; // row-major
  static Mat3 I(){ Mat3 R; R.m[0]=R.m[4]=R.m[8]=1; return R; }
};
inline void mat3Mul(const Mat3& A, const Mat3& B, Mat3& C){
  for(int r=0;r<3;r++) for(int c=0;c<3;c++){
    C.m[r*3+c] = A.m[r*3+0]*B.m[0*3+c]+A.m[r*3+1]*B.m[1*3+c]+A.m[r*3+2]*B.m[2*3+c];
  }
}
inline Mat3 skew(const Vec3& w){
  Mat3 K{}; K.m[1*3+2]=-w.x; K.m[2*3+1]= w.x;
         K.m[0*3+2]= w.y; K.m[2*3+0]=-w.y;
         K.m[0*3+1]=-w.z; K.m[1*3+0]= w.z;
  return K;
}
inline Mat3 add(const Mat3& A, const Mat3& B){
  Mat3 C; for(int i=0;i<9;i++) C.m[i]=A.m[i]+B.m[i]; return C;
}
inline Mat3 mulScalar(const Mat3& A, float s){
  Mat3 C; for(int i=0;i<9;i++) C.m[i]=A.m[i]*s; return C;
}
inline Mat3 expSO3(const Vec3& w){
  float th = std::sqrt(w.x*w.x + w.y*w.y + w.z*w.z);
  Mat3 I = Mat3::I();
  if(th < 1e-8f){
    Mat3 Kw = skew(w);
    return add(I, Kw);
  }
  Vec3 wn{w.x/th, w.y/th, w.z/th};
  Mat3 K = skew(wn);
  // I + sin(th)K + (1-cos(th))K^2
  Mat3 KK; mat3Mul(K,K,KK);
  Mat3 term1 = mulScalar(K, std::sin(th));
  Mat3 term2 = mulScalar(KK, (1.f-std::cos(th)));
  Mat3 tmp = add(I, term1); return add(tmp, term2);
}
// NOTE: we avoid full logSO3 for brevity; in practice use Eigen/Sophus.
// Here, assume small relative ΔR given by target-current; pass in a small axis-angle directly or replace with a robust logSO3.

// Pose/Plane bundle for stepping
struct PlanePose {
  Mat3  R;         // camera rotation
  Vec3  t;         // camera translation (optional)
  Vec3  n;         // ground normal (unit)
  float d;         // distance > 0
};

inline float dot(const Vec3& a, const Vec3& b){ return a.x*b.x + a.y*b.y + a.z*b.z; }
inline Vec3  add(const Vec3& a, const Vec3& b){ return {a.x+b.x,a.y+b.y,a.z+b.z}; }
inline Vec3  sub(const Vec3& a, const Vec3& b){ return {a.x-b.x,a.y-b.y,a.z-b.z}; }
inline Vec3  mul(const Vec3& a, float s){ return {a.x*s,a.y*s,a.z*s}; }
inline float norm(const Vec3& a){ return std::sqrt(dot(a,a)); }
inline Vec3  normalize(const Vec3& a){ float n=std::max(1e-9f,norm(a)); return {a.x/n,a.y/n,a.z/n}; }

struct HomographyStepperGeo {
  // Step from cur -> tgt in (R,t,n,d), then assemble H = K ( R - t n^T / d )
  // K is row-major 3x3 as Mat3. Return H_new (row-major).
  static Mat3 step(const Mat3& K,
                   const PlanePose& cur,
                   const PlanePose& tgt,
                   const StepLimits& lim,
                   bool use_t = false)
  {
    float a = std::max(0.f, std::min(1.f, lim.alpha));

    // --- Rotation step (assume small desired delta axis-angle "w_des") ---
    // In real code: compute ΔR = R_tgt * R_cur^T; w_des = logSO3(ΔR).
    // Here we approximate: if you have pitch/roll deltas, put them in w_des.
    Vec3 w_des{0,0,0}; // TODO(real): compute from (tgt.R * cur.R^T)
    float wn = norm(w_des);
    if(wn > 1e-9f){
      float cap = std::min(lim.max_rot_rad, wn);
      w_des = mul(w_des, (a * cap / wn));
    }
    Mat3 R_step = expSO3(w_des);
    Mat3 R_new; mat3Mul(R_step, cur.R, R_new);

    // --- Normal step (spherical cap) ---
    Vec3 n0 = normalize(cur.n);
    Vec3 n1 = normalize(tgt.n);
    float c = std::max(-1.f, std::min(1.f, dot(n0,n1)));
    float ang = std::acos(c);
    float ang_step = std::min(lim.max_n_rad, ang) * a;
    Vec3 n_new = n0;
    if(ang > 1e-6f){
      float s = std::sin(ang);
      float A = std::sin(ang - ang_step)/s;
      float B = std::sin(ang_step)/s;
      n_new = normalize(add(mul(n0,A), mul(n1,B)));
    }

    // --- d step (log domain) ---
    float logd0 = std::log(std::max(1e-6f, cur.d));
    float logd1 = std::log(std::max(1e-6f, tgt.d));
    float logd_delta = std::max(-lim.max_log_d, std::min(lim.max_log_d, (logd1 - logd0)*a));
    float d_new = std::exp(logd0 + logd_delta);

    // --- optional t step ---
    Vec3 t_new = cur.t;
    if(use_t){
      Vec3 dt = sub(tgt.t, cur.t);
      float dn = norm(dt);
      if(dn > 1e-9f){
        float cap = std::min(lim.max_t_meters, dn);
        dt = mul(dt, (a * cap / dn));
      }
      t_new = add(cur.t, dt);
    }

    // --- Assemble H = K ( R_new - (t_new n_new^T) / d_new )
    // Compute P = R_new - (t_new n_new^T)/d
    Mat3 P{};           // R_new copy
    for(int i=0;i<9;i++) P.m[i] = R_new.m[i];
    // Outer: t_new n_new^T
    float Tn[9] = {
      t_new.x*n_new.x, t_new.x*n_new.y, t_new.x*n_new.z,
      t_new.y*n_new.x, t_new.y*n_new.y, t_new.y*n_new.z,
      t_new.z*n_new.x, t_new.z*n_new.y, t_new.z*n_new.z
    };
    for(int i=0;i<9;i++) P.m[i] -= Tn[i]/d_new;

    Mat3 H{}; mat3Mul(K, P, H);

    // Normalize projective scale (set bottom-right or overall norm)
    float s = std::fabs(H.m[8]) > 1e-8f ? H.m[8] : std::sqrt(H.m[0]*H.m[0]+H.m[4]*H.m[4]+H.m[8]*H.m[8]);
    if(s>1e-9f) for(int i=0;i<9;i++) H.m[i] /= s;
    return H;
  }
};

// ============================ Telemetry (placeholders) ============================
struct Telemetry {
  std::atomic<size_t> q1_max{0}, q2_max{0}, q3_max{0}, q4_max{0};
  std::atomic<size_t> drops1{0}, drops2{0}, drops3{0}, drops4{0};
  // add: runtimes, mode switches, conf traces, etc.
};

// ============================ Context (threads, queues, config/state) ============================
struct Threads {
  std::thread tCap, tFeat, tGpe, tUpd, tRend;
  std::atomic<bool> stop{false};
};

struct Queues {
  SpscRing<FrameView,     64> q1;
  SpscRing<FeaturesMsg,   64> q2;
  SpscRing<GpeMsg,        64> q3;
  SpscRing<CalibUpdateMsg,64> q4;
};

struct RtConfig { int features_rate=2; int vo_rate=3; };

struct Context {
  Config cfg;
  RtConfig rcfg;
  Threads th;
  Queues  q;
  Timers  timers;
  Telemetry tm;

  // Global state glimpses (read-only for most threads; update in Update stage)
  std::atomic<float> speed_mps{0.f};
  std::atomic<float> planarity_norm{0.f};
  std::atomic<Mode>  mode{Mode::CHAIN};
  std::atomic<int>   frames_in_state{0};

  // Per-camera current pose/plane (for HomographyStepper) – simplified placeholders:
  PlanePose curPose[4]; // keep R,t,n,d per camera
  Mat3      K[4];       // intrinsics

  // Factory ground reference:
  Vec3 n0{0,1,0}; float d0{1};
};

#include <opencv2/opencv.hpp>
#include <filesystem>
namespace fs = std::filesystem;

// דוגמת utility: פונקציה לפתיחת מצלמות או קבצי וידאו
inline bool openCameras(std::array<cv::VideoCapture,4>& caps, const std::array<std::string,4>& sources) {
    for (int i = 0; i < 4; ++i) {
        if (sources[i].empty()) continue;
        if (!caps[i].open(sources[i])) {
            std::cerr << "❌ Failed to open camera/video source " << i << ": " << sources[i] << std::endl;
            return false;
        }
        caps[i].set(cv::CAP_PROP_FRAME_WIDTH,  1280);
        caps[i].set(cv::CAP_PROP_FRAME_HEIGHT, 720);
        caps[i].set(cv::CAP_PROP_FPS, 30);
    }
    return true;
}

// פונקציה ראשית: captureLoop מלא
inline void captureLoop(Context& ctx)
{
    using namespace std::chrono;

    // 📸 פתיחת מצלמות או וידאו לפי מקור
    static std::array<cv::VideoCapture,4> caps;
    static bool initialized = false;
    if (!initialized)
    {
        std::array<std::string,4> sources = {
            "0",  // מצלמה קדמית (אם מצלמה USB מחוברת)
            "1",  // מצלמה אחורית
            "2",  // מצלמה ימין
            "3"   // מצלמה שמאל
            // אפשר להחליף גם לנתיבי קבצים: "data/front.mp4", "data/rear.mp4", ...
        };

        if (!openCameras(caps, sources)) {
            std::cerr << "⚠️ Capture init failed — running dummy frames instead\n";
        }
        initialized = true;
    }

    cv::Mat frames[4];
    int frameCounter = 0;

    while (!ctx.th.stop.load())
    {
        FrameView fv{};
        fv.ts.frame_id = ctx.timers.frame_id.fetch_add(1);
        fv.ts.mono_ns  = duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();

        bool ok = true;
        for (int i = 0; i < 4; ++i)
        {
            if (caps[i].isOpened()) {
                ok &= caps[i].read(frames[i]);
                if (!ok) {
                    // אם הסתיים וידאו — לחזור להתחלה
                    caps[i].set(cv::CAP_PROP_POS_FRAMES, 0);
                    caps[i].read(frames[i]);
                }
            } else {
                // במקרה שאין מצלמה, ניצור תמונה שחורה
                frames[i] = cv::Mat::zeros(cv::Size(640,360), CV_8UC3);
                cv::putText(frames[i], "Dummy Cam "+std::to_string(i),
                            {20,200}, cv::FONT_HERSHEY_SIMPLEX, 1.0,
                            cv::Scalar(0,255,0), 2);
            }

            // נכניס את מצביעי הנתונים ל־FrameView
            fv.img[i]    = frames[i].data;
            fv.w[i]      = frames[i].cols;
            fv.h[i]      = frames[i].rows;
            fv.stride[i] = static_cast<int>(frames[i].step);
        }

        // 🌀 דחיפה לתור (queue)
        if (!ctx.q.q1.push(std::move(fv)))
            ctx.tm.drops1++;

        size_t s = ctx.q.q1.size();
        if (s > ctx.tm.q1_max) ctx.tm.q1_max = s;

        // המתנה קטנה לשמירה על קצב ~30fps
        std::this_thread::sleep_for(std::chrono::milliseconds(33));
        ++frameCounter;
    }

    // ניקוי
    for (auto& c : caps) if (c.isOpened()) c.release();
}
// ============================ Stage: Capture ============================
inline void captureLoop_(Context& ctx){
  using namespace std::chrono;
  while(!ctx.th.stop.load()){
    FrameView fv{};
    // TODO(impl): acquire 4 synced camera buffers (views from pool)
    fv.ts.frame_id = ctx.timers.frame_id.fetch_add(1);
    fv.ts.mono_ns  = duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
    if(!ctx.q.q1.push(std::move(fv))) ctx.tm.drops1++;
    size_t s = ctx.q.q1.size(); if(s>ctx.tm.q1_max) ctx.tm.q1_max=s;
  }
}

// ============================ Stage: Features ============================
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <numeric>

inline void featuresLoop(Context& ctx)
{
    int tick = 0;

    // יצירת ORB detector ו-LK parameters
    cv::Ptr<cv::ORB> orb = cv::ORB::create(1000);
    cv::Size winSize(21,21);
    cv::TermCriteria termcrit(cv::TermCriteria::COUNT|cv::TermCriteria::EPS, 30, 0.01);

    std::vector<cv::Point2f> prevPts[4], currPts[4];
    std::vector<cv::Mat> prevGray(4);

    while(!ctx.th.stop.load())
    {
        auto fv = ctx.q.q1.pop();
        if(!fv){ std::this_thread::sleep_for(std::chrono::microseconds(200)); continue; }

        FeaturesMsg fm{};
        fm.fv = *fv;
        bool runHeavy = (tick % ctx.rcfg.features_rate)==0;

        cv::Mat framesGray[4];
        for (int i = 0; i < 4; ++i)
        {
            cv::Mat img(cv::Size(fv->w[i], fv->h[i]), CV_8UC3, fv->img[i], fv->stride[i]);
            cv::cvtColor(img, framesGray[i], cv::COLOR_BGR2GRAY);
        }

        // ========== 1. detect+describe (ORB) ==========
        std::vector<cv::KeyPoint> kps[4];
        cv::Mat desc[4];
        for (int i = 0; i < 4; ++i)
        {
            if(runHeavy || prevGray[i].empty())
                orb->detectAndCompute(framesGray[i], cv::noArray(), kps[i], desc[i]);
        }

        // ========== 2. LK optical flow (per camera) ==========
        for (int i = 0; i < 4; ++i)
        {
            if(!prevGray[i].empty() && !prevPts[i].empty())
            {
                std::vector<uchar> status;
                std::vector<float> err;
                currPts[i].resize(prevPts[i].size());
                cv::calcOpticalFlowPyrLK(prevGray[i], framesGray[i],
                                          prevPts[i], currPts[i],
                                          status, err, winSize, 3, termcrit);

                // סינון נקודות לא תקפות
                std::vector<cv::Point2f> goodPrev, goodCurr;
                for(size_t j=0;j<status.size();++j)
                    if(status[j]){ goodPrev.push_back(prevPts[i][j]); goodCurr.push_back(currPts[i][j]); }

                fm.tracks[i].resize(goodPrev.size());
                for(size_t j=0;j<goodPrev.size();++j){
                    fm.tracks[i][j].p_prev = {goodPrev[j].x, goodPrev[j].y,1};
                    fm.tracks[i][j].p_cur  = {goodCurr[j].x, goodCurr[j].y,1};
                    fm.tracks[i][j].ok = true;
                }
            }
        }

        // ========== 3. seam matching (pairwise homographies) ==========
        float totalResidual = 0.0f;
        int seamPairs = 0;
        for (int i = 0; i < 4; ++i)
        {
            int j = (i+1)%4; // מצלמה סמוכה
            if(!desc[i].empty() && !desc[j].empty())
            {
                cv::BFMatcher matcher(cv::NORM_HAMMING);
                std::vector<cv::DMatch> matches;
                matcher.match(desc[i], desc[j], matches);

                // סינון התאמות רעות
                double maxDist = 0; double minDist = 100;
                for (auto& m : matches){ double dist = m.distance; if(dist<minDist) minDist=dist; if(dist>maxDist) maxDist=dist; }
                std::vector<cv::Point2f> pts1, pts2;
                for (auto& m : matches)
                    if(m.distance <= std::max(2*minDist, 30.0))
                    {
                        pts1.push_back(kps[i][m.queryIdx].pt);
                        pts2.push_back(kps[j][m.trainIdx].pt);
                    }

                if(pts1.size() >= 4)
                {
                    std::vector<uchar> mask;
                    cv::Mat H = cv::findHomography(pts1, pts2, cv::RANSAC, 3, mask);
                    float inliers = std::accumulate(mask.begin(), mask.end(), 0.0f);
                    float residual = (float)cv::norm(pts1, pts2, cv::NORM_L2) / (float)pts1.size();
                    totalResidual += residual;
                    seamPairs++;
                    fm.seamq.inliers_ratio = inliers / (float)mask.size();
                    fm.seamq.residual = residual;
                }
            }
        }
        if(seamPairs>0) totalResidual /= seamPairs;
        fm.seamq.conf = std::clamp(1.0f - totalResidual/10.0f, 0.0f, 1.0f);

        // ========== 4. light VO (Essential matrix) ==========
        for (int i = 0; i < 4; ++i)
        {
            if (fm.tracks[i].size() >= 8)
            {
                std::vector<cv::Point2f> p1, p2;
                for(auto& t : fm.tracks[i]){ p1.push_back({t.p_prev.x, t.p_prev.y}); p2.push_back({t.p_cur.x, t.p_cur.y}); }
                std::vector<uchar> mask;
                cv::Mat E = cv::findEssentialMat(p1, p2, 1.0, cv::Point2d(0,0), cv::RANSAC, 0.999, 1.0, mask);
                if(!E.empty())
                {
                    int inliers = cv::countNonZero(mask);
                    fm.voq.inliers = inliers;
                    fm.voq.residual = totalResidual;
                    fm.voq.conf = std::clamp(inliers / float(p1.size()), 0.0f, 1.0f);
                }
            }
        }

        // ========== 5. Fill Feature Quality ==========
        // מדד מרקם/פיזור לפי כמות הפיצ׳רים (יחסי לגודל התמונה)
        int totalKP = 0;
        for (int i=0;i<4;i++) totalKP += (int)kps[i].size();
        float texDensity = totalKP / (4.0f * fv->w[0] * fv->h[0]);
        fm.featq.tex_density = texDensity;
        fm.featq.tex_spread  = std::clamp(runHeavy ? 0.6f : 0.5f, 0.0f, 1.0f);

        // ========== push forward ==========
        if(!ctx.q.q2.push(std::move(fm))) ctx.tm.drops2++;
        size_t s = ctx.q.q2.size(); if(s > ctx.tm.q2_max) ctx.tm.q2_max = s;

        // לשמור פריים קודם לעקיבה
        for(int i=0;i<4;i++){
            prevGray[i] = framesGray[i].clone();
            cv::KeyPoint::convert(kps[i], prevPts[i]);
        }

        ++tick;
    }
}

inline void featuresLoop_(Context& ctx){
  int tick=0;
  while(!ctx.th.stop.load()){
    auto fv = ctx.q.q1.pop();
    if(!fv){ std::this_thread::sleep_for(std::chrono::microseconds(200)); continue; }

    FeaturesMsg fm{};
    fm.fv = *fv;
    bool runHeavy = (tick % ctx.rcfg.features_rate)==0;

    // TODO(impl):
    //  - detect+describe (ORB/FAST) within ROIs
    //  - LK optical flow to build tracks
    //  - seam matching (pairwise homographies & residuals)
    //  - light VO measurements (E/inliers/residual)
    //  - fill FeatQuality/SeamQuality/VOQuality
    fm.featq.tex_spread = runHeavy? 0.6f : 0.5f;
    fm.seamq.conf = 0.7f; fm.voq.conf = 0.5f;

    if(!ctx.q.q2.push(std::move(fm))) ctx.tm.drops2++;
    size_t s = ctx.q.q2.size(); if(s>ctx.tm.q2_max) ctx.tm.q2_max=s;
    ++tick;
  }
}

// ============================ Stage: GroundEst (mode selection + plane) ============================
inline void gpeLoop(Context& ctx){
  ModeFSM fsm(ctx.cfg.fsm);
  ConfidenceFusion confFuse;

  while(!ctx.th.stop.load()){
    auto fm = ctx.q.q2.pop();
    if(!fm){ std::this_thread::sleep_for(std::chrono::microseconds(200)); continue; }

    float speed = ctx.speed_mps.load();
    float planr = ctx.planarity_norm.load();

    // Mode decision (with hold)
    Mode cur = ctx.mode.load();
    int  held= ctx.frames_in_state.load();
    auto md  = fsm.decide(*fm, planr, speed);
    Mode nxt = fsm.allowSwitch(cur, md.target, held) ? md.target : cur;
    if(nxt!=cur){ ctx.mode.store(nxt); ctx.frames_in_state.store(0); }
    else        { ctx.frames_in_state.store(held+1); }

    GpeMsg gm{};
    gm.base = *fm; gm.mode = nxt;

    // Compute plane depending on mode (stubs):
    if(nxt==Mode::CHAIN){
      // TODO(impl): chain pairwise H cam->ref; VO(ref) → R_ref; n = R_ref*n0; d=d0
      gm.plane.n[0]=ctx.n0.x; gm.plane.n[1]=ctx.n0.y; gm.plane.n[2]=ctx.n0.z; gm.plane.d=ctx.d0;
      gm.plane.q.inliers_ratio=0.6f; gm.plane.q.residual=0.3f;
    } else if(nxt==Mode::VO){
      // TODO(impl): VO per cam, Triangulate ground points, RANSAC plane -> n,d
      gm.plane.n[0]=0; gm.plane.n[1]=1; gm.plane.n[2]=0; gm.plane.d=1;
      gm.plane.q.inliers_ratio=0.65f; gm.plane.q.residual=0.28f;
    } else { // PIECEWISE
      // TODO(impl): piecewise plane (grid) — here emit a single representative plane + mark conf
      gm.plane.n[0]=0; gm.plane.n[1]=1; gm.plane.n[2]=0; gm.plane.d=1;
      gm.plane.q.inliers_ratio=0.7f; gm.plane.q.residual=0.25f;
    }

    gm.conf = confFuse.fuse(fm->featq, fm->seamq, fm->voq, gm.plane.q);

    if(!ctx.q.q3.push(std::move(gm))) ctx.tm.drops3++;
    size_t s = ctx.q.q3.size(); if(s>ctx.tm.q3_max) ctx.tm.q3_max=s;
  }
}

// ============================ Stage: Update/H (apply ΔH with stepper) ============================
inline void updateLoop(Context& ctx){
  while(!ctx.th.stop.load()){
    auto gm = ctx.q.q3.pop();
    if(!gm){ std::this_thread::sleep_for(std::chrono::microseconds(200)); continue; }

    CalibUpdateMsg cu{};
    cu.gpe = *gm; cu.conf = gm->conf;

    // === Build per-camera target plane/pose (examples) and step ===
    // For CHAIN: derive R_ref from VO, n = R_ref*n0 (d=d0); build per-camera via H_ref->ground * H_cam->ref
    // For VO/PIECEWISE: you already estimated (R_i,t_i) and (n,d). Below we show the stepper call.

    for(int cam=0; cam<4; ++cam){
      PlanePose cur = ctx.curPose[cam];
      PlanePose tgt = cur; // start from cur, override fields from gm.plane + your VO pose

      // TODO(impl):
      //   - set tgt.R and tgt.t from your VO pose for this camera (or ref for CHAIN)
      //   - set tgt.n,tgt.d from gm.plane (or cell plane in PIECEWISE)
      tgt.n = { gm->plane.n[0], gm->plane.n[1], gm->plane.n[2] };
      tgt.d =  gm->plane.d;

      Mat3 H_new = HomographyStepperGeo::step(ctx.K[cam], cur, tgt, ctx.cfg.step, /*use_t=*/false);
      // Convert H_new into a ΔH or keep H absolute – your policy:
      // Here we interpret dH as "small delta toward target"; using H_new directly as delta:
      for(int i=0;i<9;i++) cu.dH[cam][i] = H_new.m[i]; // (policy placeholder)
    }

    // LUT policy: rebuild only periodically or if drift past threshold
    cu.lut_rebuild = false;

    if(!ctx.q.q4.push(std::move(cu))) ctx.tm.drops4++;
    size_t s = ctx.q.q4.size(); if(s>ctx.tm.q4_max) ctx.tm.q4_max=s;
  }
}

// ============================ Stage: Render (apply + BEV) ============================
inline void renderLoop(Context& ctx){
  while(!ctx.th.stop.load()){
    auto cu = ctx.q.q4.pop();
    if(!cu){ std::this_thread::sleep_for(std::chrono::microseconds(200)); continue; }

    if(cu->conf >= ctx.cfg.th.CONF_GATE){
      // Apply ΔH safely with rate limit (the stepper already bounded updates)
      // TODO(impl): update per-camera H_ground (double-buffer), rebuild LUT if cu.lut_rebuild
    }

    // TODO(impl): PROJECT_TO_BEV using current LUTs, STITCH_TILES, HUD overlays, telemetry
  }
}

// ============================ Bootstrap (start/stop threads) ============================
inline void start_all(Context& ctx){
  ctx.th.tCap  = std::thread(captureLoop, std::ref(ctx));
  ctx.th.tFeat = std::thread(featuresLoop, std::ref(ctx));
  ctx.th.tGpe  = std::thread(gpeLoop,     std::ref(ctx));
  ctx.th.tUpd  = std::thread(updateLoop,  std::ref(ctx));
  ctx.th.tRend = std::thread(renderLoop,  std::ref(ctx));
}

inline void stop_all(Context& ctx){
  ctx.th.stop = true;
  ctx.th.tRend.join(); ctx.th.tUpd.join(); ctx.th.tGpe.join();
  ctx.th.tFeat.join(); ctx.th.tCap.join();
}


