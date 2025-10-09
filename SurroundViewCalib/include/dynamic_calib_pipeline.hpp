#pragma once
/******************************************************
 * Dynamic Surround-View Calibration – Single Header
 * C++20 / OpenCV
 * Stages: Capture → Features → GroundEst → Update/H → Render
 ******************************************************/

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <array>
#include <string>
#include <vector>
#include <mutex>
#include <atomic>
#include <thread>
#include <optional>
#include <chrono>
#include <numeric>
#include <iostream>

/*==================== Basic Math Types ====================*/
struct Vec3 { float x=0,y=0,z=0; };
struct Mat3 { float m[9]{}; static Mat3 I(){ Mat3 R; R.m[0]=R.m[4]=R.m[8]=1; return R; } };

inline void mat3Mul(const Mat3& A, const Mat3& B, Mat3& C){
  for(int r=0;r<3;r++)
    for(int c=0;c<3;c++)
      C.m[r*3+c] = A.m[r*3+0]*B.m[0*3+c] + A.m[r*3+1]*B.m[1*3+c] + A.m[r*3+2]*B.m[2*3+c];
}

/*==================== Modes / Time ====================*/
enum class Mode { CHAIN, VO, PIECEWISE };
inline const char* toString(Mode m){
  switch(m){ case Mode::CHAIN: return "CHAIN"; case Mode::VO:return "VO"; case Mode::PIECEWISE:return "PIECEWISE"; }
  return "UNKNOWN";
}
struct Timestamp { int64_t mono_ns{0}; int64_t frame_id{0}; }

/*==================== Config / Thresholds ====================*/
struct Rates{ int FEATURES=2, CHAIN=3, VO=6, PIECEWISE=10, REBUILD=50; };
struct Thresholds{
  float SLOW_SPEED=8.0f, TEX_DENS_MIN=0.002f, TEX_SPREAD_MIN=0.35f, SEAM_CONF_MIN=0.25f;
  float PLANAR_BAD=0.60f, CONF_GATE=0.45f, MAX_DH_ROT_RAD=0.02f;
};
struct FSMParams{ float enter_hys=0.05f, exit_hys=0.05f; int hold_frames=15; };
struct StepLimits{ float max_rot_rad=0.02f, max_n_rad=0.02f, max_log_d=0.02f, max_t_meters=0.01f, alpha=1.0f; };
struct Config{ Rates rates; Thresholds th; FSMParams fsm; StepLimits step; int ref_cam_id=0; };

/*==================== Status/Expected ====================*/
struct Status{ bool ok{true}; std::string msg; static Status OK(){return{};} static Status Fail(std::string m){return{false,std::move(m)};} };
template<class T> struct Expected{ T value{}; Status st=Status::OK(); bool ok()const{return st.ok;} };

/*==================== DTOs (Frames/Features/GPE/Update) ====================*/
struct FrameView{
  uint8_t* img[4]{}; int w[4]{}, h[4]{}, stride[4]{};
  Timestamp ts{};
};
struct Keypoint { float x,y,score; };
struct Track    { Keypoint p_prev, p_cur; bool ok; };
struct FeatQuality{ float tex_density{0}, tex_spread{0}; };
struct SeamQuality{ float inliers_ratio{0}, residual{1}, conf{0}; };
struct VOQuality  { int inliers{0}; float residual{1}, conf{0}; };
struct PlaneQuality{ float inliers_ratio{0}, residual{1}; };

struct FeaturesMsg{
  FrameView fv;
  std::vector<Keypoint> kps[4];
  std::vector<Track>    tracks[4];
  FeatQuality featq; SeamQuality seamq; VOQuality voq;
};
struct PlaneEstimate{ float n[3]{0,1,0}; float d{1}; PlaneQuality q; };
struct GpeMsg{ FeaturesMsg base; Mode mode{Mode::CHAIN}; PlaneEstimate plane; float conf{0}; };
struct CalibUpdateMsg{ GpeMsg gpe; float dH[4][9]{}; float conf{0}; bool lut_rebuild{false}; };
struct BevFrame{ CalibUpdateMsg cu; /*optional visual output*/ };

/*==================== SPSC ring (bounded) ====================*/
template<class T, size_t N>
class SpscRing{
  static_assert((N&(N-1))==0,"N must be power of two");
public:
  bool push(T&& v) noexcept {
    auto h=head_.load(std::memory_order_relaxed);
    auto n=(h+1)&mask_;
    if(n==tail_.load(std::memory_order_acquire)) return false;
    buf_[h]=std::move(v);
    head_.store(n,std::memory_order_release); return true;
  }
  std::optional<T> pop() noexcept {
    auto t=tail_.load(std::memory_order_relaxed);
    if(t==head_.load(std::memory_order_acquire)) return std::nullopt;
    auto v=std::move(buf_[t]);
    tail_.store((t+1)&mask_,std::memory_order_release); return v;
  }
  size_t size() const noexcept {
    auto h=head_.load(std::memory_order_acquire), t=tail_.load(std::memory_order_acquire);
    return (h+N-t)&mask_;
  }
private:
  static constexpr size_t mask_=N-1;
  std::array<T,N> buf_{}; std::atomic<size_t> head_{0}, tail_{0};
};

/*==================== Timers ====================*/
struct Timers{
  std::atomic<int> frame_id{0};
  std::array<int,6> last{{-1,-1,-1,-1,-1,-1}};
  bool shouldRun(int idx,int period){
    int f=frame_id.load(std::memory_order_relaxed);
    if(f-last[idx]>=period){last[idx]=f; return true;} return false;
  }
};

/*==================== LUT + helpers ====================*/
struct Lut{ int W{0}, H{0}; cv::Mat mapX, mapY; };

inline Lut buildLUTfromH_cv(const Mat3& H_cam_ground, int bevW, int bevH,
                            float Xmin,float Xmax,float Ymin,float Ymax){
  Lut L; L.W=bevW; L.H=bevH;
  L.mapX.create(bevH,bevW,CV_32FC1); L.mapY.create(bevH,bevW,CV_32FC1);
  cv::Matx33f H(
    H_cam_ground.m[0], H_cam_ground.m[1], H_cam_ground.m[2],
    H_cam_ground.m[3], H_cam_ground.m[4], H_cam_ground.m[5],
    H_cam_ground.m[6], H_cam_ground.m[7], H_cam_ground.m[8]
  );
  for(int y=0;y<bevH;++y){
    float gy = Ymin + (Ymax-Ymin)*(y+0.5f)/bevH;
    for(int x=0;x<bevW;++x){
      float gx = Xmin + (Xmax-Xmin)*(x+0.5f)/bevW;
      cv::Vec3f g(gx,gy,1.f), p=H*g;
      float w = (std::abs(p[2])>1e-8f)?p[2]:1e-8f;
      L.mapX.at<float>(y,x)=p[0]/w; L.mapY.at<float>(y,x)=p[1]/w;
    }
  }
  return L;
}
inline cv::Mat project_to_bev(const cv::Mat& camBgr, const Lut& L){
  cv::Mat bev(L.H,L.W, camBgr.type());
  cv::remap(camBgr, bev, L.mapX, L.mapY, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));
  return bev;
}
inline cv::Mat stitch_tiles(const std::array<cv::Mat,4>& tiles){
  CV_Assert(!tiles[0].empty());
  cv::Mat acc = cv::Mat::zeros(tiles[0].size(), CV_32FC3);
  cv::Mat wsum= cv::Mat::zeros(tiles[0].size(), CV_32FC1);
  auto addw=[&](const cv::Mat&t,float w){ if(t.empty())return; cv::Mat f; t.convertTo(f,CV_32FC3); cv::accumulate(f*w,acc); wsum+=w; };
  addw(tiles[0],0.25f); addw(tiles[1],0.25f); addw(tiles[2],0.25f); addw(tiles[3],0.25f);
  cv::Mat w3; cv::cvtColor(wsum,w3,cv::COLOR_GRAY2BGR);
  cv::Mat out32 = acc/(w3+1e-6f), out; out32.convertTo(out, CV_8UC3); return out;
}

/*==================== Compose H from (K,R,t,n,d) ====================*/
inline Mat3 compose_H_from_pose_plane(const Mat3& K,const Mat3& R,const Vec3& t,const Vec3& n,float d){
  Mat3 P=R;
  float Tn[9]={ t.x*n.x,t.x*n.y,t.x*n.z, t.y*n.x,t.y*n.y,t.y*n.z, t.z*n.x,t.z*n.y,t.z*n.z };
  for(int i=0;i<9;i++) P.m[i]-=Tn[i]/std::max(d,1e-6f);
  Mat3 H; mat3Mul(K,P,H);
  float s = std::fabs(H.m[8])>1e-8f? H.m[8] : std::sqrt(H.m[0]*H.m[0]+H.m[4]*H.m[4]+H.m[8]*H.m[8]);
  if(s>1e-9f) for(int i=0;i<9;i++) H.m[i]/=s;
  return H;
}
inline Lut buildLUT_from_pose_plane(const Mat3& K,const Mat3& R,const Vec3& t,const Vec3& n,float d,
                                    int bevW,int bevH,float Xmin,float Xmax,float Ymin,float Ymax){
  Mat3 H = compose_H_from_pose_plane(K,R,t,n,d);
  return buildLUTfromH_cv(H, bevW, bevH, Xmin,Xmax,Ymin,Ymax);
}

/*==================== Confidence Fusion (simple) ====================*/
inline float clamp01(float x){ return std::max(0.f, std::min(1.f,x)); }
struct ConfidenceFusion{
  float fuse(const FeatQuality& fq,const SeamQuality& sq,const VOQuality& voq,const PlaneQuality& pq) const {
    float seam = sq.inliers_ratio * (1.f/(1.f+sq.residual));
    float vo   = clamp01(voq.inliers * (1.f/(1.f+voq.residual)) * 0.01f);
    float base = clamp01(0.5f*fq.tex_spread + 0.5f*std::max(seam,vo));
    float rans = clamp01(pq.inliers_ratio * (1.f/(1.f+pq.residual)));
    return clamp01(0.6f*base + 0.4f*rans);
  }
};

/*==================== Camera Runtime / Context ====================*/
struct CameraRuntime{
  Mat3 K = Mat3::I();
  Vec3 t{0,0,0};
  Mat3 R = Mat3::I();
  Vec3 n{0,1,0};
  float d{1.f};
  Mat3 H_ground = Mat3::I();
  std::unique_ptr<Lut> lut_active, lut_building;
  int  last_lut_build_frame{-100000};
  bool building{false};
};

template<class T,size_t N> using Ring = SpscRing<T,N>;

struct Threads{ std::thread tCap,tFeat,tGpe,tUpd,tRend; std::atomic<bool> stop{false}; };

struct Context{
  Config cfg;
  Threads th;
  struct Queues{ Ring<FrameView,64> q1; Ring<FeaturesMsg,64> q2; Ring<GpeMsg,64> q3; Ring<CalibUpdateMsg,64> q4; } q;
  struct Telemetry{ std::atomic<size_t> q1_max{0},q2_max{0},q3_max{0},q4_max{0}; std::atomic<size_t> drops1{0},drops2{0},drops3{0},drops4{0}; } tm;
  Timers timers;
  CameraRuntime cams[4];
  int frame_global{0};
  struct LatestFrames{ std::mutex m; cv::Mat bgr[4]; Timestamp ts{}; } latest;

  std::atomic<float> speed_mps{0.f};
  std::atomic<float> planarity_norm{0.f};
  std::atomic<Mode>  mode{Mode::CHAIN};
  std::atomic<int>   frames_in_state{0};
};

/*==================== Policies ====================*/
inline bool should_rebuild_LUT(const Context& ctx,const CameraRuntime& cam,const Mat3& H_new,int frame_now,
                               int period_frames=50,float drift_tau=0.05f){
  auto driftExceeded=[&](const Mat3&A,const Mat3&B){ float acc=0.f; for(int i=0;i<9;i++){ float d=B.m[i]-A.m[i]; acc+=d*d; } return std::sqrt(acc)>drift_tau; };
  bool by_period=(frame_now-cam.last_lut_build_frame)>=period_frames;
  bool by_drift = driftExceeded(cam.H_ground,H_new);
  return by_period || by_drift;
}

/*==================== Capture ====================*/
inline bool openCameras(std::array<cv::VideoCapture,4>& caps,const std::array<std::string,4>& sources){
  for(int i=0;i<4;i++){
    if(sources[i].empty()) continue;
    if(!caps[i].open(sources[i])){ std::cerr<<"❌ Failed to open source "<<i<<": "<<sources[i]<<"\n"; return false; }
    caps[i].set(cv::CAP_PROP_FRAME_WIDTH,1280); caps[i].set(cv::CAP_PROP_FRAME_HEIGHT,720); caps[i].set(cv::CAP_PROP_FPS,30);
  }
  return true;
}

inline void captureLoop(Context& ctx){
  using namespace std::chrono;
  static std::array<cv::VideoCapture,4> caps; static bool initialized=false;
  if(!initialized){
    std::array<std::string,4> sources = {"0","1","2","3"}; // אפשר להחליף ל-"data/front.mp4" וכו'
    if(!openCameras(caps,sources)) std::cerr<<"⚠️ Capture init failed — dummy frames\n";
    initialized=true;
  }
  cv::Mat frames[4];
  while(!ctx.th.stop.load()){
    FrameView fv{};
    fv.ts.frame_id = ctx.timers.frame_id.fetch_add(1);
    fv.ts.mono_ns  = duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();

    bool ok=true;
    for(int i=0;i<4;i++){
      if(caps[i].isOpened()){
        ok &= caps[i].read(frames[i]);
        if(!ok){ caps[i].set(cv::CAP_PROP_POS_FRAMES,0); caps[i].read(frames[i]); }
      }else{
        frames[i]=cv::Mat::zeros(cv::Size(640,360),CV_8UC3);
        cv::putText(frames[i], "Dummy Cam "+std::to_string(i),{20,200},cv::FONT_HERSHEY_SIMPLEX,1.0,cv::Scalar(0,255,0),2);
      }
      fv.img[i]=frames[i].data; fv.w[i]=frames[i].cols; fv.h[i]=frames[i].rows; fv.stride[i]=static_cast<int>(frames[i].step);
    }

    { std::lock_guard<std::mutex> lk(ctx.latest.m);
      for(int i=0;i<4;i++) ctx.latest.bgr[i]=frames[i].clone();
      ctx.latest.ts=fv.ts;
    }

    if(!ctx.q.q1.push(std::move(fv))) ctx.tm.drops1++;
    size_t s=ctx.q.q1.size(); if(s>ctx.tm.q1_max) ctx.tm.q1_max=s;
    std::this_thread::sleep_for(std::chrono::milliseconds(33));
  }
  for(auto& c:caps) if(c.isOpened()) c.release();
}

/*==================== Features ====================*/
inline void featuresLoop(Context& ctx){
  int tick=0;
  cv::Ptr<cv::ORB> orb=cv::ORB::create(1000);
  cv::Size win(21,21); cv::TermCriteria tc(cv::TermCriteria::COUNT|cv::TermCriteria::EPS,30,0.01);
  std::vector<cv::Point2f> prevPts[4], currPts[4]; std::vector<cv::Mat> prevGray(4);

  while(!ctx.th.stop.load()){
    auto fv=ctx.q.q1.pop();
    if(!fv){ std::this_thread::sleep_for(std::chrono::microseconds(200)); continue; }

    FeaturesMsg fm{}; fm.fv=*fv;
    bool runHeavy=(tick%ctx.cfg.rates.FEATURES)==0;

    cv::Mat gray[4]; std::vector<cv::KeyPoint> kps[4]; cv::Mat desc[4];
    for(int i=0;i<4;i++){
      cv::Mat bgr(cv::Size(fv->w[i],fv->h[i]),CV_8UC3,fv->img[i],fv->stride[i]);
      cv::cvtColor(bgr,gray[i],cv::COLOR_BGR2GRAY);
      if(runHeavy || prevGray[i].empty()) orb->detectAndCompute(gray[i],cv::noArray(),kps[i],desc[i]);
    }

    // LK per camera
    for(int i=0;i<4;i++){
      if(!prevGray[i].empty()){
        if(prevPts[i].empty()) cv::KeyPoint::convert(kps[i],prevPts[i]);
        std::vector<uchar> status; std::vector<float> err;
        currPts[i].resize(prevPts[i].size());
        cv::calcOpticalFlowPyrLK(prevGray[i], gray[i], prevPts[i], currPts[i], status, err, win,3,tc);
        std::vector<cv::Point2f> a,b;
        for(size_t j=0;j<status.size();++j) if(status[j]){ a.push_back(prevPts[i][j]); b.push_back(currPts[i][j]); }
        fm.tracks[i].resize(a.size());
        for(size_t j=0;j<a.size();++j){ fm.tracks[i][j].p_prev={a[j].x,a[j].y,1}; fm.tracks[i][j].p_cur={b[j].x,b[j].y,1}; fm.tracks[i][j].ok=true; }
      }
    }

    // seam matching (adjacent cams)
    float totalResidual=0.f; int seamPairs=0;
    for(int i=0;i<4;i++){
      int j=(i+1)%4;
      if(!desc[i].empty() && !desc[j].empty()){
        cv::BFMatcher m(cv::NORM_HAMMING);
        std::vector<cv::DMatch> matches; m.match(desc[i],desc[j],matches);
        double maxD=0,minD=100; for(auto& mm:matches){ maxD=std::max(maxD,(double)mm.distance); minD=std::min(minD,(double)mm.distance); }
        std::vector<cv::Point2f> p1,p2;
        for(auto& mm:matches) if(mm.distance<=std::max(2*minD,30.0)){ p1.push_back(kps[i][mm.queryIdx].pt); p2.push_back(kps[j][mm.trainIdx].pt); }
        if(p1.size()>=4){
          std::vector<uchar> mask; cv::Mat H=cv::findHomography(p1,p2,cv::RANSAC,3,mask);
          float inl=(float)cv::countNonZero(mask);
          float resid = (float)cv::norm(p1,p2,cv::NORM_L2)/(float)p1.size();
          fm.seamq.inliers_ratio = inl/(float)mask.size(); fm.seamq.residual=resid;
          totalResidual += resid; seamPairs++;
        }
      }
    }
    if(seamPairs>0) totalResidual/=seamPairs;
    fm.seamq.conf = clamp01(1.f - totalResidual/10.f);

    // light VO
    for(int i=0;i<4;i++){
      if(fm.tracks[i].size()>=8){
        std::vector<cv::Point2f> a,b; a.reserve(fm.tracks[i].size()); b.reserve(fm.tracks[i].size());
        for(auto&t:fm.tracks[i]){ a.push_back({t.p_prev.x,t.p_prev.y}); b.push_back({t.p_cur.x,t.p_cur.y}); }
        std::vector<uchar> mask; cv::Mat E=cv::findEssentialMat(a,b,1.0,cv::Point2d(0,0),cv::RANSAC,0.999,1.0,mask);
        if(!E.empty()){ int inl=cv::countNonZero(mask); fm.voq.inliers=inl; fm.voq.residual=totalResidual; fm.voq.conf=clamp01(inl/(float)a.size()); }
      }
    }

    // Feature quality
    int totalKP=0; for(int i=0;i<4;i++) totalKP += (int)kps[i].size();
    fm.featq.tex_density = totalKP / (4.0f * fv->w[0] * fv->h[0]);
    fm.featq.tex_spread  = clamp01((float)(runHeavy?0.6:0.5));

    if(!ctx.q.q2.push(std::move(fm))) ctx.tm.drops2++;
    size_t s=ctx.q.q2.size(); if(s>ctx.tm.q2_max) ctx.tm.q2_max=s;

    for(int i=0;i<4;i++){ prevGray[i]=gray[i].clone(); cv::KeyPoint::convert(kps[i],prevPts[i]); }
    ++tick;
  }
}

/*==================== GPE (stubs) ====================*/
inline void gpeLoop(Context& ctx){
  ConfidenceFusion fuse;
  while(!ctx.th.stop.load()){
    auto fm=ctx.q.q2.pop(); if(!fm){ std::this_thread::sleep_for(std::chrono::microseconds(200)); continue; }
    GpeMsg gm{}; gm.base=*fm;

    // החלטת מצב פשוטה (אפשר לשדרג ל-FSM)
    bool slow = (ctx.speed_mps.load()<=ctx.cfg.th.SLOW_SPEED);
    if(slow && fm->featq.tex_spread>=ctx.cfg.th.TEX_SPREAD_MIN && fm->seamq.conf>=ctx.cfg.th.SEAM_CONF_MIN) gm.mode=Mode::CHAIN;
    else { gm.mode = (ctx.planarity_norm.load()>=ctx.cfg.th.PLANAR_BAD) ? Mode::PIECEWISE : Mode::VO; }

    // Plane estimate (stub reasonable)
    if(gm.mode==Mode::CHAIN){ gm.plane={ {0,1,0}, 1.0f, {0.6f,0.3f} }; }
    else if(gm.mode==Mode::VO){ gm.plane={ {0,1,0}, 1.0f, {0.65f,0.28f} }; }
    else { gm.plane={ {0,1,0}, 1.0f, {0.7f,0.25f} }; }

    gm.conf = fuse.fuse(fm->featq, fm->seamq, fm->voq, gm.plane.q);
    if(!ctx.q.q3.push(std::move(gm))) ctx.tm.drops3++;
    size_t s=ctx.q.q3.size(); if(s>ctx.tm.q3_max) ctx.tm.q3_max=s;
  }
}

/*==================== Update (target H) ====================*/
inline void updateLoop(Context& ctx){
  while(!ctx.th.stop.load()){
    auto gm=ctx.q.q3.pop(); if(!gm){ std::this_thread::sleep_for(std::chrono::microseconds(200)); continue; }
    CalibUpdateMsg cu{}; cu.gpe=*gm; cu.conf=gm->conf;

    // בונים H יעד מכל מצלמה (כרגע מ-(K,R,t,n,d) – אפשר לשלב Stepper)
    for(int c=0;c<4;c++){
      Mat3 Ht = compose_H_from_pose_plane(ctx.cams[c].K, ctx.cams[c].R, ctx.cams[c].t, {gm->plane.n[0],gm->plane.n[1],gm->plane.n[2]}, gm->plane.d);
      for(int i=0;i<9;i++) cu.dH[c][i]=Ht.m[i];
    }
    cu.lut_rebuild=false; // rebuild לפי policy ב-render

    if(!ctx.q.q4.push(std::move(cu))) ctx.tm.drops4++;
    size_t s=ctx.q.q4.size(); if(s>ctx.tm.q4_max) ctx.tm.q4_max=s;
  }
}

/*==================== Render (commit H, rebuild LUT, project+stitch) ====================*/
inline void renderLoop(Context& ctx){
  const int   BEV_W=1024, BEV_H=1024;
  const float Xmin=-5.f, Xmax=15.f, Ymin=-10.f, Ymax=10.f;

  while(!ctx.th.stop.load()){
    auto cu=ctx.q.q4.pop(); if(!cu){ std::this_thread::sleep_for(std::chrono::microseconds(200)); continue; }

    if(cu->conf >= ctx.cfg.th.CONF_GATE){
      bool any_need_rebuild=cu->lut_rebuild;
      for(int c=0;c<4;c++){
        Mat3 Hnew{}; for(int i=0;i<9;i++) Hnew.m[i]=cu->dH[c][i];
        if(!any_need_rebuild) any_need_rebuild |= should_rebuild_LUT(ctx, ctx.cams[c], Hnew, ctx.frame_global);
        ctx.cams[c].H_ground = Hnew;
      }
      if(any_need_rebuild){
        for(int c=0;c<4;c++){
          ctx.cams[c].lut_building = std::make_unique<Lut>(
            buildLUT_from_pose_plane(ctx.cams[c].K, ctx.cams[c].R, ctx.cams[c].t, ctx.cams[c].n, ctx.cams[c].d,
                                     BEV_W,BEV_H,Xmin,Xmax,Ymin,Ymax));
        }
        for(int c=0;c<4;c++){
          ctx.cams[c].lut_active = std::move(ctx.cams[c].lut_building);
          ctx.cams[c].last_lut_build_frame = ctx.frame_global;
        }
      }
    }

    bool haveAll=true; for(int c=0;c<4;c++) if(!ctx.cams[c].lut_active){ haveAll=false; break; }
    if(!haveAll){ ctx.frame_global++; continue; }

    std::array<cv::Mat,4> camImg;
    { std::lock_guard<std::mutex> lk(ctx.latest.m);
      for(int c=0;c<4;c++) camImg[c]=ctx.latest.bgr[c];
    }
    if(camImg[0].empty()||camImg[1].empty()||camImg[2].empty()||camImg[3].empty()){ ctx.frame_global++; continue; }

    std::array<cv::Mat,4> tiles;
    for(int c=0;c<4;c++) tiles[c]=project_to_bev(camImg[c], *ctx.cams[c].lut_active);

    cv::Mat bev = stitch_tiles(tiles);
    // הצגה אופציונלית:
    // cv::imshow("BEV", bev); cv::waitKey(1);

    ctx.frame_global++;
  }
}

/*==================== Start / Stop ====================*/
inline void start_all(Context& ctx){
  ctx.th.stop=false;
  ctx.th.tCap  = std::thread(captureLoop, std::ref(ctx));
  ctx.th.tFeat = std::thread(featuresLoop,std::ref(ctx));
  ctx.th.tGpe  = std::thread(gpeLoop,    std::ref(ctx));
  ctx.th.tUpd  = std::thread(updateLoop, std::ref(ctx));
  ctx.th.tRend = std::thread(renderLoop, std::ref(ctx));
}
inline void stop_all(Context& ctx){
  ctx.th.stop=true;
  ctx.th.tRend.join(); ctx.th.tUpd.join(); ctx.th.tGpe.join(); ctx.th.tFeat.join(); ctx.th.tCap.join();
}
