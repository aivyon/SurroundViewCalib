# SurroundViewCalib

Multithreaded surround-view dynamic calibration pipeline  
for real-time automotive perception and camera fusion.

---

## 🧠 Overview
This project implements a multi-threaded pipeline skeleton in **C++20** (Visual Studio 2022) for  
real-time surround-view camera calibration and stitching.

**Stages**
Capture → Features → GroundEstimation (CHAIN / VO / PIECEWISE) → Update/H → Render



**Core components**
- SPSC lock-free queues between stages  
- Mode FSM (CHAIN / VO / PIECEWISE) with hysteresis  
- Confidence fusion for measurement weighting  
- Geometric Homography Stepper (`R,t,n,d`) for stable updates  
- Thread scheduling & rate control for real-time operation

---

## 🧰 Build
- Visual Studio 2022 (v143)
- Language Standard: **C++20**
- Windows 10/11 SDK 10.0.x
- Recommended configuration: `x64 | Release`

---

## 🚀 Next Steps
- [ ] Integrate OpenCV for ORB/LK feature tracking  
- [ ] Implement CHAIN and VO ground-plane modes  
- [ ] Add Kalman filtering for (n,d,R) smoothing  
- [ ] Stitch BEV tiles and visualize seams  

---

## 📂 Structure


include/
└─ dynamic_calib_pipeline.hpp
src/
└─ main.cpp
.gitignore
SurroundViewCalib.sln

---

## 🧾 License
MIT (free to use for research and development)



