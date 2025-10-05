---
name: M1 - Features (OpenCV)
about: ORB/FAST + LK + Homography + Essential (קל)
title: "[M1] Features Loop MVP"
labels: ["milestone:M1", "type:feature", "status:todo"]
---

### משימות
- [ ] Capture דמה: קריאת 4 תמונות מ-`/data/` (או מצלמה אחת משוכפלת).
- [ ] ORB/FAST לזיהוי + תיאור (רזולוציה מוקטנת / ROI).
- [ ] LK `calcOpticalFlowPyrLK` לעקיבה בין פריימים.
- [ ] `findHomography(..., RANSAC)` באיזורי תפר → `seamq.inliers_ratio/residual`, `seamq.conf`.
- [ ] `findEssentialMat + recoverPose` → `voq.inliers/residual`, `voq.conf`.
- [ ] טלמטריה בסיסית: הדפסת `mode` + `conf`, זמני ריצה לשלב.

### DoD
- [ ] בנייה וריצה מקומית (Debug/x64).
- [ ] `mode` מתחלף ל-CHAIN כשיש טקסטורה + seam-conf.
- [ ] אין קריסות; זמן ריצה סביר.
