---
name: M3 - VO Plane
about: VO לכל מצלמה, Triangulation, RANSAC למישור
title: "[M3] VO-based ground plane"
labels: ["milestone:M3", "type:feature", "status:todo"]
---

### משימות
- [ ] VO לכל מצלמה (E→R,t) בחלון זמן מתגלגל.
- [ ] Triangulation לנקודות כביש (ROI).
- [ ] RANSAC Plane → (n,d) + inliers/residual.
- [ ] שילוב עם Kalman (החלקת n,d,R).
- [ ] השוואה נגד CHAIN: שיפור יציבות/דיוק בזיהוי שיפוע.

### DoD
- [ ] planarity residual ↓ לעומת CHAIN בהרצה על “עלייה/ירידה”.
- [ ] Kalman מייצב Pitch/Roll ללא overshoot מורגש.
