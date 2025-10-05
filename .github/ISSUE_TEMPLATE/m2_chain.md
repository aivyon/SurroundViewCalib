---
name: M2 - CHAIN
about: Pairwise cam→ref, R_ref איטי, n=R_ref*n0, d=d0, Stepper
title: "[M2] CHAIN mode"
labels: ["milestone:M2", "type:feature", "status:todo"]
---

### משימות
- [ ] שרשור הומוגרפיות cam→ref סביב הרכב (קדמית/צד/אחורית).
- [ ] VO איטי ל-`R_ref` (כל N פריימים).
- [ ] `n = R_ref * n0`, `d = d0`.
- [ ] `HomographyStepperGeo::step` לעדכון H באופן מדורג.
- [ ] LUT rebuild כל 40–60 פריימים או drift>סף.
- [ ] Telemetry: זמן ריצה, גודל תורים, drops.

### DoD
- [ ] Seam misalignment ↓ בפריימים יציבים.
- [ ] תנועת H חלקה (ללא “קפיצות”).
