"""Fill the Image Processing Final Report Template with our actual results.

Loads the official template (preserving its fonts/sizes/styles) and replaces
the placeholder text paragraph-by-paragraph, so the output keeps the
course's required formatting. Run from the project root:

    ./venv/bin/python scripts/build_report.py
"""
import copy
import os

import docx
from docx.shared import Inches

ROOT = os.path.join(os.path.dirname(__file__), "..")
TEMPLATE = os.path.join(ROOT, "Image Processing Final Report Template (1).docx")
OUTPUT = os.path.join(ROOT, "Final_Report.docx")
ANNOTATED_IMG = os.path.join(ROOT, "test_assets", "sample_hands_annotated.png")


def set_paragraph_runs(paragraph, segments):
    """segments: list of (text, bold) tuples. Reuses the paragraph's
    existing font (name/size/italic) but overrides bold per-segment."""
    base_run = paragraph.runs[0] if paragraph.runs else None
    base_font = base_run.font if base_run else None
    for r in list(paragraph.runs):
        r._element.getparent().remove(r._element)
    for text, bold in segments:
        run = paragraph.add_run(text)
        if base_font is not None:
            run.font.name = base_font.name
            run.font.size = base_font.size
            run.font.italic = base_font.italic
        run.bold = bold


def set_body(paragraph, text):
    set_paragraph_runs(paragraph, [(text, False)])


def set_bulleted(paragraph, label, text):
    set_paragraph_runs(paragraph, [(label + " ", True), (text, False)])


def build():
    doc = docx.Document(TEMPLATE)
    p = doc.paragraphs

    set_paragraph_runs(p[0], [("Multi-User Collaborative Air Drawing on a Single Webcam: "
                                "A Practical MediaPipe + ByteTrack-Style Implementation", True)])
    set_paragraph_runs(p[1], [("Image Processing — Final Project Report", True)])
    p[1].runs[0].italic = True
    set_body(p[2], "411221420 林彥宏, 411221424 王韋峰, 411221425 陸竑宇, 411221426 劉家均")

    set_body(p[5],
        "Collaborative whiteboarding traditionally requires either a single physical board, "
        "where participants take turns, or expensive multi-touch displays. Webcam-based air "
        "drawing is a low-cost alternative, but existing systems are almost all single-user. "
        "This project builds a real-time system in which 2 to 6 users draw simultaneously on a "
        "shared canvas using one commodity webcam, with every stroke correctly attributed to "
        "its author even under hand occlusion, hand-crossing, and a user leaving and re-entering "
        "the frame. Given a live RGB webcam stream as input, the system outputs a shared drawing "
        "canvas where each stroke is rendered in its author's color and logged with a "
        "millisecond timestamp and user ID.")
    set_bulleted(p[6], "Motivation.",
        "Low-cost collaborative drawing/whiteboarding is useful in classrooms and remote "
        "settings without multi-touch hardware; the core sub-problem — robust per-identity "
        "tracking of multiple similar-looking, non-rigid objects (hands) from one camera — also "
        "generalizes to gesture-based UIs and lightweight multi-person interaction systems.")
    set_bulleted(p[7], "Problem Formulation.",
        "Input: a live 720p RGB webcam stream containing 2–6 hands. Output: a per-user "
        "attributed stroke canvas, i.e. a stream of (user_id, point, timestamp, pen-state) "
        "events that render as owner-colored polylines. This is a real-time multi-object "
        "tracking and re-identification problem layered under a gesture-driven drawing "
        "interface, rather than a single-image transform.")
    set_bulleted(p[8], "Challenges.",
        "Hands occlude each other when reaching across the frame; hands crossing paths can "
        "swap tracker identities; users exiting and re-entering the frame break naive "
        "frame-to-frame tracking; hands have far weaker, more self-similar appearance cues than "
        "whole bodies, so appearance-based re-identification is intrinsically harder; the system "
        "must run in real time on CPU only (no discrete GPU assumed).")
    set_bulleted(p[9], "Contributions.",
        "A complete, working pipeline — detection, tracking with two-stage Kalman+IoU "
        "association, appearance-based re-identification, wave-gesture user calibration, "
        "pinch-based pen control, and a shared timestamped canvas. Where the original proposal "
        "called for training a custom YOLOv8-hand detector and a triplet-loss MobileNetV3 "
        "embedding (both requiring dataset collection and GPU training time unavailable in this "
        "development environment), we substitute off-the-shelf and classical training-free "
        "components in the same architectural slots, and validate every stage with a "
        "reproducible offline test suite (8 tests, `tests/test_offline.py`) plus a real sample "
        "photo, since no webcam was accessible during development.")

    set_body(p[11],
        "The pipeline follows: webcam frame → hand detection → multi-hand tracking + "
        "re-identification → user calibration (binding) → pinch gesture → canvas rendering, "
        "matching the proposal's stage breakdown (sections 2a–2d).")
    set_bulleted(p[12], "Pipeline / System Configuration.",
        "frame → HandDetector (MediaPipe HandLandmarker, up to 6 hands, 21 landmarks each) → "
        "HandTracker (8-state constant-velocity Kalman filter per hand + two-stage Hungarian/IoU "
        "association + appearance re-ID against a lost-track pool) → CalibrationManager "
        "(wave-gesture binds a track to a user/color) → PinchDetector (1€-filtered fingertip "
        "cursor, thumb–index pinch toggles pen up/down) → Canvas (owner-colored stroke "
        "rendering, millisecond-timestamped stroke log) → GUI overlay (legend, FPS, bounding "
        "boxes).")
    set_bulleted(p[13], "Algorithm Design.",
        "(1) Detection: MediaPipe's HandLandmarker Tasks API, configured for up to 6 "
        "simultaneous hands, replaces the proposal's plan to fine-tune YOLOv8-hand on EgoHands "
        "+ 11k Hands — the off-the-shelf model already exceeds the 2-hand cap of the legacy "
        "MediaPipe API, at zero training cost. (2) Tracking: a constant-velocity Kalman filter "
        "(OpenCV `KalmanFilter`, 8-dim state) predicts each track's box; association uses "
        "SciPy's Hungarian algorithm (`linear_sum_assignment`) in two tiers — high-confidence "
        "detections matched at IoU ≥ 0.3, then leftover low-confidence detections rescued at "
        "IoU ≥ 0.1 — mirroring ByteTrack's principle of not discarding low-score boxes. "
        "(3) Re-identification: rather than a trained MobileNetV3 + triplet-loss embedding (which "
        "needs a self-collected multi-subject dataset and GPU training), each hand crop is "
        "described by a concatenated HSV color histogram (32×32 bins) and HOG descriptor on a "
        "64×64 resized crop (2788-D total), L2-normalized; lost tracks are revived by cosine "
        "similarity above a manually-tuned threshold. (4) Calibration: a 24-frame sliding-window "
        "wave detector (≥3 direction reversals, amplitude ≥ 0.6× hand width) binds any "
        "unbound confirmed track to a new user — this runs continuously, so it also covers "
        "mid-session joins. (5) Drawing: cursor = index fingertip (landmark 8), smoothed with a "
        "1€ filter (mincutoff = 1.0 Hz, β = 0.007, literature defaults from Casiez et al.); pen "
        "state = thumb–index distance (landmarks 4, 8) normalized by wrist-to-middle-MCP "
        "distance (landmarks 0, 9), with a close/open hysteresis band (0.35 / 0.45) to prevent "
        "flicker at the threshold.")
    set_bulleted(p[14], "Tools & Dataset.",
        "Python 3.12, OpenCV 4.13, MediaPipe 0.10.35 (HandLandmarker float16 model bundle), "
        "NumPy, SciPy. No training dataset was needed since every component is either "
        "off-the-shelf (detector) or classical/training-free (re-ID descriptor, calibration, "
        "pinch logic). Validation used one real photo — MediaPipe's public sample asset "
        "`woman_hands.jpg` (2 real hands) — plus procedurally generated synthetic frames for "
        "tracker/calibration/pinch/canvas unit tests, since no physical webcam or human subjects "
        "were available in the development sandbox. Collecting the proposal's ~10-session, "
        "~30-minute, CVAT-annotated multi-user dataset was out of scope for this implementation "
        "pass and is the natural next step for the user to run locally.")
    set_bulleted(p[15], "Evaluation.",
        "Because no human-subject recordings exist yet, this report evaluates each pipeline "
        "stage individually with deterministic, repeatable checks rather than the proposal's "
        "session-level MOTA/IDF1/stroke-attribution-accuracy: jitter reduction ratio (1€ filter), "
        "cosine-similarity separability (re-ID descriptor), track-ID stability across staged "
        "motion/occlusion/crossing/re-entry scenarios (tracker), gesture true/false-positive "
        "behavior (calibration), and an end-to-end exception-free run (full pipeline). The full "
        "MOT-metric ablation (IoU-only vs. ByteTrack vs. ByteTrack+ReID) from the proposal "
        "requires the not-yet-collected annotated dataset and remains future work.")

    set_body(p[17],
        "All numbers below come from `tests/test_offline.py`, run with "
        "`./venv/bin/python -m tests.test_offline`; every test passed.")
    set_bulleted(p[18], "Quantitative Results.",
        "1€ filter: raw frame-to-frame jitter std 2.59 px → filtered 1.09 px (2.4× reduction) "
        "while tracking a 150 px/s synthetic motion with only 6.80 px RMS lag error. Re-ID "
        "descriptor: cosine similarity 0.943 for two crops of the same synthetic hand color vs. "
        "0.879 for a visibly different hand color — separable, but with a modest margin "
        "consistent with a classical (non-learned) descriptor. Tracker: a single track ID was "
        "preserved through 6 frames of continuous motion, a 3-frame occlusion (Kalman-coasted), "
        "and a 40-frame occlusion followed by re-entry at a different screen location "
        "(recovered via appearance re-ID) — and across a 40-frame head-on crossing of two hands, "
        "zero ID swaps occurred. Calibration: a 29-frame oscillating wave correctly triggered "
        "user binding, while 29 frames of straight-line motion (mimicking an actual drawing "
        "stroke) correctly did not. Real-photo detection: MediaPipe's HandLandmarker found 2/2 "
        "hands in a real photo at confidence 0.94 and 0.96.")
    set_bulleted(p[19], "Qualitative Results.",
        "Figure 1 shows the real test photo with the system's output overlaid: green boxes are "
        "the detected hand bounding boxes, red dots are the 21 detected landmarks per hand. Both "
        "hands are localized correctly and the landmarks visibly align with finger joints and "
        "tips, confirming the detection stage is accurate on real (non-synthetic) input.")
    if os.path.exists(ANNOTATED_IMG):
        img_p = doc.add_paragraph()
        img_p.add_run().add_picture(ANNOTATED_IMG, width=Inches(3.2))
        cap_p = doc.add_paragraph()
        set_body(cap_p, "Figure 1. Detected hand bounding boxes and 21-point landmarks on a real "
                         "test photo (MediaPipe sample asset).")
        p[19]._p.addnext(cap_p._p)
        p[19]._p.addnext(img_p._p)
    set_bulleted(p[20], "Result Analysis.",
        "Every stage behaves correctly under controlled, repeatable synthetic stress tests "
        "(occlusion, crossing, re-entry, gesture false-positive rejection) and on one real photo "
        "for the detector. What has not been verified is true end-to-end behavior on live, "
        "simultaneous, multi-person webcam footage — no camera access was available in the "
        "development sandbox (a webcam opened successfully but returned no frames), so the "
        "wave-gesture and pinch interactions still need to be exercised by a person in front of "
        "a real camera before this can be called production-validated.")

    set_body(p[22],
        "The synthetic stress tests in Section III show each pipeline stage is individually "
        "correct, but the most useful discussion is where the design choices made to keep this "
        "implementation training-free and webcam-free during development are likely to hold up "
        "or break down in real, live, multi-person use.")

    set_bulleted(p[23], "Strengths.",
        "Kalman velocity prediction alone (no appearance) correctly disambiguated two hands "
        "during a head-on crossing test, because each track's predicted box stays closer to its "
        "own continuing trajectory than to the other hand's — exactly the mechanism the proposal "
        "relies on motion+IoU for. The re-ID descriptor, despite being classical rather than "
        "learned, was separable enough in testing to bridge a 40-frame occlusion and revive the "
        "correct identity after re-entry at a different position. The 1€ filter delivered the "
        "intended jitter-vs-latency trade-off (>2× jitter reduction, single-digit-pixel lag). "
        "The whole pipeline runs on CPU only with no training step, satisfying the proposal's "
        "real-time/no-GPU goal by construction.")
    set_bulleted(p[24], "Limitations.",
        "The classical HSV+HOG descriptor has a much smaller same-vs-different similarity "
        "margin than a trained embedding would likely produce (0.943 vs. 0.879 in testing), so "
        "`REID_COSINE_THRESH` is a manually tuned, lighting/clothing-sensitive constant — exactly "
        "the kind of \"manually tuned parameter\" limitation this report template warns about. "
        "The proposal's headline targets (≥85% stroke-attribution accuracy, IDF1 ≥ 0.80, "
        "ByteTrack+ReID beating IoU-only by ≥15 points) could not be evaluated because the "
        "~10-session annotated multi-user dataset was never collected — no webcam access existed "
        "in the development environment. The wave-gesture and pinch thresholds were tuned and "
        "validated only against synthetic motion, not real human gesture variability.")
    set_bulleted(p[25], "Failure Cases.",
        "(1) Re-ID confusion under similar appearance: two users with similarly colored sleeves "
        "could produce cosine similarity high enough to revive the wrong lost track onto a new "
        "detection, silently misattributing a hand to the wrong user — the 0.879 cross-hand "
        "similarity measured in testing shows this risk is real, not hypothetical, given the "
        "0.55 threshold. (2) Calibration sensitivity: a user waving too slowly or with low "
        "amplitude within the 24-frame window will not be recognized and must repeat the "
        "gesture; conversely, a fast back-and-forth hand correction during normal drawing could, "
        "in principle, satisfy the direction-change/amplitude conditions and trigger an "
        "unintended recalibration, though the thresholds were set conservatively to make this "
        "rare.")

    set_body(p[27],
        "This project implemented a complete real-time multi-user air-drawing pipeline — "
        "detection, Kalman+IoU tracking with appearance re-identification, wave-gesture user "
        "calibration, pinch-based drawing, and a timestamped shared canvas — following the "
        "architecture of the original proposal. Where the proposal called for training a custom "
        "YOLOv8-hand detector and a triplet-loss MobileNetV3 embedding, this implementation uses "
        "off-the-shelf and classical training-free substitutes in the same roles, because no GPU "
        "training pipeline, multi-subject capture sessions, or annotation tooling were available "
        "in the development environment. All 8 targeted offline tests pass, demonstrating each "
        "stage is individually correct on synthetic stress scenarios and one real photo. The "
        "proposal's session-level evaluation (MOTA/IDF1/stroke-attribution accuracy on annotated "
        "real multi-user recordings, and the IoU-only vs. ByteTrack vs. ByteTrack+ReID ablation) "
        "was not run, since it requires real webcam recordings this environment could not "
        "produce — that is the immediate next step, to be run locally with `python -m "
        "air_draw.main` and a real webcam. Future work: replace the classical descriptor with a "
        "trained embedding once a multi-subject dataset exists, and complete the proposed "
        "ablation on real annotated footage to confirm how much the appearance head actually "
        "contributes.")

    refs = [
        '[1] F. Zhang et al., "MediaPipe Hands: On-device Real-time Hand Tracking," '
        'CVPR Workshops, 2020.',
        '[2] Y. Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every '
        'Detection Box," ECCV, 2022.',
        '[3] A. Howard et al., "Searching for MobileNetV3," ICCV, 2019.',
        '[4] G. Casiez, N. Roussel, and D. Vogel, "1€ Filter: A Simple Speed-based '
        'Low-pass Filter for Noisy Input in Interactive Systems," CHI, 2012.',
        '[5] H. W. Kuhn, "The Hungarian method for the assignment problem," Naval '
        'Research Logistics Quarterly, vol. 2, no. 1-2, pp. 83-97, 1955.',
    ]
    refs_intro = p[29]
    refs_intro._p.getparent().remove(refs_intro._p)

    set_body(p[30], refs[0])
    ref_style = p[30].style
    last_elem = p[30]._p
    for ref in refs[1:]:
        new_p = doc.add_paragraph(style=ref_style)
        set_body(new_p, ref)
        last_elem.addnext(new_p._p)
        last_elem = new_p._p

    doc.save(OUTPUT)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    build()
