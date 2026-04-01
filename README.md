# ShotIQ 🏀

**AI-powered basketball jump shot analyzer** — works in your browser, no app download needed.

## Features

- 🎯 **Real-time pose detection** using TensorFlow.js MoveNet
- 💪 **Form scoring**: Elbow alignment, release height, follow-through
- ⚡ **Release time grading**: Elite (<250ms) / Good / Slow
- 📊 **Session stats**: Shot count, avg score, best score, breakdown
- 📱 **PWA** — works on iPhone Safari & Android Chrome full-screen

## Tech Stack

- Vanilla HTML/CSS/JS (single file)
- TensorFlow.js 4.10.0
- MoveNet SinglePose Lightning (fast mobile model)
- No backend, no dependencies to install

## Usage

1. Open the app
2. Select shooting hand (right/left)
3. Tap "Allow Camera & Start"
4. Shoot! The AI tracks your form in real-time
5. Tap "End Session" to see full stats

## Scoring

| Metric | What it measures |
|--------|-----------------|
| **Elbow** | Vertical alignment of elbow under wrist |
| **Height** | Wrist position at peak release relative to shoulder |
| **Follow** | Wrist staying elevated after release |
| **Overall** | Average of the three metrics (0–100) |

## Release Time Grades

| Grade | Time |
|-------|------|
| ⚡ Elite | < 250ms |
| 👍 Good | 250–350ms |
| 🐢 Slow | > 350ms |

## Deployment

Hosted on Vercel as a static site.
