# 🎨 AirCanvas AI

An AI-powered gesture-controlled virtual drawing canvas built using OpenCV, MediaPipe, BLIP, and Stable Diffusion.

Users can draw in the air using hand gestures, erase drawings, switch colors, apply transformations, and generate AI-created artwork from sketches in real time.

---

# 🚀 Features

- 🖐️ Real-time hand tracking using MediaPipe
- ✍️ Air drawing using index finger gestures
- 🎨 Multiple drawing colors
- 🧼 Eraser tool with permanent mask-based erasing
- 🤖 AI image generation using Stable Diffusion XL
- 🧠 Automatic prompt generation using BLIP captioning model
- ⌨️ Keyboard shortcuts for quick interaction
- 🔄 Image transformations:
  - Scale
  - Rotate
  - Shift
  - Reflect
- ✨ Dynamic sparkle visual effects
- ⚡ Optimized rendering with caching and cooldown protection

---

# 🏗️ System Architecture

```text
Camera Input
      ↓
MediaPipe Hand Tracking
      ↓
Gesture Recognition
      ↓
Drawing Engine
(strokeLayer + eraserMask)
      ↓
Canvas Save
      ↓
BLIP Caption Generation
      ↓
Prompt Enhancement
      ↓
Stable Diffusion XL
      ↓
AI Generated Artwork
