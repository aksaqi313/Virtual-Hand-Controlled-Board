# Virtual Hand-Controlled Board 🖐✏️

A real-time **virtual whiteboard** you control with your bare hands — no mouse, no stylus, just your webcam and hand gestures.

---

## Features

| Feature | Description |
|---|---|
| ✏️ **Draw** | Point your index finger to paint freely |
| 🧹 **Erase** | Raise index + middle finger (✌) to erase |
| 🎨 **8 Colors** | White, Yellow, Cyan, Green, Pink, Red, Blue, Orange |
| 🔘 **4 Brush Sizes** | 4 / 8 / 14 / 22 px |
| ✋ **Toolbar Selection** | Open palm → hover over toolbar for 12 frames to select |
| 🗑️ **Clear Canvas** | Wipe the board clean |
| 💾 **Save PNG** | Saves a timestamped screenshot of your drawing |
| 📊 **Live FPS** | Displayed in the status bar |

---

## Gestures

| Gesture | Action |
|---|---|
| ☝ Index finger only | **DRAW** |
| ✌ Index + Middle fingers | **ERASE** |
| 🖐 All fingers up | **SELECT** (hover over toolbar) |
| ✊ Fist / other | **IDLE** (lifts the pen) |

---

## Installation

```bash
pip install -r requirements.txt
```

> **Note:** MediaPipe requires Python 3.8 – 3.11. On newer Python, use a virtual environment.

---

## Run

```bash
python main.py
```

---

## Keyboard Shortcuts

| Key | Action |
|---|---|
| `Q` / `ESC` | Quit |
| `C` | Clear canvas |
| `S` | Save drawing as PNG |
| `+` / `=` | Increase brush size |
| `-` | Decrease brush size |

---

## Project Structure

```
Virtual Hand-Controlled Board/
├── main.py          # Entry point – webcam loop + orchestration
├── hand_tracker.py  # MediaPipe hand detection & gesture classification
├── canvas.py        # Persistent drawing layer & compositing
├── ui.py            # Toolbar rendering & hit-testing
├── requirements.txt # Python dependencies
└── README.md        # This file
```

---

## Troubleshooting

- **Webcam not detected** → Change `CAM_INDEX = 0` to `1` or `2` in `main.py`
- **Low FPS** → Lower `FRAME_W / FRAME_H` in `main.py` (e.g., 640×480)
- **MediaPipe install fails** → Use Python 3.10 in a `venv`
