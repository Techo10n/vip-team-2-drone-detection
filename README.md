# vip-team-2-drone-detection

Drive link containing all RT-DETR work for the semester: [https://drive.google.com/drive/folders/1KnJWp8P6UfjbysVYxzfMOl9EsPynHhhD?usp=sharing](https://drive.google.com/drive/folders/1KnJWp8P6UfjbysVYxzfMOl9EsPynHhhD?usp=sharing)


# Better Place Drones Jetson Info:

**Username:** team2

**Password:** bpdteam2

# Jetson Demo Startup Sequence

## 1. Hardware

- Plug barrel jack power into Jetson
- Plug USB-C cable from Jetson into MacBook
- Wait ~60 seconds to boot

## 2. Mac — Terminal 1 (SSH + Jupyter)

`ssh team2@192.168.55.1`

Then inside the Jetson:
`jupyter lab --no-browser`

**Leave this terminal open.**

## 3. Mac — Terminal 2 (SSH Tunnel)

`ssh -L 8000:localhost:8888 team2@192.168.55.1`

**Leave this terminal open.**

## 4. Browser

Go to: `http://localhost:8000`

Enter Jupyter password to access.

---

## Notes

- Hotspot only needed if installing packages — not required for inference
- If SSH fails, check USB-C cable is plugged in and Jetson has fully booted
- Model weights: `/home/team2/best.pt`
- Test images: `/home/team2/images/`
- Inference results: `/home/team2/runs/detect/predict/`
