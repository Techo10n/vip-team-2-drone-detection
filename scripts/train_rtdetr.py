"""
Train RT-DETR-l on SeaDronesSee dataset.
Supports both CPU (for testing) and GPU (for real training).

Usage:
  CPU test:  python scripts/train_rtdetr.py --test
  GPU train: python scripts/train_rtdetr.py
"""
import argparse
import torch
from ultralytics import RTDETR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--test', action='store_true',
        help='Run a quick CPU sanity test with mini dataset'
    )
    args = parser.parse_args()

    # Auto-detect device
    if args.test:
        device = 'cpu'
        data_config = 'configs/seadronessee_mini.yaml'
        epochs = 10
        batch = 2
        name = 'cpu_sanity_test'
        print('=== CPU SANITY TEST MODE ===')
        print('This will be very slow. We just need it to complete without errors.')
    else:
        if not torch.cuda.is_available():
            print('ERROR: No CUDA GPU detected!')
            print('For real training, you need an NVIDIA GPU.')
            print('Run with --test flag for CPU sanity testing.')
            return
        device = 0  # First GPU
        data_config = 'configs/seadronessee.yaml'
        epochs = 100
        batch = 8
        name = 'seadronessee_v1'
        print(f'=== GPU TRAINING MODE ===')
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')

    # Load pretrained RT-DETR-l (downloads weights on first run)
    model = RTDETR("rtdetr-l.pt")

    # Train
    results = model.train(
        data=data_config,
        epochs=epochs,
        imgsz=1280,          # Increased resolution for small objects
        batch=2,             # Reduced to fit VRAM at higher imgsz
        device=device,
        amp=False,           # IMPORTANT: Disable for RT-DETR stability
        patience=20,         # Early stopping (ignored in 2-epoch test)
        save=True,
        save_period=10,
        project="runs/rtdetr",
        name=name,
        pretrained=True,
        optimizer='AdamW',
        lr0=0.0001,          # Lower LR for fine-tuning
        lrf=0.01,
        warmup_epochs=3,
        cos_lr=True,
        mixup=0.0,           # Disable mixup (destructive for small objects)
        copy_paste=0.0,      # Disable copy-paste for safety with small objs
        workers=4 if args.test else 8,
        exist_ok=True,
    )

    print('\nDone!')
    print(f'Results saved to: {results.save_dir}')
    if not args.test:
        print(f'Best model: {results.save_dir}/weights/best.pt')


if __name__ == '__main__':
    main()