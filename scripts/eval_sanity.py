from ultralytics import RTDETR
import sys

def main():
    model = RTDETR('runs/detect/runs/rtdetr/cpu_sanity_test/weights/best.pt')
    metrics = model.val(data='configs/seadronessee_mini.yaml', device='cpu', split='val')
    print("\n--- PER-CLASS METRICS ---")
    for i, name in enumerate(metrics.names.values()):
        if i in metrics.ap_class_index:
            idx = list(metrics.ap_class_index).index(i)
            print(f"Class '{name}' (id {i}): mAP50 = {metrics.box.ap50[idx]:.4f}")
        else:
            print(f"Class '{name}' (id {i}): mAP50 = 0.0000")

if __name__ == '__main__':
    main()