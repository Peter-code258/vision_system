#!/usr/bin/env python3
# vision_system/src/training/train.py

import argparse, os, time, shutil
from ultralytics import YOLO

def find_latest_run(runs_dir="runs/train"):
    if not os.path.exists(runs_dir):
        return None
    exps = [os.path.join(runs_dir,d) for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir,d))]
    if not exps:
        return None
    latest = sorted(exps, key=os.path.getmtime)[-1]
    return latest

def copy_best_to_export(latest_run, out_path="vision_system/models/exported/best.pt"):
    best_path = os.path.join(latest_run, "weights", "best.pt")
    if os.path.exists(best_path):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        shutil.copy(best_path, out_path)
        print("[INFO] best.pt copied to", out_path)
        return out_path
    else:
        print("[WARN] best.pt not found in", latest_run)
        return None

def train_yolo(data_yaml, pretrained='yolov8m.pt', epochs=50, imgsz=640, batch=16,
               device='auto', project="runs/train", name="exp", resume=False, wandb_project=None, save_export=True):
    # optional wandb integration
    use_wandb = False
    if wandb_project:
        try:
            import wandb
            wandb.init(project=wandb_project, name=name, config={"epochs":epochs, "batch":batch, "imgsz":imgsz, "data":data_yaml})
            use_wandb = True
            print("[INFO] wandb enabled, project:", wandb_project)
        except Exception as e:
            print("[WARN] wandb import/init failed:", e)

    # Load model (pretrained or resume)
    model = YOLO(pretrained)

    train_kwargs = dict(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
    )
    if resume:
        train_kwargs['resume'] = True

    print("[INFO] Training with args:", train_kwargs)
    model.train(**train_kwargs)

    # after training, find latest run and copy best.pt
    latest = find_latest_run(project)
    if latest and save_export:
        copy_best_to_export(latest, out_path="vision_system/models/exported/best.pt")

    if use_wandb:
        try:
            import wandb
            wandb.finish()
        except:
            pass

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="vision_system/configs/dataset.yaml")
    p.add_argument("--pretrained", default="yolov8m.pt")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="auto")
    p.add_argument("--project", default="runs/train")
    p.add_argument("--name", default=None)
    p.add_argument("--resume", action="store_true", help="resume from last run")
    p.add_argument("--wandb_project", default=None, help="If set, init wandb logging")
    p.add_argument("--save_export", action="store_true", help="Copy best.pt to models/exported/")
    args = p.parse_args()

    # default name with timestamp if not provided
    if args.name is None:
        args.name = f"exp_{int(time.time())}"

    train_yolo(
        data_yaml=args.data,
        pretrained=args.pretrained,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        resume=args.resume,
        wandb_project=args.wandb_project,
        save_export=args.save_export
    )
