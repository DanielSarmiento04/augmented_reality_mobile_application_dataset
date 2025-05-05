wandb login 

yolo task=detect mode=train model=yolo11n.pt data=/content/data.yml epochs=100 imgsz=640 plots=True device=0 batch=-1