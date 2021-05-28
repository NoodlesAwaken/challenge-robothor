import time

import cv2
import torch
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import non_max_suppression, scale_coords, xyxy2xywh
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, time_synchronized

def load_model():
    device = select_device()
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load("weights/best.pt", map_location=device)  # load FP32 model
    if half:
        model.half()  # to FP16

    model(torch.zeros(1, 3, 640, 640).to(device).type_as(next(model.parameters())))  # run once
    
    return model

def detect(source, model):
    t0 = time.time()
    view_img, imgsz = False, 640 
    # view_img, imgsz = True, 640 
    device = select_device()

    # Initialize
    model = model
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    stride = int(model.stride.max())  # model stride
    # dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Run inference
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once


    
    img = source.transpose(2, 0, 1)
    img = torch.from_numpy(img).to(device)
    img = img.half() #if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    # pred = model(img, augment=opt.augment)[0]
    pred = model(img, False)[0]

    # Apply NMS
    # pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
    pred = non_max_suppression(pred, 0.25, 0.45, agnostic=False)
    t2 = time_synchronized()

    result = ''
    # Process detections
    for i, det in enumerate(pred):  # detections per image
        imshape = (480, 640, 3)
        # p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
        im0 = source.copy()

        # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        gn = torch.tensor(imshape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            # det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], imshape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                # line = (cls, *xywh, conf) if False else (cls, *xywh)  # label format
                line = (cls, *xywh)  # label format
                result += ('%g ' * len(line)).rstrip() % line + f' {conf:.2f}' + '\n'

                if view_img:  # Add bbox to image
                    c = int(cls)  # integer class
                    # label = None if False else (names[c] if False else f'{names[c]} {conf:.2f}')
                    label = f'{names[c]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=2)

        # Stream results
        if view_img:
            # cv2.imshow(str(p), im0)
            cv2.imshow("image", im0)
            # cv2.imwrite("result.png", im0)
            cv2.waitKey() 


    # print(f'Done. ({time.time() - t0:.3f}s)')
    # print(result)
    return result
    

# model = load_model()
# with torch.no_grad():
#     detect(source="image.png", model=model)
# with torch.no_grad():
#     detect(source="image.png", model=model)
# with torch.no_grad():
#     detect(source="image.png", model=model)
# with torch.no_grad():
#     detect(source="image.png", model=model)