
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

if __name__ == '__main__':
    # model.load('yolo11n.pt') # 加载预训练权重,改进或者做对比实验时候不建议打开，因为用预训练模型整体精度没有很明显的提升
    model = YOLO(r'E:\BaiduNetdiskDownload\githubMRFNet\ultralytics\cfg\models\11\MRFNet.yaml')
    model.train(data=r'E:\BaiduNetdiskDownload\change-ASFF-Rep\ultralytics\cfg\datasets\VisDrone.yaml',
                imgsz=640,
                epochs=200,
                batch=8,
                workers=0,
                device='0',
                patience=30,
                optimizer='SGD',
                close_mosaic=10,
                resume=False,
                project='runs/train',
                name='exp',
                single_cls=False,
                cache=False,
                )



