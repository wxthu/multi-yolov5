import torch
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.models import alexnet, vgg16, vgg13, vgg11, squeezenet1_0, squeezenet1_1
from models.common import DetectMultiBackend

import time
from tqdm import tqdm

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def time_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()



if __name__ == '__main__':
    yolov5x = DetectMultiBackend('yolov5x.pt', device=torch.device('cpu'))
    yolov5s = DetectMultiBackend('yolov5s.pt', device=torch.device('cpu'))
    warm = resnet50().eval().cuda()
    rdm_input = torch.randn(1, 3, 384, 640).to('cuda')
    t1 = time_sync()
    for _ in range(50):
        y = warm(rdm_input)
    duration = time_sync() - t1
    print(f'*** warm up finished, duration : {1000 * duration / 2000}ms ***')
    NUM = 500
    loading = [0 for _ in range(NUM)]
    inference = [0 for _ in range(NUM)]
    unloading = [0 for _ in range(NUM)]

    models = {}
    models.update({'sqt10': squeezenet1_0().eval()})
    models.update({'sqt11': squeezenet1_1().eval()})
    models.update({'rnt18': resnet18().eval()})
    models.update({'rnt34': resnet34().eval()})
    models.update({'rnt50': resnet50().eval()})
    models.update({'rnt101': resnet101().eval()})
    models.update({'rnt152': resnet152().eval()})
    models.update({'alexnet': alexnet().eval()})
    models.update({'vgg11': vgg11().eval()})
    models.update({'vgg13': vgg13().eval()})
    models.update({'vgg16': vgg16().eval()})
    models.update({'yolov5x': yolov5x.eval()})
    models.update({'yolov5s': yolov5s.eval()})

    for name, model in models.items():
        for i in tqdm(range(NUM)):
            t1 = time_sync()
            model.to('cuda')
            t2 = time_sync()
            y = model(rdm_input)
            t3 = time_sync()
            model.to('cpu')
            t4 = time_sync()
            loading[i] = 1000 * (t2 - t1)
            inference[i] = 1000 * (t3 - t2)
            unloading[i] = 1000 * (t4 - t3)
            # torch.cuda.empty_cache()
            # print(f'single round: loading {loading[i]}ms, infer {inference[i]}ms, unloading {unloading[i]}ms')

        print(f'{name} loading avg {sum(loading) / NUM}ms, infer avg {sum(inference) / NUM}ms, unloading avg : {sum(unloading) / NUM}ms')
        

    
    