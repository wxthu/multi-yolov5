import torch
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.models import alexnet, vgg16, vgg13, vgg11, squeezenet1_0, squeezenet1_1
from models.common import DetectMultiBackend

import time
from tqdm import tqdm


match_pair=dict()
def to_cuda(tensor):
    new=tensor.cuda()
    match_pair[tensor]=tensor.data
    return new

def to_cpu(tensor):
    if tensor in match_pair:
        return match_pair[tensor]
    return tensor.cpu()

if __name__ == '__main__':
    yolov5x = DetectMultiBackend('yolov5x.pt', device=torch.device('cpu'))
    yolov5s = DetectMultiBackend('yolov5s.pt', device=torch.device('cpu'))
    rdm_input = torch.randn(1, 3, 384, 640)
    NUM = 10

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
    

    # for name, model in models.items():
    #     x = rdm_input.cuda()
    #     model.to('cuda')
    #     for i in tqdm(range(NUM)):
    #         with torch.no_grad():
    #             y = model(x)
    #     print(f'{name} finished inference and to unload ...')   
    #     time.sleep(5)
    #     model.to('cpu')
    #     torch.cuda.empty_cache()
    #     time.sleep(5)

    for name, model in models.items():
        x = rdm_input.cuda()
        model._apply(to_cuda)
        for i in tqdm(range(NUM)):
            with torch.no_grad():
                y = model(x)
        print(f'{name} finished inference and to unload ...')   
        time.sleep(5)
        model._apply(to_cpu)
        match_pair.clear()
        torch.cuda.empty_cache()
        time.sleep(5)


    
    