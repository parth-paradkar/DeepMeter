from typing import List
import torch.nn as nn
from torchvision.transforms.functional import resize
from base.layer import Layer


class SingleLayerModel(nn.Module):
    def __init__(self, layer: Layer) -> None:
        super().__init__()
        self.layer = layer

    def __call(self, x, func, reshape_dim=None):
        if reshape_dim:
            x = resize(x, [reshape_dim[2], reshape_dim[3]])
            x = x.expand(reshape_dim)
            print(x.shape)
        x = func(x)
        return x

    def forward(self, x):
        x_ = x
        x_ = self.__call(x_, self.layer.layer_object, self.layer.reshape_dim)
        print(f"Ran {self.layer.layer_module_name}")

    

class SandboxModel(nn.Module):
    def __init__(self, layers:List[Layer], layerwise=False) -> None:
        super().__init__()
        self.layers = layers
        self.layerwise = layerwise
        if self.layers:
            self.active_layer = None
            self.__layer_iter = iter(self.layers)
        else:
            raise Exception("Layers not provided!")

    def __iter__(self):
        self.active_layer = self.layers[0]
        return self

    def __next__(self):
        curr = self.active_layer
        # import pdb; pdb.set_trace()
        if not curr:
            raise StopIteration
        self.active_layer = next(self.__layer_iter)
        return curr
    
    def __call(self, x, func, reshape_dim=None):
        if reshape_dim:
            x = resize(x, [reshape_dim[2], reshape_dim[3]])
            x = x.expand(reshape_dim)
            print(x.shape)
        x = func(x)
        return x


    def set_active_layer(self, layer_index):
        self.active_layer = self.layers[layer_index]

    def forward(self, x):
        x_ = x
        if not self.layerwise:
            for layer in self.layers:
                x_ = self.__call(x_, layer.layer_object, layer.reshape_dim)
                print(f"Ran {layer.layer_module_name}")
        else:
            x_ = self.__call(x_, self.active_layer.layer_object, self.active_layer.reshape_dim)
            print(f"Ran {self.active_layer.layer_module_name}")