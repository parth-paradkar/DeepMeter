#!/opt/intel/oneapi/intelpython/latest/bin/python

# from base.layer import Layer

# from processors.input_parser import InputInterface
from argparse import ArgumentParser
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Resize, Compose
from torchvision.transforms.functional import resize

import torch.nn as nn
import json


parser = ArgumentParser("File input for model configuration")
parser.add_argument("--file", dest="filename")
parser.add_argument("--layer", dest="layer")
parser.add_argument("--batch", dest="batch")

args = parser.parse_args()
filename = args.filename
layer_index = int(args.layer)
batch_size = int(args.batch)



class Layer:
    def __init__(self, kwargs) -> None:
        self.params = kwargs

        self.type_ = kwargs.get("type")
        if not self.type_:
            raise Exception("'type' key missing. Please specify type of action")

        if self.type_ == "layer":
            torch_module = kwargs.get("torch_module")
            if torch_module:
                self.layer_module_name = torch_module.split(".")[-1]
            else:
                raise Exception("Required parameter: torch_module")

            self.activation_functions = ["relu", "tanh", "selu", "leakyrelu"]
            self.layer_module = getattr(nn, self.layer_module_name)
            self.layer_object = self.__get_layer()
            self.reshape_dim = kwargs.get("reshape_dim")
        elif self.type_ == "reshape":
            self.layer_object = self.__buffer()
            self.reshape_dim = kwargs.get("reshape_dim")

    def __buffer(x):
        return x

    def __get_layer(self):
        try:
            if self.layer_module_name.lower() in self.activation_functions:
                return self.layer_module()
            elif "linear" in self.layer_module_name.lower():
                return self.layer_module(in_features=self.params['in_features'], out_features=self.params['out_features'])
            elif "pool" in self.layer_module_name.lower():
                return self.layer_module(kernel_size=self.params['kernel_size'], stride=self.params['stride'])
            elif "conv" in self.layer_module_name.lower():
                return self.layer_module(
                    in_channels=self.params['in_channels'], 
                    out_channels=self.params['out_channels'], 
                    kernel_size=self.params['kernel_size'], 
                    stride=self.params['stride'],
                    padding=self.params['padding']
                )
        except KeyError as e:
            raise Exception(f"Missing parameter: {e.args[0]}")
        except:
            raise 

    def __str__(self) -> str:
        return str(self.layer_object)

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


class InputInterface:
    def __init__(self, filepath) -> None:
        with open(filepath, "r") as f:
            self.file_content = f.read()
        self.parsed_input = json.loads(self.file_content)
        self.tup_attrs = ["kernel_size", "padding", "stride"]

    def __get_arg_tuple(self, value: str) -> tuple:
        # split_vals = value[1:-1].split(",")
        # assert len(split_vals) == 2
        # return (val.strip() for val in split_vals)
        assert isinstance(value, list)
        return tuple(value)
    
    def __get_layer_from_dict(self, layer_info: dict) -> Layer:
        for key in layer_info.keys():
            if key in self.tup_attrs:
                layer_info[key] = self.__get_arg_tuple(layer_info[key])
        return Layer(layer_info)


    def get_layers(self):
        return [self.__get_layer_from_dict(layer_info) for layer_info in self.parsed_input]

input_interface = InputInterface(filename)
layers = input_interface.get_layers()

current_layer = layers[layer_index]

# model = SandboxModel(layers, layerwise=True)
model = SingleLayerModel(current_layer)

dataset = MNIST(
    root="./data", 
    download=True, 
    transform=Compose(
        [Resize(32), ToTensor()]
    )
)

model.eval()

for index in range(batch_size):
    output = model(dataset[index][0].unsqueeze(0))
