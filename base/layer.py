import torch.nn as nn

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