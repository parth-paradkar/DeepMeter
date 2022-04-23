from base.layer import Layer

"""
Input format example
[
    {
        "torch_module": "nn.Linear",
        "in_features": 10
        "out_features": 20
    },

]
"""
import json

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