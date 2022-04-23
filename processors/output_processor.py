import os
import csv

class OutputFileInterface:
    def __init__(self) -> None:
        pass

    def get_command_output(self, command):
        print(os.getcwd())
        output = os.popen(command)
        return output.readlines()

    def __format_data(self, data:dict, headers:dict):
        # import pdb; pdb.set_trace()
        full_header = []
        for value in headers.values():
            full_header += value
        accumulated_data = list(data.values())[0]
        for value in list(data.values())[1:]:
            assert len(value) == len(accumulated_data)
            for i in range(len(value)):
                accumulated_data[i] += value[i]
        for layer in accumulated_data:
            assert len(full_header) == len(layer)
        return full_header, accumulated_data
    
    def save_data(self, data:dict, headers:dict, csv_file_name):
        full_header, full_data = self.__format_data(data, headers)
        with open(csv_file_name, "w") as f:
            writer = csv.writer(f)
            writer.writerow(full_header)
            writer.writerows(full_data)
            print(f"Data written to {csv_file_name}")


class OutputParser(OutputFileInterface):
    
    def __init__(self):
        super()

    def parse_output(self, output, parse_func, metrics) -> list:
        iter_data = list()
        for metric in metrics:
            found = False
            for line in output:
                if metric in line:
                    found = True
                    iter_data += parse_func(line)
                    break
            if not found:
                print(f"{metric} not found!")
                iter_data.append([0, 0])
            print(f"Layer: Data {iter_data}")
        return iter_data


    def get_layer_iter_data(self, iter, command, metrics, parse_func):
        """After layer has been activated in file"""
        output = self.get_command_output(command)
        iter_data = self.parse_output(output, parse_func, metrics)
        return iter_data
            

    def insert_db(self, iter_data, db):
        assert len(iter_data) == len(db)
        for i, data in enumerate(iter_data):
            for j, value in enumerate(data):
                db[i][j] += float(value)
        return db


    def get_average_data(self, db):
        average_layer_data = list()
        for data in db:
            if data[1]:
                average_layer_data.append(data[0]/data[1])
            else:
                average_layer_data.append(0)
        return average_layer_data   

class ParseFlops():
    def __init__(self) -> None:
        self.metrics = ["SP GFLOPS", "DP GFLOPS", "x87 GFLOPS", "Elapsed Time", "Average CPU Frequency"]
        self.command = "vtune -collect performance-snapshot ./{}"
        self.run = "flops"
        self.db = [[0, 0] for _ in range(len(self.metrics))]

    def parse_line(self, line):
        data = []
        value = line.split(": ")[1]
        data.append([value[:5].strip(), 1])
        return data

    @property
    def header_row(self):
        return self.metrics

    def flush_db(self):
        self.db = [[0, 0] for _ in range(len(self.metrics))]

class ParseEnergy():
    def __init__(self, file, layer, batch) -> None:
        self.metrics = ["CPU/Package_0", "CPU/Package_1", "DRAM/DRAM_0", "DRAM/DRAM_1"]
        self.command = f"sudo $SOCWATCH_DIR/socwatch -f power -p ./controller/run.py --file ./{file} --layer {layer} --batch {batch} && cat SoCWatchOutput.csv"
        self.run = "energy"
        self.db = [[0, 0] for _ in range(2 * len(self.metrics))]

    def parse_line(self, line):
        data = []
        values = line.split(",")
        data.append([values[2].strip(), 1])
        data.append([values[3].strip(), 1])
        return data

    @property
    def header_row(self):
        header_row = []
        for metric in self.metrics:
            header_row.append(metric + "_power")
            header_row.append(metric + "_energy")
        return header_row

    def flush_db(self):
        self.db = [[0, 0] for _ in range(2 * len(self.metrics))]


def get_parser(to_run):
    if to_run == "flops":
        return ParseFlops()
    elif to_run == "energy":
        return ParseEnergy()

