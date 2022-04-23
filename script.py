#!/opt/intel/oneapi/intelpython/latest/bin/python

from argparse import ArgumentParser
from processors.output_processor import OutputParser, ParseEnergy

parser = ArgumentParser("File input for model configuration")
parser.add_argument("--file", dest="filename", help="JSON file with model configuration")
parser.add_argument("--layer", dest="layer", help="Selected layer")
parser.add_argument("--batch", dest="batch", help="Batch size")
parser.add_argument("--runs", dest="runs", help="Number of runs")


def get_data(file, layer, batch, num_runs):
    to_run = ["energy"]
    
    output_parser = OutputParser()
    parser = ParseEnergy(file, layer, batch)
    
    headers = {parser.run:parser.header_row}
    data = {r:[] for r in to_run}
        
    for iter in range(num_runs):
        iter_data = output_parser.get_layer_iter_data(iter, parser.command, parser.metrics, parser.parse_line)
        parser.db = output_parser.insert_db(iter_data, parser.db)
    
    average_data = output_parser.get_average_data(parser.db)
    data[parser.run].append(average_data)
    parser.flush_db()
        

    output_parser.save_data(data, headers, "result.csv")

if __name__ == "__main__":
    args = parser.parse_args()
    
    file = args.filename
    layer = args.layer
    batch = int(args.batch)
    runs = int(args.runs)

    get_data(file, layer, batch, runs)