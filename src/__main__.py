import sys
from model.drop_original_multi_channel_parallel_crnn import DropOriginalMultiChannelParallelCRNN
from model.multi_channel_parallel_crnn import MultiChannelParallelCRNN
from model.multi_input_parallel_crnn import MultiInputParallelCRNN



def main():
    model = DropOriginalMultiChannelParallelCRNN('/datashare_small/osterburg_data/processed/fma/', sys.argv[1])
    model.train()
    model.evaluate()


if __name__ == "__main__":
    main()
