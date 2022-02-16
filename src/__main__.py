import sys
from model.multi_channel_parallel_crnn import ParallelCRNN
from model.multi_input_parallel_crnn import ParallelCRNN


def main():
    model = ParallelCRNN('/data/osterburg_data/processed/', sys.argv[1])
    model.train()
    model.evaluate()


if __name__ == "__main__":
    main()
