import sys
from model.parallel_crnn import ParallelCRNN

def main():
    model = ParallelCRNN('/datashare_small/osterburg_data/processed/', sys.argv[1])
    model.train()
    model.evaluate()


if __name__ == "__main__":
    main()
