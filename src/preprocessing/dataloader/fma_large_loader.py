from .dataset_loader import AbstractDatasetLoader

class FmaLargeLoader(AbstractDatasetLoader):

    def __init__(self):
        self._PATH = '~/Datasets/fma/'
        self._DATA_SET = 'fma_small/'
        self._META_DATA = 'fma_metadata/'
    

    def load(self):
        df = self._pd.read_csv(self._META_DATA, index_col=0, header=[0, 1])
        df = df[[('track', 'genre_top')]]

        self._pd.DataFrame(
            { 'file': self._os.path.join(self._DATA_SET, "{:06d}".format(index)[:3], "{:06d}".format(index) + ".mp3"), 'label': label } for (index, label) in df.itertuples()
        )
