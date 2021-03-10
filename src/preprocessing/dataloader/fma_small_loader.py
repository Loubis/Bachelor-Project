from .dataset_loader import AbstractDatasetLoader

class FmaSmallLoader(AbstractDatasetLoader):

    def __init__(self, base_path: str):
        super().__init__()
        self.destination = self._os.path.join(base_path, 'processed' ,'fma_small')
        if not self._os.path.exists(self.destination):
            self._os.makedirs(self.destination)

        self._PATH = self._os.path.join(base_path, 'fma')
        self._DATA_SET = self._os.path.join(self._PATH, 'fma_small')
        self._META_DATA = self._os.path.join(self._PATH, 'fma_metadata', 'tracks.csv')

    
    def load(self):
        df = self._pd.read_csv(self._META_DATA, index_col=0, header=[0, 1])
        df = df[df[('set', 'subset')] == 'small']
        df = df[[('track', 'genre_top')]]

        genres_dict = {}
        genres = df[("track", "genre_top")].dropna().unique()
        for label in genres:
            genres_dict.update({label: len(genres_dict)})
        
        # Swap dict keys and values and save
        genres_dict_swapped = dict([(value, key) for key, value in genres_dict.items()]) 
        self._np.save(self.destination + '/encoded_labels.npy', self._np.array(list(genres_dict_swapped.items())))


        return self._pd.DataFrame(
            { 'file': self._os.path.join(self._DATA_SET, "{:06d}".format(index)[:3], "{:06d}".format(index) + ".mp3"), 'label': genres_dict[label] } for (index, label) in df.itertuples()
        )
