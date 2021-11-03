from .dataset_loader import AbstractDatasetLoader

class FmaMediumLoader(AbstractDatasetLoader):

    def __init__(self, base_path: str, pipeline_str: str) -> None:
        super().__init__()
        self.destination = self._os.path.join(base_path, 'processed', 'fma', f'fma_medium{pipeline_str}')
        if not self._os.path.exists(self.destination):
            self._os.makedirs(self.destination)

        self._PATH = self._os.path.join(base_path, 'raw' ,'fma')
        self._DATA_SET = self._os.path.join(self._PATH, 'fma_medium')
        self._META_DATA = self._os.path.join(self._PATH, 'fma_metadata', 'tracks.csv')

    
    def load(self):
        df = self._pd.read_csv(self._META_DATA, index_col=0, header=[0, 1])
        df = df[df[('set', 'subset')] == 'medium']
        df = df[[('track', 'genre_top')]]

        genres_dict = {}
        genres = df[("track", "genre_top")].dropna().unique()
        for label in genres:
            genres_dict.update({ label: len(genres_dict) })
        

        metadata = {
            'label_count': len(genres_dict),
            'labels': genres_dict
        }

        return self._pd.DataFrame(
            { 'file': self._os.path.join(self._DATA_SET, "{:06d}".format(index)[:3], "{:06d}".format(index) + ".mp3"), 'label': genres_dict[label] } for (index, label) in df.itertuples()
        ), metadata
