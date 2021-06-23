from .dataset_loader import AbstractDatasetLoader


class GtzanLoader(AbstractDatasetLoader):

    def __init__(self, base_path: str, pipeline_str: str) -> None:
        super().__init__()
        self.destination = self._os.path.join(base_path, 'processed', f'gtzan{pipeline_str}')
        if not self._os.path.exists(self.destination):
            self._os.makedirs(self.destination)
        self._PATH = self._os.path.join(base_path, 'raw', 'gtzan')
        self._DATA_SET = self._os.path.join(self._PATH, 'genres_original')
        self._META_DATA_30_SEC = self._os.path.join(self._PATH, 'features_30_sec.csv')

    
    def load(self):
        df = self._pd.read_csv(self._META_DATA_30_SEC)
        df = df[['filename', 'label']]

        for (index, filename_col, label_col) in df.itertuples():
            df.loc[[index],['filename']] = self._os.path.join(self._DATA_SET, label_col, filename_col)
        
        genres_dict = {}
        genres = df['label'].unique()
        for label in genres:
            genres_dict.update({label: len(genres_dict)})

        df['label'] = df['label'].apply(lambda label: genres_dict[label])
        
        metadata = {
            'label_count': len(genres_dict),
            'labels': genres_dict
        }

        return df.rename(columns={ 'filename': 'file' }), metadata
