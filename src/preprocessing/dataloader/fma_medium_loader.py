from .dataset_loader import AbstractDatasetLoader

# List of songs empty or shorter songs
# Source: https://github.com/mdeff/fma/wiki#excerpts-shorter-than-30s-and-erroneous-audio-length-metadata (last visit 07.02.2022)
excluded_shorter_tracks = [
    "1486",
    "5574",
    "65753",
    "80391",
    "98558",
    "98559",
    "98560",
    "98565",
    "98566",
    "98567",
    "98568",
    "98569",
    "98571",
    "99134",
    "105247",
    "108924",
    "108925",
    "126981",
    "127336",
    "133297",
    "143992",
]

class FmaMediumLoader(AbstractDatasetLoader):

    def __init__(self, base_path: str, pipeline_str: str) -> None:
        super().__init__()
        self.destination = self._os.path.join(base_path, "processed", "fma", f"fma_medium{pipeline_str}")
        if not self._os.path.exists(self.destination):
            self._os.makedirs(self.destination)
        self._PATH = self._os.path.join(base_path, "raw" ,"fma")
        self._DATA_SET = self._os.path.join(self._PATH, "fma_medium")
        self._META_DATA = self._os.path.join(self._PATH, "fma_metadata", "tracks.csv")


    def load(self):
        df = self._pd.read_csv(self._META_DATA, index_col=0, header=[0, 1])
        # Filter small + medium data sub set
        df = df[df[("set", "subset")].isin(["small","medium"])]
        # Filter malformed songs
        df[~df.index.isin(excluded_shorter_tracks)]

        genres_dict = {}
        genres = df[("track", "genre_top")].dropna().unique()
        for label in genres:
            genres_dict.update({ label: len(genres_dict) })

        metadata = {
            "label_count": len(genres_dict),
            "labels": genres_dict
        }

        split_sets = {
            "training":   df[df[("set", "split")] == "training"],
            "validation": df[df[("set", "split")] == "validation"],
            "test":       df[df[("set", "split")] == "test"],
        }

        # Filter uneeded columns
        for key in split_sets:
            split_sets[key] = split_sets[key][[("track", "genre_top")]]

        return_sets = {}
        for key in split_sets:
            return_sets[key] = self._pd.DataFrame(
                { 
                    "file": self._os.path.join(self._DATA_SET, "{:06d}".format(index)[:3], "{:06d}".format(index) + ".mp3"),
                    "label": genres_dict[label]
                } 
                for (index, label) in split_sets[key].itertuples()
            )

        return return_sets, metadata
