from preprocessing.audio.audio_processor import AbstractAudioPreprocessor


class GainChangerProcessor(AbstractAudioPreprocessor):
    def __init__(self, gain):
        self._gain = self._flaot_to_db(gain)

    def process(self, data):
        ...

    def _db_to_float(self, value):
        return 10 ** (value / 20)


class GainChanger10dBUpProcessor(GainChangerProcessor):
    def __init__(self):
        super().__init__(10)


class GainChanger10dBDownProcessor(GainChangerProcessor):
    def __init__(self):
        super().__init__(-10)
