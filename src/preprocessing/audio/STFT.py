from tqdm import tqdm
import numpy as np
import librosa as lr


class LibrosaCPUSTFT():
    def convert(self, data):
        for index, file in enumerate(tqdm(data)):
            for audio_key, waveform in file[1].items():
                try:
                    data[index][1][audio_key] = self._stft_librosa(self._to_mono(waveform))
                except Exception as e:
                    print(e)
        return data


    def _to_mono(self, waveform):
        return lr.to_mono(
            waveform.reshape( 
                (waveform.shape[1], waveform.shape[0]) 
            )
        )


    def _stft_librosa(self, waveform):
        return lr.power_to_db(
            lr.feature.melspectrogram(
                waveform,
                sr=44100,
                n_fft=2048,
                hop_length=1024
            ),
            ref=np.max
        )


class TensorflowGPUSTFT():
    def convert(self, data):
        for index, file in enumerate(tqdm(data)):
            for audio_key, waveform in file[1].items():
                data[index][1][audio_key] = self._stft_backend(self._to_mono(waveform))
        return data


    def _to_mono(self, waveform):
        return lr.to_mono(
            waveform.reshape( 
                (waveform.shape[1], waveform.shape[0]) 
            )
        )
