import spleeter
from spleeter.separator import Separator
from spleeter.audio.adapter import AudioAdapter

import librosa
import soundfile as sf


def combine_tracks(vocal_waveform, accom_waveform, sample_rate=22050, hop_length=1024):
    separator = Separator('spleeter:2stems', stft_backend=spleeter.audio.STFTBackend.LIBROSA)

    vocals = separator.separate(vocal_waveform)['vocals'].mean(1)
    accom = separator.separate(accom_waveform)['accompaniment'].mean(1)

    tempo_vocals, beat_samples_vocals = librosa.beat.beat_track(y=vocals, sr=sample_rate, hop_length=hop_length, units='samples')
    tempo_accom, beat_samples_accom = librosa.beat.beat_track(y=accom, sr=sample_rate, hop_length=hop_length, units='samples')

    def calc_stretch(beat_times_from, beat_times_to):
        time_per_beat_from = (beat_times_from[1:] - beat_times_from[:-1]).mean()
        time_per_beat_to = (beat_times_to[1:] - beat_times_to[:-1]).mean()
        return time_per_beat_to / time_per_beat_from

    mult = calc_stretch(beat_samples_vocals, beat_samples_accom)
    accom_spedup = librosa.effects.time_stretch(accom, mult)

    tempo_accom_spedup, beat_samples_accom_spedup = librosa.beat.beat_track(y=accom_spedup, sr=sample_rate, hop_length=hop_length, start_bpm=tempo_vocals, units='samples')

    shift = beat_samples_vocals[1] - beat_samples_accom_spedup[0]
    if shift >= 0:
        vocals_spedup_shifted = vocals[shift:]
        accom_spedup_shifted = accom_spedup
    else:
        accom_spedup_shifted = accom_spedup[-shift:]
        vocals_spedup_shifted = vocals

    common_length = min(accom_spedup_shifted.shape[0], vocals_spedup_shifted.shape[0])

    return accom_spedup_shifted[:common_length] + vocals_spedup_shifted[:common_length]


if __name__ == '__main__':
    sample_rate = 22050
    audio_loader = AudioAdapter.default()
    kanye_waveform, _ = audio_loader.load('data/kanye.mp3', sample_rate=sample_rate)
    eminem_waveform, _ = audio_loader.load('data/eminem.mp3', sample_rate=sample_rate)
    
    combined_waveform = combine_tracks(kanye_waveform, eminem_waveform, sample_rate)
    sf.write('combined.wav', combined_waveform, sample_rate)
    