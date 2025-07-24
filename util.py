import os
import pickle
import numpy as np
import librosa
import math
from typing import List, Tuple, Dict
from mutagen import File

def read_metadata(audio_fp):
    audio = File(audio_fp, easy=True)
    if audio is None:
        return {}

    return audio.tags if audio.tags else {}
def set_bpm(audio_fp, min_tempo = 100, max_tempo = 200, maxstep = 12, bpm_method = 'DDCL'):
    if bpm_method == 'DDCL':
        beat_dict = extract_BPM(audio_fp, min_tempo, max_tempo)
        bpm = beat_dict['BPM']
        beats = beat_dict['beats']

        beat_intervals = list(beat_dict['beat_intervals'])
        beat_intervals = [inty*100 for inty in beat_intervals]
        bpm_intervals = [60/gap for gap in beat_intervals]

        avg_point = 0
        bpm_shift_times = []
        first_line = True
        for i in range(1,len(bpm_intervals)):
            cur_bpm = bpm_intervals[i]
            avg_bpm = 60/np.mean(beat_intervals[avg_point:i])
            if abs(cur_bpm-avg_bpm)>25 or i == len(bpm_intervals)-1:
                if first_line:
                    bpm_str = '0.0={}'.format(avg_bpm)
                    first_line=False
                else:
                    bpm_str += '\n'+',{}={}'.format(avg_point, avg_bpm)
                bpm_shift_times.append((avg_point, avg_bpm))
                beats[avg_point:i] = np.linspace(beats[avg_point], beats[i], endpoint = False, num = i-avg_point)
                avg_point = i
                
        if first_line:
            bpm_str = '0.0={}'.format(avg_bpm)

        print('bpm shifts: {}'.format(bpm_shift_times))

        offset = beats[0]

        song_length = beats[-1]/.6

        subdiv_beats = []
        for j in range(len(beats)-1):
            for i in range(maxstep):
                subdiv_beats.append(((beats[j]*(maxstep-i))/maxstep)+(beats[j+1]*i)/maxstep)
        subdiv_beats.append(beats[-1])

        print('offset: {}'.format(offset))

        return beats, np.array(subdiv_beats), bpm_shift_times, offset, bpm_str, song_length, bpm
    elif bpm_method == 'AV':
        beat_dict, _ = arrow_vortex_get_bpm(audio_fp, bpm_range = (min_tempo, max_tempo))
        bpm = beat_dict[0]['bpm']
        offset = beat_dict[0]['offset']
        beats = beat_dict[0]['beats']
        bpm_shift_times = [(0,bpm)]
        subdiv_beats = []
        for j in range(len(beats)-1):
            for i in range(maxstep):
                subdiv_beats.append(((beats[j]*(maxstep-i))/maxstep)+(beats[j+1]*i)/maxstep)
        subdiv_beats.append(beats[-1])
        bpm_str = '0.0={}'.format(bpm)
        song_length = beats[-1]/60
        return beats, np.array(subdiv_beats), bpm_shift_times, offset, bpm_str, song_length, bpm
    elif bpm_method == 'SMEdit':
        beat_dict = smedit_analyze_audio(audio_fp, bpm_range = (min_tempo, max_tempo))
        bpm = beat_dict['bpm_results'][0]['bpm']
        offset = beat_dict['offset_results'][0]['offset']
        beats = beat_dict['beat_times'][bpm]
        bpm_shift_times = [(0,bpm)]
        subdiv_beats = []
        for j in range(len(beats)-1):
            for i in range(maxstep):
                subdiv_beats.append(((beats[j]*(maxstep-i))/maxstep)+(beats[j+1]*i)/maxstep)
        subdiv_beats.append(beats[-1])
        bpm_str = '0.0={}'.format(bpm)
        song_length = beats[-1]/60
        return beats, np.array(subdiv_beats), bpm_shift_times, offset, bpm_str, song_length, bpm

def weighted_median(data):
    sorted_data = np.sort(data)
    n = len(sorted_data)
    if n % 2 == 1:
        return sorted_data[n // 2]
    else:
        return (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2.0

def hamming_window(w):
    return [0.54 - 0.46 * math.cos(2 * math.pi * n / (w - 1)) for n in range(w)]

def arrow_vortex_get_bpm(
    audio_file,
    window_size=1024,
    hop_size=256,
    bpm_range=(89, 205),
    silence_threshold=-70,
    threshold_weight=0.1,
    threshold_window_size=7
):
    # 1. Load audio
    audio, sr = librosa.load(audio_file, sr=None, mono=True)
    
    # 2. Compute spectral flux (onset detection function)
    #    We'll compute magnitude spectrogram and then differences
    S = np.abs(librosa.stft(audio, n_fft=window_size, hop_length=hop_size, window='hann'))
    flux = np.sqrt(np.sum(np.diff(S, axis=1).clip(min=0)**2, axis=0))
    
    # Pad flux so that len is consistent with frames produced in Essentia version
    detection_function = np.concatenate(([0], flux))
    
    # 3. Adaptive thresholding
    thresholds = np.zeros_like(detection_function)
    L = threshold_window_size
    for n in range(len(detection_function)):
        start = max(0, n - L + 1)
        end   = n + 1
        window = detection_function[start:end]
        thresholds[n] = weighted_median(window) + threshold_weight * np.mean(window)
    
    is_peak = (
        (detection_function[1:-1] > thresholds[1:-1]) &
        (detection_function[1:-1] > detection_function[:-2]) &
        (detection_function[1:-1] > detection_function[2:])
    )
    onsets = np.where(is_peak)[0] + 1
    
    # 4. Silence filtering on onsets
    filtered = []
    for o in onsets:
        start_sample = o * hop_size
        end_sample   = min(start_sample + window_size, len(audio))
        frame = audio[start_sample:end_sample]
        if len(frame) == 0:
            continue
        if len(frame) < window_size:
            frame = np.pad(frame, (0, window_size - len(frame)))
        # compute mean energy in dB
        stft_f = librosa.stft(frame, n_fft=window_size, hop_length=hop_size, window='hann')
        me = np.mean(np.sum(np.abs(stft_f)**2, axis=0))
        me_db = 10 * np.log10(max(me, 1e-10))
        if me_db > silence_threshold:
            filtered.append(o)
    onsets = np.array(filtered)
    onset_times = onsets * hop_size / sr
    last_onset_time = onset_times[-1] if len(onset_times) else 0.0
    
    # 5. BPM detection via interval histogram
    frame_rate = sr / hop_size
    i_min = int(frame_rate * 60 / bpm_range[1])
    i_max = int(frame_rate * 60 / bpm_range[0])
    test_intervals = np.arange(i_min, i_max+1, 10)
    
    fitness_scores = {}
    for interval in test_intervals:
        hist = np.zeros(interval, dtype=int)
        for o in onsets:
            hist[o % interval] += 1
        
        # sliding Hamming based evidence
        hsize = min(interval // 4, 10)
        win = np.hamming(hsize)
        evidence = np.zeros(interval)
        for p in range(interval):
            for n in range(hsize):
                idx = (p - hsize//2 + n) % interval
                evidence[p] += win[n] * hist[idx]
        
        confidence = np.zeros(interval)
        for p in range(interval):
            confidence[p] = evidence[p] + 0.5 * evidence[(p + interval//2) % interval]
        
        fitness_scores[interval] = np.max(confidence)
    
    if not fitness_scores:
        raise RuntimeError("No fitness scores; check input track or bpm_range")
    
    intervals = np.array(list(fitness_scores.keys()))
    scores = np.array(list(fitness_scores.values()))
    if len(intervals) >= 4:
        poly = np.poly1d(np.polyfit(intervals, scores, 3))
        for iv in intervals:
            fitness_scores[iv] -= poly(iv)
    
    max_fit = max(fitness_scores.values())
    thr = 0.4 * max_fit
    
    cands = [(frame_rate * 60 / iv, fitness_scores[iv], iv)
             for iv in fitness_scores if fitness_scores[iv] > thr]
    cands.sort(key=lambda x: x[1], reverse=True)
    
    # Deduplicate within 0.1 BPM
    final = []
    for bpm, fit, iv in cands:
        if not any(abs(bpm - fbpm) < 0.1 for fbpm, *_ in final):
            final.append((bpm, fit, iv))
        if len(final) >= 5:
            break
    
    # 6. Offset & beat timing
    results = []
    for bpm, fit, iv in final:
        interval = iv
        hist = np.zeros(interval, dtype=int)
        for o in onsets:
            hist[o % interval] += 1
        
        hsize = min(interval // 4, 10)
        win = np.hamming(hsize)
        conf = np.zeros(interval)
        for p in range(interval):
            e1 = sum(win[n] * hist[(p - hsize//2 + n) % interval] for n in range(hsize))
            e2 = sum(win[n] * hist[(p + interval//2 - hsize//2 + n) % interval] for n in range(hsize))
            conf[p] = e1 + 0.5 * e2
        
        p_max = np.argmax(conf)
        offset_s = p_max * hop_size / sr
        inter_s = interval * hop_size / sr
        
        # Beat vs offbeat energy slope
        slopes_beat = []
        slopes_off = []
        window_samples = int(0.05 * sr)
        t = offset_s
        while t < len(audio)/sr - 0.1:
            b = int(t * sr)
            ob = int((t + inter_s/2) * sr)
            if b-window_samples >= 0 and b+window_samples < len(audio):
                slopes_beat.append(
                    sum(np.abs(audio[b+1:b+window_samples+1])) -
                    sum(np.abs(audio[b-window_samples:b]))
                )
            if ob-window_samples >= 0 and ob+window_samples < len(audio):
                slopes_off.append(
                    sum(np.abs(audio[ob+1:ob+window_samples+1])) -
                    sum(np.abs(audio[ob-window_samples:ob]))
                )
            t += inter_s
        
        if slopes_beat and slopes_off and np.mean(slopes_off) > np.mean(slopes_beat):
            offset_s += inter_s/2
        
        # Generate beat times
        beat_step = 60.0 / bpm
        times = []
        b = offset_s
        while b > 0:
            b -= beat_step
        while b < last_onset_time + beat_step:
            if b >= 0:
                times.append(b)
            b += beat_step
        
        results.append({
            'bpm': bpm,
            'offset': offset_s,
            'fitness': fit,
            'confidence': np.max(conf),
            'beats': times
        })
    
    best = results[0]
    print(f"Best bpm, offset = {best['bpm']:.2f}, {best['offset']:.3f}s")
    return results, onset_times

class SMEditAudioSyncDetector:
    def __init__(self, 
                 window_step: int = 512,
                 fft_size: int = 1024,
                 tempo_fft_size: int = 4096,
                 tempo_step: int = 2,
                 min_bpm: float = 125,
                 max_bpm: float = 250,
                 sample_rate: int = 44100):
        self.window_step = window_step
        self.fft_size = fft_size
        self.tempo_fft_size = tempo_fft_size
        self.tempo_step = tempo_step
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm
        self.sample_rate = sample_rate

        self.AVERAGE_WINDOW_RADIUS = 3
        self.TEMPOGRAM_SMOOTHING = 3
        self.TEMPOGRAM_OFFSET_THRESHOLD = 0.02
        self.TEMPOGRAM_GROUPING_WINDOW = 6
        self.OFFSET_LOOKAHEAD = 800

        self.weight_data = self.weight_data = [
            (20, 0.4006009013520281), (25, 0.4258037044922291), (31.5, 0.4536690484291709),
            (40, 0.4840856831659204), (50, 0.5142710208279764), (63, 0.5473453749315819),
            (80, 0.5841121495327103), (100, 0.6214074879602299), (125, 0.6601749463607856),
            (160, 0.7054673721340388), (200, 0.7489234225800412), (250, 0.7936507936507937),
            (315, 0.8406893652795292), (400, 0.889284126278346), (500, 0.9291521486643438),
            (630, 0.9675858732462506), (800, 0.9985022466300548), (1000, 0.9997500624843789),
            (1250, 0.9564801530368244), (1600, 0.9409550693954364), (2000, 1.0196278358399185),
            (2500, 1.0955902492467817), (3150, 1.1232799775344005), (4000, 1.0914051841746248),
            (5000, 0.9997500624843789), (6300, 0.8727907484180668), (8000, 0.7722007722007722),
            (10000, 0.7369196757553427), (12500, 0.7768498737618955), (16000, 0.7698229407236336),
            (20000, 0.4311738708634257), (22550, 0.2), (25000, 0)
        ]
        self.spectro_weights = None
        self.audio_data = None
        self.spectrogram = []
        self.spectogram_difference = []
        self.novelty_curve = []
        self.novelty_curve_isolated = []
        self.tempogram = []
        self.tempogram_groups = []

    def load_audio(self, audio_path: str) -> np.ndarray:
        self.audio_data, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        return self.audio_data

    def _calculate_spectro_weights(self):
        weights = np.zeros(self.fft_size // 2 + 1)
        for i in range(len(weights)):
            freq = i * self.sample_rate / self.fft_size
            for j, (wf, wv) in enumerate(self.weight_data):
                if wf > freq:
                    break
            else:
                j = len(self.weight_data)
            if j == 0 or j > len(self.weight_data)-1:
                weights[i] = 0
            else:
                lo_f, lo_w = self.weight_data[j-1]
                hi_f, hi_w = self.weight_data[j]
                lf, lh = np.log1p(lo_f), np.log1p(hi_f)
                f = np.log1p(freq)
                t = (f-lf)/(lh-lf) if lh!=lf else 0
                weights[i] = lo_w + t*(hi_w - lo_w)
        return weights

    def _render_block(self, block_num: int):
        start = block_num * self.window_step - self.fft_size//2
        end   = start + self.fft_size
        slice_ = np.zeros(self.fft_size, dtype=float)
        a_s = max(0, start); a_e = min(len(self.audio_data), end)
        s_s = a_s - start; s_e = s_s + (a_e - a_s)
        if a_e > a_s:
            slice_[s_s:s_e] = self.audio_data[a_s:a_e]
        mag = np.abs(librosa.stft(slice_, n_fft=self.fft_size, hop_length=self.fft_size+1, window='hann')[:,0])
        return np.log1p(mag)

    def _calc_difference(self, block_num, curr):
        prev = self.spectrogram[block_num-1] if block_num>0 else np.zeros_like(curr)
        return np.maximum(0, curr - prev) * self.spectro_weights

    def _calc_isolated_novelty(self, i):
        lo = max(0, i - self.AVERAGE_WINDOW_RADIUS)
        hi = min(len(self.novelty_curve), i + self.AVERAGE_WINDOW_RADIUS + 1)
        avg = np.mean(self.novelty_curve[lo:hi])
        while len(self.novelty_curve_isolated) <= i:
            self.novelty_curve_isolated.append(0)
        self.novelty_curve_isolated[i] = np.log1p(max(0, self.novelty_curve[i] - avg))

    def detect_onsets(self, threshold: float = 0.3) -> List[float]:
        if self.audio_data is None:
            raise ValueError
        self.spectro_weights = self._calculate_spectro_weights()
        total_blocks = int(np.ceil(len(self.audio_data)/self.window_step))
        self.spectrogram.clear()
        self.spectogram_difference.clear()
        self.novelty_curve.clear()
        self.novelty_curve_isolated.clear()

        for i in range(total_blocks):
            sp = self._render_block(i)
            self.spectrogram.append(sp)
            diff = self._calc_difference(i, sp)
            self.spectogram_difference.append(diff)
            self.novelty_curve.append(np.sum(diff))
            self._calc_isolated_novelty(i)

        peaks = []
        for i in range(1, len(self.novelty_curve_isolated)):
            if (self.novelty_curve_isolated[i] > threshold and
                self.novelty_curve_isolated[i] > self.novelty_curve_isolated[i-1]):
                peaks.append(i * self.window_step / self.sample_rate)
        return peaks

    def detect_tempo_and_offset(self) -> Tuple[List[Dict], List[Dict]]:
        if not self.novelty_curve_isolated:
            raise ValueError
        nov = np.array(self.novelty_curve_isolated)
        if nov.max() > 0:
            nov /= nov.max()

        max_tb = int(np.ceil(len(nov)/self.tempo_step))
        self.tempogram.clear()
        self.tempogram_groups.clear()
        for b in range(max_tb):
            start = b*self.tempo_step - self.tempo_fft_size//2
            end = start + self.tempo_fft_size
            slice_ = np.zeros(self.tempo_fft_size, dtype=float)
            ds = max(0, start); de = min(len(nov), end)
            ss = ds - start; se = ss + (de-ds)
            if de > ds:
                slice_[ss:se] = nov[ds:de]
            windowed = slice_ * np.hanning(self.tempo_fft_size)
            resp = np.log1p(np.abs(np.fft.rfft(windowed)))
            tempos = {}
            for i, v in enumerate(resp):
                tmp = (self.sample_rate * 60)/(self.window_step * self.tempo_fft_size) * i
                if tmp>self.max_bpm*4 or tmp<self.min_bpm/4:
                    continue
                while tmp>self.max_bpm:
                    tmp /= 2
                while tmp<self.min_bpm:
                    tmp *= 2
                bpm = np.round(tmp, 3)
                tempos[bpm] = tempos.get(bpm, 0) + v
            tlist = [{'bpm':k,'value':v} for k,v in tempos.items()]
            tlist.sort(key=lambda x: x['value'], reverse=True)
            self.tempogram.append(tlist[:10])
            groups = []
            for t in tlist[:10]:
                grp = next((g for g in groups if abs(g['center'] - t['bpm']) < self.TEMPOGRAM_GROUPING_WINDOW), None)
                if grp is None:
                    groups.append({'center': t['bpm'], 'groups':[t], 'avg': t['bpm']})
                else:
                    grp['groups'].append(t)
                    total = sum(g['value'] for g in grp['groups'])
                    grp['avg'] = sum(g['bpm']*g['value'] for g in grp['groups'])/total
            self.tempogram_groups.append(groups)

        return self._calculate_bpm_and_offset()
    
    def _calculate_bpm_and_offset(self) -> Tuple[List[Dict], List[Dict]]:
        """Calculate final BPM and offset results"""
        print("Calculating BPM and offset results...")
        
        # Find most consistent BPM
        bpm_counts = {}
        peak_scan_start = 0
        
        for i, groups in enumerate(self.tempogram_groups):
            candidates = [g for g in groups if g['groups'][0]['value'] >= self.TEMPOGRAM_OFFSET_THRESHOLD]
            if not candidates:
                continue
                
            peak_scan_start = i
            
            for group in candidates:
                # Smooth the BPM estimate
                total_blocks = 0
                bpm_total = 0
                
                for j in range(max(0, i - self.TEMPOGRAM_SMOOTHING), 
                              min(len(self.tempogram_groups), i + self.TEMPOGRAM_SMOOTHING + 1)):
                    closest_group = None
                    for other_group in self.tempogram_groups[j]:
                        if (abs(other_group['center'] - group['center']) < self.TEMPOGRAM_GROUPING_WINDOW):
                            closest_group = other_group
                            break
                    
                    if closest_group:
                        bpm_total += closest_group['avg']
                        total_blocks += 1
                
                if total_blocks > 0:
                    bpm = np.round(bpm_total / total_blocks)
                    bpm_counts[bpm] = bpm_counts.get(bpm, 0) + 1
        
        # Get top BPMs
        bpm_results = []
        if bpm_counts:
            sorted_bpms = sorted(bpm_counts.items(), key=lambda x: x[1], reverse=True)
            total_confidence = sum(count for _, count in sorted_bpms[:5])
            
            for bpm, count in sorted_bpms[:5]:
                confidence = count / total_confidence if total_confidence > 0 else 0
                bpm_results.append({
                    'bpm': bpm,
                    'confidence': confidence
                })
        
        # Calculate offset for the top BPM
        offset_results = []
        if bpm_results:
            first_bpm = bpm_results[0]['bpm']
            offset_results = self._calculate_offset_for_bpm(first_bpm, peak_scan_start)
        
        return bpm_results, offset_results
    
    def _calculate_offset_for_bpm(self, bpm: float, peak_scan_start: int) -> List[Dict]:
        """Calculate offset options for a given BPM"""
        if not self.novelty_curve_isolated:
            return []
            
        beat_length_blocks = (60 / bpm) * (self.sample_rate / self.window_step)
        
        # Create analysis wave for beat tracking
        analyze_wave = []
        for i in range(self.OFFSET_LOOKAHEAD):
            beat_block = (i % beat_length_blocks) / beat_length_blocks
            t = 0
            n = 0
            for j in range(1, 5):  # harmonics 1-4
                weight = 1 / j
                phase_weight = max(1 - abs(np.round(beat_block * j) / j - beat_block) * 12, 0)
                n += phase_weight * weight
                t += weight
            analyze_wave.append(n / t if t > 0 else 0)
        
        # Test different offset positions
        options = []
        end_scan = min(len(self.novelty_curve_isolated), 
                      int(peak_scan_start + beat_length_blocks))
        
        for i in range(peak_scan_start, end_scan):
            if i + self.OFFSET_LOOKAHEAD >= len(self.novelty_curve_isolated):
                break
                
            # Calculate correlation with beat pattern
            response = 0
            for j in range(self.OFFSET_LOOKAHEAD):
                if i + j < len(self.novelty_curve_isolated):
                    response += analyze_wave[j] * self.novelty_curve_isolated[i + j]
            
            offset = -((i * self.window_step) / self.sample_rate) % (60 / bpm)
            options.append({
                'offset': offset,
                'response': response
            })
        
        # Sort by response strength
        options.sort(key=lambda x: x['response'], reverse=True)
        
        # Return top 5 with confidence percentages
        if options:
            total_response = sum(opt['response'] for opt in options[:5])
            offset_results = []
            
            for opt in options[:5]:
                confidence = opt['response'] / total_response if total_response > 0 else 0
                offset_results.append({
                    'offset': np.round(opt['offset'], 3),
                    'confidence': confidence
                })
            
            return offset_results
        
        return []
    
    def calculate_beat_times(self, bpm: float, offset: float, audio_duration: float = None) -> List[float]:
        """Calculate beat times for a given BPM and offset"""
        if audio_duration is None:
            if self.audio_data is None:
                raise ValueError("Audio data not loaded and no duration provided")
            audio_duration = len(self.audio_data) / self.sample_rate
        
        beat_interval = 60.0 / bpm  # seconds per beat
        beat_times = []
        
        # Start from the offset and generate beats
        current_time = offset
        while current_time < audio_duration:
            if current_time >= 0:  # Only include positive times
                beat_times.append(np.round(current_time, 3))
            current_time += beat_interval
        
        return beat_times

def smedit_analyze_audio(audio_path: str, threshold: float = 0.3, bpm_range = [80,200]):
    """Analyze audio file for tempo and offset"""
    detector = SMEditAudioSyncDetector(min_bpm=bpm_range[0], max_bpm=bpm_range[1])
    
    # Load audio
    detector.load_audio(audio_path)
    
    # Detect onsets
    onsets = detector.detect_onsets(threshold=threshold)
    
    # Detect tempo and offset
    bpm_results, offset_results = detector.detect_tempo_and_offset()
    
    # Calculate beat times for each BPM estimate
    audio_duration = len(detector.audio_data) / detector.sample_rate
    beat_times = {}
    
    for bpm_result in bpm_results:
        bpm = bpm_result['bpm']
        # Use the best offset for this BPM (first one if available)
        if offset_results:
            offset = offset_results[0]['offset']
        else:
            offset = 0.0
        
        beats = detector.calculate_beat_times(bpm, offset, audio_duration)
        beat_times[bpm] = beats
    print(f"Best bpm, offset: {bpm_results[0]}, {offset_results[0]}")
    
    return {
        'bpm_results': bpm_results,
        'offset_results': offset_results,
        'onsets': onsets,
        'beat_times': beat_times,
        'detector': detector
    }

def get_template():
    template = """\
#TITLE:{title};
#ARTIST:{artist};
#MUSIC:{music_fp};
#OFFSET:{offset};
#BPMS:{bpm};
#STOPS:;
{charts}\
"""
    return template

def get_chart_template():
    chart_template = """\
    #NOTES:
    dance-single:
    DDCL:
    {ccoarse}:
    {cfine}:
    0.0,0.0,0.0,0.0,0.0:
    {measures};\
    """
    return chart_template

def ez_name(x):
    """
    Cleans and formats a given string x by removing spaces, replacing non-alphanumeric characters with underscores (_), 
    and ensuring it returns a string suitable for naming files or directories.
    _______
    ez_name("Hello World! 2023")  # Returns 'HelloWorld_2023'
    """
    x = ''.join(x.strip().split())
    x_clean = []
    for char in x:
        if char.isalnum():
            x_clean.append(char)
        else:
            x_clean.append('_')
    return ''.join(x_clean)

def get_subdirs(root, choose=False):
    """
    Lists the names of all subdirectories within a given root directory. 
    Optionally, it allows the user to select specific subdirectories to return.
    """
    subdir_names = sorted([x for x in os.listdir(root) if os.path.isdir(os.path.join(root, x))])
    if choose:
        for i, subdir_name in enumerate(subdir_names):
            print('{}: {}'.format(i, subdir_name))
        subdir_idxs = [int(x) for x in input('Which subdir(s)? ').split(',')]
        subdir_names = [subdir_names[i] for i in subdir_idxs]
    return subdir_names
    

def open_dataset_fps(*args):
    datasets = []
    for data_fp in args:
        if not data_fp:
            datasets.append([])
            continue

        with open(data_fp, 'r') as f:
            song_fps = f.read().split()
        dataset = []
        for song_fp in song_fps:
            with open(song_fp, 'rb') as f:
                dataset.append(pickle.load(f))
        datasets.append(dataset)
    return datasets[0] if len(datasets) == 1 else datasets

def windowize(dataset, frames = 7, front_set = 'None', go_backwards = False, take_windows = None, return_type = 'numpy'):
    if type(dataset) == list:
        dataset = np.array(dataset)
    dslen = dataset.shape[0]
    ds_shape = list(dataset.shape)
    ds_shape[0] = frames
    if front_set == 'None':
        front = np.zeros(ds_shape)
    elif front_set == 'min':
        front = np.ones(ds_shape)*np.min(dataset)
    else:
        front = np.array([front_set for i in range(frames)])
    if go_backwards:
        dataset = np.append(dataset, front, axis = 0)
    else:
        dataset = np.append(front, dataset, axis = 0)
    if take_windows is not None:
        new_ds = [dataset[i:i+1+frames] for i in take_windows]
    else:
        new_ds = [dataset[i:i+1+frames] for i in range(dslen)]
    if return_type == 'list':
        return new_ds
    elif return_type == 'numpy':
        return np.array(new_ds)
    else:
        raise NotImplementedError('o_o')

def sparse_to_categorical(val, max_val):
    a = np.zeros(max_val+1)
    a[val] = 1
    return a

def unfoldify(listy, base = 4):
    out = np.zeros(base*len(listy))
    for i in range(len(listy)):
        if listy[i] != 0:
            out[int(base*i+listy[i]-1)] = 1
    return out

def sparceify(listy, base = 4):
    out = 0
    for i in range(len(listy)):
        if listy[i] != 0:
            out += (listy[i])*(base**i)
    return out

def get_dataset_fp_list(*args):
    datasets = []
    for data_fp in args:
        if not data_fp:
            datasets.append([])
            continue

        with open(data_fp, 'r') as f:
            song_fps = f.read().split()
        dataset = []
        for song_fp in song_fps:
            dataset.append(song_fp)
        datasets.append(dataset)
    return datasets[0] if len(datasets) == 1 else datasets


def make_onset_feature_context(song_features, frame_idx, radius, left_radius = False):
    nframes = song_features.shape[0]
    
    assert nframes > 0
    if left_radius:
        frame_idxs = range(frame_idx - left_radius, frame_idx + radius + 1)
    else:
        frame_idxs = range(frame_idx - radius, frame_idx + radius + 1)
    context = np.zeros((len(frame_idxs),) + song_features.shape[1:], dtype=song_features.dtype)
    for i, frame_idx in enumerate(frame_idxs):
        if frame_idx >= 0 and frame_idx < nframes:
            context[i] = song_features[frame_idx]
        else:
            context[i] = np.ones_like(song_features[0])*np.log(1e-16)

    return context

def make_onset_feature_context_range(song_features, start_idx, end_idx, radius = 0, frame_density = 32):
    nframes = song_features.shape[0]

    start_idx, end_idx = int(start_idx*100), int(end_idx*100)
    
    assert nframes > 0

    frame_idxs = np.linspace(start_idx - radius, end_idx + radius, num = frame_density, endpoint = False).astype(int)
    context = np.zeros((len(frame_idxs),) + song_features.shape[1:], dtype=song_features.dtype)
    for i, frame_idx in enumerate(frame_idxs):
        if frame_idx >= 0 and frame_idx < nframes:
            context[i] = song_features[frame_idx]
        else:
            context[i] = np.ones_like(song_features[0])*np.log(1e-16)

    return context

def front_null(dataset, frames = 7, front_set = 'None', back_null = False):
    if type(dataset) == list:
        dataset = np.array(dataset)
    ds_shape = list(dataset.shape)
    ds_shape[0] = frames

    if front_set == 'min':
        front = np.ones(ds_shape) * np.min(dataset)
    elif front_set == 'None':
        front = np.zeros(ds_shape)
    else:
        front = np.array([front_set for i in range(frames)])
    dataset = np.append(front, dataset, axis = 0)
    if back_null:
        dataset = np.append(dataset, front, axis = 0)
    return dataset

def create_analyzers(fs=44100,
                     nhop=512,
                     nffts=[1024, 2048, 4096],
                     mel_nband=80,
                     mel_freqlo=27.5,
                     mel_freqhi=16000.0):
    analyzers = []
    for nfft in nffts:
        analyzer = {
            'nfft': nfft,
            'mel_basis': librosa.filters.mel(sr=fs, n_fft=nfft, n_mels=mel_nband,
                                             fmin=mel_freqlo, fmax=mel_freqhi),
            'window': np.blackman(nfft)  # closest to blackmanharris62
        }
        analyzers.append(analyzer)
    return analyzers

def extract_mel_feats(audio_fp, analyzers, fs=44100, nhop=512, nffts=[1024, 2048, 4096], log_scale=True):
    y, _ = librosa.load(audio_fp, sr=fs, mono=True)
    feat_channels = []

    for analyzer in analyzers:
        nfft = analyzer['nfft']
        mel_basis = analyzer['mel_basis']
        window = analyzer['window']

        # Compute STFT
        stft = librosa.stft(y, n_fft=nfft, hop_length=nhop, window=window, center=True)
        mag = np.abs(stft)**2  # Power spectrum

        # Apply mel filterbank
        mel_spec = mel_basis @ mag

        # Transpose to (frames, bands)
        mel_spec = mel_spec.T

        feat_channels.append(mel_spec)

    # Stack all channels (n_frames, n_mels, n_ffts)
    feat_channels = np.stack(feat_channels, axis=-1)

    if log_scale:
        feat_channels = np.log(feat_channels + 1e-16)

    return feat_channels

def extract_BPM(audio_fp, min_tempo=100, max_tempo=200):
    y, sr = librosa.load(audio_fp, sr=None, mono=True)

    # Estimate tempo
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, start_bpm=(min_tempo + max_tempo) / 2,
                                           tightness=1000, trim=False, units='time')

    # Convert to time
    beat_times = librosa.frames_to_time(beats, sr=sr)

    # Estimate beat intervals (spacing between beats)
    beat_intervals = np.diff(beat_times)

    # Estimate offset as in original: 2*beat0 - beat1
    offset = 2 * beat_times[0] - beat_times[1] if len(beat_times) > 1 else 0.0

    return {
        'BPM': tempo,
        'offset': offset,
        'beats': beat_times.tolist(),
        'beat_intervals': beat_intervals.tolist()
    }

def quick_reducify(ds, indices):
    return [[a[i] for i in indices] for a in ds]

def weighted_pick(weights):
    t = np.cumsum(weights)
    s = np.sum(weights)
    return(min(int(np.searchsorted(t, np.random.rand(1)*s)), 256))

def unravel_onehot(onehot, base):
    if onehot == 0:
        return '0000'
    else:
        out = ''
        while onehot:
            onehot, r = divmod(onehot, base)
            out += str(r)
        while len(out)<4:
            out += '0'
        return out

def downsample(dataset, down_key=2):
    return [[data[i] for i in range(0,len(data),down_key)] for data in dataset]

def label_to_vect_dict(labels):
    values = [list(a) for a in labels]
    max_len = np.lcm.reduce(np.unique([len(a) for a in values if len(a) != 0]))

    new_values = [np.zeros((max_len,)) for a in values]
    for i in range(len(new_values)):
        num_ticks = len(values[i])
        if num_ticks != 0:
            step = max_len/num_ticks
            for j in range(0, max_len, int(step)):
                new_values[i][j] = values[i][int(j/step)]
    new_dict = {k : v for k , v in zip(labels, new_values)}

    return new_dict

def ddc_string_to_step(string, narrow_types = 4):
    listy = [int(a) for a in list(string)]
    out = np.zeros((4*narrow_types))
    for i in range(len(listy)):
        out[narrow_types*i + listy[i]] = 1
    return out
    
def pickle_box(fp):
    #Load into a temporary namespace to avoid reference pollution
    with open(fp, 'rb') as f:
        data = pickle.load(f)
    return data