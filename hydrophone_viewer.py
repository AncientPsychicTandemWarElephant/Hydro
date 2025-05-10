# === Hydrophone Viewer with Audio Support and Visual Tracking ===
# Version 1.7.0 - Complete with Audio Tracking

import os
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.io import wavfile
from matplotlib.widgets import Button, SpanSelector, RangeSlider
import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.transforms import blended_transform_factory
import logging
from datetime import datetime
import pickle
import sys
import threading
import time

# Configure logging
logging.basicConfig(
    filename='error_log.txt',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# === Global State Variables ===
spec_img = None
ax_fft = None
ax_spec = None
ax_filelist = None
ax_log = None
fig = None
file_ranges = []
file_texts = []
freq_markers = [(None, None, None, None, None), (None, None, None, None, None)]
fft_patch = None
file_patch = None
spec_click_line = None
spec_click_text = None
time_labels_all = []
data_global = None
freqs_global = None
selected_range = None
comments = []
log_entries = []
comment_buttons = []
scroll_position = 0
gain_slider = None
fft_ymin = 0
fft_ymax = 120
vmin = 0
vmax = 1

# === Audio Globals ===
audio_data = None
audio_sample_rate = None
audio_timeline = []
audio_playback_line = None
audio_playing = False
audio_stop_flag = False
audio_thread = None
ax_audio_load = None
ax_audio_play = None
btn_audio_load = None
btn_audio_play = None
ax_time_display = None

# === FFT Data Parser ===
def parse_hydrophone_file(file_path):
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        start_idx = next(i for i, line in enumerate(lines) if line.startswith("Time") and "Data Points" in line)
        header_tokens = lines[start_idx].strip().split()
        freqs = [float(tok) for tok in header_tokens if tok.replace('.', '', 1).isdigit()]
        time_labels = []
        spec = []
        time_objects = []
        for line in lines[start_idx + 1:]:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            time_labels.append(parts[0])
            time_objects.append(datetime.strptime(parts[0], "%H:%M:%S"))
            amplitudes = [float(val) for val in parts[5:5+len(freqs)]]
            spec.append(amplitudes)
        spec_array = np.array(spec)
        if spec_array.shape[1] != len(freqs):
            raise ValueError("Mismatch in frequency bin count")
        return time_labels, freqs, spec_array, time_objects
    except Exception as e:
        logging.error(f"Error parsing file {file_path}", exc_info=True)
        print(f"Error parsing file {file_path}. Check error_log.txt for details.")
        raise

# === WAV File Loader with Timing ===
def load_and_merge_audio_with_timing(wav_paths, fft_file_paths=None):
    """Load audio files and calculate proper time alignment"""
    global audio_data, audio_sample_rate, audio_timeline
    
    audio_chunks = []
    audio_timeline = []
    
    for i, wav_path in enumerate(sorted(wav_paths)):
        rate, data = wavfile.read(wav_path)
        
        if audio_sample_rate is None:
            audio_sample_rate = rate
        elif audio_sample_rate != rate:
            raise ValueError(f"Inconsistent sample rate in {wav_path}")
        
        # Convert to mono if stereo
        if len(data.shape) > 1:
            data = data.mean(axis=1)
        
        audio_chunks.append(data)
        
        # Calculate time offset for this chunk
        if i == 0:
            offset = 0
        else:
            offset = sum(len(chunk) for chunk in audio_chunks[:-1]) / audio_sample_rate
        
        audio_timeline.append(offset)
    
    audio_data = np.concatenate(audio_chunks).astype(np.float32)
    # Normalize
    max_val = np.max(np.abs(audio_data))
    if max_val > 0:
        audio_data /= max_val
    
    add_log_entry(f"Audio total duration: {len(audio_data)/audio_sample_rate:.1f} seconds")

# === Update Functions ===
def update_fft(idx, freqs, data):
    global ax_fft, freq_markers, fft_ymin, fft_ymax
    ax_fft.clear()
    ax_fft.set_facecolor('black')
    ax_fft.set_title('FFT Slice', fontsize=12, color='#ffffff')
    ax_fft.grid(True, axis='y', linestyle='--', color='gray', alpha=0.3)
    ax_fft.set_ylim(fft_ymin, fft_ymax)
    ax_fft.plot(freqs, data[idx], color='lime')
    ax_fft.set_xlim(freqs[0], freqs[-1])
    
    for i, (_, _, _, freq, hline) in enumerate(freq_markers):
        if freq is not None:
            color = 'red' if i == 0 else 'blue'
            line = ax_fft.axvline(freq, color=color)
            label = ax_fft.text(freq, ax_fft.get_ylim()[1] * 1.01, f"{freq:.1f} Hz", 
                               rotation=90, va='bottom', ha='center', color=color)
            freq_markers[i] = (line, label, i, freq, hline)
    plt.draw()

def update_fft_range(start, end, freqs, data):
    global ax_fft, fft_ymin, fft_ymax, freq_markers
    ax_fft.clear()
    ax_fft.set_facecolor('black')
    ax_fft.set_title(f'Stacked FFTs {start}–{end}', fontsize=12, color='#ffffff')
    ax_fft.grid(True, axis='y', linestyle='--', color='gray', alpha=0.3)
    ax_fft.set_ylim(fft_ymin, fft_ymax)
    for i in range(start, end+1):
        ax_fft.plot(freqs, data[i], color='lime', alpha=0.2)
    ax_fft.set_xlim(freqs[0], freqs[-1])

    for i, (_, _, _, freq, hline) in enumerate(freq_markers):
        if freq is not None:
            color = 'red' if i == 0 else 'blue'
            line = ax_fft.axvline(freq, color=color)
            label = ax_fft.text(freq, 1.01, f"{freq:.1f} Hz", rotation=90, 
                               va='bottom', ha='center', color=color, 
                               transform=blended_transform_factory(ax_fft.transData, ax_fft.transAxes))
            freq_markers[i] = (line, label, i, freq, hline)
    plt.draw()

def update_gain(val):
    global spec_img
    lo, hi = val
    spec_img.set_clim(lo, hi)
    plt.draw()

def step_min(delta):
    global gain_slider, vmin
    lo, hi = gain_slider.val
    new_lo = np.clip(lo + delta, vmin, hi - 0.01)
    gain_slider.set_val((new_lo, hi))

def step_max(delta):
    global gain_slider, vmax
    lo, hi = gain_slider.val
    new_hi = np.clip(hi + delta, lo + 0.01, vmax)
    gain_slider.set_val((lo, new_hi))

def update_marker(n, xpos):
    global freq_markers, ax_fft, ax_spec
    freq = xpos
    
    # Remove old marker
    if freq_markers[n][0]:
        try:
            freq_markers[n][0].remove()
        except Exception:
            pass
    if freq_markers[n][1]:
        try:
            freq_markers[n][1].remove()
        except Exception:
            pass
    
    # Remove from spectrogram
    for artist in ax_spec.lines[:]:
        if hasattr(artist, 'is_marker') and artist.is_marker == n:
            try:
                artist.remove()
            except Exception:
                pass
    for artist in ax_spec.texts[:]:
        if hasattr(artist, 'is_marker') and artist.is_marker == n:
            try:
                artist.remove()
            except Exception:
                pass
    
    color = 'red' if n == 0 else 'blue'
    line = ax_fft.axvline(freq, color=color)
    label = ax_fft.text(freq, 1.01, f"{freq:.1f} Hz", rotation=90, 
                       va='bottom', ha='center', color=color, 
                       transform=blended_transform_factory(ax_fft.transData, ax_fft.transAxes))
    hline = ax_spec.axhline(freq, color=color, linestyle='--', linewidth=1)
    label_spec = ax_spec.text(ax_spec.get_xlim()[1] + 5, freq, f"{freq:.1f} Hz", 
                             va='center', ha='left', fontsize=8, color=color, clip_on=False)
    label_spec.is_marker = n
    hline.is_marker = n
    freq_markers[n] = (line, label, n, freq, hline)
    plt.draw()

def add_log_entry(msg):
    """Add entry to log display"""
    global ax_log, log_entries
    log_entries.append(msg)
    print(f"Log: {msg}")
    if ax_log:
        ax_log.clear()
        ax_log.set_title("Log", fontsize=9, pad=4, color='black')
        ax_log.axis("off")
        # Show only last 5 entries
        display_entries = log_entries[-5:]
        for i, entry in enumerate(display_entries):
            ax_log.text(0.02, 0.8 - i*0.15, entry, transform=ax_log.transAxes, 
                       fontsize=8, va='top')
        plt.draw()

def update_time_display(current_time, total_time):
    """Update the time display with current playback position"""
    global ax_time_display
    
    if ax_time_display:
        ax_time_display.clear()
        ax_time_display.set_title("Playback Time", fontsize=9, pad=4)
        ax_time_display.axis("off")
        
        current_str = time.strftime('%H:%M:%S', time.gmtime(current_time))
        total_str = time.strftime('%H:%M:%S', time.gmtime(total_time))
        
        ax_time_display.text(0.5, 0.5, f"{current_str} / {total_str}", 
                            transform=ax_time_display.transAxes, 
                            fontsize=10, ha='center', va='center')
        plt.draw()

def update_play_button_text():
    """Update play button text based on state"""
    global btn_audio_play, audio_playing
    if btn_audio_play:
        if audio_playing:
            btn_audio_play.label.set_text('Stop Audio')
        else:
            btn_audio_play.label.set_text('Play Audio')
        plt.draw()

# === Audio Functions ===
def on_load_audio(event):
    """Load audio files with better timing alignment"""
    global audio_data, audio_sample_rate, btn_audio_play, file_paths
    
    wav_paths = filedialog.askopenfilenames(
        title="Select WAV files",
        filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
    )
    
    if wav_paths:
        try:
            # Pass FFT file paths if available for better time alignment
            load_and_merge_audio_with_timing(wav_paths, file_paths if 'file_paths' in globals() else None)
            add_log_entry(f"Loaded {len(wav_paths)} audio file(s)")
            if btn_audio_play:
                btn_audio_play.set_active(True)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load audio: {str(e)}")
            logging.error(f"Error loading audio files", exc_info=True)

def on_play_audio(event):
    """Play audio for selected time range with visual tracking"""
    global audio_data, audio_sample_rate, selected_range, time_labels_all
    global audio_playback_line, audio_playing, audio_stop_flag, audio_thread
    
    # Stop any currently playing audio
    if audio_playing:
        audio_stop_flag = True
        sd.stop()
        if audio_thread and audio_thread.is_alive():
            audio_thread.join(timeout=1.0)
        audio_playing = False
        if audio_playback_line:
            audio_playback_line.remove()
            audio_playback_line = None
        update_play_button_text()
        plt.draw()
        add_log_entry("Audio stopped")
        return
    
    if audio_data is None:
        add_log_entry("No audio loaded")
        return
    
    if selected_range is None:
        add_log_entry("Select a time range first")
        return
    
    start_idx, end_idx = selected_range
    
    # Map FFT indices to audio samples
    # Calculate the time per FFT sample (assuming 1 second per FFT)
    fft_time_per_sample = 1.0  # seconds
    
    # Calculate start and end times in seconds
    start_time = start_idx * fft_time_per_sample
    end_time = end_idx * fft_time_per_sample
    
    # Convert to audio samples
    start_sample = int(start_time * audio_sample_rate)
    end_sample = int(end_time * audio_sample_rate)
    
    # Ensure bounds are valid
    start_sample = max(0, min(start_sample, len(audio_data) - 1))
    end_sample = max(0, min(end_sample, len(audio_data)))
    
    if start_sample >= end_sample:
        add_log_entry("Invalid audio range")
        return
    
    # Extract audio segment
    segment = audio_data[start_sample:end_sample]
    
    # Create playback tracking line
    if audio_playback_line:
        audio_playback_line.remove()
    audio_playback_line = ax_spec.axvline(start_idx, color='yellow', linewidth=2, 
                                         linestyle='-', alpha=0.8)
    
    # Start playback
    audio_playing = True
    audio_stop_flag = False
    update_play_button_text()
    
    def play_and_track():
        global audio_playback_line, audio_playing, audio_stop_flag
        
        try:
            # Start playing audio
            sd.play(segment, audio_sample_rate)
            add_log_entry(f"Playing audio: {start_idx}-{end_idx}")
            
            # Track playback position
            duration = len(segment) / audio_sample_rate
            start_play_time = time.time()
            
            while audio_playing and not audio_stop_flag:
                elapsed = time.time() - start_play_time
                
                if elapsed >= duration:
                    break
                
                # Calculate current position in FFT indices
                progress = elapsed / duration
                current_idx = start_idx + (end_idx - start_idx) * progress
                
                # Update playback line position
                if audio_playback_line:
                    audio_playback_line.set_xdata([current_idx, current_idx])
                    plt.draw()
                
                # Update time display
                update_time_display(elapsed, duration)
                
                time.sleep(0.05)  # Update every 50ms
            
            # Cleanup after playback
            sd.stop()
            if audio_playback_line:
                audio_playback_line.remove()
                audio_playback_line = None
            plt.draw()
            
        except Exception as e:
            add_log_entry(f"Audio error: {str(e)}")
            logging.error(f"Error during audio playback", exc_info=True)
        finally:
            audio_playing = False
            update_play_button_text()
            add_log_entry("Audio finished")
    
    # Start playback in a separate thread
    audio_thread = threading.Thread(target=play_and_track, daemon=True)
    audio_thread.start()

def on_key_press_audio(event):
    """Handle keyboard shortcuts for audio control"""
    global audio_playing
    
    if event.key == ' ':  # Spacebar for play/stop
        on_play_audio(None)
    elif event.key == 'escape' and audio_playing:  # Escape to stop
        global audio_stop_flag
        audio_stop_flag = True
        sd.stop()
        audio_playing = False

def setup_viewer(file_paths):
    global fig, ax_fft, ax_spec, spec_img, ax_filelist, ax_log, ax_clear
    global file_ranges, file_texts, fft_patch, file_patch
    global spec_click_line, spec_click_text
    global time_labels_all, data_global, freqs_global, selected_range
    global fft_ymin, fft_ymax, gain_slider, vmin, vmax
    global ax_audio_load, ax_audio_play, btn_audio_load, btn_audio_play
    global ax_time_display
    
    print(f"Setting up viewer with {len(file_paths)} files")
    
    fft_ymin, fft_ymax = 0, 120
    time_labels_all = []
    data_list = []
    file_ranges = []
    freqs_global = None
    idx_offset = 0
    
    last_time = None
    for path in file_paths:
        try:
            print(f"Processing {path}")
            t, f, d, t_objs = parse_hydrophone_file(path)
            if last_time and (t_objs[0] - last_time).total_seconds() > 1.5:
                gap_len = int((t_objs[0] - last_time).total_seconds())
                if gap_len > 1:
                    gap_array = np.full((gap_len, d.shape[1]), np.nan)
                    data_list.append(gap_array)
                    time_labels_all.extend(["GAP"] * gap_len)
                    idx_offset += gap_len
            last_time = t_objs[-1]
            time_labels_all.extend(t)
            data_list.append(d)
            file_ranges.append((idx_offset, idx_offset + len(d) - 1))
            idx_offset += len(d)
            if freqs_global is None:
                freqs_global = f
            elif freqs_global != f:
                raise ValueError("Frequency bins don't match")
        except Exception as e:
            logging.error(f"Error processing file {path}", exc_info=True)
            print(f"Error processing file {path}: {str(e)}")
            continue
    
    if not data_list:
        print("No valid data loaded")
        messagebox.showerror("Error", "No valid data could be loaded from the selected files")
        return
    
    data_global = np.vstack(data_list)
    comments.clear()
    comment_buttons.clear()
    
    print("Creating figure")
    
    # Create figure
    fig = plt.figure(figsize=(22, 9))
    ax_log = fig.add_axes([0.12, 0.03, 0.7, 0.07], frameon=True, facecolor='lightgray')
    ax_log.set_title("Log", fontsize=9, pad=4, color='black')
    ax_log.axis("off")
    
    # Gain controls
    gain_slider_left = 0.045
    gain_slider_bottom = 0.2
    gain_slider_width = 0.02
    gain_slider_height = 0.25
    ax_gain = fig.add_axes([gain_slider_left, gain_slider_bottom, gain_slider_width, gain_slider_height])
    vmin = np.nanmin(data_global)
    vmax = np.nanmax(data_global)
    gain_slider = RangeSlider(ax_gain, 'Gain', vmin, vmax, valinit=(vmin, vmax), orientation='vertical')
    
    # Gain adjustment buttons
    btn_width = 0.035
    btn_height = 0.04
    btn_left = gain_slider_left - btn_width - 0.005
    btn_y = [
        gain_slider_bottom + gain_slider_height * 0.75,
        gain_slider_bottom + gain_slider_height * 0.60,
        gain_slider_bottom + gain_slider_height * 0.35,
        gain_slider_bottom + gain_slider_height * 0.20,
    ]
    
    ax_max_up = fig.add_axes([btn_left, btn_y[0], btn_width, btn_height])
    btn_max_up = Button(ax_max_up, '+Max')
    ax_max_down = fig.add_axes([btn_left, btn_y[1], btn_width, btn_height])
    btn_max_down = Button(ax_max_down, '-Max')
    ax_min_up = fig.add_axes([btn_left, btn_y[2], btn_width, btn_height])
    btn_min_up = Button(ax_min_up, '+Min')
    ax_min_down = fig.add_axes([btn_left, btn_y[3], btn_width, btn_height])
    btn_min_down = Button(ax_min_down, '-Min')
    
    btn_min_up.on_clicked(lambda event: step_min(1.0))
    btn_min_down.on_clicked(lambda event: step_min(-1.0))
    btn_max_up.on_clicked(lambda event: step_max(1.0))
    btn_max_down.on_clicked(lambda event: step_max(-1.0))
    gain_slider.on_changed(update_gain)
    
    # Menu button
    menu_btn_ax = fig.add_axes([0.005, 0.945, 0.05, 0.03])
    menu_btn = Button(menu_btn_ax, '▼ File')
    menu_btn_ax._button_panel = []
    
    def toggle_file_menu(event):
        if menu_btn_ax._button_panel:
            for btn in menu_btn_ax._button_panel:
                btn.ax.remove()
            menu_btn_ax._button_panel.clear()
        else:
            ax_open = fig.add_axes([0.005, 0.91, 0.1, 0.03])
            btn_open = Button(ax_open, 'Open Files')
            btn_open.on_clicked(on_open)
            menu_btn_ax._button_panel.append(btn_open)
            
            ax_save = fig.add_axes([0.005, 0.87, 0.1, 0.03])
            btn_save = Button(ax_save, 'Save Project')
            btn_save.on_clicked(on_save_project)
            menu_btn_ax._button_panel.append(btn_save)
            
            ax_load = fig.add_axes([0.005, 0.83, 0.1, 0.03])
            btn_load = Button(ax_load, 'Load Project')
            btn_load.on_clicked(on_load_project)
            menu_btn_ax._button_panel.append(btn_load)
        plt.draw()
    
    menu_btn.on_clicked(toggle_file_menu)
    
    # Version label
    fig.text(0.99, 0.01, 'v1.7.0', fontsize=8, color='gray', ha='right', va='bottom')
    fig.subplots_adjust(top=0.88)
    
    # Main plots
    ax_fft = fig.add_axes([0.1, 0.55, 0.7, 0.35])
    ax_spec = fig.add_axes([0.1, 0.2, 0.7, 0.25])
    ax_filelist = fig.add_axes([0.82, 0.3, 0.14, 0.55], frameon=True, facecolor='lightgray')
    
    # File list
    ax_filelist.clear()
    ax_filelist.set_title("Files", fontsize=9, pad=8)
    ax_filelist.axis("off")
    file_texts.clear()
    for i, path in enumerate(file_paths):
        y = 1 - (i + 1) * (1.0 / max(len(file_paths), 10))
        txt = ax_filelist.text(0.05, y, os.path.basename(path), 
                              transform=ax_filelist.transAxes, fontsize=8, 
                              verticalalignment='top', picker=True)
        file_texts.append(txt)
    
    # FFT plot setup
    ax_fft.set_facecolor('black')
    ax_fft.set_title('FFT Slice')
    ax_fft.grid(True, axis='y', linestyle='--', color='gray', alpha=0.3)
    ax_fft.plot(freqs_global, data_global[0], color='lime')
    ax_fft.set_ylim(fft_ymin, fft_ymax)
    
    # Spectrogram
    spec_img = ax_spec.imshow(
        data_global.T,
        aspect='auto', origin='lower',
        extent=[0, data_global.shape[0]-1, freqs_global[0], freqs_global[-1]],
        cmap='viridis'
    )
    
    # Selection span
    def ctrl_select(xmin, xmax):
        global fft_patch, selected_range
        start, end = int(xmin), int(xmax)
        selected_range = (start, end)
        if fft_patch:
            fft_patch.remove()
        fft_patch = ax_spec.axvspan(start, end, color='red', alpha=0.3)
        update_fft_range(start, end, freqs_global, data_global)
    
    span = SpanSelector(ax_spec, lambda *args: None, 'horizontal', 
                       useblit=True, props=dict(alpha=0.3, facecolor='red'))
    span.set_active(False)
    
    # Event handlers
    def on_key_press(event):
        if event.key == 'control':
            span.set_active(True)
    
    def on_key_release(event):
        if event.key == 'control':
            span.set_active(False)
    
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    fig.canvas.mpl_connect('key_release_event', on_key_release)
    fig.canvas.mpl_connect('key_press_event', on_key_press_audio)  # Add audio key handling
    
    def on_span_release(start, end):
        ctrl_select(start, end)
    span.onselect = on_span_release
    
    def on_spec_click(event):
        global spec_click_line, spec_click_text, fft_patch, selected_range
        if event.inaxes == ax_spec:
            idx = max(0, min(int(event.xdata), data_global.shape[0] - 1))
            if event.button == 1:
                if fft_patch:
                    fft_patch.remove()
                    fft_patch = None
                selected_range = None
                update_fft(idx, freqs_global, data_global)
                if spec_click_line:
                    spec_click_line.remove()
                if spec_click_text:
                    spec_click_text.remove()
                spec_click_line = ax_spec.axvline(idx, color='white', linewidth=1)
                time_str = time_labels_all[idx]
                spec_click_text = ax_spec.text(idx, 0, time_str,
                                             transform=blended_transform_factory(ax_spec.transData, ax_spec.transAxes),
                                             color='white', rotation=90, va='top', ha='center', clip_on=False)
            elif event.button == 3:
                if spec_click_line:
                    start = int(spec_click_line.get_xdata()[0])
                    end = idx
                    if start > end:
                        start, end = end, start
                    if fft_patch:
                        fft_patch.remove()
                    fft_patch = ax_spec.axvspan(start, end, color='red', alpha=0.3)
                    selected_range = (start, end)
                    update_fft_range(start, end, freqs_global, data_global)
        plt.draw()
    
    fig.canvas.mpl_connect('button_press_event', on_spec_click)
    
    def on_click(event):
        if event.key == 'control' and event.inaxes == ax_fft:
            if event.button == 1:
                update_marker(0, event.xdata)
            elif event.button == 3:
                update_marker(1, event.xdata)
    
    fig.canvas.mpl_connect('button_press_event', on_click)
    
    def on_pick(event):
        global file_patch
        if event.artist in file_texts:
            idx = file_texts.index(event.artist)
            start, end = file_ranges[idx]
            try:
                if file_patch:
                    file_patch.remove()
            except ValueError:
                file_patch = None
            file_patch = ax_spec.axvspan(start, end, color='blue', alpha=0.2)
            for i, txt in enumerate(file_texts):
                txt.set_backgroundcolor('yellow' if i == idx else None)
            plt.draw()
    
    fig.canvas.mpl_connect('pick_event', on_pick)
    
    def on_open(event):
        new_paths = filedialog.askopenfilenames(filetypes=[('Text', '*.txt')])
        if new_paths:
            plt.close(fig)
            setup_viewer(list(new_paths))
    
    def on_save_project(event):
        save_path = filedialog.asksaveasfilename(defaultextension=".hproj", 
                                               filetypes=[("Hydrophone Project", "*.hproj")])
        if not save_path:
            return
        state = {
            "comments": comments,
            "y_scale": (fft_ymin, fft_ymax),
            "gain": spec_img.get_clim(),
            "file_paths": file_paths,
            "freq_markers": [(m[3] if m else None) for m in freq_markers],
            "log_entries": log_entries,
            "click_index": int(spec_click_line.get_xdata()[0]) if spec_click_line else None,
        }
        with open(save_path, 'wb') as f:
            pickle.dump(state, f)
        add_log_entry(f"Project saved to {os.path.basename(save_path)}")
    
    def on_load_project(event):
        global fft_ymin, fft_ymax, spec_click_line, spec_click_text
        load_path = filedialog.askopenfilename(filetypes=[("Hydrophone Project", "*.hproj")])
        if not load_path:
            return
        with open(load_path, 'rb') as f:
            state = pickle.load(f)
        
        fft_ymin, fft_ymax = state["y_scale"]
        setup_viewer(state["file_paths"])
        
        for c in state.get("comments", []):
            comments.append(c)
        
        spec_img.set_clim(*state["gain"])
        log_entries.extend(state.get("log_entries", []))
        
        if "click_index" in state and state["click_index"] is not None:
            update_fft(state["click_index"], freqs_global, data_global)
            idx = state["click_index"]
            spec_click_line = ax_spec.axvline(idx, color='white', linewidth=1)
            time_str = time_labels_all[idx]
            spec_click_text = ax_spec.text(idx, 0, time_str,
                                         transform=blended_transform_factory(ax_spec.transData, ax_spec.transAxes),
                                         color='white', rotation=90, va='top', ha='center', clip_on=False)
        
        if "freq_markers" in state:
            for i, freq in enumerate(state["freq_markers"]):
                if freq is not None:
                    update_marker(i, freq)
    
    def on_key(event):
        global spec_click_line, spec_click_text, fft_patch
        if spec_click_line is None or event.key not in ['left', 'right']:
            return
        if fft_patch:
            fft_patch.remove()
            fft_patch = None
        current = spec_click_line.get_xdata()[0]
        idx = int(current) + (-1 if event.key == 'left' else 1)
        idx = max(0, min(idx, data_global.shape[0] - 1))
        update_fft(idx, freqs_global, data_global)
        spec_click_line.set_xdata([idx, idx])
        time_str = time_labels_all[idx]
        spec_click_text.set_text(time_str)
        spec_click_text.set_position((idx, 0))
        plt.draw()
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    def clear_file_highlight(event):
        global file_patch
        if file_patch:
            file_patch.remove()
            file_patch = None
        for txt in file_texts:
            txt.set_backgroundcolor(None)
        plt.draw()
    
    ax_clear = fig.add_axes([0.75, 0.13, 0.1, 0.04])
    btn_clear = Button(ax_clear, 'Clear Highlight')
    btn_clear.on_clicked(clear_file_highlight)
    
    def adjust_fft_scale(delta):
        global fft_ymin, fft_ymax
        fft_ymax = max(10, fft_ymax + delta)
        ax_fft.set_ylim(fft_ymin, fft_ymax)
        plt.draw()
    
    ax_up = fig.add_axes([0.05, 0.55, 0.03, 0.04])
    btn_up = Button(ax_up, '+Y')
    btn_up.on_clicked(lambda e: adjust_fft_scale(10))
    
    ax_down = fig.add_axes([0.05, 0.5, 0.03, 0.04])
    btn_down = Button(ax_down, '-Y')
    btn_down.on_clicked(lambda e: adjust_fft_scale(-10))
    
    # Audio controls
    ax_audio_load = fig.add_axes([0.82, 0.13, 0.06, 0.04])
    btn_audio_load = Button(ax_audio_load, 'Load Audio')
    btn_audio_load.on_clicked(on_load_audio)
    
    ax_audio_play = fig.add_axes([0.89, 0.13, 0.06, 0.04])
    btn_audio_play = Button(ax_audio_play, 'Play Audio')
    btn_audio_play.on_clicked(on_play_audio)
    
    # Time display
    ax_time_display = fig.add_axes([0.82, 0.08, 0.14, 0.03], frameon=True, facecolor='lightgray')
    ax_time_display.set_title("Playback Time", fontsize=9, pad=4)
    ax_time_display.axis("off")
    ax_time_display.text(0.5, 0.5, "00:00:00 / 00:00:00", 
                        transform=ax_time_display.transAxes, 
                        fontsize=10, ha='center', va='center')
    
    print("Showing figure")
    plt.show()

# Main execution
if __name__ == '__main__':
    try:
        print("Starting Hydrophone Viewer")
        root = tk.Tk()
        root.withdraw()  # Hide the main tkinter window
        
        print("Opening file dialog")
        file_paths = filedialog.askopenfilenames(
            title="Select Hydrophone Data Files",
            filetypes=[('Text files', '*.txt'), ('All files', '*.*')]
        )
        
        if file_paths:
            print(f"Selected {len(file_paths)} files")
            setup_viewer(list(file_paths))
        else:
            print("No files selected")
            sys.exit()
            
    except Exception as e:
        logging.error("An error occurred while launching the viewer", exc_info=True)
        print(f"An error occurred: {str(e)}")
        print("Please check the error_log.txt file for details.")
        messagebox.showerror("Error", f"An error occurred: {str(e)}")
        sys.exit(1)