# === Hydrophone Viewer (Restored Version with All Features) ===
# Version 1.5.1
import os
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import matplotlib.animation as animation
from scipy.signal import istft
from matplotlib.widgets import Button, SpanSelector, RangeSlider
import tkinter as tk
from tkinter import filedialog
from matplotlib.transforms import blended_transform_factory
import logging

# Configure logging
logging.basicConfig(
    filename='error_log.txt',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

spec_img = None
ax_fft = None
ax_spec = None
ax_filelist = None
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

# Ensure `scroll_position` is initialized globally
scroll_position = 0  # Initial scroll position

from datetime import datetime, timedelta

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


def update_fft(idx, freqs, data):
    ax_fft.clear()
    ax_fft.set_facecolor('black')
    ax_fft.set_title('FFT Slice', fontsize=12, color='#ffffff')
    ax_fft.grid(True, axis='y', linestyle='--', color='gray', alpha=0.3)
    ax_fft.set_ylim(fft_ymin, fft_ymax)
    ax_fft.plot(freqs, data[idx], color='lime')
    ax_fft.set_xlim(freqs[0], freqs[-1])
    for marker in freq_markers:
        if len(marker) != 5:
            continue
        line, label, _, freq, hline = marker
        for artist in (line, label):
            if artist:
                try:
                    artist.remove()
                except Exception:
                    pass
    for i, (_, _, _, freq, hline) in enumerate(freq_markers):
        if freq is not None:
            color = 'red' if i == 0 else 'blue'
            line = ax_fft.axvline(freq, color=color)
            label = ax_fft.text(freq, ax_fft.get_ylim()[1] * 1.01, f"{freq:.1f} Hz", rotation=90, va='bottom', ha='center', color=color)
            freq_markers[i] = (line, label, i, freq, hline)
    plt.draw()


def update_fft_range(start, end, freqs, data):
    global fft_ymin, fft_ymax
    ax_fft.clear()
    ax_fft.set_facecolor('black')
    ax_fft.set_title(f'Stacked FFTs {start}–{end}', fontsize=12, color='#ffffff')
    ax_fft.grid(True, axis='y', linestyle='--', color='gray', alpha=0.3)
    ax_fft.set_ylim(fft_ymin, fft_ymax)
    for i in range(start, end+1):
        ax_fft.plot(freqs, data[i], color='lime', alpha=0.2)
    ax_fft.set_xlim(freqs[0], freqs[-1])

    # Redraw frequency markers
    for i, (_, _, _, freq, hline) in enumerate(freq_markers):
        if freq is not None:
            color = 'red' if i == 0 else 'blue'
            line = ax_fft.axvline(freq, color=color)
            y_top = ax_fft.get_ylim()[1]
            label = ax_fft.text(freq, 1.01, f"{freq:.1f} Hz", rotation=90, va='bottom', ha='center', color=color, transform=blended_transform_factory(ax_fft.transData, ax_fft.transAxes))
            freq_markers[i] = (line, label, i, freq, hline)

    plt.draw()


def update_gain(val):
    lo, hi = val
    spec_img.set_clim(lo, hi)
    plt.draw()


def update_marker(n, xpos):
    global freq_markers
    freq = xpos
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
    # Remove any previous ax_spec horizontal lines for this marker
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
    y_top = ax_fft.get_ylim()[1]
    label = ax_fft.text(freq, 1.01, f"{freq:.1f} Hz", rotation=90, va='bottom', ha='center', color=color, transform=blended_transform_factory(ax_fft.transData, ax_fft.transAxes))
    hline = ax_spec.axhline(freq, color=color, linestyle='--', linewidth=1)
    label_spec = ax_spec.text(ax_spec.get_xlim()[1] + 5, freq, f"{freq:.1f} Hz", va='center', ha='left', fontsize=8, color=color, clip_on=False)
    label_spec.is_marker = n
    hline.is_marker = n  # custom flag to allow identification later
    freq_markers[n] = (line, label, n, freq, hline)
    plt.draw()
    plt.draw()


active_animations = []

def setup_viewer(file_paths):
    global ax_clear
    global fft_ymin, fft_ymax
    fft_ymin, fft_ymax = 0, 120
    global fig, ax_fft, ax_spec, spec_img, ax_filelist, ax_comments, comment_buttons
    global file_ranges, file_texts, fft_patch, file_patch
    global spec_click_line, spec_click_text
    global time_labels_all, data_global, freqs_global, selected_range

    time_labels_all = []
    data_list = []
    file_ranges = []
    freqs_global = None
    idx_offset = 0

    last_time = None
    for path in file_paths:
        try:
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
            print(f"Error processing file {path}. Check error_log.txt for details.")

    data_global = np.vstack(data_list)
    comments.clear()
    comment_buttons.clear()

    fig = plt.figure(figsize=(18, 9))
    ax_log = fig.add_axes([0.1, 0.03, 0.7, 0.07], frameon=True, facecolor='lightgray')
    ax_log.set_title("Log", fontsize=9, pad=4, color='black')
    ax_log.axis("off")

    def add_log_entry(text):
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_entries.append(f"[{timestamp}] {text}")
        ax_log.clear()
        ax_log.set_title("Log", fontsize=9, pad=4, color='black')
        ax_log.axis("off")
        for i, entry in enumerate(reversed(log_entries[-6:])):
            ax_log.text(0.01, 0.95 - i * 0.15, entry, transform=ax_log.transAxes, fontsize=7, color='black', va='top')
        plt.draw()
    # Create menu as overlay text
    menu_btn_ax = fig.add_axes([0.005, 0.945, 0.05, 0.03])
    menu_btn = Button(menu_btn_ax, '▼ File')  # ▼ File
    menu_btn_ax._button_panel = []

    def toggle_file_menu(event):
        global ax_clear
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
    fig.text(0.99, 0.01, 'v1.3.2', fontsize=8, color='gray', ha='right', va='bottom')
    fig.subplots_adjust(top=0.88)
    ax_fft = fig.add_axes([0.1, 0.55, 0.7, 0.35])
    ax_fft.grid(True, axis='y', linestyle='--', color='gray', alpha=0.3)
    ax_spec = fig.add_axes([0.1, 0.2, 0.7, 0.25])
    ax_gain = fig.add_axes([0.02, 0.2, 0.02, 0.25])
    ax_filelist = fig.add_axes([0.82, 0.3, 0.14, 0.55], frameon=True, facecolor='lightgray')
    ax_filelist.clear()
    ax_filelist.set_title("Files", fontsize=9, pad=8)
    ax_filelist.axis("off")
    file_texts.clear()
    for i, path in enumerate(file_paths):
        y = 1 - (i + 1) * (1.0 / max(len(file_paths), 10))  # auto-spacing based on count
        txt = ax_filelist.text(0.05, y, os.path.basename(path), transform=ax_filelist.transAxes, fontsize=8, verticalalignment='top', picker=True)
        file_texts.append(txt)
    
    ax_fft.set_facecolor('black')
    ax_fft.set_title('FFT Slice')
    ax_fft.plot(freqs_global, data_global[0], color='lime')

    spec_img = ax_spec.imshow(
        data_global.T,
        aspect='auto', origin='lower',
        extent=[0, data_global.shape[0]-1, freqs_global[0], freqs_global[-1]],
        cmap='viridis'
    )

    vmin = np.nanmin(data_global)
    vmax = np.nanmax(data_global)
    gain_slider = RangeSlider(ax_gain, 'Gain', vmin, vmax,
                              valinit=(vmin, vmax), orientation='vertical')
    gain_slider.on_changed(lambda v: update_gain(v))

    def ctrl_select(xmin, xmax):
        global fft_patch, selected_range
        start, end = int(xmin), int(xmax)
        selected_range = (start, end)
        if fft_patch:
            fft_patch.remove()
        fft_patch = ax_spec.axvspan(start, end, color='red', alpha=0.3)
        update_fft_range(start, end, freqs_global, data_global)

    span = SpanSelector(ax_spec, lambda *args: None, 'horizontal', useblit=True, props=dict(alpha=0.3, facecolor='red'))
    span.set_active(False)
    fig.canvas.mpl_connect('key_press_event', lambda e: span.set_active(e.key == 'control'))
    fig.canvas.mpl_connect('key_release_event', lambda e: span.set_active(False))
    def on_span_release(start, end):
        ctrl_select(start, end)
    span.onselect = on_span_release

    def on_spec_click(event):
        global spec_click_line, spec_click_text, fft_patch, selected_range
        if event.inaxes == ax_spec:
            idx = max(0, min(int(event.xdata), data_global.shape[0] - 1))

            if event.button == 1:
                # Left click: single FFT
                if fft_patch:
                    fft_patch.remove(); fft_patch = None
                selected_range = None
                update_fft(idx, freqs_global, data_global)
                if spec_click_line: spec_click_line.remove()
                if spec_click_text: spec_click_text.remove()
                spec_click_line = ax_spec.axvline(idx, color='white', linewidth=1)
                time_str = time_labels_all[idx]
                spec_click_text = ax_spec.text(idx, 0, time_str, transform=blended_transform_factory(ax_spec.transData, ax_spec.transAxes), color='white', rotation=90, va='top', ha='center', clip_on=False)

            elif event.button == 3:
                # Right click: define end point and stack
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

    # Update the file list to highlight the selected file in yellow
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

            # Highlight the selected file name in yellow
            for i, txt in enumerate(file_texts):
                if i == idx:
                    txt.set_backgroundcolor('yellow')
                else:
                    txt.set_backgroundcolor(None)

            plt.draw()

    fig.canvas.mpl_connect('pick_event', on_pick)

    def on_open(event):
        new_paths = filedialog.askopenfilenames(filetypes=[('Text', '*.txt')])
        if new_paths:
            plt.close(fig)
            setup_viewer(list(new_paths))

    def on_save_project(event):
        import pickle
        from tkinter import filedialog
        save_path = filedialog.asksaveasfilename(defaultextension=".hproj", filetypes=[("Hydrophone Project", "*.hproj")])
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
        import pickle
        from tkinter import filedialog
        load_path = filedialog.askopenfilename(filetypes=[("Hydrophone Project", "*.hproj")])
        if not load_path:
            return
        with open(load_path, 'rb') as f:
            state = pickle.load(f)
        global fft_ymin, fft_ymax
        fft_ymin, fft_ymax = state["y_scale"]
        setup_viewer(state["file_paths"])
        for c in state.get("comments", []):
            comments.append(c)
        update_comment_panel()
        spec_img.set_clim(*state["gain"])
        log_entries.extend(state.get("log_entries", []))
        if "click_index" in state:
            update_fft(state["click_index"], freqs_global, data_global)
            global spec_click_line, spec_click_text
            idx = state["click_index"]
            spec_click_line = ax_spec.axvline(idx, color='white', linewidth=1)
            time_str = time_labels_all[idx]
            spec_click_text = ax_spec.text(idx, 0, time_str, transform=blended_transform_factory(ax_spec.transData, ax_spec.transAxes), color='white', rotation=90, va='top', ha='center', clip_on=False)
        if "freq_markers" in state:
            for i, freq in enumerate(state["freq_markers"]):
                if freq is not None:
                    update_marker(i, freq)
        for c in state["comments"]:
            comments.append(c)
        update_comment_panel()
        spec_img.set_clim(*state["gain"])
        add_log_entry(f"Project loaded from {os.path.basename(load_path)}")

    

    def on_key(event):
        global spec_click_line, spec_click_text, fft_patch
        if spec_click_line is None or event.key not in ['left', 'right']:
            return
        if fft_patch:
            fft_patch.remove()
            fft_patch = None
        current = spec_click_line.get_xdata()[0]
        idx = int(current) + (-1 if event.key == 'left' else 1)
        idx = max(0, min(idx, data_global.shape[0]-1))
        update_fft(idx, freqs_global, data_global)
        spec_click_line.set_xdata([idx, idx])
        time_str = time_labels_all[idx]
        spec_click_text.set_text(time_str)
        spec_click_text.set_position((idx, 0))
        plt.draw()

    fig.canvas.mpl_connect('key_press_event', on_key)

    # Update the clear_file_highlight function to also clear filename highlighting
    def clear_file_highlight(event):
        global file_patch
        if file_patch:
            file_patch.remove()
            file_patch = None
        # Clear filename highlighting
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

    plt.show()


if __name__ == '__main__':
    root = tk.Tk(); root.withdraw()
    try:
        file_paths = filedialog.askopenfilenames(filetypes=[('Text', '*.txt')])
        if file_paths:
            setup_viewer(file_paths)
    except Exception as e:
        logging.error("An error occurred while launching the viewer", exc_info=True)
        print("An error occurred. Please check the error_log.txt file for details.")
