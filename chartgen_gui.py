import tkinter as tk
import threading
import os
from tkinter import filedialog, messagebox
from tkinter import ttk

from models import generate_charts

class TqdmTkProgress:
    def __init__(self, root, progressbar, total):
        self.root = root
        self.progressbar = progressbar
        self.total = total
        self.current = 0
        self.progressbar["mode"] = "determinate"
        self.progressbar["maximum"] = total
        self.progressbar["value"] = 0

    def update(self, n=1):
        self.current += n
        self.root.after(0, lambda: self.progressbar.configure(value=self.current))

    def close(self):
        self.root.after(0, lambda: self.progressbar.configure(value=self.total))



class ChartGenApp:
    def __init__(self, root):
        self.root = root
        root.title("DDCL Chart Generation")

        self.params = {
            'onset_model_fp': tk.StringVar(value='trained_models/onset_model.keras'),
            'sym_model_fp': tk.StringVar(value='trained_models/sym_model.keras'),
            'batch_size': tk.IntVar(value=32),
            'model_frame_density': tk.IntVar(value=32),
            'onset_history_len': tk.IntVar(value=15),
            'threshold': tk.DoubleVar(value=0.5),
            'in_directory': tk.StringVar(value='input_songs_for_generation'),
            'out_directory': tk.StringVar(value='generated_charts'),
            'diffs': tk.StringVar(value='Beginner,Easy,Medium,Hard,Challenge'),
            'maxstep': tk.IntVar(value=12),
            'use_song_length': tk.BooleanVar(value=False),
            'bpm_method': tk.StringVar(value='DDCL')
        }

        self.create_widgets()

    def create_widgets(self):
        frame = ttk.Frame(self.root, padding=10)
        frame.grid(row=0, column=0, sticky='nsew')

        def add_entry(label_text, var, row, browse=False, directory=False):
            ttk.Label(frame, text=label_text).grid(row=row, column=0, sticky='w')
            entry = ttk.Entry(frame, textvariable=var, width=50)
            entry.grid(row=row, column=1, sticky='ew')
            if browse:
                def browse_fn():
                    if directory:
                        path = filedialog.askdirectory()
                    else:
                        path = filedialog.askopenfilename()
                    if path:
                        var.set(path)
                ttk.Button(frame, text="Browse", command=browse_fn).grid(row=row, column=2)

        row = 0
        add_entry("Onset Model File", self.params['onset_model_fp'], row, browse=True)
        row += 1
        add_entry("Symbolic Model File", self.params['sym_model_fp'], row, browse=True)
        row += 1
        add_entry("Input Directory", self.params['in_directory'], row, browse=True, directory=True)
        row += 1
        add_entry("Output Directory", self.params['out_directory'], row, browse=True, directory=True)
        row += 1

        ttk.Label(frame, text="Batch Size").grid(row=row, column=0, sticky='w')
        ttk.Entry(frame, textvariable=self.params['batch_size']).grid(row=row, column=1)
        row += 1

        ttk.Label(frame, text="Model Frame Density").grid(row=row, column=0, sticky='w')
        ttk.Entry(frame, textvariable=self.params['model_frame_density']).grid(row=row, column=1)
        row += 1

        ttk.Label(frame, text="Onset History Length").grid(row=row, column=0, sticky='w')
        ttk.Entry(frame, textvariable=self.params['onset_history_len']).grid(row=row, column=1)
        row += 1

        ttk.Label(frame, text="Threshold").grid(row=row, column=0, sticky='w')
        ttk.Entry(frame, textvariable=self.params['threshold']).grid(row=row, column=1)
        row += 1

        self.difficulty_vars = {
            'Beginner': tk.BooleanVar(value=True),
            'Easy': tk.BooleanVar(value=True),
            'Medium': tk.BooleanVar(value=True),
            'Hard': tk.BooleanVar(value=True),
            'Challenge': tk.BooleanVar(value=True)  # or "Expert" if you prefer
        }

        ttk.Label(frame, text="Select Difficulties:").grid(row=row, column=0, sticky='nw')
        diff_frame = ttk.Frame(frame)
        diff_frame.grid(row=row, column=1, sticky='w')

        for i, (label, var) in enumerate(self.difficulty_vars.items()):
            ttk.Checkbutton(diff_frame, text=label, variable=var).grid(row=i, column=0, sticky='w')
        row += 1
        ttk.Label(frame, text="BPM Method").grid(row=row, column=0, sticky='w')
        ttk.Combobox(frame, textvariable=self.params['bpm_method'], values=['DDCL', 'AV', 'SMEdit']).grid(row=row, column=1)
        row += 1
        self.progress = ttk.Progressbar(frame, orient='horizontal', mode='determinate', length=300)
        self.progress.grid(row=row, column=0, columnspan=3, pady=5)
        row += 1
        ttk.Button(frame, text="Generate Charts", command=self.start_generation).grid(row=row, column=0, columnspan=3, pady=10)

    def run_generation_thread(self):
        try:
            diffs = [label for label, var in self.difficulty_vars.items() if var.get()]
            in_dir = self.params['in_directory'].get()
            song_count = len([f for f in os.listdir(in_dir) if f.endswith(('.mp3', '.wav', '.ogg', '.aiff'))])

            progress_tracker = TqdmTkProgress(self.root, self.progress, song_count)

            generate_charts(
                onset_model_fp=self.params['onset_model_fp'].get(),
                sym_model_fp=self.params['sym_model_fp'].get(),
                batch_size=self.params['batch_size'].get(),
                model_frame_density=self.params['model_frame_density'].get(),
                onset_history_len=self.params['onset_history_len'].get(),
                threshold=self.params['threshold'].get(),
                in_directory=in_dir,
                out_directory=self.params['out_directory'].get(),
                diffs=diffs,
                maxstep=self.params['maxstep'].get(),
                use_song_length=self.params['use_song_length'].get(),
                bpm_method=self.params['bpm_method'].get(),
                progress_callback=progress_tracker  # NEW
            )

            self.root.after(0, lambda: messagebox.showinfo("Success", "Chart generation completed."))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"An error occurred: {str(e)}"))
        finally:
            self.root.after(0, self.enable_ui)

    def start_generation(self):
        self.progress["value"] = 0
        self.disable_ui()
        threading.Thread(target=self.run_generation_thread).start()

    def disable_ui(self):
        for child in self.root.winfo_children():
            self._set_state_recursive(child, 'disabled')

    def enable_ui(self):
        for child in self.root.winfo_children():
            self._set_state_recursive(child, 'normal')

    def _set_state_recursive(self, widget, state):
        try:
            widget.configure(state=state)
        except:
            pass
        for child in widget.winfo_children():
            self._set_state_recursive(child, state)

def main():
    root = tk.Tk()
    app = ChartGenApp(root)
    root.mainloop()

if __name__ == '__main__':
    main()
