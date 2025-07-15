import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

from models import generate_charts

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

        ttk.Button(frame, text="Generate Charts", command=self.run_generation).grid(row=row, column=0, columnspan=3, pady=10)

    def run_generation(self):
        try:
            diffs = [label for label, var in self.difficulty_vars.items() if var.get()]
            generate_charts(
                onset_model_fp=self.params['onset_model_fp'].get(),
                sym_model_fp=self.params['sym_model_fp'].get(),
                batch_size=self.params['batch_size'].get(),
                model_frame_density=self.params['model_frame_density'].get(),
                onset_history_len=self.params['onset_history_len'].get(),
                threshold=self.params['threshold'].get(),
                in_directory=self.params['in_directory'].get(),
                out_directory=self.params['out_directory'].get(),
                diffs=diffs,
                maxstep=self.params['maxstep'].get(),
                use_song_length=self.params['use_song_length'].get(),
                bpm_method=self.params['bpm_method'].get()
            )
            messagebox.showinfo("Success", "Chart generation completed.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

def main():
    root = tk.Tk()
    app = ChartGenApp(root)
    root.mainloop()

if __name__ == '__main__':
    main()
