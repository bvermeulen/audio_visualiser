import sys
import time
import threading
import struct
from pathlib import Path
from enum import auto, IntEnum
from tkinter import (
    Tk, TclError, Frame, Label, Button, Radiobutton, Scale, Entry, ttk,
    filedialog, IntVar
)
import wave
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import pyaudio


DEFAULT_FREQUENCY = 223  # frequency in Hz
DEFAULT_DURATION = 5.0   # length of sound stream in seconds
INTERVAL = 100           # plot interval in millisecond
PACKAGE_LENGTH = 1024    # number of samples in sound package
VOLUME_RESOLUTION = 0.02 # resolution in volume scale (0 - 1)

PADY_OUTER = (2, 15)
PADX_OUTER = (5, 10)
PADY_INNER_1 = (2, 4)
PADY_INNER_2 = (0, 4)
PADY_INNER_3 = (0, 10)


class SoundType(IntEnum):
    NOTE = auto()
    DESIGN = auto()
    FILE = auto()


class SamplingRate(IntEnum):
    sr2048 = 11
    sr4096 = 12
    sr8192 = 13
    sr16384 = 14
    sr32768 = 15


class SoundVisualiser:
    ''' Methods to play and visualise audio sound
    '''
    def __init__(self, width=5, height=4, dpi=100):
        self.audio = pyaudio.PyAudio()
        self.out = ''
        self._fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        self._canvas = None
        self.fs = None
        self.stream = None
        self._start_time = None
        self._volume = None
        self.full_sound_stream = None
        self.sound_stream = None

    @property
    def volume(self):
        return self._volume

    @volume.setter
    def volume(self, value):
        self._volume = value

    @property
    def fig(self):
        return self._fig

    @property
    def canvas(self):
        return self._canvas

    @canvas.setter
    def canvas(self, value):
        self._canvas = value

    def generate_sound_stream(self, selected_type, frequency, fs, duration, wave_data=None):
        self.fs = fs
        self.duration = duration
        match selected_type:
            case SoundType.NOTE:
                self.full_sound_stream = (
                    np.sin(2 * np.pi * frequency / self.fs * np.arange(fs * duration))
                ).astype(np.float32)

            case SoundType.DESIGN:
                self.full_sound_stream = (
                    (0.5 * np.sin(2 * np.pi * 325 / self.fs * np.arange(fs * duration))) +
                    (0.1 * np.sin(2 * np.pi * 330 / self.fs * np.arange(fs * duration))) +
                    (0.5 * np.sin(2 * np.pi * 340 / self.fs * np.arange(fs * duration))) + 0
                ).astype(np.float32)

            case SoundType.FILE:
                if wave_data is None:
                    return self.fs, self.duration

                frames = wave_data.getnframes()
                channels = wave_data.getnchannels()
                sample_width = wave_data.getsampwidth()
                self.fs = wave_data.getframerate()
                sound_byte_str = wave_data.readframes(frames)
                self.duration = frames / self.fs * channels

                if sample_width == 1:
                    self.fmt = f'{int(frames * channels)}B'

                else:
                    self.fmt = f'{int(frames * channels)}h'

                a = struct.unpack(self.fmt, sound_byte_str)
                a = [float(val) for val in a]
                sound_stream = np.array(a).astype(np.float32)
                scale_factor = max(
                    abs(np.min(sound_stream)), abs(np.max(sound_stream))
                )
                self.full_sound_stream = sound_stream / scale_factor

            case other:
                raise ValueError(f'check selected_type invalid value {self.selected_type}')

        self.ax.set_ylim(
            1.1 * np.min(self.full_sound_stream), 1.1 * np.max(self.full_sound_stream)
        )
        self.ax.set_xlim(1000 * PACKAGE_LENGTH / self.fs, 0)
        self.canvas.draw()
        return self.fs, self.duration

    def callback(self, in_data, frame_count, time_info, status):
        self.out = self.sound_stream[:frame_count]
        self.sound_stream = self.sound_stream[frame_count:]
        return self.out*self.volume, pyaudio.paContinue

    def update_frame(self, frame):
        samples = len(self.out)
        if samples == PACKAGE_LENGTH:
            self.line.set_data(self.xdata, self.out)

        elif samples > 0:
            xdata = np.linspace(0, 1000 * samples / self.fs, samples)
            self.line.set_data(xdata, self.out)

        else:
            return

        self.canvas.draw()

    def init_sound_visualisation(self):
        self.stream = self.audio.open(
            format=pyaudio.paFloat32, channels=1, rate=self.fs, output=True,
            frames_per_buffer=PACKAGE_LENGTH, stream_callback=self.callback
        )
        self.sound_stream = self.full_sound_stream
        self.xdata = np.linspace(0, 1000 * PACKAGE_LENGTH / self.fs, PACKAGE_LENGTH)
        self.line, = self.ax.plot([], [], lw=2)

        duration_range = np.arange(0, self.duration, INTERVAL / 1000)
        self.visualisation = FuncAnimation(
            self.fig, self.update_frame, frames=duration_range, interval=INTERVAL, repeat=False
        )
        self.canvas.draw()
        return time.time()

    def control_sound_stream(self, control_word):
        match control_word:
            case 'is_active':
                return self.stream.is_active()

            case 'start':
                self.stream.start_stream()

            case 'stop':
                self.stream.stop_stream()

            case 'close':
                self.stream.close()

            case other:
                raise ValueError(f'Incorrect control word given: {control_word}')

    def control_visualiser(self, control_word):
        match control_word:
            case 'init':
                return self.init_sound_visualisation()

            case 'start':
                try:
                    self.visualisation.event_source.start()
                except AttributeError:
                    pass

            case 'stop':
                try:
                    self.visualisation.event_source.stop()
                except AttributeError:
                    pass

            case other:
                raise ValueError(f'Incorrect control word given: {control_word}')

    def set_plot(self, fs):
        self.ax.set_ylim(-1.1, 1.1)
        self.ax.set_xlim(1000 * PACKAGE_LENGTH / fs, 0)
        self.canvas.draw()

    def remove_plot(self):
        for line in self.ax.lines:
            line.remove()

        self.canvas.draw()


class TkViewControl(Tk):
    ''' Tkinter view and control of audio visualiser
    '''
    def __init__(self, sv):
        super().__init__()
        self.sv = sv
        self.geometry('800x500')
        self.protocol('WM_DELETE_WINDOW', self.quit)

        self.title("Sound Visualiser")
        self.columnconfigure(0, weight=1)

        self.duration = 100
        self.set_volume(0.25)
        self.fs = None
        self.running = False
        self.stopped = True
        self.error_message = ''
        self.start_time = None
        self.pause_start_time = None
        self.pause_time = None
        self.poll_id = None
        self.plot_area()
        self.main_buttons()
        self.control_buttons()

    def plot_area(self):
        plot_frame = Frame(self)
        plot_frame.grid(row=0, column=0, sticky='nw')
        canvas = FigureCanvasTkAgg(self.sv.fig, plot_frame)
        canvas.get_tk_widget().pack()
        self.sv.canvas = canvas

    def main_buttons(self):
        bottom_frame = Frame(self)
        bottom_frame.grid(row=1, column=0, rowspan=2, sticky='new')

        self.start_pause_button = Button(
            bottom_frame, text='Start', command=self.control_start_pause)
        self.start_pause_button.pack(side='left')

        self.stop_button = Button(
            bottom_frame, text='Stop', command=self.stop)
        self.stop_button.pack(side='left')

        self.quit_button = Button(
            bottom_frame, text='Quit', command=self.quit)
        self.quit_button.pack(side='left')

    @staticmethod
    def read_entry(entry_field, default_value):
        try:
            value = float(entry_field.get())

        except ValueError:
            value = default_value
            entry_field.insert(0, default_value)

        except TclError:
            value = default_value

        return value

    def control_start_pause(self):
        if self.error_message:
            return

        if self.stopped:
            # start playing and visualising sound
            self.sv.remove_plot()

            if self.selected_type == SoundType.NOTE:
                frequency = self.read_entry(self.frequency_entry, DEFAULT_FREQUENCY)
                self.duration = self.read_entry(self.duration_entry, DEFAULT_DURATION)
                self.sv.generate_sound_stream(self.selected_type, frequency, self.fs, self.duration)

            if self.selected_type == SoundType.DESIGN:
                self.duration = self.read_entry(self.duration_entry, DEFAULT_DURATION)
                self.sv.generate_sound_stream(self.selected_type, 0, self.fs, self.duration)

            # TODO: the minus 1 second from self.duration is to align the playing of the
            # sound and the progress bar, under investigation
            self.time_progress['maximum'] = 1000 * (self.duration - 1.0)
            self.time_progress['value'] = 0
            self.running = True
            self.stopped = False
            self.start_time = self.sv.control_visualiser('init')
            self.pause_time = 0.0
            self.update_progress_bar()

            # start audio deamon in a seperate thread as otherwise audio and
            # plot will not be at the same time
            self.x = threading.Thread(target=self.play_sound)
            self.x.daemon = True
            self.x.start()

            self.start_pause_button.config(text='Pause')

        else:
            # sound is playing: if running then pause, or if pause run
            if self.running:
                self.sv.control_visualiser('stop')
                self.pause_start_time = time.time()
                self.start_pause_button.config(text='Run')
                self.stop_progress_bar()

            else:
                self.pause_time += time.time() - self.pause_start_time
                self.sv.control_visualiser('start')
                self.start_pause_button.config(text='Pause')
                self.update_progress_bar()

            self.running = not self.running

    def play_sound(self):
        # pause audio when self.running is False; close audio when stopped
        while self.sv.control_sound_stream('is_active'):
            if self.stopped:
                break

            # pause loop
            while not self.running:
                if self.sv.control_sound_stream('is_active'):
                    self.sv.control_sound_stream('stop')

                if self.stopped:
                    break

            if self.running and not self.sv.control_sound_stream('is_active'):
                self.sv.control_sound_stream('start')

        # sound has stopped, reset values to start again
        self.sv.control_sound_stream('stop')
        self.sv.control_sound_stream('close')
        self.running = False
        self.stopped = True
        self.stop_progress_bar()
        self.start_pause_button.config(text='Start')

    def update_progress_bar(self):
        elapsed_time = time.time() - self.start_time - self.pause_time
        self.time_progress['value'] = elapsed_time * 1000
        self.poll_id = self.after(250, self.update_progress_bar)

    def stop_progress_bar(self):
        self.after_cancel(self.poll_id)

    def stop(self):
        self.sv.control_visualiser('stop')
        self.stopped = True

    def quit(self):
        # TODO: check if all these delays are necessary for the processes to
        # stop in an orderly fashion
        self.stop()
        # TODO: check need for self.audo.terminate()
        time.sleep(1)
        self.after(3, self.destroy)
        sys.exit()

    def control_buttons(self):
        self.control_frame = Frame(self)
        self.control_frame.grid(row=0, column=1, sticky='nw')

        self.control_wave_type()
        self.control_sampling_rate()
        self.control_volume_time()

        self.r_type.set(1)
        self.select_type()

    def control_wave_type(self):
        type_outer_frame = Frame(self.control_frame, bd=1, relief='groove')
        type_outer_frame.grid(
            row=0, column=0, sticky='ew', pady=PADY_OUTER, padx=PADX_OUTER
        )
        self.r_type = IntVar()

        Label(type_outer_frame, text='Sound type').grid(
            row=0, column=0, stick='w', pady=PADY_INNER_1)

        for column, mode in enumerate(SoundType):
            Radiobutton(
                type_outer_frame, text=mode.name.lower(), width=6, variable=self.r_type,
                value=mode.value, command=self.select_type
            ).grid(row=1, column=column, stick='w', pady=PADY_INNER_2)

        self.type_frame = Frame(type_outer_frame)
        self.type_frame.grid(
            row=2, column=0, columnspan=3, sticky='w', pady=PADY_INNER_3)

    def select_type(self):
        self.error_message = ''

        self.selected_type = self.r_type.get()

        if self.selected_type == SoundType.NOTE:
            self.note_options()

        elif self.selected_type == SoundType.DESIGN:
            self.design_options()

        elif self.selected_type == SoundType.FILE:
            self.file_options()

        else:
            raise ValueError(f'check selected_type invalid value {self.selected_type}')

        self.select_sampling_display()

    def note_options(self):
        for widget in self.type_frame.winfo_children():
            widget.destroy()

        Label(self.type_frame, text='Frequency').pack(side='left')
        self.frequency_entry = Entry(self.type_frame, width=5)
        self.frequency_entry.insert(0, DEFAULT_FREQUENCY)
        self.frequency_entry.pack(side='left')

        Label(self.type_frame, text='Duration').pack(side='left')
        self.duration_entry = Entry(self.type_frame, width=5)
        self.duration_entry.insert(0, DEFAULT_DURATION)
        self.duration_entry.pack(side='left')

    def design_options(self):
        for widget in self.type_frame.winfo_children():
            widget.destroy()

        Label(self.type_frame, text='Duration').pack(side='left')
        self.duration_entry = Entry(self.type_frame, width=5)
        self.duration_entry.insert(0, DEFAULT_DURATION)
        self.duration_entry.pack(side='left')

    def file_options(self):
        for widget in self.type_frame.winfo_children():
            widget.destroy()

        sound_file = filedialog.askopenfile(
            title='Select sound file',
            filetypes=(('wav files', '*.wav'), ('all files', '*'))
        )
        try:
            sound_file = Path(sound_file.name)
            file_name_text = f'Sound file: {sound_file.name}  '

        except AttributeError:
            self.error_message = 'No file selected ...'
            Label(self.type_frame, text=self.error_message).pack(anchor='w')
            file_name_text = ''

        Label(self.type_frame, text=file_name_text).pack(anchor='w')

        if file_name_text:

            try:
                wave_data = wave.open(str(sound_file))

            except (wave.Error, EOFError, AttributeError):
                self.error_message = 'Invalid wav file ...'
                Label(self.type_frame, text=self.error_message).pack(anchor='w')

            self.fs, self.duration = self.sv.generate_sound_stream(
                self.selected_type, 0, 0, 0, wave_data=wave_data
            )

    def control_sampling_rate(self):
        sampling_outer_frame = Frame(self.control_frame, bd=1, relief='groove')
        sampling_outer_frame.grid(
            row=1, column=0, sticky='ew', pady=PADY_OUTER, padx=PADX_OUTER
        )
        Label(sampling_outer_frame, text='Sampling frequency').grid(
            row=0, column=0, stick='w', pady=PADY_INNER_1)

        self.sampling_frame = Frame(sampling_outer_frame)
        self.sampling_frame.grid(row=1, column=0, stick='w', pady=PADY_INNER_3)

    def select_sampling_display(self):
        if self.selected_type in [SoundType.NOTE, SoundType.DESIGN]:
            self.display_sampling_options()

        elif self.selected_type == SoundType.FILE:
            self.display_sampling_rate()

        else:
            raise ValueError(f'check selected_type invalid value {self.selected_type}')

    def display_sampling_options(self):
        for widget in self.sampling_frame.winfo_children():
            widget.destroy()

        self.r_fs = IntVar()
        self.r_fs.set(14)
        self.select_fs()

        for i, sampling_rate in enumerate(SamplingRate):
            Radiobutton(
                self.sampling_frame, text=sampling_rate.name[2:],
                width=6, variable=self.r_fs, value=sampling_rate.value,
                command=self.select_fs
            ).grid(row=int(i / 3), column=(i % 3), sticky='w')

    def select_fs(self):
        self.fs = 2**self.r_fs.get()
        self.sv.set_plot(self.fs)

    def display_sampling_rate(self):
        for widget in self.sampling_frame.winfo_children():
            widget.destroy()

        Label(self.sampling_frame, text=f'Sampling rate: {self.fs} Hz').grid(
            row=0, column=0, stick='w')

        Label(self.sampling_frame, text=f'Duration: {self.duration:.1f} seconds').grid(
            row=1, column=0, stick='w')

    def control_volume_time(self):
        volume_outer_frame = Frame(self.control_frame, bd=1, relief='groove')
        volume_outer_frame.grid(
            row=2, column=0, sticky='ew', pady=PADY_OUTER, padx=PADX_OUTER
        )
        Label(volume_outer_frame, text='Volume').grid(
            row=0, column=0, stick='w', pady=PADY_INNER_1)

        volume_slider = Scale(
            volume_outer_frame, from_=0, to_=1, resolution=VOLUME_RESOLUTION,
            orient='horizontal', command=self.set_volume, showvalue=0,
        )
        volume_slider.set(self.volume)
        volume_slider.grid(row=1, column=0, sticky='w', pady=PADY_INNER_3)

        Label(volume_outer_frame, text='Time').grid(
            row=0, column=1, stick='w', pady=PADY_INNER_1, padx=(20, 0))

        self.time_progress = ttk.Progressbar(
            volume_outer_frame, orient='horizontal', length=100, mode='determinate'
        )
        self.time_progress.grid(
            row=1, column=1, sticky='w', pady=PADY_INNER_3, padx=(20, 0))

    def set_volume(self, value):
        self.volume = float(value)
        self.sv.volume = self.volume


def app():
    sv = SoundVisualiser()
    control_view = TkViewControl(sv)
    control_view.mainloop()


if __name__ == '__main__':
    app()
