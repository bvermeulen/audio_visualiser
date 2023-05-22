import sys
import time
import threading
import struct
from pathlib import Path
from enum import auto, IntEnum
from PyQt5 import uic, QtWidgets
import wave
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import pyaudio


DEFAULT_FREQUENCY = 223  # frequency in Hz
DEFAULT_DURATION = 5.0   # length of sound stream in seconds
INTERVAL = 100           # plot interval in millisecond
PACKAGE_LENGTH = 1024    # number of samples in sound package
VOLUME_RESOLUTION = 0.02 # resolution in volume scale (0 - 1)

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

    def __init__(self):
        super().__init__()
        self.audio = pyaudio.PyAudio()
        self.out = ''

    def generate_sound_stream(self):
        if self.selected_type == SoundType.NOTE:
            self.sound_stream = (
                (np.sin(2 * np.pi * self.frequency / self.fs *
                        np.arange(self.fs * self.duration))).astype(np.float32)
            ).astype(np.float32)

        elif self.selected_type == SoundType.DESIGN:
            self.sound_stream = (
                (0.5 * np.sin(2 * np.pi * 325 / self.fs *
                              np.arange(self.fs * self.duration))) +
                (0.1 * np.sin(2 * np.pi * 330 / self.fs *
                              np.arange(self.fs * self.duration))) +
                (0.5 * np.sin(2 * np.pi * 340 / self.fs *
                              np.arange(self.fs * self.duration))) + 0
            ).astype(np.float32)

        elif self.selected_type == SoundType.FILE:
            a = struct.unpack(self.fmt, self.sound_byte_str)
            a = [float(val) for val in a]
            self.sound_stream = np.array(a).astype(np.float32)
            scale_factor = max(abs(np.min(self.sound_stream)),
                               abs(np.max(self.sound_stream)))
            self.sound_stream = self.sound_stream / scale_factor

        else:
            raise ValueError(f'check selected_type invalid value {self.selected_type}')

        # print(f'{np.min(self.sound_stream)}, {np.max(self.sound_stream)}, '
        #       f'{len(self.sound_stream)}')
        self.ax.set_xlim(1000 * PACKAGE_LENGTH / self.fs, 0)

    def callback(self, in_data, frame_count, time_info, status):
        self.out = self.sound_stream[:frame_count]
        self.sound_stream = self.sound_stream[frame_count:]
        return self.out*self.volume, pyaudio.paContinue

    def play_sound(self):
        self.stream = self.audio.open(format=pyaudio.paFloat32,
                                      channels=1,
                                      rate=self.fs,
                                      output=True,
                                      frames_per_buffer=PACKAGE_LENGTH,
                                      stream_callback=self.callback)

        self.stream.start_stream()

        # pause audio when self.running is False; close audio when stopped
        while self.stream.is_active():
            if self.stopped:
                break

            # pause loop
            while not self.running:
                if self.stream.is_active:
                    self.stream.stop_stream()

                if self.stopped:
                    break

            if self.running and not self.stream.is_active():
                self.stream.start_stream()

        self.stream.stop_stream()
        self.stream.close()

        self.running = False
        self.stopped = True
        self.action_start_pause.setText('Start')

    def update_frame(self, frame):
        samples = len(self.out)
        if samples == PACKAGE_LENGTH:
            self.line.set_data(self.xdata, self.out)

        elif samples > 0:
            xdata = np.linspace(0, 1000 * samples / self.fs, samples)
            self.line.set_data(xdata, self.out)

        else:
            return

        elapsed_time = time.time() - self.start_time - self.pause_time
        self.time_progress_bar.setValue(elapsed_time * 1000)

        return self.line,

    def start_visualisation(self):
        self.generate_sound_stream()
        self.xdata = np.linspace(0, 1000 * PACKAGE_LENGTH / self.fs, PACKAGE_LENGTH)
        self.ax.set_ylim(1.1 * np.min(self.sound_stream),
                         1.1 * np.max(self.sound_stream))

        self.line, = self.ax.plot([], [], lw=2)

        duration_range = np.arange(0, self.duration, INTERVAL / 1000)
        self.start_time = time.time()
        self.pause_time = 0
        self.visualisation = FuncAnimation(self.fig,
                                           self.update_frame,
                                           frames=duration_range,
                                           interval=INTERVAL,
                                           repeat=False)

        # start audio deamon in a seperate thread as otherwise audio and
        # plot will not be at the same time
        x = threading.Thread(target=self.play_sound)
        x.daemon = True
        x.start()

        # self.root.after(1, self.fig.canvas.draw())


class PyqtMainWindow(QtWidgets.QMainWindow, SoundVisualiser):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        uic.loadUi(Path(__file__).parent / 'audio_visualiser.ui', self)
        self.action_exit.triggered.connect(self.quit)
        self.action_start_pause.triggered.connect(self.control_start_pause)
        self.action_stop.triggered.connect(self.stop)
        self.rb_note.toggled.connect(self.note_options)
        self.rb_design.toggled.connect(self.design_options)
        self.rb_file.toggled.connect(self.file_options)
        self.rb_2048.toggled.connect(lambda: self.set_sampling_rate(SamplingRate.sr2048))
        self.rb_4096.toggled.connect(lambda: self.set_sampling_rate(SamplingRate.sr4096))
        self.rb_8192.toggled.connect(lambda: self.set_sampling_rate(SamplingRate.sr8192))
        self.rb_16384.toggled.connect(lambda: self.set_sampling_rate(SamplingRate.sr16384))
        self.rb_32768.toggled.connect(lambda: self.set_sampling_rate(SamplingRate.sr32768))
        self.volume_slider.setMinimum(0)
        self.volume_slider.setMaximum(100)
        self.volume_slider.setTickInterval(5)
        self.volume_slider.valueChanged.connect(self.set_volume)

        self.duration = 100
        self.running = False
        self.stopped = True
        self.error_message = ''
        self.pause_time = None
        self.visualisation = None
        self.plot_area()
        self.rb_note.setChecked(True)
        self.rb_4096.setChecked(True)
        self.volume_slider.setValue(25)

    def plot_area(self):
        self.fig, self.ax = plt.subplots(figsize=(5, 4))
        canvas = FigureCanvasTkAgg(self.fig)

    def control_start_pause(self):
        if self.error_message:
            return

        if self.stopped:
            try:
                self.ax.lines.pop(0)

            except IndexError:
                pass

            self.set_frequency_duration()

            # TODO: the minus 1 second from self.duration is to align the playing of the
            # sound and the progress bar, under investigation
            self.time_progress_bar.setMaximum = 1000 * (self.duration - 1.0)
            self.time_progress_bar.setValue = 0
            self.running = True
            self.stopped = False
            self.action_start_pause.setText('Pause')
            self.start_visualisation()  #pylint: disable=no-member
            return

        if self.running:
            self.visualisation.event_source.stop()
            self.pause_start_time = time.time()
            self.action_start_pause.setText('Run')

        else:
            self.pause_time += time.time() - self.pause_start_time
            self.visualisation.event_source.start()
            self.action_start_pause.setText('Pause')

        self.running = not self.running

    def stop(self):
        try:
            self.visualisation.event_source.stop()

        except AttributeError:
            # just pass as stop button was pressed before visualisation had
            # actually started
            pass

        self.stopped = True

    def quit(self):
        self.stop()
        time.sleep(1)
        sys.exit()

    def set_frequency_duration(self):
        if self.selected_type == SoundType.NOTE:
            try:
                self.frequency = int(self.input_note_frequency.text())

            except ValueError:
                self.frequency = DEFAULT_FREQUENCY
                self.input_note_frequency.setText(str(DEFAULT_FREQUENCY))

            try:
                self.duration = float(self.input_note_duration.text())

            except ValueError:
                self.duration = DEFAULT_DURATION
                self.input_note_duration.setText(str(DEFAULT_DURATION))

        if self.selected_type == SoundType.DESIGN:
            try:
                self.duration = float(self.input_design_duration.text())

            except ValueError:
                self.duration = DEFAULT_DURATION
                self.input_design_duration.setText(str(DEFAULT_DURATION))

    def note_options(self):
        self.selected_type = SoundType.NOTE
        self.stacked_widget_sound_type.setCurrentIndex(0)
        self.stacked_widget_status.setCurrentIndex(0)
        self.set_frequency_duration()

    def design_options(self):
        self.selected_type = SoundType.DESIGN
        self.stacked_widget_sound_type.setCurrentIndex(1)
        self.stacked_widget_status.setCurrentIndex(0)
        self.set_frequency_duration()

    def file_options(self):
        self.selected_type = SoundType.FILE
        self.stacked_widget_sound_type.setCurrentIndex(2)
        self.stacked_widget_status.setCurrentIndex(1)

        # for widget in self.type_frame.winfo_children():
        #     widget.destroy()

        # sound_file = filedialog.askopenfile(
        #     title='Select sound file',
        #     filetypes=(('wav files', '*.wav'), ('all files', '*')))

        # try:
        #     file_name_text = f'Sound file: {Path(sound_file.name).name}  '

        # except AttributeError:
        #     self.error_message = 'No file selected ...'
        #     Label(self.type_frame, text=self.error_message).pack(anchor='w')
        #     file_name_text = ''

        # Label(self.type_frame, text=file_name_text).pack(anchor='w')

        # if file_name_text:
        #     try:
        #         w = wave.open(sound_file.name)
        #     except (wave.Error, EOFError, AttributeError):
        #         self.error_message = 'Invalid wav file ...'
        #         Label(self.type_frame, text=self.error_message).pack(anchor='w')
        #         return

        #     frames = w.getnframes()
        #     channels = w.getnchannels()
        #     sample_width = w.getsampwidth()
        #     self.fs = w.getframerate()
        #     self.sound_byte_str = w.readframes(frames)
        #     self.duration = frames / self.fs * channels

        #     if sample_width == 1:
        #         self.fmt = f'{int(frames * channels)}B'

        #     else:
        #         self.fmt = f'{int(frames * channels)}h'

        #     print(f'frames: {frames}, channels: {channels}, '
        #           f'sample width: {sample_width}, framerate: {self.fs}')

    def set_sampling_rate(self, sampling_rate):
        self.fs = 2**sampling_rate
        print(f'{self.fs = }')
        self.ax.set_xlim(1000 * PACKAGE_LENGTH / self.fs, 0)


    def control_volume_time(self):
        ...

        # volume_outer_frame = Frame(self.control_frame, bd=1, relief='groove')
        # volume_outer_frame.grid(
        #     row=2, column=0, sticky='ew', pady=PADY_OUTER)

        # Label(volume_outer_frame, text='Volume').grid(
        #     row=0, column=0, stick='w', pady=PADY_INNER_1)

        # volume_slider = Scale(volume_outer_frame,
        #                       from_=0, to_=1, resolution=VOLUME_RESOLUTION,
        #                       orient='horizontal',
        #                       command=self.set_volume,
        #                       showvalue=0,
        #                      )
        # volume_slider.set(self.volume)
        # volume_slider.grid(row=1, column=0, sticky='w', pady=PADY_INNER_3)

        # Label(volume_outer_frame, text='Time').grid(
        #     row=0, column=1, stick='w', pady=PADY_INNER_1, padx=(20, 0))

        # self.time_progress = ttk.Progressbar(volume_outer_frame,
        #                                      orient='horizontal',
        #                                      length=100,
        #                                      mode='determinate'
        #                                     )
        # self.time_progress.grid(
        #     row=1, column=1, sticky='w', pady=PADY_INNER_3, padx=(20, 0))

    def set_volume(self, value):
        self.volume = float(value) / 200.0


def start_app():
    app = QtWidgets.QApplication([])
    main_window = PyqtMainWindow()
    main_window.show()
    app.exec()


if __name__ == '__main__':
    start_app()
