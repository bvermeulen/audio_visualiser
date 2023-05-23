import sys
import time
import threading
import struct
from pathlib import Path
from enum import auto, IntEnum
from PyQt5 import uic, QtWidgets
from PyQt5.QtCore import QTimer
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.animation import FuncAnimation
import wave
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


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)

    def get_fig_ax(self):
        return self.fig, self.ax


class SoundVisualiser:

    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.out = ''
        self._canvas = MplCanvas(self)
        self.fig, self.ax = self.canvas.get_fig_ax()
        self.fs = None
        self._stream = None
        self._start_time = None
        self._volume = None

    @property
    def volume(self):
        return self._volume

    @volume.setter
    def volume(self, value):
        self._volume = value

    @property
    def canvas(self):
        return self._canvas

    @property
    def stream(self):
        return self._stream

    def generate_sound_stream(self, selected_type, frequency, fs, duration, wave_data=None):
        self.fs = fs
        self.duration = duration
        if selected_type == SoundType.NOTE:
            self.full_sound_stream = (
                (np.sin(2 * np.pi * frequency / self.fs * np.arange(fs * duration))).astype(np.float32)
            ).astype(np.float32)

        elif selected_type == SoundType.DESIGN:
            self.full_sound_stream = (
                (0.5 * np.sin(2 * np.pi * 325 / self.fs * np.arange(fs * duration))) +
                (0.1 * np.sin(2 * np.pi * 330 / self.fs * np.arange(fs * duration))) +
                (0.5 * np.sin(2 * np.pi * 340 / self.fs * np.arange(fs * duration))) + 0
            ).astype(np.float32)

        elif selected_type == SoundType.FILE:
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
            self.sound_stream = np.array(a).astype(np.float32)
            scale_factor = max(
                abs(np.min(self.sound_stream)), abs(np.max(self.sound_stream))
            )
            self.full_sound_stream = self.sound_stream / scale_factor

        else:
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

    def start_visualisation(self):
        self._stream = self.audio.open(
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

    def set_plot(self, fs):
        self.ax.set_ylim(-1.1, 1.1)
        self.ax.set_xlim(1000 * PACKAGE_LENGTH / fs, 0)
        self.canvas.draw()

    def remove_plot(self):
        try:
            self.ax.lines.pop(0)

        except IndexError:
            pass

        self.canvas.draw()


class PyqtMainWindow(QtWidgets.QMainWindow):

    def __init__(self, sv_instance, *args, **kwargs):
        super().__init__(*args, **kwargs)
        uic.loadUi(Path(__file__).parent / 'audio_visualiser.ui', self)
        self.sv = sv_instance
        self.action_exit.triggered.connect(self.quit)
        self.action_start_pause.triggered.connect(self.control_start_pause)
        self.action_stop.triggered.connect(self.stop)
        self.rb_note.clicked.connect(self.note_options)
        self.rb_design.clicked.connect(self.design_options)
        self.rb_file.clicked.connect(self.file_options)
        self.rb_2048.toggled.connect(lambda: self.set_sampling_rate(SamplingRate.sr2048))
        self.rb_4096.toggled.connect(lambda: self.set_sampling_rate(SamplingRate.sr4096))
        self.rb_8192.toggled.connect(lambda: self.set_sampling_rate(SamplingRate.sr8192))
        self.rb_16384.toggled.connect(lambda: self.set_sampling_rate(SamplingRate.sr16384))
        self.rb_32768.toggled.connect(lambda: self.set_sampling_rate(SamplingRate.sr32768))
        self.volume_slider.setMinimum(0)
        self.volume_slider.setMaximum(100)
        self.volume_slider.setTickInterval(5)
        self.volume_slider.valueChanged.connect(self.set_volume)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_progress_bar)
        self.timer.stop()
        self.start_time = None
        self.duration = 100
        self.running = False
        self.stopped = True
        self.error_message = None
        self.pause_time = 0.0
        self.visualisation = None
        self.start_time = None
        self.rb_4096.setChecked(True)
        self.volume_slider.setValue(25)
        self.plot_layout.addWidget(self.sv.canvas)
        self.rb_note.setChecked(True)
        self.note_options()

    def control_start_pause(self):
        if self.error_message:
            return

        if self.stopped:
            self.sv.remove_plot()
            self.set_frequency_duration()

            self.time_progress_bar.setValue(0)
            self.running = True
            self.stopped = False
            self.action_start_pause.setText('Pause')
            self.start_time = self.sv.start_visualisation()  #pylint: disable=no-member
            self.timer.start(250)

            # start audio deamon in a seperate thread as otherwise audio and
            # plot will not be at the same time
            self.x = threading.Thread(target=self.play_sound)
            self.x.daemon = True
            self.x.start()
            return

        if self.running:
            self.sv.visualisation.event_source.stop()
            self.pause_start_time = time.time()
            self.action_start_pause.setText('Run')
            self.timer.stop()

        else:
            self.pause_time += time.time() - self.pause_start_time
            self.sv.visualisation.event_source.start()
            self.action_start_pause.setText('Pause')
            self.timer.start(250)

        self.running = not self.running

    def play_sound(self):
        # pause audio when self.running is False; close audio when stopped
        while self.sv.stream.is_active():
            if self.stopped:
                break

            # pause loop
            while not self.running:
                if self.sv.stream.is_active:
                    self.sv.stream.stop_stream()

                if self.stopped:
                    break

            if self.running and not self.sv.stream.is_active():
                self.sv.stream.start_stream()

        self.sv.stream.stop_stream()
        self.sv.stream.close()

        self.running = False
        self.stopped = True
        self.action_start_pause.setText('Start')

    def stop(self):
        try:
            self.sv.visualisation.event_source.stop()

        except AttributeError:
            # just pass as stop button was pressed before visualisation had
            # actually started
            pass
        self.stopped = True
        self.timer.stop()

    def update_progress_bar(self):
        elapsed_time = 100 * (time.time() - self.pause_time - self.start_time) / self.duration
        self.time_progress_bar.setValue(int(elapsed_time))

    def quit(self):
        self.stop()
        time.sleep(1)
        sys.exit()

    def set_sampling_rate(self, sampling_rate):
        self.fs = 2**sampling_rate
        self.sv.set_plot(self.fs)

    def set_frequency_duration(self):
        if self.selected_type == SoundType.NOTE:
            try:
                self.frequency = int(self.input_note_frequency.text())
                if self.frequency < 10 or self.frequency > 10_000:
                    self.frequency = DEFAULT_FREQUENCY

            except ValueError:
                self.frequency = DEFAULT_FREQUENCY

            try:
                self.duration = float(self.input_note_duration.text())
                if self.duration < 1.0:
                    self.duration = DEFAULT_DURATION

            except ValueError:
                self.duration = DEFAULT_DURATION

            self.input_note_frequency.setText(str(self.frequency))
            self.input_note_duration.setText(str(self.duration))

        if self.selected_type == SoundType.DESIGN:
            try:
                self.duration = float(self.input_design_duration.text())
                if self.duration < 1.0:
                    self.duration = DEFAULT_DURATION

            except ValueError:
                self.duration = DEFAULT_DURATION

            self.input_design_duration.setText(str(self.duration))

    def note_options(self):
        self.selected_type = SoundType.NOTE
        self.stacked_widget_sound_type.setCurrentIndex(0)
        self.stacked_widget_status.setCurrentIndex(0)
        self.set_frequency_duration()
        self.sv.generate_sound_stream(self.selected_type, self.frequency, self.fs, self.duration)

    def design_options(self):
        self.selected_type = SoundType.DESIGN
        self.stacked_widget_sound_type.setCurrentIndex(1)
        self.stacked_widget_status.setCurrentIndex(0)
        self.set_frequency_duration()
        self.sv.generate_sound_stream(self.selected_type, self.frequency, self.fs, self.duration)

    def file_options(self):
        self.selected_type = SoundType.FILE
        self.stacked_widget_sound_type.setCurrentIndex(2)
        self.stacked_widget_status.setCurrentIndex(1)
        sound_file = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Open file','.','Sound files (*.wav, *.*)'
        )
        sound_file = Path(sound_file[0])
        if (sound_file_txt := sound_file.name):
            file_name_text = f'Sound file: {sound_file_txt}  '
            self.input_file.setText(file_name_text)
            self.error_message = None

        else:
            self.error_message = 'No file selected ...'
            self.input_file.setText(self.error_message)
            file_name_text = None

        if file_name_text:
            try:
                wave_data = wave.open(sound_file.name)
                self.error_message = None

            except (wave.Error, EOFError, AttributeError):
                self.error_message = 'Invalid wav file ...'
                self.input_file.setText(self.error_message)
                return

            self.fs, self.duration = self.sv.generate_sound_stream(
                self.selected_type, 0, 0, 0, wave_data=wave_data
            )
            self.input_status_sampling_rate.setText(f'{self.fs}')
            self.input_status_duration.setText(f'{self.duration:.1f}')

    def set_volume(self, value):
        self.sv.volume = float(value) / 200.0


def start_app():
    app = QtWidgets.QApplication([])
    sv = SoundVisualiser()
    main_window = PyqtMainWindow(sv)
    main_window.show()
    app.exec()


if __name__ == '__main__':
    start_app()
