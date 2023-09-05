import sys
import time
import io
from functools import partial
import threading
import struct
from pathlib import Path
from enum import auto, IntEnum
from PyQt6.QtCore import QTimer
from PyQt6 import uic, QtWidgets
import numpy as np
import matplotlib

matplotlib.use("Qt5Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.animation import FuncAnimation
import wave
import pyaudio

DEFAULT_FREQUENCY = 223  # frequency in Hz
DEFAULT_DURATION = 5.0  # length of sound stream in seconds
INTERVAL = 100  # plot interval in millisecond
PACKAGE_LENGTH = 2048  # number of samples in sound package


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


def wav_generator(wav_file, chunk):
    if isinstance(wav_file, Path):
        wav_file = str(wav_file)
    else:
        wav_file.seek(0)

    with wave.open(wav_file, "rb") as wf:
        frames = wf.getnframes()
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        fr = wf.getframerate()
        chunk = chunk
        duration = frames / fr
        if sample_width == 1:
            fmt_id = "B"
        else:
            fmt_id = "h"

        yield frames, channels, sample_width, fr, chunk, duration, fmt_id

        data = b"dummy"
        while data != b"":
            data = wf.readframes(chunk)
            yield data


class MplCanvas(FigureCanvas):
    """Class that creates a canvas instance"""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)

    def get_fig_ax(self):
        return self.fig, self.ax


class SoundVisualiser:
    """Methods to play and visualise audio sound"""

    def __init__(self):
        self.out = ""
        self._canvas = MplCanvas()
        self.fig, self.ax = self.canvas.get_fig_ax()
        self.fr = None
        self.channels = None
        self.stream = None
        self._start_time = None
        self._volume = None
        self.volume_factor = None
        self.sound_stream = None
        self.wav_file = None

    @property
    def volume(self):
        return self._volume

    @volume.setter
    def volume(self, value):
        self._volume = value

    @property
    def canvas(self):
        return self._canvas

    def handle_sound_wav(self, selected_type, frequency, fr, duration, sound_file=None):
        valid = True

        def write_stream_to_wav_file(file_name, sound_stream, fr, volume_factor):
            sound_stream = (sound_stream * (2**15 - 1) * volume_factor).astype(
                "h"
            )
            with wave.open(file_name, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(fr)
                wf.writeframes(sound_stream.tobytes())

        match selected_type:
            case SoundType.NOTE:
                self.volume_factor = 1e-2 * 0.5
                sound_stream = (
                    np.sin(2 * np.pi * frequency / fr * np.arange(fr * duration))
                ).astype(np.float32)
                self.wav_file = io.BytesIO()
                write_stream_to_wav_file(
                    self.wav_file, sound_stream, fr, self.volume_factor
                )

            case SoundType.DESIGN:
                self.volume_factor = 1e-2 * 0.5
                sound_stream = (
                    (0.5 * np.sin(2 * np.pi * 325 / fr * np.arange(fr * duration)))
                    + (0.1 * np.sin(2 * np.pi * 330 / fr * np.arange(fr * duration)))
                    + (0.5 * np.sin(2 * np.pi * 340 / fr * np.arange(fr * duration)))
                    + 0
                ).astype(np.float32)
                self.wav_file = io.BytesIO()
                write_stream_to_wav_file(
                    self.wav_file, sound_stream, fr, self.volume_factor
                )

            case SoundType.FILE:
                self.wav_file = sound_file
                self.volume_factor = 1e-5 * 3.0
                if self.wav_file is None:
                    return False

                try:
                    wave_data = wave.open(str(self.wav_file))
                    wave_data.close()

                except (wave.Error, EOFError, AttributeError):
                    return False

            case other:
                raise ValueError(
                    f"check selected_type invalid value {self.selected_type}"
                )
        return valid

    def set_up_stream_with_callback(self, channels, fr, chunk, fmt_id, read_audio):
        audio = pyaudio.PyAudio()
        if not channels:
            return

        def callback(in_data, frame_count, time_info, status, fmt_id, read_audio):
            byte_stream = next(read_audio)
            fmt = f"{len(byte_stream) // 2}{fmt_id}"
            a = struct.unpack(fmt, byte_stream)
            self.out = np.array(a).astype(np.float32) * self.volume_factor
            return self.out * self.volume, pyaudio.paContinue

        stream = audio.open(
            format=pyaudio.paFloat32,
            channels=channels,
            rate=fr,
            output=True,
            frames_per_buffer=chunk,
            stream_callback=partial(callback, fmt_id=fmt_id, read_audio=read_audio),
        )
        return stream

    def update_frame(self, frame):
        samples = len(self.out)
        if samples == PACKAGE_LENGTH:
            self.line.set_data(self.xdata, self.out)

        elif samples > 0:
            xdata = np.linspace(0, 1000 * samples / self.fr, samples)
            self.line.set_data(xdata, self.out)

        else:
            return

        self.canvas.draw()

    def init_sound_visualisation(self):
        read_wav = wav_generator(self.wav_file, PACKAGE_LENGTH)
        (
            _,
            channels,
            _,
            self.fr,
            chunk,
            duration,
            fmt_id,
        ) = next(read_wav)
        self.stream = self.set_up_stream_with_callback(
            channels, self.fr, chunk, fmt_id, read_wav
        )
        self.ax.set_ylim(-1.1, 1.1)
        self.ax.set_xlim(1000 * PACKAGE_LENGTH / self.fr, 0)
        self.canvas.draw()
        self.xdata = np.linspace(0, 1000 * PACKAGE_LENGTH / self.fr, PACKAGE_LENGTH)
        (self.line,) = self.ax.plot([], [], lw=2)

        duration_range = np.arange(0, duration, INTERVAL / 1000)
        self.visualisation = FuncAnimation(
            self.fig,
            self.update_frame,
            frames=duration_range,
            interval=INTERVAL,
            repeat=False,
        )
        self.canvas.draw()
        return time.time(), self.fr, duration

    def control_sound_stream(self, control_word):
        match control_word:
            case "is_active":
                return self.stream.is_active()

            case "start":
                self.stream.start_stream()

            case "stop":
                self.stream.stop_stream()

            case "close":
                self.stream.close()

            case other:
                raise ValueError(f"Incorrect control word given: {control_word}")

    def control_visualiser(self, control_word):
        match control_word:
            case "init":
                return self.init_sound_visualisation()

            case "start":
                self.visualisation.event_source.start()

            case "stop":
                self.visualisation.event_source.stop()

            case other:
                raise ValueError(f"Incorrect control word given: {control_word}")

    def set_plot(self, fr):
        self.ax.set_ylim(-1.1, 1.1)
        self.ax.set_xlim(1000 * PACKAGE_LENGTH / fr, 0)
        self.canvas.draw()

    def remove_plot(self):
        for line in self.ax.lines:
            line.remove()

        self.canvas.draw()


class PyqtViewControl(QtWidgets.QMainWindow):
    """PyQt view and control"""

    def __init__(self, sv_instance, *args, **kwargs):
        super().__init__(*args, **kwargs)
        uic.loadUi(Path(__file__).parent / "audio_visualiser.ui", self)
        self.sv = sv_instance
        self.plot_layout.addWidget(self.sv.canvas)
        self.action_exit.triggered.connect(self.quit)
        self.action_start_pause.triggered.connect(self.control_start_pause)
        self.action_stop.triggered.connect(self.stop)
        self.start_pause_button.clicked.connect(self.control_start_pause)
        self.stop_button.clicked.connect(self.stop)
        self.rb_note.clicked.connect(self.note_options)
        self.rb_design.clicked.connect(self.design_options)
        self.rb_file.clicked.connect(self.file_options)
        self.input_note_frequency.editingFinished.connect(self.note_options)
        self.input_note_duration.editingFinished.connect(self.note_options)
        self.input_design_duration.editingFinished.connect(self.design_options)
        self.selected_type = SoundType.NOTE
        self.rb_2048.toggled.connect(
            lambda: self.set_sampling_rate(SamplingRate.sr2048)
        )
        self.rb_4096.toggled.connect(
            lambda: self.set_sampling_rate(SamplingRate.sr4096)
        )
        self.rb_8192.toggled.connect(
            lambda: self.set_sampling_rate(SamplingRate.sr8192)
        )
        self.rb_16384.toggled.connect(
            lambda: self.set_sampling_rate(SamplingRate.sr16384)
        )
        self.rb_32768.toggled.connect(
            lambda: self.set_sampling_rate(SamplingRate.sr32768)
        )
        self.volume_slider.setMinimum(0)
        self.volume_slider.setMaximum(100)
        self.volume_slider.setTickInterval(5)
        self.volume_slider.valueChanged.connect(self.set_volume)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_progress_bar)
        self.timer.stop()
        self.start_time = None
        self.duration = None
        self.fr = None
        self.running = False
        self.stopped = True
        self.error_message = None
        self.pause_time = 0.0
        self.visualisation = None
        self.start_time = None
        self.rb_4096.setChecked(True)
        self.volume_slider.setValue(30)
        self.rb_note.setChecked(True)
        self.note_options()

    def enable_disable_buttons(self, enable_buttons=False):
        self.rb_note.setEnabled(enable_buttons)
        self.rb_design.setEnabled(enable_buttons)
        self.rb_file.setEnabled(enable_buttons)

    def control_start_pause(self):
        if self.error_message:
            return

        if self.stopped:
            self.sv.remove_plot()
            self.set_frequency_duration()
            self.running = True
            self.stopped = False
            self.time_progress_bar.setValue(0)
            self.start_time, self.fr, self.duration = self.sv.control_visualiser("init")
            self.enable_disable_buttons(enable_buttons=False)
            self.pause_time = 0.0
            self.timer.start(250)

            # start audio deamon in a seperate thread as otherwise audio and
            # plot will not be at the same time
            self.x = threading.Thread(target=self.play_sound)
            self.x.daemon = True
            self.x.start()

            self.action_start_pause.setText("Pause")
            self.start_pause_button.setText("Pause")
            self.input_status_sampling_rate.setText(f"{self.fr}")
            self.input_status_duration.setText(f"{self.duration:.1f}")

        else:
            if self.running:
                self.sv.control_visualiser("stop")
                self.pause_start_time = time.time()
                self.timer.stop()
                self.action_start_pause.setText("Run")
                self.start_pause_button.setText("Run")

            else:
                self.pause_time += time.time() - self.pause_start_time
                self.sv.control_visualiser("start")
                self.timer.start(250)
                self.action_start_pause.setText("Pause")
                self.start_pause_button.setText("Pause")

            self.running = not self.running

    def play_sound(self):
        # pause audio when self.running is False; close audio when stopped
        while self.sv.control_sound_stream("is_active"):
            if self.stopped:
                break

            # pause loop
            while not self.running:
                if self.sv.control_sound_stream("is_active"):
                    self.sv.control_sound_stream("stop")

                if self.stopped:
                    break

            if self.running and not self.sv.control_sound_stream("is_active"):
                self.sv.control_sound_stream("start")

        self.sv.control_sound_stream("stop")
        self.sv.control_sound_stream("close")
        self.running = False
        self.stopped = True
        self.action_start_pause.setText("Start")
        self.start_pause_button.setText("Start")

    def stop(self):
        try:
            self.sv.control_visualiser("stop")

        except AttributeError:
            # just pass as stop button was pressed before visualisation had
            # actually started
            pass
        self.stopped = True
        self.enable_disable_buttons(enable_buttons=True)
        self.timer.stop()

    def update_progress_bar(self):
        elapsed_time = (
            100 * (time.time() - self.pause_time - self.start_time) / self.duration
        )
        self.time_progress_bar.setValue(int(elapsed_time))

    def set_sampling_rate(self, sampling_rate):
        self.fr = 2**sampling_rate
        match self.selected_type:
            case SoundType.NOTE:
                self.note_options()
            case SoundType.DESIGN:
                self.design_options()
            case other:
                pass
        self.sv.set_plot(self.fr)

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
        if not self.stopped:
            return
        self.error_message = None
        self.selected_type = SoundType.NOTE
        self.stacked_widget_sound_type.setCurrentIndex(0)
        self.stacked_widget_status.setCurrentIndex(0)
        self.set_frequency_duration()
        self.sv.handle_sound_wav(
            self.selected_type, self.frequency, self.fr, self.duration
        )

    def design_options(self):
        if not self.stopped:
            return
        self.error_message = None
        self.selected_type = SoundType.DESIGN
        self.stacked_widget_sound_type.setCurrentIndex(1)
        self.stacked_widget_status.setCurrentIndex(0)
        self.set_frequency_duration()
        self.sv.handle_sound_wav(
            self.selected_type, self.frequency, self.fr, self.duration
        )

    def file_options(self):
        if not self.stopped:
            return
        self.error_message = None
        self.selected_type = SoundType.FILE
        self.stacked_widget_sound_type.setCurrentIndex(2)
        self.stacked_widget_status.setCurrentIndex(1)
        sound_file = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open file", ".", "Sound files (*.wav)"
        )
        sound_file = Path(sound_file[0])
        if sound_file_txt := sound_file.name:
            file_name_text = f"Sound file: {sound_file_txt}  "
            self.input_file.setText(file_name_text)

        else:
            self.error_message = "No file selected ..."
            self.input_file.setText(self.error_message)
            file_name_text = None

        if file_name_text:
            valid = self.sv.handle_sound_wav(
                self.selected_type, 0, 0, 0, sound_file=sound_file
            )
            if not valid:
                self.error_message = "Invalid wav file ..."
                self.input_file.setText(self.error_message)

    def set_volume(self, value):
        self.sv.volume = float(value) / 200.0

    def quit(self):
        self.stop()
        time.sleep(1)
        sys.exit()


def start_app():
    app = QtWidgets.QApplication([])
    sv = SoundVisualiser()
    view_control = PyqtViewControl(sv)
    view_control.show()
    app.exec()


if __name__ == "__main__":
    start_app()
