#!/usr/bin/env python3

# Copyright (C) 2022 Deepwave Digital, Inc - All Rights Reserved
# You may use, distribute and modify this code under the terms of the DEEPWAVE DIGITAL SOFTWARE
# SOURCE CODE TERMS OF USE, which is provided with the code. If a copy of the license was not
# received, please write to support@deepwavedigital.com

import os
import pathlib
import random
from typing import Any, Callable, Dict, List, Union

from matplotlib import pyplot as plt
import numpy as np
import onnxruntime
import scipy.special

_script_dir = pathlib.Path(__file__).parent.absolute()
_airpack_root = _script_dir.parent.parent


def get_file_pars(filename: Union[str, os.PathLike]) -> Dict[str, Any]:
    """ File names are of the format: key0=val0_key1=val1_key2=val2.bin so that we can
    easily parse the file name to get the file parameters, e.g., snr=10. This will allow
    us to only plot the desired SNR values.

    :param filename:    filename string to parse
    :return:            Dictionary of file parameters
    """

    # Split the sets of key, value pairs
    file_base = os.path.splitext(os.path.basename(filename))[0]

    # Loop over sets of key, value pars and form a dictionary of parameters
    file_pars = {}
    for str in file_base.split('_'):
        key, val = str.split('=')
        file_pars[key] = int(val) if val.lstrip('-').isnumeric() else val
    return file_pars


def setup_inference_function(saver_path: pathlib.Path,
                             file_name: str = 'saved_model.onnx') \
                            -> Callable[[np.ndarray], List[float]]:
    """ Sets up a tensorflow session from an onnx file to perform inference
    saved_model
    :param saver_path:  Path to onnx file
    :param file_name:   onnx file name
    :return:            Callable inference function
    """
    onnx_file = saver_path / file_name
    sess = onnxruntime.InferenceSession(str(onnx_file))
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name

    def infer_func(x):
        return sess.run([label_name], {input_name: x})[0]
    return infer_func


def infer(data_folder: Union[str, os.PathLike],
          plot_snr: int = 12,
          fs: float = 31.25e6) -> List[float]:
    """ This script will re-initialize a trained PyTorch model for inference. It will
    look through the test_data_folder and find one signal file for each label and for
    the SNR value defined by plot_snr.

    .. note ::
        For the provided data set, plot_snr may range from -5 to 20 dB and the accuracy
        of the trained model may be shown to go down as the SNR is decreased.

    :param data_folder:     Location of data
    :param plot_snr:        Define desired SNR to plot
    :param fs:              Define sample rate
    :return:                Inference results
    """

    # Define location of training data
    test_data_folder = pathlib.Path(data_folder) / "test"
    # Define location of saved model
    model_save_folder = _airpack_root / "output" / "pytorch"
    # Define data type
    dtype = np.int16
    # Define scalar multiplier to normalize the data. This should be the same value used for the
    # normalize_scalar variable in run_training.py. This value must also be used in deployment. It
    # may be though of as a digital gain to get the signal close to (-1, 1).
    normalize_scalar = 1000

    # Restore model for inference
    perform_inference = setup_inference_function(model_save_folder)

    # Get one file for each label with snr = plot_snr
    labels = os.listdir(test_data_folder)
    labels.sort()
    files = []
    for label in labels:
        label_folder = test_data_folder / label
        label_files = os.listdir(label_folder)
        random.shuffle(label_files)
        for label_file in label_files:
            pars = get_file_pars(label_file)
            if pars['snr'] == plot_snr:
                files.append(test_data_folder / label / label_file)
                break

    # Read I/Q data from each file and perform inference
    sigs = []
    results = []
    for file in files:
        # Read I/Q data from the file
        sig_interleaved_int16 = np.fromfile(file, dtype=dtype)

        # Convert raw I/Q data to float and scale it to be between [-1, 1]
        sig_interleaved = sig_interleaved_int16.astype(np.float32) / np.iinfo(dtype).max

        # Normalize the signal
        sig_scaled = sig_interleaved * normalize_scalar

        # Input to model must have dims of (batch_size, 4096). Since we have batch_size = 1, we must
        # expand the dimensions so that sig.shape = (1, 4096)
        sig_input = np.expand_dims(sig_scaled, 0)

        # Feed the signal into the neural network for inference
        # result_array = sess.run(pred, feed_dict={input_layer: sig_input})
        result_array = perform_inference(sig_input)

        # Normalize output
        result_prob = scipy.special.softmax(result_array)

        # Determine which label has the highest probability
        result = np.argmax(result_prob)

        # Save the signal and the result for plotting
        sig = sig_scaled[::2] + 1j*sig_scaled[1::2]
        sigs.append(sig)
        results.append(result)

    # Plot signal and spectrogram
    plt.style.use('dark_background')
    fig, axs = plt.subplots(3, len(results), figsize=(20, 10),
                            gridspec_kw={'wspace': 0.05, 'hspace': 0.25})
    axs = np.rollaxis(axs, 1)  # shift dimensions of axes for clearer loop iteration

    # Set the y axis labels for the first column only
    axs[0, 0].set_ylabel('Amp (Norm)')  # Row 0 - Amplitude vs. Time
    axs[0, 1].set_ylabel('Freq (MHz)')  # Row 1 - Spectrogram
    axs[0, 2].set_ylabel('PSD (dB)')  # Row 2 - Power vs. Frequency
    for i, (sig, label, result, ax) in enumerate(zip(sigs, labels, results, axs)):

        # Plot time domain signal
        t = np.arange(len(sig)) / fs / 1e-6  # time in microseconds
        ax[0].plot(t, sig.real, 'c')  # Plot in-phase (real) component of signal
        ax[0].plot(t, sig.imag, 'm')  # Plot quadrature (imag) component of signal

        ax[0].set_xlim([t[0], t[-1]])  # Set axis limits
        ax[0].set_xticks(np.arange(0, t[-1], 20))
        ax[0].set_title('Truth = {}\nResult = {}'.format(int(label), result))
        ax[0].set_xlabel('Time (us)')

        # Plot spectrogram of signal
        ax[1].specgram(sig, NFFT=32, Fs=fs / 1e6, noverlap=16, mode='psd', scale='dB', vmin=-100,
                       vmax=-20)
        ax[1].set_xlim([t[0], t[-1]])
        ax[1].set_xticks(np.arange(0, t[-1], 20))
        ax[1].set_xlabel('Time (us)')

        # Plot power spectral density
        nfft = len(sig)
        f = (np.arange(0, fs, fs / nfft) - (fs / 2) + (fs / nfft)) / 1e6  # Freq array
        y0 = np.fft.fftshift(np.fft.fft(sig))  # FFT
        y = 10 * np.log10((y0 * y0.conjugate()).real / (nfft ** 2))  # PSD
        ax[2].plot(f, y)
        ax[2].set_xlabel('Freq (MHz)')
        ax[2].set_ylim([-70, 30])

        if i > 0:  # Don't y show tick marks except for 1st column
            [ax_row.set_yticks([]) for ax_row in ax]

    # Save figure
    save_plot_name = model_save_folder / 'saved_model_plot.png'
    plt.savefig(str(save_plot_name))
    print(f'Output plot saved to: {save_plot_name}')

    return results


if __name__ == '__main__':
    _default_data_folder = '/data'
    infer(_default_data_folder)
