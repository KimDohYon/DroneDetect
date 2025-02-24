#! /usr/bin/env python3

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch

class WaveformDataset(torch.utils.data.Dataset):
    _default_device: torch.device = torch.device('cpu')

    def __init__(self,
                 root: Union[Path, str],
                 dtype: Any,
                 num_samples: Optional[int] = None,
                 device: torch.device = torch.device('cpu'),
                 transform: Optional[torch.nn.Module] = None,
                 seed: Union[None, int, np.random.SeedSequence] = None,
                 shuffle: bool = False) -> None:
        self.num_samples = num_samples
        self.device = device
        self.dtype = dtype
        self.root = Path(root).expanduser()
        self.rng = np.random.default_rng(seed)
        self.transform = transform
        self.file_list = sorted(self.root.rglob('*.fc32'), key=lambda p: p.stem)

        if len(self.file_list) == 0:
            raise FileNotFoundError("No data files found in {}".format(root))

        if shuffle:
            self.rng.shuffle(self.file_list)

    def __getitem__(self, n: int) -> Tuple[torch.Tensor, Dict]:
        """
        데이터셋의 n번째 아이템(파일)을 로드합니다.

        :param n: 로드할 아이템의 인덱스
        :return: tuple: (waveform, label)
        """
        path = self.file_list[n].resolve()
        waveform, label = self._load_item(path.stem, path.suffix, path.parent)

        # 파일에 포함된 전체 샘플 수가 지정한 num_samples보다 클 경우,
        # 임의의 위치에서 길이 num_samples인 윈도우를 추출합니다.
        if self.num_samples:
            total_samples = waveform.shape[-1]
            if total_samples < self.num_samples:
                error_fmt = ("File {} has insufficient samples: "
                             "required {}, found {}")
                raise ValueError(error_fmt.format(path, self.num_samples, total_samples))
            elif total_samples > self.num_samples:
                # 임의의 시작 위치를 선택하여 num_samples 길이의 윈도우 추출
                start = self.rng.integers(0, total_samples - self.num_samples + 1)
                waveform = waveform[:, start:start + self.num_samples]

        if self.transform is not None:
            with torch.no_grad():
                waveform = self.transform(waveform)

        # 복소수 텐서라면 실수 표현(리얼, 이미지)으로 변환 후 flatten
        if torch.is_complex(waveform):
            # torch.view_as_real: (1, samples) -> (1, samples, 2)
            waveform = torch.view_as_real(waveform).view(-1)

        return waveform, label

    def __len__(self) -> int:
        """데이터셋의 샘플 개수를 반환합니다."""
        return len(self.file_list)

    def _load_item(self, fileid: str, suffix: str, path: Path) -> Tuple[torch.Tensor, Dict]:
        """
        파일과 해당 레이블을 로드합니다.
        파일의 랜덤한 9375KB 블록만 읽어와서 데이터를 생성합니다.

        파일 경로의 가장 하위 폴더명이 레이블(숫자)로 사용됩니다.
        반환되는 텐서는 torch.complex64 타입입니다.

        :param fileid: 확장자 없는 파일명
        :param suffix: 파일 확장자
        :param path: 파일이 위치한 디렉토리
        :return: tuple: (waveform, label)
        """
        filename = os.path.join(path, fileid + suffix)
        label = int(os.path.basename(os.path.dirname(filename)))
        
        # 원하는 블록 크기: 9375 KB
        desired_bytes = 93750 * 1024  # 9375 KB in bytes
        dtype_itemsize = np.dtype(self.dtype).itemsize
        
        file_size = os.path.getsize(filename)
        if desired_bytes > file_size:
            raise ValueError(f"File {filename} is smaller than the desired block size of {desired_bytes} bytes.")
        
        total_elements = file_size // dtype_itemsize
        num_elements_to_read = desired_bytes // dtype_itemsize
        
        # 랜덤 시작 인덱스 선택
        max_start = total_elements - num_elements_to_read
        start_element = self.rng.integers(0, max_start + 1)
        
        # I/Q interleaved 데이터일 경우, 시작 인덱스가 짝수여야 함.
        if self.dtype != np.complex64 and (start_element % 2 != 0):
            start_element = max(0, start_element - 1)
        
        offset_bytes = start_element * dtype_itemsize
        
        # np.memmap을 사용해 원하는 구간만 읽어옴
        waveform = np.memmap(filename, dtype=self.dtype, mode='r',
                             offset=offset_bytes, shape=(num_elements_to_read,))
        waveform = np.array(waveform)  # 메모리로 로드
        
        # 복소수 변환: self.dtype가 np.complex64가 아니면 I/Q interleaved로 가정
        if self.dtype == np.complex128:
            waveform = np.asarray(waveform, dtype=np.complex64)
        elif self.dtype != np.complex64:
            waveform = np.asarray(waveform[0::2] + 1j * waveform[1::2],
                                  dtype=np.complex64)
        
        # 단일 채널 데이터 텐서 생성
        num_channels = 1
        samples_per_channel = waveform.shape[0]
        tensor_shape = (num_channels, samples_per_channel)
        tensor = torch.as_tensor(waveform, device=torch.device('cpu')).view(tensor_shape)
        
        return tensor, label

def load_waveform(data_folder: str, num_samples: int, shuffle: bool) -> torch.utils.data.Dataset:
    """
    :param data_folder: 신호 데이터가 있는 디렉토리
    :param num_samples: 복소수 샘플 단위의 신호 윈도우 길이
    :param shuffle: 샘플 섞기 여부
    :return: PyTorch 데이터셋
    """
    data = WaveformDataset(root=data_folder,
                           dtype=np.float32,
                           shuffle=shuffle,
                           num_samples=num_samples)
    return data

if __name__ == "__main__":
    load_waveform("/data", 24000000, True)


