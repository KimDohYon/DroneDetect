#!/usr/bin/env python3

# Copyright (C) 2022 Deepwave Digital, Inc - All Rights Reserved
# You may use, distribute and modify this code under the terms of the DEEPWAVE DIGITAL SOFTWARE
# SOURCE CODE TERMS OF USE, which is provided with the code. If a copy of the license was not
# received, please write to support@deepwavedigital.com

import os
import pathlib
from typing import Union  # NameError 해결
from tqdm import tqdm
import torch
from torch.optim.lr_scheduler import StepLR  # StepLR 스케줄러 추가

_script_dir = pathlib.Path(__file__).parent.absolute()
_airpack_root = _script_dir.parent.parent

try:
    from airpack.pytorch import fileio, model
except ModuleNotFoundError as e:
    import sys
    _msg = "{0}\nPlease run:\n  pip install -e {1}".format(e, _airpack_root)
    raise ModuleNotFoundError(_msg).with_traceback(sys.exc_info()[2]) from None

DEVICE = torch.device("cpu")

def train(network: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer, criterion: torch.nn.Module, scalar: int) -> tuple:
    network.train()
    running_loss = 0.0
    running_accuracy = 0.0
    for data_in, labels in dataloader:
        data_in = data_in.to(DEVICE)
        labels = labels.to(DEVICE)
        data_fix = data_in * scalar
        pred = network(data_fix)
        loss = criterion(pred, labels)
        running_loss += loss.item()
        _, pred_label = torch.max(pred, 1)
        running_accuracy += torch.sum(pred_label == labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss = running_loss / len(dataloader.dataset)
    train_acc = running_accuracy / len(dataloader.dataset)
    return train_loss, train_acc

def validate(network: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
             criterion: torch.nn.Module, scalar: int) -> tuple:
    network.eval()
    running_loss = 0.0
    running_accuracy = 0.0
    with torch.no_grad():
        for data_in, labels in dataloader:
            data_in = data_in.to(DEVICE)
            labels = labels.to(DEVICE)
            data_fix = data_in * scalar
            pred = network(data_fix)
            loss = criterion(pred, labels)
            running_loss += loss.item()
            _, pred_label = torch.max(pred, 1)
            running_accuracy += torch.sum(pred_label == labels)
    val_loss = running_loss / len(dataloader.dataset)
    val_acc = running_accuracy / len(dataloader.dataset)
    return val_loss, val_acc

def main(data_folder: Union[str, os.PathLike], n_epoch: int = 1000) -> float:
    data_folder = pathlib.Path(data_folder)
    train_data_folder = data_folder / "train"
    test_data_folder = data_folder / "test"
    model_save_folder = _airpack_root / "output" / "pytorch"
    os.makedirs(model_save_folder, exist_ok=True)
    
    input_len = 1200000  
    output_len = 20
    learning_rate = 1e-4
    batch_size = 4  
    normalize_scalar = 1

    train_data = fileio.load_waveform(train_data_folder, input_len, True)
    test_data = fileio.load_waveform(test_data_folder, input_len, False)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    classifier = model.default_network(input_len, output_len).to(DEVICE)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    
    # ** StepLR 스케줄러 추가 **
    scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
    # 100 에포크마다 러닝레이트를 0.1배로 감소

    for epoch in tqdm(range(n_epoch), total=n_epoch, desc='Training Progress', unit='epoch'):
        print(f'\nEpoch {epoch + 1} of {n_epoch}')
        train_epoch_loss, train_epoch_accr = train(classifier, train_loader, optimizer, criterion, normalize_scalar)
        val_epoch_loss, val_epoch_accr = validate(classifier, test_loader, criterion, normalize_scalar)
        
        print(f'Train Loss: {train_epoch_loss:.4f}, Train Accuracy: {train_epoch_accr:.4f}')
        print(f'Val Loss: {val_epoch_loss:.4f}, Val Accuracy: {val_epoch_accr:.4f}')
        
        # ** 스케줄러 업데이트 **
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Current Learning Rate: {current_lr}")

    dummy_input = torch.randn(batch_size, input_len * 2).to(DEVICE)
    output_file = os.path.join(model_save_folder, 'saved_model.onnx')
    torch.onnx.export(classifier, dummy_input, output_file, export_params=True,
                      opset_version=10, do_constant_folding=True,
                      input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {0: 'batch_size'},
                                    "output": {0: 'batch_size'}})
    return train_epoch_accr

if __name__ == '__main__':
    _default_data_folder = '/data'
    main(_default_data_folder)


