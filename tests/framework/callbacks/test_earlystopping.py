"""
Copyright (c) 2022 Poutyne and all respective contributors.

Each contributor holds copyright over their respective contributions. The project versioning (Git)
records all such contribution source information.

This file is part of Poutyne.

Poutyne is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
version.

Poutyne is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with Poutyne. If not, see
<https://www.gnu.org/licenses/>.
"""

import pytest
import torch
import torch.nn as nn

from poutyne import Callback, EarlyStopping, Metric, Model
from tests.framework.tools import some_data_generator


class EarlyStoppingDummyMetric(Metric):
    __name__ = 'dummy'

    def __init__(self, values) -> None:
        super().__init__()
        self.values = values
        self.current_epoch = None

    def update(self, y_pred, y_true):
        pass

    def compute(self):
        return self.values[self.current_epoch - 1]

    def reset(self) -> None:
        pass


class DummyMetricCallback(Callback):
    def __init__(self, metric):
        super().__init__()
        self.metric = metric

    def on_epoch_begin(self, epoch_number: int, logs: dict):
        self.metric.current_epoch = epoch_number


class EarlyStoppingTest:
    batch_size = 20

    def setup_method(self):
        torch.manual_seed(42)
        self.pytorch_network = nn.Linear(1, 1)
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.pytorch_network.parameters(), lr=1e-3)

    def test_integration(self):
        train_gen = some_data_generator(20)
        valid_gen = some_data_generator(20)
        earlystopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=False)

        model = Model(self.pytorch_network, self.optimizer, self.loss_function)
        model.fit_generator(train_gen, valid_gen, epochs=10, steps_per_epoch=5, callbacks=[earlystopper])

    def test_early_stopping_patience_of_1(self):
        earlystopper = EarlyStopping(monitor='val_dummy', min_delta=0, patience=1, verbose=False)

        dummy_metric_values = [8, 4, 5, 2]
        early_stop_epoch = 3
        self._test_early_stopping(earlystopper, dummy_metric_values, early_stop_epoch)

    def test_early_stopping_with_delta(self):
        earlystopper = EarlyStopping(monitor='val_dummy', min_delta=3, patience=2, verbose=False)

        dummy_metric_values = [8, 4, 5, 2, 2]
        early_stop_epoch = 4
        self._test_early_stopping(earlystopper, dummy_metric_values, early_stop_epoch)

    def test_early_stopping_with_max(self):
        earlystopper = EarlyStopping(monitor='val_dummy', mode='max', min_delta=0, patience=2, verbose=False)

        dummy_metric_values = [2, 8, 4, 5, 2]
        early_stop_epoch = 4
        self._test_early_stopping(earlystopper, dummy_metric_values, early_stop_epoch)

    def test_mode_not_min_max_raise_error(self):
        with pytest.raises(ValueError):
            EarlyStopping(monitor='val_dummy', mode='a_mode', min_delta=0, patience=2, verbose=False)

    def test_early_stopping_with_verbose(self, capsys):
        earlystopper = EarlyStopping(monitor='val_dummy', mode='max', min_delta=0, patience=2, verbose=True)

        dummy_metric_values = [2, 8, 4, 5, 2]
        early_stop_epoch = 4

        self._test_early_stopping(earlystopper, dummy_metric_values, early_stop_epoch)

        captured = capsys.readouterr()
        assert 'Epoch 4: early stopping' in captured.out.strip()

    def _test_early_stopping(self, earlystopper, dummy_metric_values, early_stop_epoch):
        dummy_metric = EarlyStoppingDummyMetric(dummy_metric_values)
        dummy_metric_callback = DummyMetricCallback(dummy_metric)

        train_gen = some_data_generator(EarlyStoppingTest.batch_size)
        valid_gen = some_data_generator(EarlyStoppingTest.batch_size)

        model = Model(self.pytorch_network, self.optimizer, self.loss_function, epoch_metrics=[dummy_metric])
        history = model.fit_generator(
            train_gen, valid_gen, epochs=10, steps_per_epoch=5, callbacks=[dummy_metric_callback, earlystopper]
        )
        assert len(history) == early_stop_epoch
        assert history[-1]['epoch'] == early_stop_epoch
