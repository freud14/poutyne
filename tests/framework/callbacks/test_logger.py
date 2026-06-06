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

import csv
import os
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, call

import pytest
import torch
import torch.nn as nn

from tests.framework.tools import some_data_generator

try:
    from torch.utils.tensorboard import SummaryWriter as TorchSummaryWriter
except ImportError:
    TorchSummaryWriter = None

try:
    from tensorboardX import SummaryWriter as XSummaryWriter
except ImportError:
    XSummaryWriter = None

from poutyne import AtomicCSVLogger, Callback, Model, TensorBoardLogger
from poutyne import CSVLogger as NonAtomicCSVLogger


class History(Callback):
    def on_epoch_end(self, epoch_number, logs):
        self.history.append(logs)

    def on_train_batch_end(self, batch_number, logs):
        self.history.append(logs)

    def on_train_begin(self, logs):
        self.history = []


class CSVLoggerTestBase:
    __test__ = False
    # pylint: disable=not-callable,no-member
    CSVLogger = None
    batch_size = 20
    lr = 1e-3
    num_epochs = 10

    def setup_method(self):
        torch.manual_seed(42)
        self.pytorch_network = nn.Linear(1, 1)
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.pytorch_network.parameters(), lr=CSVLoggerTestBase.lr)
        self.model = Model(self.pytorch_network, self.optimizer, self.loss_function)
        self.temp_dir_obj = TemporaryDirectory()
        self.csv_filename = os.path.join(self.temp_dir_obj.name, 'my_log.csv')

    def teardown_method(self):
        self.temp_dir_obj.cleanup()

    def test_logging(self):
        train_gen = some_data_generator(20)
        valid_gen = some_data_generator(20)
        logger = self.CSVLogger(self.csv_filename)
        history = self.model.fit_generator(
            train_gen, valid_gen, epochs=self.num_epochs, steps_per_epoch=5, callbacks=[logger]
        )
        self._test_logging(history)

    def test_logging_with_batch_granularity(self):
        train_gen = some_data_generator(20)
        valid_gen = some_data_generator(20)
        logger = self.CSVLogger(self.csv_filename, batch_granularity=True)
        history = History()
        self.model.fit_generator(
            train_gen, valid_gen, epochs=self.num_epochs, steps_per_epoch=5, callbacks=[logger, history]
        )
        self._test_logging(history.history)

    def test_logging_append(self):
        train_gen = some_data_generator(20)
        valid_gen = some_data_generator(20)
        logger = self.CSVLogger(self.csv_filename)
        history = self.model.fit_generator(
            train_gen, valid_gen, epochs=self.num_epochs, steps_per_epoch=5, callbacks=[logger]
        )
        logger = self.CSVLogger(self.csv_filename, append=True)
        history2 = self.model.fit_generator(
            train_gen, valid_gen, epochs=20, steps_per_epoch=5, initial_epoch=self.num_epochs, callbacks=[logger]
        )
        self._test_logging(history + history2)

    def test_logging_overwrite(self):
        train_gen = some_data_generator(20)
        valid_gen = some_data_generator(20)
        logger = self.CSVLogger(self.csv_filename)
        self.model.fit_generator(train_gen, valid_gen, epochs=self.num_epochs, steps_per_epoch=5, callbacks=[logger])
        logger = self.CSVLogger(self.csv_filename, append=False)
        history = self.model.fit_generator(
            train_gen, valid_gen, epochs=20, steps_per_epoch=5, initial_epoch=self.num_epochs, callbacks=[logger]
        )
        self._test_logging(history)

    def test_multiple_learning_rates(self):
        train_gen = some_data_generator(20)
        valid_gen = some_data_generator(20)
        logger = self.CSVLogger(self.csv_filename)
        lrs = [CSVLoggerTestBase.lr, CSVLoggerTestBase.lr / 2]
        optimizer = torch.optim.SGD(
            [
                {'params': [self.pytorch_network.weight], 'lr': lrs[0]},
                {'params': [self.pytorch_network.bias], 'lr': lrs[1]},
            ]
        )
        model = Model(self.pytorch_network, optimizer, self.loss_function)
        history = model.fit_generator(
            train_gen, valid_gen, epochs=self.num_epochs, steps_per_epoch=5, callbacks=[logger]
        )
        self._test_logging(history, lrs=lrs)

    def _test_logging(self, history, lrs=None):
        if lrs is None:
            lrs = [CSVLoggerTestBase.lr]
        with open(self.csv_filename, encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = []
            for row in reader:
                if row['epoch'] != '':
                    if len(lrs) == 1:
                        assert float(row['lr']) == pytest.approx(lrs[0], abs=5e-8)
                        del row['lr']
                    else:
                        for i, lr in enumerate(lrs):
                            assert float(row[f'lr_group_{i}']) == pytest.approx(lr, abs=5e-8)
                            del row[f'lr_group_{i}']

                rows.append(row)
        assert len(rows) == len(history)
        for raw_row, hist_entry in zip(rows, history):
            row = {k: v for k, v in raw_row.items() if v != ''}
            assert row.keys() == hist_entry.keys()
            for k in row:
                if isinstance(hist_entry[k], float):
                    assert float(row[k]) == pytest.approx(hist_entry[k], abs=5e-8)
                else:
                    assert str(row[k]) == str(hist_entry[k])


class NonAtomicCSVLoggerTest(CSVLoggerTestBase):
    __test__ = True
    CSVLogger = NonAtomicCSVLogger


class AtomicCSVLoggerTest(CSVLoggerTestBase):
    __test__ = True
    CSVLogger = AtomicCSVLogger


class TensorBoardLoggerTestBase:
    __test__ = False
    SummaryWriter = None
    batch_size = 20
    lr = 1e-3
    num_epochs = 10

    def setup_method(self):
        torch.manual_seed(42)
        self.pytorch_network = nn.Linear(1, 1)
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.pytorch_network.parameters(), lr=TensorBoardLoggerTestBase.lr)
        self.model = Model(self.pytorch_network, self.optimizer, self.loss_function)
        self.temp_dir_obj = TemporaryDirectory()
        # pylint: disable=not-callable
        self.writer = self.SummaryWriter(self.temp_dir_obj.name)
        self.writer.add_scalars = MagicMock()

    def teardown_method(self):
        self.temp_dir_obj.cleanup()

    def test_logging(self):
        train_gen = some_data_generator(20)
        valid_gen = some_data_generator(20)
        logger = TensorBoardLogger(self.writer)
        history = self.model.fit_generator(
            train_gen, valid_gen, epochs=self.num_epochs, steps_per_epoch=5, callbacks=[logger]
        )
        self._test_logging(history)

    def test_multiple_learning_rates(self):
        train_gen = some_data_generator(20)
        valid_gen = some_data_generator(20)
        logger = TensorBoardLogger(self.writer)
        lrs = [CSVLoggerTestBase.lr, CSVLoggerTestBase.lr / 2]
        optimizer = torch.optim.SGD(
            [
                {'params': [self.pytorch_network.weight], 'lr': lrs[0]},
                {'params': [self.pytorch_network.bias], 'lr': lrs[1]},
            ]
        )
        model = Model(self.pytorch_network, optimizer, self.loss_function)
        history = model.fit_generator(
            train_gen, valid_gen, epochs=self.num_epochs, steps_per_epoch=5, callbacks=[logger]
        )
        self._test_logging(history, lrs=lrs)

    def _test_logging(self, history, lrs=None):
        if lrs is None:
            lrs = [CSVLoggerTestBase.lr]
        calls = []
        for h in history:
            calls.append(call('loss', {'loss': h['loss'], 'val_loss': h['val_loss']}, h['epoch']))
            if len(lrs) == 1:
                calls.append(call('lr', {'lr': self.lr}, h['epoch']))
            else:
                calls.append(call('lr', {f'lr_group_{i}': lr for i, lr in enumerate(lrs)}, h['epoch']))
        self.writer.add_scalars.assert_has_calls(calls, any_order=True)


@pytest.mark.skipif(XSummaryWriter is None, reason="Needs tensorboardX to run this test")
class TensorboardXLoggerTest(TensorBoardLoggerTestBase):
    __test__ = True
    SummaryWriter = XSummaryWriter


@pytest.mark.skipif(TorchSummaryWriter is None, reason="Unable to import SummaryWriter from torch")
class TorchTensorboardLoggerTest(TensorBoardLoggerTestBase):
    __test__ = True
    SummaryWriter = TorchSummaryWriter


class TensorBoardLoggerWithSplitTrainValTestBase:
    __test__ = False
    SummaryWriter = None
    batch_size = 20
    lr = 1e-3
    num_epochs = 10

    def setup_method(self):
        torch.manual_seed(42)
        self.pytorch_network = nn.Linear(1, 1)
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.pytorch_network.parameters(), lr=TensorBoardLoggerTestBase.lr)
        self.model = Model(self.pytorch_network, self.optimizer, self.loss_function)
        self.temp_dir_obj = TemporaryDirectory()
        # pylint: disable=not-callable
        self.writer = self.SummaryWriter(self.temp_dir_obj.name)
        self.writer.add_scalar = MagicMock()

    def teardown_method(self):
        self.temp_dir_obj.cleanup()

    def test_logging(self):
        train_gen = some_data_generator(20)
        valid_gen = some_data_generator(20)
        logger = TensorBoardLogger(self.writer, split_train_val=True)
        history = self.model.fit_generator(
            train_gen, valid_gen, epochs=self.num_epochs, steps_per_epoch=5, callbacks=[logger]
        )
        self._test_logging(history)

    def test_multiple_learning_rates(self):
        train_gen = some_data_generator(20)
        valid_gen = some_data_generator(20)
        logger = TensorBoardLogger(self.writer, split_train_val=True)
        lrs = [CSVLoggerTestBase.lr, CSVLoggerTestBase.lr / 2]
        optimizer = torch.optim.SGD(
            [
                {'params': [self.pytorch_network.weight], 'lr': lrs[0]},
                {'params': [self.pytorch_network.bias], 'lr': lrs[1]},
            ]
        )
        model = Model(self.pytorch_network, optimizer, self.loss_function)
        history = model.fit_generator(
            train_gen, valid_gen, epochs=self.num_epochs, steps_per_epoch=5, callbacks=[logger]
        )
        self._test_logging(history, lrs=lrs)

    def _test_logging(self, history, lrs=None):
        if lrs is None:
            lrs = [CSVLoggerTestBase.lr]

        calls = []
        for h in history:
            for k, v in h.items():
                calls.append(call(k, v, h['epoch']))

            if len(lrs) == 1:
                calls.append(call('lr', self.lr, h['epoch']))
            else:
                for i, lr in enumerate(lrs):
                    calls.append(call(f'lr_group_{i}', lr, h['epoch']))

        self.writer.add_scalar.assert_has_calls(calls, any_order=True)


@pytest.mark.skipif(XSummaryWriter is None, reason="Needs tensorboardX to run this test")
class TensorboardXLoggerWithSplitTrainValTest(TensorBoardLoggerWithSplitTrainValTestBase):
    __test__ = True
    SummaryWriter = XSummaryWriter


@pytest.mark.skipif(TorchSummaryWriter is None, reason="Unable to import SummaryWriter from torch")
class TorchTensorboardLoggerWithSplitTrainValTest(TensorBoardLoggerWithSplitTrainValTestBase):
    __test__ = True
    SummaryWriter = TorchSummaryWriter
