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

# pylint: disable=not-callable,no-member
import csv
import os
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from poutyne import AtomicCSVGradientLogger, GradientLoggerBase, MemoryGradientLogger, Model, TensorBoardGradientLogger
from poutyne import CSVGradientLogger as NonAtomicCSVGradientLogger
from tests.framework.tools import some_data_generator


class GradientLoggerTestBase:
    __test__ = False
    GradientLogger = None
    batch_size = 20
    num_epochs = 10

    def setup_method(self):
        torch.manual_seed(42)
        self.pytorch_network = nn.Sequential(nn.Linear(1, 2), nn.Linear(2, 1))
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.pytorch_network.parameters(), lr=1e-3)
        self.model = Model(self.pytorch_network, self.optimizer, self.loss_function)
        self.temp_dir_obj = TemporaryDirectory()
        self.csv_filename = os.path.join(self.temp_dir_obj.name, 'layer_{}.csv')

    def teardown_method(self):
        self.temp_dir_obj.cleanup()


class GradientLoggerBaseTest(GradientLoggerTestBase):
    __test__ = True
    GradientLogger = GradientLoggerBase

    def test_log_stats_raise_a_NotImplementedError(self):
        logger = self.GradientLogger()
        a_epoch_number = 1
        a_batch_number = 1
        a_logs = {}
        a_layer_dict = {}
        with pytest.raises(NotImplementedError):
            logger.log_stats(
                epoch_number=a_epoch_number, batch_number=a_batch_number, logs=a_logs, layer_stats=a_layer_dict
            )

    def test_on_train_begin_no_bias_does_not_keep_it(self):
        logger = self.GradientLogger(keep_bias=False)
        logger.set_model(self.model)

        a_logs = {}
        logger.on_train_begin(a_logs)
        actual = logger.layers
        expected = ['0.weight', '1.weight']
        assert actual == expected
        assert len(actual) == 2

    def test_on_train_begin_with_bias_then_keep_it(self):
        logger = self.GradientLogger(keep_bias=True)
        logger.set_model(self.model)

        a_logs = {}
        logger.on_train_begin(a_logs)
        actual = logger.layers
        expected = ['0.weight', '0.bias', '1.weight', '1.bias']
        assert actual == expected
        assert len(actual) == 4


class CSVGradientLoggerTestBase(GradientLoggerTestBase):
    __test__ = False

    def test_logging(self):
        train_gen = some_data_generator(self.batch_size)
        valid_gen = some_data_generator(self.batch_size)
        memgrad = MemoryGradientLogger()
        logger = self.GradientLogger(self.csv_filename)
        self.model.fit_generator(
            train_gen, valid_gen, epochs=self.num_epochs, steps_per_epoch=5, callbacks=[memgrad, logger]
        )
        self._test_logging(memgrad.history)

    def test_logging_append(self):
        train_gen = some_data_generator(self.batch_size)
        valid_gen = some_data_generator(self.batch_size)
        logger = self.GradientLogger(self.csv_filename)
        memgrad = MemoryGradientLogger()
        self.model.fit_generator(
            train_gen, valid_gen, epochs=self.num_epochs, steps_per_epoch=5, callbacks=[memgrad, logger]
        )
        memgrad2 = MemoryGradientLogger()
        logger = self.GradientLogger(self.csv_filename, append=True)
        self.model.fit_generator(
            train_gen,
            valid_gen,
            epochs=20,
            steps_per_epoch=5,
            initial_epoch=self.num_epochs,
            callbacks=[memgrad2, logger],
        )
        history = {layer: stats1 + memgrad2.history[layer] for layer, stats1 in memgrad.history.items()}
        self._test_logging(history)

    def test_logging_overwrite(self):
        train_gen = some_data_generator(self.batch_size)
        valid_gen = some_data_generator(self.batch_size)
        logger = self.GradientLogger(self.csv_filename)
        self.model.fit_generator(train_gen, valid_gen, epochs=self.num_epochs, steps_per_epoch=5, callbacks=[logger])
        memgrad = MemoryGradientLogger()
        logger = self.GradientLogger(self.csv_filename, append=False)
        self.model.fit_generator(
            train_gen,
            valid_gen,
            epochs=20,
            steps_per_epoch=5,
            initial_epoch=self.num_epochs,
            callbacks=[memgrad, logger],
        )
        self._test_logging(memgrad.history)

    def _test_logging(self, history):
        for layer, stats in history.items():
            filename = self.csv_filename.format(layer)
            with open(filename, encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                rows = list(reader)
            assert len(rows) == len(stats)
            for row, stats_entry in zip(rows, stats):
                assert row.keys() == stats_entry.keys()
                for k in row:
                    assert float(row[k]) == pytest.approx(stats_entry[k], abs=5e-8)


class NonAtomicCSVGradientLoggerTest(CSVGradientLoggerTestBase):
    __test__ = True
    GradientLogger = NonAtomicCSVGradientLogger


class AtomicCSVGradientLoggerTest(CSVGradientLoggerTestBase):
    __test__ = True
    GradientLogger = AtomicCSVGradientLogger


class TensorboardGradientLoggerTest(GradientLoggerTestBase):
    __test__ = True
    GradientLogger = TensorBoardGradientLogger

    def test_logging(self):
        train_gen = some_data_generator(self.batch_size)
        valid_gen = some_data_generator(self.batch_size)
        memgrad = MemoryGradientLogger()
        tensorboard_writer_mock = MagicMock()
        logger = self.GradientLogger(tensorboard_writer_mock)
        self.model.fit_generator(
            train_gen, valid_gen, epochs=self.num_epochs, steps_per_epoch=5, callbacks=[memgrad, logger]
        )
        self._test_logging(memgrad.history, tensorboard_writer_mock)

    def _test_logging(self, history, writer_mock):
        # Each layer has 7 metric computed on them (mean, var, min, abs_min, max, abs_max and l2 norm)
        # We do 50 steps (10 epoch times 5) = 700 methods call
        assert len(writer_mock.method_calls) == 700

        # We group the seven stats entries for a layer_i intro a list
        grouped_metrics_calls = list(zip(*[iter(writer_mock.method_calls)] * 7))

        # We will use an idx to group the stats in layer calls
        for history_idx, stats in enumerate(history.values()):
            for mock_calls, stats_entry in zip(grouped_metrics_calls[history_idx::2], stats):
                # We remove non-tracked history by tensorboard
                stats_entry.pop("epoch")
                stats_entry.pop("batch")

                assert len(mock_calls) == len(stats_entry)
                for stats_idx, k in enumerate(stats_entry.keys()):
                    assert mock_calls[stats_idx][1][1][k] == pytest.approx(stats_entry[k], abs=5e-8)
