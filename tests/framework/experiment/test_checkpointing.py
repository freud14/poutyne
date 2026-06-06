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

import os
import pickle
from tempfile import TemporaryDirectory
from typing import ClassVar

import pytest

try:
    import pandas  # pylint: disable=unused-import  # noqa: F401

    pandas_available = True
except ImportError:
    pandas_available = False
import torch
from torch import nn

from poutyne import Experiment
from tests.framework.experiment.utils import ConstantMetric, ConstantMetricCallback
from tests.framework.tools import SomeDataGeneratorWithLen


@pytest.mark.skipif(not pandas_available, reason="pandas is not available")
class ExperimentCheckpointLoadingTestBase:
    __test__ = False
    # pylint: disable=no-member
    NUM_EPOCHS = None
    METRIC_VALUES = None
    CHECKPOINT_EPOCHS = None
    NO_CHECKPOINT_EPOCHS = None
    MONITOR_MODE = None

    def setup_method(self):
        self.temp_dir_obj = TemporaryDirectory()
        self.test_checkpoints_path = self.temp_dir_obj.name

        self.metric = ConstantMetric()
        self.metric_callback = ConstantMetricCallback(self.METRIC_VALUES, self.metric)

        expt = Experiment(
            self.test_checkpoints_path,
            nn.Linear(1, 1),
            optimizer='sgd',
            loss_function='mse',
            batch_metrics=[self.metric],
            monitor_metric="const",
            monitor_mode=self.MONITOR_MODE,
        )
        train_generator = SomeDataGeneratorWithLen(2, 32, 0)
        expt.train(train_generator, epochs=self.NUM_EPOCHS, callbacks=[self.metric_callback])

        self.checkpoint_paths = [
            os.path.join(self.test_checkpoints_path, f"checkpoint_epoch_{epoch}.ckpt")
            for epoch in self.CHECKPOINT_EPOCHS
        ]
        self.no_checkpoint_paths = [
            os.path.join(self.test_checkpoints_path, f"checkpoint_epoch_{epoch}.ckpt")
            for epoch in self.NO_CHECKPOINT_EPOCHS
        ]
        self.last_checkpoint_path = os.path.join(self.test_checkpoints_path, "checkpoint.ckpt")
        self.test_experiment = Experiment(
            self.test_checkpoints_path,
            nn.Linear(1, 1),
            optimizer='sgd',
            loss_function='mse',
            batch_metrics=[self.metric],
            monitor_metric="const",
            monitor_mode=self.MONITOR_MODE,
        )

    def teardown_method(self):
        self.temp_dir_obj.cleanup()

    def test_load_checkpoint_with_int(self):
        index = self.CHECKPOINT_EPOCHS[0]
        filename = self.checkpoint_paths[0]
        self.test_experiment.load_checkpoint(index)

        assert self.test_experiment.model.network.state_dict() == torch.load(
            filename, pickle_module=pickle, map_location="cpu"
        )

    def test_load_checkpoint_best(self):
        filename = self.checkpoint_paths[-1]
        self.test_experiment.load_checkpoint("best")

        assert self.test_experiment.model.network.state_dict() == torch.load(
            filename, pickle_module=pickle, map_location="cpu"
        )

    def test_load_checkpoint_last(self):
        self.test_experiment.load_checkpoint("last")

        assert self.test_experiment.model.network.state_dict() == torch.load(
            self.last_checkpoint_path, pickle_module=pickle, map_location="cpu"
        )

    def test_load_checkpoint_using_path(self):
        cpkt_path = os.path.join(self.test_checkpoints_path, "test_model_weights_state_dict.p")
        torch.save(
            torch.load(self.checkpoint_paths[0], pickle_module=pickle, map_location="cpu"),
            cpkt_path,
            pickle_module=pickle,
        )  # change the ckpt path
        self.test_experiment.load_checkpoint(cpkt_path)

        assert self.test_experiment.model.network.state_dict() == torch.load(
            cpkt_path, pickle_module=pickle, map_location="cpu"
        )

    def test_load_invalid_checkpoint(self):
        with pytest.raises(ValueError):
            self.test_experiment.load_checkpoint(self.NO_CHECKPOINT_EPOCHS[0])

        with pytest.raises(FileNotFoundError):
            self.test_experiment.load_checkpoint(self.no_checkpoint_paths[0])

        temp_dir_obj = TemporaryDirectory()
        test_checkpoints_path = temp_dir_obj.name
        expt = Experiment(test_checkpoints_path, nn.Linear(1, 1), optimizer='sgd', loss_function='mse')
        with pytest.raises(ValueError):
            expt.load_checkpoint('best')

        with pytest.raises(ValueError):
            expt.load_checkpoint('last')


class MonitorMinCheckpointExperimentTest(ExperimentCheckpointLoadingTestBase):
    __test__ = True
    NUM_EPOCHS = 5
    METRIC_VALUES: ClassVar[list] = [9, 3, 6, 2, 3]
    CHECKPOINT_EPOCHS: ClassVar[list] = [1, 2, 4]
    NO_CHECKPOINT_EPOCHS: ClassVar[list] = [3, 5]
    MONITOR_MODE = "min"


class MonitorMaxCheckpointExperimentTest(ExperimentCheckpointLoadingTestBase):
    __test__ = True
    NUM_EPOCHS = 5
    METRIC_VALUES: ClassVar[list] = [4, 3, 6, 2, 7]
    CHECKPOINT_EPOCHS: ClassVar[list] = [1, 3, 5]
    NO_CHECKPOINT_EPOCHS: ClassVar[list] = [2, 4]
    MONITOR_MODE = "max"
