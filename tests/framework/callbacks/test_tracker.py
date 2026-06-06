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

import math
from tempfile import TemporaryDirectory
from unittest.mock import ANY, MagicMock, call

import pytest
import torch
import torch.nn as nn

from tests.framework.tools import some_data_generator

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from poutyne import GradientTracker, Model, TensorBoardGradientTracker, WeightsGradientsStatsTracker


class GradientStatsTrackerTest:
    def setup_method(self):
        self.tracker = WeightsGradientsStatsTracker(number_layers=2)

        self.absolute_min_both_layer = 0.00
        self.layer_1_min = -0.15
        self.layer_1_max = 0.24
        self.layer_1_gradients = torch.Tensor([[self.layer_1_max], [self.absolute_min_both_layer], [self.layer_1_min]])

        self.layer_2_min = -0.25
        self.layer_2_max = 0.16
        self.layer_2_gradients = torch.Tensor([[self.layer_2_max], [self.layer_2_min], [self.absolute_min_both_layer]])

        self.layer_1_name = "fake_name_1"
        self.layer_2_name = "fake_name_2"
        self.layer_names = [self.layer_1_name, self.layer_2_name]

        # The value have been compute manual according to the Welford's online algorithm
        # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
        # We refer to the WelfordCompute document in tests/framework/callbacks for the expected value computation.
        mean_1 = 0.13
        mean_2 = 0.1366666
        s2_1 = 0
        s2_2 = 0
        self.first_batch_expected_stats = {
            self.layer_1_name: {
                "mean": mean_1,
                "mean_std_dev_up": mean_1 + math.sqrt(s2_1),
                "mean_std_dev_down": mean_1 - math.sqrt(s2_1),
                "min": self.layer_1_min,
                "abs_min": self.absolute_min_both_layer,
                "max": self.layer_1_max,
                "abs_max": self.layer_1_max,
            },
            self.layer_2_name: {
                "mean": mean_2,
                "mean_std_dev_up": mean_2 + math.sqrt(s2_2),
                "mean_std_dev_down": mean_2 - math.sqrt(s2_2),
                "min": self.layer_2_min,
                "abs_min": self.absolute_min_both_layer,
                "max": self.layer_2_max,
                "abs_max": math.fabs(self.layer_2_min),  # since the absolute value is higher than the normal max
            },
        }

        mean_1 = 0.195
        mean_2 = 0.205
        s2_1 = 0.00845
        s2_2 = 0.0093889
        # *2 for min and max since the weights are 3 times the layers gradients
        self.second_batch_expected_stats = {
            self.layer_1_name: {
                "mean": mean_1,
                "mean_std_dev_up": mean_1 + math.sqrt(s2_1),
                "mean_std_dev_down": mean_1 - math.sqrt(s2_1),
                "min": self.layer_1_min * 2,
                "abs_min": self.absolute_min_both_layer,
                "max": self.layer_1_max * 2,
                "abs_max": self.layer_1_max * 2,
            },
            self.layer_2_name: {
                "mean": mean_2,
                "mean_std_dev_up": mean_2 + math.sqrt(s2_2),
                "mean_std_dev_down": mean_2 - math.sqrt(s2_2),
                "min": self.layer_2_min * 2,
                "abs_min": self.absolute_min_both_layer,
                "max": self.layer_2_max * 2,
                "abs_max": math.fabs(self.layer_2_min * 2),  # since the absolute value is higher than the normal max
            },
        }

        mean_1 = 0.26
        mean_2 = 0.27333333333
        s2_1 = 0.0169
        s2_2 = 0.018677778
        # *3 for min and max since the weights are 3 times the layers gradients
        self.third_batch_expected_stats = {
            self.layer_1_name: {
                "mean": mean_1,
                "mean_std_dev_up": mean_1 + math.sqrt(s2_1),
                "mean_std_dev_down": mean_1 - math.sqrt(s2_1),
                "min": self.layer_1_min * 3,
                "abs_min": self.absolute_min_both_layer,
                "max": self.layer_1_max * 3,
                "abs_max": self.layer_1_max * 3,
            },
            self.layer_2_name: {
                "mean": mean_2,
                "mean_std_dev_up": mean_2 + math.sqrt(s2_2),
                "mean_std_dev_down": mean_2 - math.sqrt(s2_2),
                "min": self.layer_2_min * 3,
                "abs_min": self.absolute_min_both_layer,
                "max": self.layer_2_max * 3,
                "abs_max": math.fabs(self.layer_2_min * 3),  # since the absolute value is higher than the normal max
            },
        }

    def test_compute_two_layers_statistic(self):
        self._run_n_batch(num_batch=1)
        self._test_batch_stats(self.first_batch_expected_stats)

        self._run_n_batch(num_batch=2)
        self._test_batch_stats(self.second_batch_expected_stats)

        self._run_n_batch(num_batch=3)
        self._test_batch_stats(self.third_batch_expected_stats)

    def _run_n_batch(self, num_batch):
        for batch_number in range(1, num_batch + 1):
            layer_1_params = MagicMock()
            layer_1_params.grad = self.layer_1_gradients * batch_number
            layer_2_params = MagicMock()
            layer_2_params.grad = self.layer_2_gradients * batch_number

            named_parameters = (
                (n, p) for n, p in [(self.layer_1_name, layer_1_params), (self.layer_2_name, layer_2_params)]
            )

            self.tracker.batch_statistic_upgrade(named_parameters=named_parameters)

    def _test_batch_stats(self, batch_expected):
        batch_actual_stats = self.tracker.get_stats(self.layer_names)

        self._test_stats(batch_expected[self.layer_1_name], batch_actual_stats[self.layer_1_name])

        self._test_stats(batch_expected[self.layer_2_name], batch_actual_stats[self.layer_2_name])

    def _test_stats(self, expected, actual):
        assert len(expected) == len(actual)
        assert expected.keys() == actual.keys()
        for expected_value, actual_value in zip(expected.values(), actual.values()):
            assert float(expected_value) == pytest.approx(float(actual_value), abs=5e-4)


class GradientTrackerTest:
    def test_keep_good_layer(self):
        # pylint: disable=protected-access
        gradient_tracker = GradientTracker(keep_bias=False)
        layer_to_keep_params = MagicMock()
        layer_to_keep_params.requires_grad = True
        assert gradient_tracker._keep_layer(layer_to_keep_params, "fake_layer_name_to_keep")
        assert not gradient_tracker._keep_layer(layer_to_keep_params, "bias_name_not_to_keep")

        layer_not_to_keep_params = MagicMock()
        layer_not_to_keep_params.requires_grad = False
        assert not gradient_tracker._keep_layer(layer_not_to_keep_params, "fake_layer_name_not_to_keep")
        assert not gradient_tracker._keep_layer(layer_not_to_keep_params, "bias_name_not_to_keep")

        gradient_tracker = GradientTracker(keep_bias=True)
        layer_to_keep_params = MagicMock()
        layer_to_keep_params.requires_grad = True
        assert gradient_tracker._keep_layer(layer_to_keep_params, "fake_layer_name_to_keep")
        assert gradient_tracker._keep_layer(layer_to_keep_params, "bias_name_to_keep")

        layer_not_to_keep_params = MagicMock()
        layer_not_to_keep_params.requires_grad = False
        assert not gradient_tracker._keep_layer(layer_not_to_keep_params, "fake_layer_name_not_to_keep")
        assert not gradient_tracker._keep_layer(layer_not_to_keep_params, "bias_name_not_to_keep")

    def test_integration(self):
        train_gen = some_data_generator(20)
        valid_gen = some_data_generator(20)
        torch.manual_seed(42)
        self.pytorch_network = nn.Linear(1, 1)
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.pytorch_network.parameters(), lr=1e-3)
        device = torch.device("cpu")
        self.model = Model(self.pytorch_network, self.optimizer, self.loss_function, device=device)
        gradient_tracker = GradientTracker()

        self.model.fit_generator(train_gen, valid_gen, epochs=10, steps_per_epoch=5, callbacks=[gradient_tracker])

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="no gpu available")
    def test_integration_on_gpu(self):
        train_gen = some_data_generator(20)
        valid_gen = some_data_generator(20)
        torch.manual_seed(42)
        self.pytorch_network = nn.Linear(1, 1)
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.pytorch_network.parameters(), lr=1e-3)
        device = torch.device("cuda:0")
        self.model = Model(self.pytorch_network, self.optimizer, self.loss_function, device=device)
        gradient_tracker = GradientTracker()

        self.model.fit_generator(train_gen, valid_gen, epochs=10, steps_per_epoch=5, callbacks=[gradient_tracker])


@pytest.mark.skipif(SummaryWriter is None, reason="Unable to import SummaryWriter from torch")
class TensorBoardGradientTrackerTest:
    batch_size = 20
    lr = 1e-3
    num_epochs = 10

    def setup_method(self):
        torch.manual_seed(42)
        self.loss_function = nn.MSELoss()
        self.temp_dir_obj = TemporaryDirectory()
        # pylint: disable=not-callable
        self.writer = SummaryWriter(self.temp_dir_obj.name)
        self.writer.add_scalars = MagicMock()

    def teardown_method(self):
        self.temp_dir_obj.cleanup()

    def test_tracking_one_layer_model(self):
        self.num_layer = 1
        self.pytorch_network = nn.Linear(1, 1)
        self.optimizer = torch.optim.SGD(self.pytorch_network.parameters(), lr=self.lr)
        self.model = Model(self.pytorch_network, self.optimizer, self.loss_function)

        keep_bias = False
        train_gen = some_data_generator(20)
        valid_gen = some_data_generator(20)
        tracker = TensorBoardGradientTracker(self.writer, keep_bias=keep_bias)
        self.model.fit_generator(train_gen, valid_gen, epochs=self.num_epochs, steps_per_epoch=5, callbacks=[tracker])
        self._test_tracking(keep_bias)

    def test_tracking_one_layer_model_with_bias(self):
        self.num_layer = 1
        self.pytorch_network = nn.Linear(1, 1)
        self.optimizer = torch.optim.SGD(self.pytorch_network.parameters(), lr=self.lr)
        self.model = Model(self.pytorch_network, self.optimizer, self.loss_function)

        keep_bias = True
        train_gen = some_data_generator(20)
        valid_gen = some_data_generator(20)
        tracker = TensorBoardGradientTracker(self.writer, keep_bias=keep_bias)
        self.model.fit_generator(train_gen, valid_gen, epochs=self.num_epochs, steps_per_epoch=5, callbacks=[tracker])
        self._test_tracking(keep_bias)

    def test_tracking_two_layers_model(self):
        self.num_layer = 2
        self.pytorch_network = nn.Sequential(nn.Linear(1, 1), nn.Linear(1, 1))
        self.optimizer = torch.optim.SGD(self.pytorch_network.parameters(), lr=self.lr)
        self.model = Model(self.pytorch_network, self.optimizer, self.loss_function)

        keep_bias = False
        train_gen = some_data_generator(20)
        valid_gen = some_data_generator(20)
        tracker = TensorBoardGradientTracker(self.writer, keep_bias=keep_bias)
        self.model.fit_generator(train_gen, valid_gen, epochs=self.num_epochs, steps_per_epoch=5, callbacks=[tracker])
        self._test_tracking(keep_bias)

    def test_tracking_two_layers_shallow_model(self):
        self.num_layer = 2
        self.pytorch_network = nn.Sequential(nn.Linear(1, 4), nn.Linear(4, 1))
        self.optimizer = torch.optim.SGD(self.pytorch_network.parameters(), lr=self.lr)
        self.model = Model(self.pytorch_network, self.optimizer, self.loss_function)

        keep_bias = False
        train_gen = some_data_generator(20)
        valid_gen = some_data_generator(20)
        tracker = TensorBoardGradientTracker(self.writer, keep_bias=keep_bias)
        self.model.fit_generator(train_gen, valid_gen, epochs=self.num_epochs, steps_per_epoch=5, callbacks=[tracker])
        self._test_tracking(keep_bias)

    def test_tracking_N_layers_model_with_bias(self):
        self.num_layer = 4
        self.pytorch_network = nn.Sequential(nn.Linear(1, 1), nn.Linear(1, 1), nn.Linear(1, 1), nn.Linear(1, 1))
        self.optimizer = torch.optim.SGD(self.pytorch_network.parameters(), lr=self.lr)
        self.model = Model(self.pytorch_network, self.optimizer, self.loss_function)

        keep_bias = True
        train_gen = some_data_generator(20)
        valid_gen = some_data_generator(20)
        tracker = TensorBoardGradientTracker(self.writer, keep_bias=keep_bias)
        self.model.fit_generator(train_gen, valid_gen, epochs=self.num_epochs, steps_per_epoch=5, callbacks=[tracker])
        self._test_tracking(keep_bias)

    def _test_tracking(self, keep_bias):
        expected_calls = []
        for epoch in range(1, self.num_epochs + 1):
            layer_names = [""]
            if self.num_layer > 1:
                layer_names = []
                for layer_idx in range(self.num_layer):
                    layer_names.append(f"{layer_idx}.")
            for layer_name in layer_names:
                expected_calls.append(call(f'gradient_distributions/{layer_name}weight', {'mean': ANY}, epoch))
                expected_calls.append(
                    call(f'gradient_distributions/{layer_name}weight', {'mean_std_dev_up': ANY}, epoch)
                )
                expected_calls.append(
                    call(f'gradient_distributions/{layer_name}weight', {'mean_std_dev_down': ANY}, epoch)
                )
                expected_calls.append(call(f'other_gradient_stats/{layer_name}weight', {'min': ANY}, epoch))
                expected_calls.append(call(f'other_gradient_stats/{layer_name}weight', {'max': ANY}, epoch))

                if keep_bias:
                    expected_calls.append(call(f'gradient_distributions/{layer_name}bias', {'mean': ANY}, epoch))
                    expected_calls.append(
                        call(f'gradient_distributions/{layer_name}bias', {'mean_std_dev_up': ANY}, epoch)
                    )
                    expected_calls.append(
                        call(f'gradient_distributions/{layer_name}bias', {'mean_std_dev_down': ANY}, epoch)
                    )
                    expected_calls.append(call(f'other_gradient_stats/{layer_name}bias', {'min': ANY}, epoch))
                    expected_calls.append(call(f'other_gradient_stats/{layer_name}bias', {'max': ANY}, epoch))

        method_calls = self.writer.add_scalars.mock_calls
        assert len(method_calls) == len(expected_calls)
        assert method_calls == expected_calls
