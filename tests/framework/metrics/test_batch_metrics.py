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

import numpy as np
import pytest
import torch

from poutyne import Accuracy, BinaryAccuracy, TopKAccuracy, acc, bin_acc, topk


class AccuracyTest:
    def setup_method(self):
        self.predictions = torch.Tensor(
            [
                [0.35, 0.25, 0.1, 0.1, 0.2],
                [0.1, 0.6, 0.1, 0.2, 0.0],
                [0.1, 0.6, 0.1, 0.2, 0.0],
                [0.1, 0.5, 0.1, 0.2, 0.0],
                [0.1, 0.2, 0.1, 0.7, 0.0],
                [0.1, 0.6, 0.1, 0.2, 0.0],
            ]
        )
        self.label_predictions = torch.Tensor([0, 1, 1, 1, 3, 1])
        self.targets = torch.Tensor([0, 4, 1, 0, 3, 0])
        self.accuracy_none = (self.label_predictions == self.targets).float() * 100
        self.accuracy = self.accuracy_none.mean()

    def test_standard(self):
        accuracy = Accuracy()
        actual = accuracy(self.predictions, self.targets)
        assert actual.shape == ()
        np.testing.assert_almost_equal(self.accuracy, actual)

    def test_functional(self):
        actual = acc(self.predictions, self.targets)
        assert actual.shape == ()
        np.testing.assert_almost_equal(self.accuracy, actual)

    def test_hundred(self):
        accuracy = Accuracy()
        actual = accuracy(self.predictions, self.label_predictions)
        assert actual.shape == ()
        np.testing.assert_almost_equal(100.0, actual)

    def test_sum(self):
        accuracy = Accuracy(reduction='sum')
        actual = accuracy(self.predictions, self.targets)
        assert actual.shape == ()
        np.testing.assert_almost_equal(self.accuracy * len(self.predictions), actual)

    def test_functional_sum(self):
        actual = acc(self.predictions, self.targets, reduction='sum')
        assert actual.shape == ()
        np.testing.assert_almost_equal(self.accuracy * len(self.predictions), actual)

    def test_sum_hundred(self):
        accuracy = Accuracy(reduction='sum')
        actual = accuracy(self.predictions, self.label_predictions)
        assert actual.shape == ()
        np.testing.assert_almost_equal(100.0 * len(self.predictions), actual)

    def test_none(self):
        accuracy = Accuracy(reduction='none')
        actual = accuracy(self.predictions, self.targets)
        assert self.targets.shape == actual.shape
        assert torch.all(self.accuracy_none == actual)

    def test_functional_none(self):
        actual = acc(self.predictions, self.targets, reduction='none')
        assert self.targets.shape == actual.shape
        assert torch.all(self.accuracy_none == actual)

    def test_none_hundred(self):
        accuracy = Accuracy(reduction='none')
        actual = accuracy(self.predictions, self.label_predictions)
        assert self.targets.shape == actual.shape
        assert torch.all(100.0 * torch.ones_like(self.label_predictions).float() == actual)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="no gpu available")
    def test_on_gpu(self):
        accuracy = Accuracy()
        accuracy.cuda()
        actual = accuracy(self.predictions.cuda(), self.targets.cuda())
        assert actual.shape == ()
        np.testing.assert_almost_equal(self.accuracy, actual.cpu())

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="no gpu available")
    def test_functional_on_gpu(self):
        actual = acc(self.predictions.cuda(), self.targets.cuda())
        assert actual.shape == ()
        np.testing.assert_almost_equal(self.accuracy, actual.cpu())


class IgnoreIndexAccuracyTest:
    def setup_method(self):
        self.predictions = torch.Tensor(
            [
                [0.35, 0.25, 0.1, 0.1, 0.2],
                [0.1, 0.6, 0.1, 0.2, 0.0],
                [0.1, 0.6, 0.1, 0.2, 0.0],
                [0.1, 0.5, 0.1, 0.2, 0.0],
                [0.1, 0.2, 0.1, 0.7, 0.0],
                [0.1, 0.6, 0.1, 0.2, 0.0],
            ]
        )
        self.label_predictions = torch.Tensor([0, 1, 1, 1, 3, 1])
        self.targets = torch.Tensor([-100, 4, 1, -100, 3, 0])
        self.accuracy = 50.0

    def test_standard(self):
        accuracy = Accuracy()
        actual = accuracy(self.predictions, self.targets)
        assert actual.shape == ()
        np.testing.assert_almost_equal(self.accuracy, actual)

    def test_functional_standard(self):
        actual = acc(self.predictions, self.targets)
        assert actual.shape == ()
        np.testing.assert_almost_equal(self.accuracy, actual)

    def test_ignore_index_with_different_value(self):
        accuracy = Accuracy(ignore_index=-1)
        actual = accuracy(self.predictions, self.targets)
        assert actual.shape == ()
        assert float(actual) == pytest.approx(1 / 3 * 100, abs=5e-6)

    def test_functional_ignore_index_with_different_value(self):
        actual = acc(self.predictions, self.targets, ignore_index=-1)
        assert actual.shape == ()
        assert float(actual) == pytest.approx(1 / 3 * 100, abs=5e-6)

    def test_ignore_index_with_valid_index(self):
        accuracy = Accuracy(ignore_index=1)
        actual = accuracy(self.predictions, self.targets)
        assert actual.shape == ()
        np.testing.assert_almost_equal(20.0, actual)

    def test_functional_ignore_index_with_valid_index(self):
        actual = acc(self.predictions, self.targets, ignore_index=1)
        assert actual.shape == ()
        np.testing.assert_almost_equal(20.0, actual)


class BinaryAccuracyTest:
    def setup_method(self):
        self.predictions = torch.Tensor([-0.45, -0.66, 0.8, -1.65, 0.42, -0.04, -0.99, -0.46, 1.12, 1.93])
        self.label_predictions = torch.Tensor([0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0])
        self.targets = torch.Tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0])
        self.binary_accuracy_none = (self.label_predictions == self.targets).float() * 100
        self.binary_accuracy = self.binary_accuracy_none.mean()

    def test_standard(self):
        binary_accuracy = BinaryAccuracy()
        actual = binary_accuracy(self.predictions, self.targets)
        assert actual.shape == ()
        np.testing.assert_almost_equal(self.binary_accuracy, actual, decimal=5)

    def test_functional(self):
        actual = bin_acc(self.predictions, self.targets)
        assert actual.shape == ()
        np.testing.assert_almost_equal(self.binary_accuracy, actual, decimal=5)

    def test_hundred(self):
        binary_accuracy = BinaryAccuracy()
        actual = binary_accuracy(self.predictions, self.label_predictions)
        assert actual.shape == ()
        np.testing.assert_almost_equal(100.0, actual, decimal=5)

    def test_sum(self):
        binary_accuracy = BinaryAccuracy(reduction='sum')
        actual = binary_accuracy(self.predictions, self.targets)
        assert actual.shape == ()
        np.testing.assert_almost_equal(self.binary_accuracy_none.sum(), actual)

    def test_functional_sum(self):
        actual = bin_acc(self.predictions, self.targets, reduction='sum')
        assert actual.shape == ()
        np.testing.assert_almost_equal(self.binary_accuracy_none.sum(), actual)

    def test_sum_hundred(self):
        binary_accuracy = BinaryAccuracy(reduction='sum')
        actual = binary_accuracy(self.predictions, self.label_predictions)
        assert actual.shape == ()
        np.testing.assert_almost_equal(100.0 * len(self.predictions), actual)

    def test_none(self):
        binary_accuracy = BinaryAccuracy(reduction='none')
        actual = binary_accuracy(self.predictions, self.targets)
        assert self.targets.shape == actual.shape
        assert torch.all(self.binary_accuracy_none == actual)

    def test_functional_none(self):
        actual = bin_acc(self.predictions, self.targets, reduction='none')
        assert self.targets.shape == actual.shape
        assert torch.all(self.binary_accuracy_none == actual)

    def test_none_hundred(self):
        binary_accuracy = BinaryAccuracy(reduction='none')
        actual = binary_accuracy(self.predictions, self.label_predictions)
        assert self.targets.shape == actual.shape
        assert torch.all(100.0 * torch.ones_like(self.label_predictions).float() == actual)

    def test_threshold(self):
        threshold = -0.5
        binary_accuracy = BinaryAccuracy(threshold=threshold)
        actual = binary_accuracy(self.predictions, self.targets)
        label_predictions = (self.predictions > threshold).float()
        assert actual.shape == ()
        np.testing.assert_almost_equal((label_predictions == self.targets).float().mean() * 100, actual)

    def test_functional_threshold(self):
        threshold = -0.5
        actual = bin_acc(self.predictions, self.targets, threshold=threshold)
        label_predictions = (self.predictions > threshold).float()
        assert actual.shape == ()
        np.testing.assert_almost_equal((label_predictions == self.targets).float().mean() * 100, actual)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="no gpu available")
    def test_on_gpu(self):
        binary_accuracy = BinaryAccuracy()
        binary_accuracy.cuda()
        actual = binary_accuracy(self.predictions.cuda(), self.targets.cuda())
        assert actual.shape == ()
        np.testing.assert_almost_equal(self.binary_accuracy, actual.cpu(), decimal=5)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="no gpu available")
    def test_functional_on_gpu(self):
        actual = bin_acc(self.predictions.cuda(), self.targets.cuda())
        assert actual.shape == ()
        np.testing.assert_almost_equal(self.binary_accuracy, actual.cpu(), decimal=5)


class TopKAccuracyTest:
    def setup_method(self):
        self.predictions = torch.Tensor(
            [
                [0.35, 0.25, 0.1, 0.1, 0.2],
                [0.1, 0.7, 0.0, 0.2, 0.0],
                [0.0, 0.7, 0.1, 0.2, 0.0],
                [0.1, 0.5, 0.0, 0.3, 0.0],
                [0.0, 0.3, 0.1, 0.7, 0.0],
                [0.1, 0.6, 0.0, 0.3, 0.0],
            ]
        )
        self.top_3_label_predictions = torch.Tensor([[0, 1, 4], [1, 3, 0], [1, 3, 0], [1, 3, 0], [3, 1, 2], [1, 3, 0]])
        self.targets = torch.Tensor([0, 4, 1, 2, 3, 0])
        self.top_k_acc_none = (self.top_3_label_predictions == self.targets.unsqueeze(1)).any(1).float() * 100
        self.top_k_acc = self.top_k_acc_none.mean()

    def test_standard(self):
        top_k_acc = TopKAccuracy(3)
        actual = top_k_acc(self.predictions, self.targets)
        assert actual.shape == ()
        np.testing.assert_almost_equal(self.top_k_acc, actual, decimal=5)

    def test_functional(self):
        actual = topk(self.predictions, self.targets, 3)
        assert actual.shape == ()
        np.testing.assert_almost_equal(self.top_k_acc, actual, decimal=5)

    def test_hundred(self):
        top_k_acc = TopKAccuracy(3)
        actual = top_k_acc(self.predictions, self.top_3_label_predictions[:, 0])
        assert actual.shape == ()
        np.testing.assert_almost_equal(100.0, actual)

    def test_sum(self):
        top_k_acc = TopKAccuracy(3, reduction='sum')
        actual = top_k_acc(self.predictions, self.targets)
        assert actual.shape == ()
        np.testing.assert_almost_equal(self.top_k_acc * len(self.predictions), actual, decimal=0)

    def test_functional_sum(self):
        actual = topk(self.predictions, self.targets, 3, reduction='sum')
        assert actual.shape == ()
        np.testing.assert_almost_equal(self.top_k_acc * len(self.predictions), actual, decimal=0)

    def test_sum_hundred(self):
        top_k_acc = TopKAccuracy(3, reduction='sum')
        actual = top_k_acc(self.predictions, self.top_3_label_predictions[:, 0])
        assert actual.shape == ()
        np.testing.assert_almost_equal(100.0 * len(self.predictions), actual, decimal=0)

    def test_none(self):
        top_k_acc = TopKAccuracy(3, reduction='none')
        actual = top_k_acc(self.predictions, self.targets)
        assert self.targets.shape == actual.shape
        assert torch.all(self.top_k_acc_none == actual)

    def test_functional_none(self):
        actual = topk(self.predictions, self.targets, 3, reduction='none')
        assert self.targets.shape == actual.shape
        assert torch.all(self.top_k_acc_none == actual)

    def test_none_hundred(self):
        top_k_acc = TopKAccuracy(3, reduction='none')
        actual = top_k_acc(self.predictions, self.top_3_label_predictions[:, 0])
        assert self.targets.shape == actual.shape
        assert torch.all(100.0 * torch.ones_like(self.targets).float() == actual)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="no gpu available")
    def test_on_gpu(self):
        top_k_acc = TopKAccuracy(3)
        top_k_acc.cuda()
        actual = top_k_acc(self.predictions.cuda(), self.targets.cuda())
        assert actual.shape == ()
        np.testing.assert_almost_equal(self.top_k_acc, actual.cpu(), decimal=5)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="no gpu available")
    def test_functional_on_gpu(self):
        actual = topk(self.predictions.cuda(), self.targets.cuda(), 3)
        assert actual.shape == ()
        np.testing.assert_almost_equal(self.top_k_acc, actual.cpu(), decimal=5)


class IgnoreIndexTopKAccuracyTest:
    def setup_method(self):
        self.predictions = torch.Tensor(
            [
                [0.35, 0.25, 0.1, 0.1, 0.2],
                [0.1, 0.7, 0.0, 0.2, 0.0],
                [0.0, 0.7, 0.1, 0.2, 0.0],
                [0.1, 0.5, 0.0, 0.3, 0.0],
                [0.0, 0.3, 0.1, 0.7, 0.0],
                [0.1, 0.6, 0.0, 0.3, 0.0],
            ]
        )
        self.top_3_label_predictions = torch.Tensor([[0, 1, 4], [1, 3, 0], [1, 3, 0], [1, 3, 0], [3, 1, 2], [1, 3, 0]])
        self.targets = torch.Tensor([-100, 4, 1, -100, 3, 0])
        self.top_k_acc = 75.0

    def test_standard(self):
        top_k_acc = TopKAccuracy(3)
        actual = top_k_acc(self.predictions, self.targets)
        assert actual.shape == ()
        np.testing.assert_almost_equal(self.top_k_acc, actual)

    def test_functional_standard(self):
        actual = topk(self.predictions, self.targets, 3)
        assert actual.shape == ()
        np.testing.assert_almost_equal(self.top_k_acc, actual)

    def test_ignore_index_with_different_value(self):
        top_k_acc = TopKAccuracy(3, ignore_index=-1)
        actual = top_k_acc(self.predictions, self.targets)
        assert actual.shape == ()
        assert float(actual) == pytest.approx(50.0, abs=5e-6)

    def test_functional_ignore_index_with_different_value(self):
        actual = topk(self.predictions, self.targets, 3, ignore_index=-1)
        assert actual.shape == ()
        assert float(actual) == pytest.approx(50.0, abs=5e-6)

    def test_ignore_index_with_valid_index(self):
        top_k_acc = TopKAccuracy(3, ignore_index=1)
        actual = top_k_acc(self.predictions, self.targets)
        assert actual.shape == ()
        np.testing.assert_almost_equal(40.0, actual)

    def test_functional_ignore_index_with_valid_index(self):
        actual = topk(self.predictions, self.targets, 3, ignore_index=1)
        assert actual.shape == ()
        np.testing.assert_almost_equal(40.0, actual)
