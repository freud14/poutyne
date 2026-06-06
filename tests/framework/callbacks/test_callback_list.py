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

from unittest.mock import MagicMock

from poutyne import Callback, CallbackList


class CallbackListTest:
    def setup_method(self) -> None:
        self.initial_callback = MagicMock(spec=Callback)
        self.callback_list = CallbackList([self.initial_callback])

    def test_append_callback(self):
        assert len(self.callback_list.callbacks) == 1
        a_callback = MagicMock(spec=Callback)
        self.callback_list.append(a_callback)

        assert len(self.callback_list.callbacks) == 2

    def test_iterator(self):
        a_callback = MagicMock(spec=Callback)
        self.callback_list.append(a_callback)
        assert len(list(self.callback_list)) == 2
