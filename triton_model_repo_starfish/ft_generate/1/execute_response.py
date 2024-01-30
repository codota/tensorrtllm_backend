#  Copyright (c) Tabnine LTD  2023.
#  All Rights Reserved.

import numpy as np


class ExecuteResponse:
    def __init__(self, output, generated_length, lp_data):
        self.output = output
        self.generated_length = generated_length
        self.lp_data = lp_data

    def slice(self, length):
        return ExecuteResponse(
            self.output[:, :length],
            np.minimum(self.generated_length, [length] * len(self.generated_length)),
            self.lp_data[:, :length] if self.lp_data is not None else None,
        )

    def max_generated_length(self):
        return np.max(self.generated_length)
