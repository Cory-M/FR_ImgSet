import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lrs


class ExponentialDecay(lrs.StepLR):
    def get_lr(self):
        return [base_lr * self.gamma ** (self.last_epoch / self.step_size)
                for base_lr in self.base_lrs]


class View(nn.Module):
    def __init__(self):
        super(View, self).__init__()
   
    def forward(self, x):
        return x.view(x.size()[0], -1) 


class Conv2d_SAME(nn.Conv2d):
    def forward(self, input):
        input_rows = input.size(2)
        input_cols = input.size(3)
        filter_rows = self.weight.size(2)
        filter_cols = self.weight.size(3)
        effective_filter_size_rows = (filter_rows - 1) * self.dilation[0] + 1
        effective_filter_size_cols = (filter_cols - 1) * self.dilation[1] + 1
        out_rows = (input_rows + self.stride[0] - 1) // self.stride[0]
        out_cols = (input_cols + self.stride[1] - 1) // self.stride[1]
        padding_rows = max(0, (out_rows - 1) * self.stride[0] +
                            effective_filter_size_rows - input_rows)
        rows_odd = (padding_rows % 2 != 0)
        padding_cols = max(0, (out_cols - 1) * self.stride[1] +
                            effective_filter_size_cols - input_cols)
        cols_odd = (padding_cols % 2 != 0)

        if rows_odd or cols_odd:
            input = F.pad(input, [0, int(cols_odd), 0, int(rows_odd)])

        return F.conv2d(input, self.weight, self.bias, self.stride,
                    padding=(padding_rows // 2, padding_cols // 2),
                    dilation=self.dilation, groups=self.groups)


class MaxPool2d_SAME(nn.MaxPool2d):
    def forward(self, input):
        input_rows = input.size(2)
        input_cols = input.size(3)
        filter_rows = self.kernel_size
        filter_cols = self.kernel_size
        effective_filter_size_rows = (filter_rows - 1) * self.dilation + 1
        effective_filter_size_cols = (filter_cols - 1) * self.dilation + 1
        out_rows = (input_rows + self.stride - 1) // self.stride
        out_cols = (input_cols + self.stride - 1) // self.stride
        padding_rows = max(0, (out_rows - 1) * self.stride +
                            effective_filter_size_rows - input_rows)
        rows_odd = (padding_rows % 2 != 0)
        padding_cols = max(0, (out_cols - 1) * self.stride +
                            effective_filter_size_cols - input_cols)
        cols_odd = (padding_cols % 2 != 0)

        if rows_odd or cols_odd:
            input = F.pad(input, [0, int(cols_odd), 0, int(rows_odd)])

        return F.max_pool2d(input, self.kernel_size, self.stride,
                        padding=(padding_rows // 2, padding_cols // 2),
                        dilation=self.dilation, ceil_mode=self.ceil_mode,
                        return_indices=self.return_indices)