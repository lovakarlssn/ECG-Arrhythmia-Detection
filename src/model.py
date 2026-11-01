import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

class BaselineClassifier:
    def __init__(self, C=1.0, random_state=42):
        self.model = LogisticRegression(class_weight='balanced', C=C, max_iter=1000, random_state=random_state, solver='lbfgs')

    def fit(self, X, y):
        if X.ndim == 3:
            X = X.squeeze(1)
        self.model.fit(X, y)

    def predict(self, X):
        if X.ndim == 3:
            X = X.squeeze(1)
        return self.model.predict(X)

    def get_params_count(self):
        return int(self.model.coef_.size + self.model.intercept_.size)

class FakeQuantizeInt8(nn.Module):
    def __init__(self, num_bits=8):
        super().__init__()
        self.qmin = -(2 ** (num_bits - 1))
        self.qmax = (2 ** (num_bits - 1)) - 1

    def forward(self, x):
        if not self.training and not getattr(self, "force_quant", False):
            return x
        max_val = torch.max(torch.abs(x)).detach()
        if max_val == 0:
            return x
        scale = max_val / self.qmax
        x_q = torch.clamp(torch.round(x / scale), self.qmin, self.qmax)
        x_deq = x_q * scale
        return x + (x_deq - x).detach()

class QuantizedConv1dReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, pool_size=2):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.relu = nn.ReLU(inplace=False)
        self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size) if pool_size > 1 else nn.Identity()
        self.quant_enabled = False
        self.weight_quant = FakeQuantizeInt8(8)
        self.act_quant = FakeQuantizeInt8(8)

    def set_quantization(self, enabled=True):
        self.quant_enabled = enabled
        self.weight_quant.force_quant = enabled
        self.act_quant.force_quant = enabled

    def forward(self, x):
        if self.quant_enabled:
            w = self.weight_quant(self.conv.weight)
            b = self.weight_quant(self.conv.bias) if self.conv.bias is not None else None
            x = F.conv1d(x, w, b, stride=self.conv.stride, padding=self.conv.padding)
        else:
            x = self.conv(x)
        x = self.pool(x)
        x = self.relu(x)
        if self.quant_enabled:
            x = self.act_quant(x)
        return x

class MAX78002_1DCNN(nn.Module):
    def __init__(self, num_classes=4, input_channels=1, base_filters=16, dropout=0.1):
        super().__init__()
        c1 = base_filters
        c2 = base_filters * 2
        c3 = base_filters * 4
        c4 = base_filters * 4

        self.block1 = QuantizedConv1dReLU(input_channels, c1, kernel_size=3, padding=1, pool_size=2)
        self.block2 = QuantizedConv1dReLU(c1, c2, kernel_size=3, padding=1, pool_size=2)
        self.block3 = QuantizedConv1dReLU(c2, c3, kernel_size=3, padding=1, pool_size=3)
        self.block4 = QuantizedConv1dReLU(c3, c4, kernel_size=3, padding=1, pool_size=3)

        self.drop = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(c4 * 5, num_classes, bias=True)
        self.quant_enabled = False
        self.weight_quant = FakeQuantizeInt8(8)
        self.act_quant = FakeQuantizeInt8(8)

    def set_quantization(self, enabled=True):
        self.quant_enabled = enabled
        self.weight_quant.force_quant = enabled
        self.act_quant.force_quant = enabled
        self.block1.set_quantization(enabled)
        self.block2.set_quantization(enabled)
        self.block3.set_quantization(enabled)
        self.block4.set_quantization(enabled)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.view(x.size(0), -1)
        x = self.drop(x)
        if self.quant_enabled:
            w = self.weight_quant(self.fc.weight)
            b = self.weight_quant(self.fc.bias) if self.fc.bias is not None else None
            x = F.linear(x, w, b)
            x = self.act_quant(x)
        else:
            x = self.fc(x)
        return x

def audit_model_memory(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fp32_size_kb = (total_params * 4) / 1024.0
    int8_size_kb = (total_params * 1) / 1024.0
    print(f"Model Parameter Count: {total_params}")
    print(f"FP32 Footprint: {fp32_size_kb:.2f} KB")
    print(f"INT8 Footprint (MAX78002): {int8_size_kb:.2f} KB")
    assert int8_size_kb < 2048.0, "Model exceeds 2MB memory ceiling"
    return total_params, fp32_size_kb, int8_size_kb