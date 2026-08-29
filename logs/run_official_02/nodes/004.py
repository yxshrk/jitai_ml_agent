"""Run the frozen best-known stack with exponential moving-average weights.

The parent implementation is preserved verbatim through its public CLI. The only
change is that floating-point optimizer parameters are averaged with EMA decay
0.998, and checkpoint state_dict calls receive the averaged parameters. Thus the
parent's validation checkpoint selection and output-writing behavior are retained.
"""
import importlib.util
import runpy
import sys

import torch


_EMA_BY_PTR = {}
_EMA_DECAY = 0.998


def _patch_optimizer(cls):
    original_step = cls.step

    def ema_step(self, *args, **kwargs):
        result = original_step(self, *args, **kwargs)
        with torch.no_grad():
            for group in self.param_groups:
                for param in group["params"]:
                    if param is None or not param.requires_grad or not param.is_floating_point():
                        continue
                    ptr = param.data_ptr()
                    average = _EMA_BY_PTR.get(ptr)
                    if average is None or average.shape != param.shape or average.device != param.device:
                        _EMA_BY_PTR[ptr] = param.detach().clone()
                    else:
                        average.mul_(_EMA_DECAY).add_(param.detach(), alpha=1.0 - _EMA_DECAY)
        return result

    cls.step = ema_step


def _patch_state_dict():
    original_state_dict = torch.nn.Module.state_dict

    def ema_state_dict(self, *args, **kwargs):
        state = original_state_dict(self, *args, **kwargs)
        for key, value in list(state.items()):
            if not torch.is_tensor(value) or not value.is_floating_point():
                continue
            average = _EMA_BY_PTR.get(value.data_ptr())
            if average is not None and average.shape == value.shape:
                state[key] = average.detach().clone()
        return state

    torch.nn.Module.state_dict = ema_state_dict


def main():
    spec = importlib.util.find_spec("zoo.ablate_fields")
    if spec is None or not spec.origin:
        raise SystemExit("cannot locate zoo.ablate_fields on PYTHONPATH")

    _patch_optimizer(torch.optim.Adam)
    _patch_optimizer(torch.optim.AdamW)
    _patch_state_dict()

    sys.argv = [spec.origin, "--field-level", "0", "--regularized", *sys.argv[1:]]
    runpy.run_path(spec.origin, run_name="__main__")


if __name__ == "__main__":
    main()
