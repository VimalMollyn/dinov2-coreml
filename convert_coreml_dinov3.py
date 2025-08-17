import torch
import coremltools as ct

model = torch.hub.load('facebookresearch/dinov3', 'dinov3_vits16', source='github', weights="https://dinov3.llamameta.net/dinov3_vits16/dinov3_vits16_pretrain_lvd1689m-08c60483.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoieGJ3dzM4M2o4djlnbmQ0d3NhNHI5c3dkIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NTUzNzA0Mjl9fX1dfQ__&Signature=F%7E7LBWyNTy0c4nSnOIjIaGuiUd-AHIgIQzmfwivW28ducAs3kLadd2wj2taoY4K588BkPsp4OItjzYXImnLUGYnPGhzISTR%7EgDgyRhF9yukwsC8uCumExqnzM8sB9KBP0Lracpo3BdCRe9wWgucX21we09jH%7EBuCleOoDVeLZeve9%7EUasCwr-miQhu0vmHirfWdUZ4A%7E%7EHrJ1KU-IxigsCwAq6pfE66tCRIF9wtZaw62RbZi29Frr3AEDrCu3aDJzV6OPemu%7EOX0L-BPliKIh-HZhUI38sDMmvJ2bKKOGArKVuZKuw60rJx0N7FAmiKRjJMfAd-hqK8Qf3vzdj%7EsIQ__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1082201730553524")

img_size = 512 
example_input = torch.randn(1, 3, img_size, img_size)

# create a wrapper for the model
class DINOv2Wrapper(torch.nn.Module):
    def __init__(self, model):
        super(DINOv2Wrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model.forward_features(x)["x_norm_patchtokens"]

wrapper = DINOv2Wrapper(model.eval())

# trace the wrapper
with torch.no_grad():
    wrapper._requires_grad = False
    wrapper.eval()

    traced_model = torch.jit.trace(wrapper, example_input)

from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil import register_torch_op
from coremltools.converters.mil.frontend.torch.torch_op_registry import _TORCH_OPS_REGISTRY

@register_torch_op
def _upsample_bicubic2d_aa(context, node):
    a = context[node.inputs[0]]
    output_size = context[node.inputs[1]].val
    align_corners = context[node.inputs[2]].val
    scale = context[node.inputs[3]]
    if scale is None:
        input_height = a.shape[-2]
        input_width = a.shape[-1]
        scale_h = output_size[0] / input_height
        scale_w = output_size[1] / input_width
    else:
        scale_h = scale.val[0]
        scale_w = scale.val[1]
    
    x = mb.upsample_bilinear(x=a, scale_factor_height=scale_h, scale_factor_width=scale_w, align_corners=align_corners, name=node.name)
    context.add(x)

# convert to coreml
model_from_trace = ct.convert(
    traced_model,
    inputs=[ct.TensorType(name="image", shape=example_input.shape)],
    outputs=[ct.TensorType(name="x_norm_patchtokens")],
    compute_precision=ct.precision.FLOAT16
)


# save the model
model_from_trace.save("dinov3_vits16_518_fp16.mlpackage")