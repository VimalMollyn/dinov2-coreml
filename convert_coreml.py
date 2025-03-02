import torch
import coremltools as ct

model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')

img_size = 518 # only 518 supported now, due to upsample_bicubic2d_aa not being supported by coremltools for other sizes
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
model_from_trace.save("dinov2_small_14_registers_518_fp16.mlpackage")