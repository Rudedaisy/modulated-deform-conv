#include <torch/extension.h>
 
// CUDA forward declarations

int deform_conv2d_forward_cuda(at::Tensor input, at::Tensor weight, at::Tensor bias,
			       at::Tensor offset, at::Tensor output,
			       const int kernel_h,const int kernel_w, const int stride_h, const int stride_w,
			       const int pad_h, const int pad_w, const int dilation_h,const int dilation_w,
			       const int group, const int deformable_group, const int KG, const int in_step,
			       const bool with_bias);

int deform_conv2d_backward_cuda(at::Tensor input, at::Tensor weight, at::Tensor bias,at::Tensor offset,
				at::Tensor grad_input, at::Tensor grad_weight, at::Tensor grad_bias,
				at::Tensor grad_offset, at::Tensor grad_output,
				const int kernel_h,const int kernel_w,const int stride_h,const int stride_w,
				const int pad_h,const int pad_w,const int dilation_h,const int dilation_w,
				const int group,const int deformable_group, const int KG, const int in_step,const bool with_bias);

int fused_deform_conv2d_forward_cuda(at::Tensor input, at::Tensor weight, at::Tensor bias, at::Tensor offset_weight,
                               at::Tensor offset, at::Tensor output,
                               const int kernel_h,const int kernel_w, const int stride_h, const int stride_w,
                               const int pad_h, const int pad_w, const int dilation_h,const int dilation_w,
                               const int group, const int deformable_group, const int KG, const int in_step,
                               const bool with_bias);

//int fused_deform_conv2d_backward_cuda(at::Tensor input, at::Tensor weight, at::Tensor bias,at::Tensor offset,
//                                at::Tensor grad_input, at::Tensor grad_weight, at::Tensor grad_bias,
//                                at::Tensor grad_offset, at::Tensor grad_output,
//                                const int kernel_h,const int kernel_w,const int stride_h,const int stride_w,
//                                const int pad_h,const int pad_w,const int dilation_h,const int dilation_w,
//                                const int group,const int deformable_group, const int KG, const int in_step,const bool with_bias);

at::Tensor modulated_deform_conv2d_forward_cuda(at::Tensor input, at::Tensor weight, at::Tensor bias,
						at::Tensor offset, at::Tensor mask,
						const int kernel_h,const int kernel_w, const int stride_h, const int stride_w,
						const int pad_h, const int pad_w, const int dilation_h,const int dilation_w,
						const int group, const int deformable_group,const int in_step,
						const bool with_bias);

py::tuple modulated_deform_conv2d_backward_cuda(at::Tensor input, at::Tensor weight, at::Tensor bias,at::Tensor offset, at::Tensor mask,
						at::Tensor grad_output,
						const int kernel_h,const int kernel_w,const int stride_h,const int stride_w,
						const int pad_h,const int pad_w,const int dilation_h,const int dilation_w,
						const int group,const int deformable_group, const int in_step,const bool with_bias);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("deform_conv2d_forward_cuda", &deform_conv2d_forward_cuda,
	"deform_conv2d_forward_cuda");
  m.def("deform_conv2d_backward_cuda", &deform_conv2d_backward_cuda,
	"deform_conv2d_backward_cuda");
  m.def("fused_deform_conv2d_forward_cuda", &fused_deform_conv2d_forward_cuda,
        "fused_deform_conv2d_forward_cuda");
  //m.def("fused_deform_conv2d_backward_cuda", &fused_deform_conv2d_backward_cuda,
  //      "fused_deform_conv2d_backward_cuda");
  
  //m.def("deform_conv3d_forward_cuda", &deform_conv3d_forward_cuda,
  //	"deform_conv3d_forward_cuda");
  //m.def("deform_conv3d_backward_cuda", &deform_conv3d_backward_cuda,
  //	"deform_conv3d_backward_cuda");
  
  //  m.def("modulated_deform_conv3d_forward_cuda", &modulated_deform_conv3d_forward_cuda,
  //	"modulated_deform_conv3d_forward_cuda");
  //m.def("modulated_deform_conv3d_backward_cuda", &modulated_deform_conv3d_backward_cuda,
  //	"modulated_deform_conv3d_backward_cuda");
  
  m.def("modulated_deform_conv2d_forward_cuda", &modulated_deform_conv2d_forward_cuda,
	"modulated_deform_conv2d_forward_cuda");
  m.def("modulated_deform_conv2d_backward_cuda", &modulated_deform_conv2d_backward_cuda,
	"modulated_deform_conv2d_backward_cuda");
}
