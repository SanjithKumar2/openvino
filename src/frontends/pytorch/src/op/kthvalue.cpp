#include "openvino/op/topk.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/squeeze.hpp"
#include "../utils.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset3.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/opsets/opset12.hpp"

namespace ov{
namespace frontend{
namespace pytorch{
namespace op{
    OutputVector translate_kthvalue(const NodeContext& context){
        num_inputs_check(context, 3, 3);
        auto input = context.get_input(0);
        auto k = context.get_input(1);
        auto dim = context.get_input(2);
        bool keepdim = false;

        auto tensor_rank = std::get<1>(get_shape_rank(context, input, false));
        auto tensor_rank_correct_type = context.mark_node(std::make_shared<ov::op::v1::ConvertLike>(tensor_rank, dim));
        auto normalised_axis = normalize_axis(context, dim, tensor_rank_correct_type);
        auto topk = context.mark_node(std::make_shared<ov::op::v3::TopK>(
            input,
            k,
            normalised_axis,
            ov::op::v3::TopK::Mode::MIN,
            ov::op::v3::TopK::SortType::NONE,
            element::i64
        ));

        Output<Node> values = topk -> output(0);
        Output<Node> indices = topk -> output(1);
        if (!keepdim){
        auto const_axis = context.mark_node(ov::op::v0::Constant::create(element::i32, Shape{1}, {1}));
        auto values = context.mark_node(std::make_shared<ov::op::v0::Squeeze>(topk->output(0), const_axis));
        auto indices = context.mark_node(std::make_shared<ov::op::v0::Squeeze>(topk->output(1), const_axis));
        }
        return {values, indices};
    }
} 
} // namespace pytorch
} // namespace frontend
} // namespace ov