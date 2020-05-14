// Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#pragma once

#include <vespa/eval/eval/tensor_function.h>

namespace vespalib::tensor {

/**
 * Tensor function for simple join operations on dense tensors.
 **/
class DenseSimpleJoinFunction : public eval::tensor_function::Join
{
    using Super = eval::tensor_function::Join;
public:
    enum class Primary : uint8_t { LHS, RHS };
    enum class Overlap : uint8_t { INNER, OUTER, FULL };
    using join_fun_t = ::vespalib::eval::tensor_function::join_fun_t;
private:
    Primary _primary;
    Overlap _overlap;
    bool    _inplace;
public:
    DenseSimpleJoinFunction(const eval::ValueType &result_type,
                            const TensorFunction &lhs,
                            const TensorFunction &rhs,
                            join_fun_t function_in,
                            Primary primary_in,
                            Overlap overlap_in,
                            bool inplace_in);
    ~DenseSimpleJoinFunction() override;
    Primary primary() const { return _primary; }
    Overlap overlap() const { return _overlap; }
    bool inplace() const { return _inplace; }
    size_t factor() const;
    eval::InterpretedFunction::Instruction compile_self(const eval::TensorEngine &engine, Stash &stash) const override;
    static const eval::TensorFunction &optimize(const eval::TensorFunction &expr, Stash &stash);
};

} // namespace vespalib::tensor
