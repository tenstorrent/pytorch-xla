from copy import copy
from typing import Any, Optional
from weakref import WeakKeyDictionary
import torch
import torch_xla
from torch.utils._pytree import tree_flatten
from torch_xla._internal.jax_workarounds import jax_env_context, jax_import_guard, requires_jax, maybe_get_torchax
import torch_xla.debug.profiler as xp
import abc


class Type:
  F32 = 'f32'
  F64 = 'f64'
  BF16 = 'bf16'
  F16 = 'f16'
  U8 = 'u8'
  S8 = 's8'
  U16 = 'u16'
  S16 = 's16'
  U32 = 'u32'
  S32 = 's32'
  U64 = 'u64'
  S64 = 's64'
  C64 = 'c64'
  C128 = 'c128'
  PRED = 'pred'


_XLA_PT_TYPE_MAP = {
    Type.F32: torch.float32,
    Type.F64: torch.float64,
    Type.BF16: torch.bfloat16,
    Type.F16: torch.float16,
    Type.U8: torch.uint8,
    Type.S8: torch.int8,
    Type.U16: torch.uint16,
    Type.S16: torch.int16,
    Type.U32: torch.uint32,
    Type.S32: torch.int32,
    Type.U64: torch.uint64,
    Type.S64: torch.int64,
    Type.C64: torch.complex64,
    Type.C128: torch.complex128,
    Type.PRED: torch.bool,
}

_PT_XLA_TYPE_MAP = {
    torch.float32: Type.F32,
    torch.float64: Type.F64,
    torch.bfloat16: Type.BF16,
    torch.float16: Type.F16,
    torch.uint8: Type.U8,
    torch.int8: Type.S8,
    torch.uint16: Type.U16,
    torch.int16: Type.S16,
    torch.uint32: Type.U32,
    torch.int32: Type.S32,
    torch.uint64: Type.U64,
    torch.int64: Type.S64,
    torch.complex64: Type.C64,
    torch.complex128: Type.C128,
    torch.bool: Type.PRED,
}


class Shape(object):
  """Wraps a core XLA shape object to provide a more friendly API."""

  def __init__(self, shape):
    self._shape = shape

  @classmethod
  def create(cls, dtype, sizes, dynamic_dimensions=None):
    if dynamic_dimensions is None:
      return Shape({'type': str(dtype), 'sizes': tuple(sizes)})
    return Shape({
        'type': str(dtype),
        'sizes': tuple(sizes),
        'dynamic_dimensions': tuple(dynamic_dimensions)
    })

  @property
  def shape(self):
    return self._shape

  def is_tuple(self):
    return isinstance(self._shape, (list, tuple))

  def tuple_size(self):
    assert self.is_tuple()
    return len(self._shape)

  def tuple_shape(self, index):
    assert self.is_tuple()
    return self._shape[index]

  def is_dynamic(self):
    assert not self.is_tuple()
    return 'dynamic_dimensions' in self._shape

  def as_scalar(self):
    return Shape.create(self.dtype, ())

  @property
  def rank(self):
    assert not self.is_tuple()
    return len(self._shape['sizes'])

  @property
  def sizes(self):
    assert not self.is_tuple()
    return self._shape['sizes']

  @property
  def dynamic_dimensions(self):
    assert not self.is_tuple()
    return self._shape.get('dynamic_dimensions', None)

  @property
  def dtype(self):
    assert not self.is_tuple()
    return self._shape['type']


class Op(object):
  """Wraps an `xla::XlaOp` XLA core operation and provide APIs to build them.

  The APIs exposed by this class are close to an exact match of the API
  documented here:

    https://www.tensorflow.org/xla/operation_semantics

  And here:

    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/client/xla_builder.h

  Args:
    op (_XLAC.XlaOp): The core XLA operation wrapped.
  """

  def __init__(self, op):
    self.op = op

  def shape(self):
    return Shape(torch_xla._XLAC._xla_op_shape(self.op))

  def builder(self):
    return torch_xla._XLAC._xla_op_builder(self.op)

  def build(self, name):
    return torch_xla._XLAC._xla_op_build(name, self.op)

  def __add__(self, rhs):
    return mkop('Add', (self.op, rhs.op))

  def __sub__(self, rhs):
    return mkop('Sub', (self.op, rhs.op))

  def __mul__(self, rhs):
    return mkop('Mul', (self.op, rhs.op))

  def __matmul__(self, rhs):
    return mkop('Dot', (self.op, rhs.op))

  def __truediv__(self, rhs):
    return mkop('Div', (self.op, rhs.op))

  def __pow__(self, rhs):
    return mkop('Pow', (self.op, rhs.op))

  def __mod__(self, rhs):
    return mkop('Rem', (self.op, rhs.op))

  def __neg__(self):
    return mkop('Neg', (self.op,))

  def __not__(self):
    return mkop('Not', (self.op,))

  def __and__(self, rhs):
    return mkop('And', (self.op, rhs.op))

  def __or__(self, rhs):
    return mkop('Or', (self.op, rhs.op))

  def __xor__(self, rhs):
    return mkop('Xor', (self.op, rhs.op))

  def __eq__(self, rhs):
    return mkop('Eq', (self.op, rhs.op))

  def __ne__(self, rhs):
    return mkop('Ne', (self.op, rhs.op))

  def __le__(self, rhs):
    return mkop('Le', (self.op, rhs.op))

  def __lt__(self, rhs):
    return mkop('Lt', (self.op, rhs.op))

  def __ge__(self, rhs):
    return mkop('Ge', (self.op, rhs.op))

  def __gt__(self, rhs):
    return mkop('Gt', (self.op, rhs.op))

  def __lshift__(self, rhs):
    return mkop('ShiftLeft', (self.op, rhs.op))

  def __rshift__(self, rhs):
    return mkop('ShiftRight', (self.op, rhs.op))

  def reshape(self, sizes, dimensions=None, inferred_dimension=None):
    return mkop(
        'Reshape', (self.op,),
        sizes=sizes,
        dimensions=dimensions,
        inferred_dimension=inferred_dimension)

  def dynamic_reshape(self, sizes):
    return mkop('DynamicReshape', (self.op,), sizes=sizes)

  def broadcast(self, sizes):
    return mkop('Broadcast', (self.op,), sizes=sizes)

  def broadcast_in_dim(self, sizes, dimensions):
    return mkop(
        'BroadcastInDim', (self.op,), sizes=sizes, dimensions=dimensions)

  def slice(self, start_indices, limit_indices, strides=None):
    if strides is None:
      strides = [1] * len(start_indices)
    return mkop(
        'Slice', (self.op,),
        start_indices=start_indices,
        limit_indices=limit_indices,
        strides=strides)

  def slice_in_dim(self, start_index, limit_index, dimno, stride=1):
    return mkop(
        'SliceInDim', (self.op,),
        start_index=start_index,
        limit_index=limit_index,
        dimno=dimno,
        stride=stride)

  def dynamic_slice(self, start_indices, slice_sizes):
    start_indices = [x.op for x in start_indices]
    return mkop(
        'DynamicSlice', (self.op,),
        start_indices=start_indices,
        slice_sizes=slice_sizes)

  def dynamic_update_slice(self, update, start_indices):
    start_indices = [x.op for x in start_indices]
    return mkop(
        'DynamicUpdateSlice', (self.op, update.op), start_indices=start_indices)

  def gather(self,
             start_indices,
             offset_dims,
             collapsed_slice_dims,
             start_index_map,
             index_vector_dim,
             indices_are_sorted=None):
    return mkop(
        'Gather', (self.op, start_indices.op),
        offset_dims=offset_dims,
        collapsed_slice_dims=collapsed_slice_dims,
        start_index_map=start_index_map,
        index_vector_dim=index_vector_dim,
        indices_are_sorted=indices_are_sorted)

  def scatter(self,
              scatter_indices,
              updates,
              computation,
              update_window_dims,
              inserted_window_dims,
              scatter_dims_to_operand_dims,
              index_vector_dim,
              indices_are_sorted=None,
              unique_indices=None):
    return mkop(
        'Scatter', (self.op, scatter_indices.op, updates.op),
        computation=computation,
        update_window_dims=update_window_dims,
        inserted_window_dims=inserted_window_dims,
        scatter_dims_to_operand_dims=scatter_dims_to_operand_dims,
        index_vector_dim=index_vector_dim,
        indices_are_sorted=indices_are_sorted,
        unique_indices=unique_indices)

  def conv(self,
           kernel,
           window_strides,
           feature_group_count=1,
           batch_group_count=1,
           padding='valid',
           precision_config=None):
    return mkop(
        'Conv', (self.op, kernel.op),
        window_strides=window_strides,
        feature_group_count=feature_group_count,
        batch_group_count=batch_group_count,
        padding=padding,
        precision_config=precision_config)

  def conv_with_general_padding(self,
                                kernel,
                                window_strides,
                                padding,
                                feature_group_count=1,
                                batch_group_count=1,
                                precision_config=None):
    return mkop(
        'ConvWithGeneralPadding', (self.op, kernel.op),
        window_strides=window_strides,
        feature_group_count=feature_group_count,
        batch_group_count=batch_group_count,
        padding=padding,
        precision_config=precision_config)

  def conv_with_general_dimensions(self,
                                   kernel,
                                   window_strides,
                                   input_batch_dimension,
                                   input_feature_dimension,
                                   kernel_input_feature_dimension,
                                   kernel_output_feature_dimension,
                                   output_batch_dimension,
                                   output_feature_dimension,
                                   input_spatial_dimensions,
                                   kernel_spatial_dimensions,
                                   output_spatial_dimensions,
                                   padding='valid',
                                   feature_group_count=1,
                                   batch_group_count=1,
                                   precision_config=None):
    return mkop(
        'ConvWithGeneralDimensions', (self.op, kernel.op),
        window_strides=window_strides,
        input_batch_dimension=input_batch_dimension,
        input_feature_dimension=input_feature_dimension,
        kernel_input_feature_dimension=kernel_input_feature_dimension,
        kernel_output_feature_dimension=kernel_output_feature_dimension,
        output_batch_dimension=output_batch_dimension,
        output_feature_dimension=output_feature_dimension,
        input_spatial_dimensions=input_spatial_dimensions,
        kernel_spatial_dimensions=kernel_spatial_dimensions,
        output_spatial_dimensions=output_spatial_dimensions,
        feature_group_count=feature_group_count,
        batch_group_count=batch_group_count,
        padding=padding,
        precision_config=precision_config)

  def conv_general(self,
                   kernel,
                   window_strides,
                   padding,
                   input_batch_dimension,
                   input_feature_dimension,
                   kernel_input_feature_dimension,
                   kernel_output_feature_dimension,
                   output_batch_dimension,
                   output_feature_dimension,
                   input_spatial_dimensions,
                   kernel_spatial_dimensions,
                   output_spatial_dimensions,
                   feature_group_count=1,
                   batch_group_count=1,
                   precision_config=None):
    return mkop(
        'ConvGeneral', (self.op, kernel.op),
        window_strides=window_strides,
        padding=padding,
        input_batch_dimension=input_batch_dimension,
        input_feature_dimension=input_feature_dimension,
        kernel_input_feature_dimension=kernel_input_feature_dimension,
        kernel_output_feature_dimension=kernel_output_feature_dimension,
        output_batch_dimension=output_batch_dimension,
        output_feature_dimension=output_feature_dimension,
        input_spatial_dimensions=input_spatial_dimensions,
        kernel_spatial_dimensions=kernel_spatial_dimensions,
        output_spatial_dimensions=output_spatial_dimensions,
        feature_group_count=feature_group_count,
        batch_group_count=batch_group_count,
        precision_config=precision_config)

  def conv_general_dilated(self,
                           kernel,
                           window_strides,
                           padding,
                           lhs_dilation,
                           rhs_dilation,
                           input_batch_dimension,
                           input_feature_dimension,
                           kernel_input_feature_dimension,
                           kernel_output_feature_dimension,
                           output_batch_dimension,
                           output_feature_dimension,
                           input_spatial_dimensions,
                           kernel_spatial_dimensions,
                           output_spatial_dimensions,
                           feature_group_count=1,
                           batch_group_count=1,
                           precision_config=None):
    return mkop(
        'ConvGeneralDilated', (self.op, kernel.op),
        window_strides=window_strides,
        padding=padding,
        lhs_dilation=lhs_dilation,
        rhs_dilation=rhs_dilation,
        input_batch_dimension=input_batch_dimension,
        input_feature_dimension=input_feature_dimension,
        kernel_input_feature_dimension=kernel_input_feature_dimension,
        kernel_output_feature_dimension=kernel_output_feature_dimension,
        output_batch_dimension=output_batch_dimension,
        output_feature_dimension=output_feature_dimension,
        input_spatial_dimensions=input_spatial_dimensions,
        kernel_spatial_dimensions=kernel_spatial_dimensions,
        output_spatial_dimensions=output_spatial_dimensions,
        feature_group_count=feature_group_count,
        batch_group_count=batch_group_count,
        precision_config=precision_config)

  def cast(self, to_type):
    return mkop('Convert', (self.op,), to_type=to_type)

  def bitcast(self, to_type):
    return mkop('BitcastConvert', (self.op,), to_type=to_type)

  def pad(self, value, config):
    return mkop('Pad', (self.op, value.op), config=config)

  def select_and_scatter(self,
                         source,
                         init_value,
                         window_dimensions,
                         window_strides,
                         select_computation,
                         scatter_computation,
                         padding='valid'):
    scalar_shape = self.shape().as_scalar()
    select_computation = Op.make_computation('Select', select_computation,
                                             (scalar_shape, scalar_shape))
    scatter_computation = Op.make_computation('Scatter', scatter_computation,
                                              (scalar_shape, scalar_shape))
    return mkop(
        'SelectAndScatter', (self.op, source.op, init_value.op),
        window_dimensions=window_dimensions,
        window_strides=window_strides,
        select_computation=select_computation,
        scatter_computation=scatter_computation,
        padding=padding)

  def select_and_scatter_with_general_padding(self, source, init_value,
                                              window_dimensions, window_strides,
                                              select_computation,
                                              scatter_computation, padding):
    scalar_shape = self.shape().as_scalar()
    select_computation = Op.make_computation('Select', select_computation,
                                             (scalar_shape, scalar_shape))
    scatter_computation = Op.make_computation('Scatter', scatter_computation,
                                              (scalar_shape, scalar_shape))
    return mkop(
        'SelectAndScatterWithGeneralPadding',
        (self.op, source.op, init_value.op),
        window_dimensions=window_dimensions,
        window_strides=window_strides,
        select_computation=select_computation,
        scatter_computation=scatter_computation,
        padding=padding)

  def max_pool(self,
               kernel_size,
               stride,
               batch_dimension,
               feature_dimension,
               spatial_dimensions,
               padding='valid'):
    return mkop(
        'MaxPool', (self.op,),
        kernel_size=kernel_size,
        stride=stride,
        batch_dimension=batch_dimension,
        feature_dimension=feature_dimension,
        spatial_dimensions=spatial_dimensions,
        padding=padding)

  def reduce(self, init_value, computation, dimensions):
    scalar_shape = self.shape().as_scalar()
    computation = Op.make_computation('Reduce', computation,
                                      (scalar_shape, scalar_shape))
    return mkop(
        'Reduce', (self.op, init_value.op),
        computation=computation,
        dimensions=dimensions)

  def reduce_all(self, init_value, computation):
    scalar_shape = self.shape().as_scalar()
    computation = Op.make_computation('ReduceAll', computation,
                                      (scalar_shape, scalar_shape))
    return mkop('ReduceAll', (self.op, init_value.op), computation=computation)

  def reduce_window(self,
                    init_value,
                    computation,
                    window_dimensions,
                    window_strides,
                    padding='valid'):
    scalar_shape = self.shape().as_scalar()
    computation = Op.make_computation('ReduceWindow', computation,
                                      (scalar_shape, scalar_shape))
    return mkop(
        'ReduceWindow', (self.op, init_value.op),
        computation=computation,
        window_dimensions=window_dimensions,
        window_strides=window_strides,
        padding=padding)

  def select(self, true_value, false_value):
    return mkop('Select', (self.op, true_value.op, false_value.op))

  def transpose(self, permutation):
    return mkop('Transpose', (self.op,), permutation=permutation)

  def triangualr_solve(self,
                       b,
                       left_side=None,
                       lower=None,
                       unit_diagonal=None,
                       transpose_a=None):
    return mkop(
        'TriangularSolve', (self.op, b.op),
        left_side=left_side,
        lower=lower,
        unit_diagonal=unit_diagonal,
        transpose_a=transpose_a)

  def clamp(self, min_value, max_value):
    return mkop('Clamp', (self.op, min_value.op, max_value.op))

  def get_tuple_element(self, index):
    return mkop('GetTupleElement', (self.op,), index=index)

  def conditional(self, true_operand, false_operand, true_computation,
                  false_computation):
    true_computation = Op.make_computation('CondTrue', true_computation,
                                           (true_operand,))
    false_computation = Op.make_computation('CondFalse', false_computation,
                                            (false_operand,))
    return mkop(
        'Conditional', (self.op, true_operand.op, false_operand.op),
        true_computation=true_computation,
        false_computation=false_computation)

  @classmethod
  def wrap_function(cls, fn):

    def wrapper(*args, **kwargs):
      if len(args) == 1:
        arg = args[0]
        shape = arg.shape()
        if shape.is_tuple():
          args = [
              arg.get_tuple_element(i) for i in range(0, shape.tuple_size())
          ]
      result = fn(*args, **kwargs)
      return Op.tuple(result) if isinstance(result, (tuple, list)) else result

    return wrapper

  @classmethod
  def make_computation(cls, name, computation, args_or_shapes, **kwargs):
    if not callable(computation):
      return computation
    shapes = []
    for arg in args_or_shapes:
      shapes.append(arg if isinstance(arg, Shape) else arg.shape())
    return create_computation(name, Op.wrap_function(computation), shapes,
                              **kwargs)

  def mkconditional(self, ops, true_fn, false_fn, **kwargs):
    input_tuple = Op.tuple(ops)
    return self.conditional(input_tuple, input_tuple, true_fn, false_fn)

  def while_loop(self, condition_computation, body_computation):
    condition_computation = Op.make_computation('Condition',
                                                condition_computation, (self,))
    body_computation = Op.make_computation('Body', body_computation, (self,))
    return mkop(
        'While', (self.op,),
        condition_computation=condition_computation,
        body_computation=body_computation)

  @classmethod
  def mkwhile(self, ops, condition_fn, body_fn, **kwargs):
    input_tuple = Op.tuple(ops)
    return input_tuple.while_loop(
        condition_computation=condition_fn, body_computation=body_fn)

  def get_dimension_size(self, dimension):
    return mkop('GetDimensionSize', (self.op,), dimension=dimension)

  def set_dimension_size(self, size, dimension):
    return mkop('SetDimensionSize', (self.op, size.op), dimension=dimension)

  def rev(self, dimensions):
    return mkop('Rev', (self.op,), dimensions=dimensions)

  def acos(self):
    return mkop('Acos', (self.op,))

  def asin(self):
    return mkop('Asin', (self.op,))

  def atan(self):
    return mkop('Atan', (self.op,))

  def ceil(self):
    return mkop('Ceil', (self.op,))

  def cos(self):
    return mkop('Cos', (self.op,))

  def cosh(self):
    return mkop('Cosh', (self.op,))

  def erf(self):
    return mkop('Erf', (self.op,))

  def erfc(self):
    return mkop('Erfc', (self.op,))

  def erfinf(self):
    return mkop('ErfInv', (self.op,))

  def exp(self):
    return mkop('Exp', (self.op,))

  def expm1(self):
    return mkop('Expm1', (self.op,))

  def floor(self):
    return mkop('Floor', (self.op,))

  def log(self):
    return mkop('Log', (self.op,))

  def log1p(self):
    return mkop('Log1p', (self.op,))

  def sqrt(self):
    return mkop('Sqrt', (self.op,))

  def real(self):
    return mkop('Real', (self.op,))

  def imag(self):
    return mkop('Imag', (self.op,))

  def clz(self):
    return mkop('Clz', (self.op,))

  def conj(self):
    return mkop('Conj', (self.op,))

  def rsqrt(self):
    return mkop('Rsqrt', (self.op,))

  def sin(self):
    return mkop('Sin', (self.op,))

  def sinh(self):
    return mkop('Sinh', (self.op,))

  def tan(self):
    return mkop('Tan', (self.op,))

  def tanh(self):
    return mkop('Tanh', (self.op,))

  def atan2(self, other):
    return mkop('Atan2', (self.op, other.op))

  def max(self, other):
    return mkop('Max', (self.op, other.op))

  def min(self, other):
    return mkop('Min', (self.op, other.op))

  def scalar_like(self, value):
    shape = self.shape()
    v = Op.scalar(self.builder(), value, dtype=shape.dtype)
    return v.broadcast(shape.sizes)

  def zeros_like(self):
    return self.scalar_like(0)

  def ones_like(self):
    return self.scalar_like(1)

  @classmethod
  def _extract_xla_ops(cls, ops):
    return [x.op for x in ops]

  @classmethod
  def tuple(cls, ops, builder=None):
    return mkop('Tuple', Op._extract_xla_ops(ops), builder=builder)

  @classmethod
  def concat_in_dim(cls, ops, dimension, builder=None):
    return mkop(
        'ConcatInDim',
        Op._extract_xla_ops(ops),
        builder=builder,
        dimension=dimension)

  @classmethod
  def call(cls, computation, ops, builder=None):
    computation = Op.make_computation('Call', computation, ops)
    return mkop(
        'Call',
        Op._extract_xla_ops(ops),
        computation=computation,
        builder=builder)

  @classmethod
  def constant(cls, builder, value):
    return mkleaf('Constant', builder, value=value)

  @classmethod
  def scalar(cls, builder, value, dtype=None):
    return mkleaf(
        'Constant',
        builder,
        value=torch.tensor(value, dtype=cls.to_torch_type(dtype)))

  @classmethod
  def zero(cls, builder, dtype=None):
    return cls.scalar(builder, 0, dtype=dtype)

  @classmethod
  def one(cls, builder, dtype=None):
    return cls.scalar(builder, 1, dtype=dtype)

  @classmethod
  def iota(cls, builder, shape, iota_dimension):
    return mkleaf(
        'Iota', builder, shape=shape.shape, iota_dimension=iota_dimension)

  @classmethod
  def sort(cls, ops, comparator, dimension=None, is_stable=None):
    return mkop(
        'Sort',
        Op._extract_xla_ops(ops),
        comparator=comparator,
        dimension=dimension,
        is_stable=is_stable)

  @classmethod
  def map(cls, ops, computation, dimensions, static_operands=(), builder=None):
    return mkop(
        'Map',
        Op._extract_xla_ops(ops),
        builder=builder,
        computation=computation,
        dimensions=dimensions,
        static_operands=Op._extract_xla_ops(static_operands))

  @classmethod
  def to_torch_type(cls, dtype):
    return _XLA_PT_TYPE_MAP[dtype] if dtype else torch.float32

  @classmethod
  def from_torch_type(cls, dtype):
    return _PT_XLA_TYPE_MAP[dtype]


def create_builder(name):
  return torch_xla._XLAC._xla_op_create_builder(name)


def mkshape(dtype, sizes, dynamic_dimensions=None):
  return Shape.create(dtype, sizes, dynamic_dimensions=dynamic_dimensions)


def mkop(name, ops, **kwargs):
  builder = kwargs.get('builder', None)
  if builder is None:
    assert ops
    builder = torch_xla._XLAC._xla_op_builder(ops[0])
  return Op(torch_xla._XLAC._xla_op_create(builder, name, ops, kwargs))


def mkleaf(name, builder, **kwargs):
  return Op(torch_xla._XLAC._xla_op_create(builder, name, (), kwargs))


def mkparam(builder, param_no, shape):
  return Op(torch_xla._XLAC._xla_op_param(builder, param_no, shape.shape))


def tensor_shape(tensor, device=''):
  if isinstance(tensor, (list, tuple)):
    return [
        Shape(torch_xla._XLAC._xla_op_tensor_shape(t, device)) for t in tensor
    ]
  return Shape(torch_xla._XLAC._xla_op_tensor_shape(tensor, device))


def create_computation(name, fn, shapes, **kwargs):
  builder = create_builder(name)
  params = []
  for shape in shapes:
    p = mkparam(builder, len(params), shape)
    params.append(p)
  root = fn(*params, **kwargs)
  return root.build(name)


def computation_from_module_proto(name, proto):
  return torch_xla._XLAC._xla_op_computation_from_module_proto(name, proto)


def get_computation_hlo(computation):
  return torch_xla._XLAC._xla_computation_text(computation)


def xla_computation_as_func(computation, name=None):
  """Converts an XlaComputation object to a callable that takes XLATensors."""
  name = name or "xla::computation"
  if not '::' in name:
    name = 'xla::' + name

  def fn(input_tensors):
    result = torch_xla._XLAC._xla_user_computation(name, input_tensors,
                                                   computation)
    return result

  return fn


class FlattenedInputFunc:
  """Wraps a Python function to provide an interface for calling it with
  flattened PyTorch tensor arguments and handling flattened outputs.

  This class splits the call to the original function into 3 methods:
  1. preprocess
  2. flat_call
  3. propost_process
  
  The invariant to hold is:
  ```
    flattened = FlattenedInputFunc(orig_function, sample_args, sample_kwargs)
    orig_function(*sample_args, **sample_kwargs) == flattend.post_process(
        flattend.flat_call(
          flattened.preprocess(
              sample_args, sample_kwargs)))
  ```

  This class is useful for integrating functions with complex, nested argument
  structures (pytrees) into systems that expect inputs and outputs as flat
  lists of tensors, such as certain JAX transformations or model serving
  frameworks.

  The structure of the inputs is determined from `sample_args` and
  `sample_kwargs` provided during initialization. Non-tensor arguments
  from these samples are stored and automatically re-inserted when the
  function is called with new tensor inputs.
  """

  def __init__(self, orig_func):
    self.tx = maybe_get_torchax()
    self.orig_func = orig_func
    self.non_tensors = None
    self.in_spec = None
    self.out_spec = None
    # This is used to mark position of tensor inputs
    self._sentinel = object()

  def preprocess(self, args, kwargs=None):
    with jax_env_context():
      import jax
      kwargs = kwargs or {}
      flattened_inputs, spec = jax.tree.flatten((args, kwargs))
      tensors = tuple(
          a for a in flattened_inputs if isinstance(a, torch.Tensor))
      self.non_tensors = tuple(
          self._sentinel if isinstance(a, torch.Tensor) else a
          for a in flattened_inputs)
      # Note: saving the non_tensors and in_spec here
      # because flat_call needs to take those as closure, not as inputs
      # flat_call is meant to be processed by, say, jax.jit.
      self.in_spec = spec
      return tensors

  def flat_call(self, flat_input):
    with jax_env_context():
      import jax
      assert self.in_spec is not None, 'flat call only makes sense after preprocess is called'

      # Put the tensor input and the non tensor input together
      new_flattened = list(self.non_tensors)
      tensor_args_iter = iter(flat_input)
      for i in range(len(new_flattened)):
        if new_flattened[i] is self._sentinel:
          new_flattened[i] = next(tensor_args_iter)

      args, kwargs = jax.tree.unflatten(self.in_spec, new_flattened)
      res = self.orig_func(*args, **kwargs)
      flattened_out, spec = jax.tree.flatten(res)
      self.out_spec = spec
      return flattened_out

  def postprocess(self, res_flattened):
    with jax_env_context():
      import jax
      assert self.out_spec is not None, 'post process only makes sense after flat_call is called'
      res = jax.tree.unflatten(self.out_spec, res_flattened)
      return res


class CompiledCallableWithCache(abc.ABC):
  """This class is meant to be subclassed.
  
  Given a function that can create shape-specialized computations
  one can treat it as a shape-generic computations if one accepts that
  shape changes will trigger recompile.
  
  This class captures this idea by having a cache for shapes.
  the function that specializes are "passed in" via subclassing.
  """

  def __init__(self, flat_input_func: FlattenedInputFunc):
    self._cache = {}
    self._flat_input_func = flat_input_func

  def __call__(self, *args, **kwargs):
    tensor_input_flattened = self._flat_input_func.preprocess(args, kwargs)

    abstract_inputs = tuple((a.shape, a.dtype) if a is not None else a
                            for a in tensor_input_flattened)

    cache_key = (abstract_inputs, tuple(self._flat_input_func.non_tensors),
                 self._flat_input_func.in_spec)
    if cache_key not in self._cache:
      self._cache[cache_key] = self.specialize(tensor_input_flattened)

    flat_callable = self._cache[cache_key]

    output = flat_callable(tensor_input_flattened)
    return self._flat_input_func.postprocess(output)

  @abc.abstractmethod
  def specialize(self, sample_flat_args):
    pass


class JaxFlattenedInputFunc(FlattenedInputFunc):
  """When we know that the original function is a jax function, 
  
  we need to do more preprocessing. In particular, translate dtypes from
  torch.dtype to jax.dtype
  """

  def preprocess(self, args, kwargs=None):
    res = super().preprocess(args, kwargs)
    tx = maybe_get_torchax()
    self.non_tensors = tuple(
        tx.ops.mappings.t2j_dtype(a) if isinstance(a, torch.dtype) else a
        for a in self.non_tensors)
    return res


class JaxCallable(CompiledCallableWithCache):

  def __init__(self, jax_func):
    super().__init__(JaxFlattenedInputFunc(jax_func))

  def specialize(self, sample_flat_args):
    import jax
    tx = maybe_get_torchax()
    sample_flat_args = tuple(
        jax.ShapeDtypeStruct(a.shape, tx.ops.mappings.t2j_dtype(a.dtype)
                            ) if a is not None else None
        for a in sample_flat_args)

    with xp.Trace('jax_to_xla_computation'):
      lowered = jax.jit(
          self._flat_input_func.flat_call,
          keep_unused=True).lower(sample_flat_args)
      hlo_ir = lowered.compiler_ir('hlo')
      # Get a protobuf representation of the HLO. `as_serialized_hlo_module_proto` is
      # mentioned at https://github.com/jax-ml/jax/discussions/22266
      hlo_module = hlo_ir.as_serialized_hlo_module_proto()  # type: ignore
      computation = computation_from_module_proto('jax_callable', hlo_module)
      return xla_computation_as_func(computation, 'jax_func')


class XlaCallable(CompiledCallableWithCache):
  """XlaCallable lets you implement LazyTensor callables using the Python XlaBuilder API."""

  def __init__(self, xla_func):
    """xla_func is a function that takes XlaOp as the placeholder and expresses
    math using xla builder python API above
    """
    super().__init__(FlattenedInputFunc(xla_func))

  def specialize(self, sample_flat_args):
    sample_args_shapes = tuple(
        Shape.create(Op.from_torch_type(a.dtype), a.shape)
        for a in sample_flat_args)

    with xp.Trace('xla::computation'):
      name = 'xla::computation'
      builder = create_builder(name)
      params = []
      for a in sample_flat_args:
        p = mkparam(builder, len(params),
                    mkshape(Op.from_torch_type(a.dtype), a.shape))
        params.append(p)
      root = Op.tuple(self._flat_input_func.flat_call(params))
      computation = root.build(name)
      return xla_computation_as_func(computation, name)


def _jax_to_xla_computation_cache_elements() -> int:
  size = 0
  for func in _JAX_TO_XLA_COMPUTATION_CACHE.values():
    size += len(func._cache)
  return size


_JAX_TO_XLA_COMPUTATION_CACHE = {}


@requires_jax
def call_jax(jax_func,
             args: tuple[Any, ...],
             kwargs: Optional[dict[str, Any]] = None,
             name=None,
             override_hash=None):
  """
  Call a JAX function `jax_func` with the given `args` and `kwargs` that may contain
  XLA tensors.

  Args:
    jax_func: a functionally pure Python callable that does some math on JAX arrays.
              It needs to be `jax.jit` traceable.

    args: a tuple of arguments to pass to `jax_func`. Any XLA tensors are turned into
          JAX arrays before being passed to `jax_func`.

    kwargs: a dictionary of keyword arguments to pass to `jax_func`. Any XLA tensors are
          turned into JAX arrays before being passed to `jax_func`.

    name: Name of the graph given to xla computation.
    
    override_hash: Optionally set a value for to be used as hash key to cache the
    precompiled callable. By default we will use the id of jax_func as the key.
    If jax_func is generated in a closure, it's id will change. So one can override
    the hash key to be the id of the parent function to not have collisions.

  ## Example

      >>> import torch
      >>> import torch_xla
      >>> import torch_xla.core.xla_builder as xb
      >>>
      >>> def f(a, b):
      >>>   # Call any JAX functionality here.
      >>>   import jax.numpy as jnp
      >>>   return a + jnp.sin(b)
      >>>
      >>> # Pass PyTorch/XLA tensors to JAX function this way.
      >>> a = torch.ones((3, 3), device='xla')
      >>> b = xb.call_jax(f, (a, a))
      >>>
      >>> # Result is the same as if we ran the equivalent torch ops.
      >>> torch.testing.assert_close(b.cpu(), torch.sin(torch.ones(3, 3)) + 1)

  ## Caching

  In order to call `jax_func`, we will jit compile it into HLO, which involves tracing
  the function. The address of `jax_func` and the shapes of `args` and `kwargs` is used
  as the key into a cache to avoid repeated tracing/compilation, similar to how `jax.jit`
  works. If you get tracing overhead, check if `jax_func` is being redefined all the time.
  A common mistake is defining `jax_func` as a local function, e.g. during a training step.
  """
  import jax
  from jax._src import config

  tx = maybe_get_torchax()
  flattened, _ = jax.tree.flatten((args, kwargs))
  kwargs = kwargs or {}
  if tx is not None and any(isinstance(a, tx.tensor.Tensor) for a in flattened):
    return tx.interop.call_jax(jax_func, *args, **kwargs)

  hash_key = (override_hash or id(jax_func), config.trace_context())
  if hash_key not in _JAX_TO_XLA_COMPUTATION_CACHE:
    _JAX_TO_XLA_COMPUTATION_CACHE[hash_key] = JaxCallable(jax_func)

  wrapped_jax_callable = _JAX_TO_XLA_COMPUTATION_CACHE[hash_key]
  kwargs = kwargs or {}
  return wrapped_jax_callable(*args, **kwargs)


def create_placeholder_tensor(shape, dtype):
  """
  Creates a placeholder tensor that does not hold any device buffer.
  This is primarily useful for staging out the HLO of a user computation.
  Accessing the value of the tensor will panic.
  """
  dtype = Op.from_torch_type(dtype)
  shape = mkshape(dtype, shape)
  return torch_xla._XLAC._xla_create_placeholder_tensor(shape.shape)
