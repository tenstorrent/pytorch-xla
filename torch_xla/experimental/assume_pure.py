from functools import wraps, partial
from typing import Dict

import torch
import torch.nn as nn
from torch.utils._pytree import tree_flatten, tree_unflatten
import torch_xla
from torch_xla._internal.jax_workarounds import requires_jax
import torch_xla.core.xla_builder as xb

_XLA_COMPUTATION_CACHE = {}


@requires_jax
def assume_pure(fn, *, add_rng_seed_argument=False):
  """Decorates a pure PyTorch/XLA function to skip expensive re-tracing.

  Returns a new function that will only be traced once for each unique
  input tensor shapes or non-tensor input argument values. This is useful
  for removing Lazy Tensor tracing overhead.

  The decorated function must be pure (i.e. no side-effects, behavior
  only depends on inputs).

  Limitations:
  - The decorated function can only use upstream PyTorch operators e.g.
    `torch.einsum`, `torch.nn.functional.layer_norm`, and a few PyTorch/XLA operators:
    * `torch_xla.experimental.assume_pure` (recursive `assume_pure`)
    * `torch_xla.distributed.spmd.mark_sharding`

  - Other custom PyTorch/XLA operations such as `flash_attention` are not
    supported. This limitation may be lifted in the future.

  Args:
    fn: Callable, the function that is assumed to be pure.
      A pure function means, if the inputs are fixed then the output is also fixed
      ie. a mathematical function. NOTE: functions that does randomness generation
      are NOT pure by this definition.
      
    add_rng_seed_argument: bool, if true, then the returned function will take 
      an extra 'rng_seed' argument. A function with different rng_seed can produce
      different result, so the lifted function becomes pure. rng_seed must be int

      
  Example:
  
  ```
  def add_randn(a):
    return a + torch.randn_like(a)
  ```
  
  add_randn is not a pure function; but assume_pure(add_randn) assumes it is pure
  and hardcodes the rng key at tracing time; making add_randn behaves differently 
  (thus being incorrect).
  
  if we do add_randn_p = assume_pure(add_randn, add_rng_seed_argument=True), then
  we can call add_randn_p(a, rng_seed=0) to get one result and add_randn_p(a, rng_seed=0)
  to get another result.
  """
  from torchax.interop import jax_view
  import torchax
  if add_rng_seed_argument:

    def new_fn(*args, **kwargs):
      env = torchax.default_env()
      rng_seed = args[0]
      args = args[1:]
      env.manual_seed(rng_seed._elem)
      return fn(*args, **kwargs)

    jitted = j2t_autograd(jax_view(new_fn))

    def func_to_return(*args, **kwargs):
      rng_seed = kwargs.get('rng_seed')
      assert rng_seed is not None, 'Missing keyword argument rng_seed.'
      kwargs.pop('rng_seed')
      if isinstance(rng_seed, int):
        rng_seed = torch.tensor(rng_seed, dtype=torch.uint32, device='xla')
      args = (rng_seed, *args)
      result = jitted(*args, **kwargs)
      return result

    return func_to_return
  else:
    return j2t_autograd(jax_view(fn))


@requires_jax
def j2t_autograd(fn):
  """Given a JAX function, returns a PyTorch autograd function implemented with `jax.vjp(fn)`.

  It wraps `fn` with `jax.vjp` to compute both the output and residuals (intermediate
  activations). The wrapped function is then run via `call_jax` and integrated into
  the PyTorch autograd framework by saving the residuals into the context object.
  """
  import torchax.interop
  # When j2t_autograd calls call_jax, the first arg is vjp, and the second
  # arg is the actual function. So we want the hash key to be based on the second
  # arg.
  return torchax.interop.j2t_autograd(
      fn, call_jax=lambda fn, *args: xb.call_jax(fn, args))


class PureModule(nn.Module):
  """Wraps a module whose forward pass is known to be free of side-effects and whose
  behavior only depends on the inputs.

  It behaves as if decorating the wrapped module's functionalized forward pass with `@assume_pure`.

  This wrapper has a few advantages over the underlying module:
  - `PureModule`s will only be traced once.
  - Framework profile scopes added via `xp.Trace` will show up in both the forward
    and the backward pass.
  """

  def __init__(self, module: nn.Module) -> None:
    super().__init__()
    self._module = module
    self._pure_forward = assume_pure(partial(_pure_forward, self._module))

  def forward(self, *args, **kwargs):
    params = dict(self._module.named_parameters())
    buffers = dict(self._module.named_buffers())
    return self._pure_forward(params, buffers, args, kwargs)


def _pure_forward(module, params, buffers, args, kwargs):
  return torch.func.functional_call(module, (params, buffers), args, kwargs)


def make_fake_inputs(input):
  """Creates a fake input for the given input torch tensor. If the input
  is not a tensor, it returns the input as is.
  """
  if isinstance(input, torch.Tensor):
    t = xb.create_placeholder_tensor(input.shape, input.dtype)
    return t.requires_grad_(input.requires_grad)
  return input


def prepare_computation_inputs(fn_ctx, flat_fake_inputs, flat_inputs):
  """Prepares the computation inputs for the XLA computation.

  fn_ctx contains the mapping fake tensors in flat_fake_inputs to the input 
  parameter id in xla computation. We use this mapping to pick actual inputs
  from flat_inputs to create the computation inputs.
  
  Args:
  fn_ctx: The lowering context for the function.
  flat_fake_inputs: The flattened fake inputs for the function.
  flat_inputs: The flattened actual inputs for the function.
  Returns:
  computation_inputs: The computation inputs for the XLA computation.
  hoisted_vars_map: The hoisted variables map for the XLA computation.
  hlo_input_id_to_input_index_map: The mapping from HLO input IDs to input
    indices for flat_inputs.
  """
  all_hlo_input_vars_map: Dict[
      int, torch.Tensor] = fn_ctx.device_parameter_id_tensor_mapping()
  hlo_input_id_to_input_index_map: Dict[int, int] = {}
  computation_inputs = [None] * len(all_hlo_input_vars_map)
  for i, t in enumerate(flat_fake_inputs):
    if isinstance(t, torch.Tensor):
      param_id = fn_ctx.tensor_parameter_id(t)
      if param_id != -1:
        computation_inputs[param_id] = flat_inputs[i]
        hlo_input_id_to_input_index_map[param_id] = i
        del all_hlo_input_vars_map[param_id]

  # The remaining variables in all_input_vars_map are the hoisted variables
  # that are not present in flat_inputs.
  hoisted_vars_map = all_hlo_input_vars_map
  for i, var in hoisted_vars_map.items():
    computation_inputs[i] = var

  return computation_inputs, hoisted_vars_map, hlo_input_id_to_input_index_map


def assume_pure_torch(func, use_cache=False):
  """Decorator to mark a function as pure for PyTorch/XLA.
  This decorator builds an XLA computation from the function and caches it.
  The decorated function must be pure (i.e. no side-effects, behavior
  only depends on inputs). 
  Args:
    func: The function to be decorated.
    use_cache: If True, caches the XLA computation for the function with
      the same name as the function. It is the user's responsibility to ensure
      that the function is called with the same input shapes and types each time
      when using this.
  NOTE: This decorator only works for forward pass.
  """
  assert not torch.is_grad_enabled()

  @wraps(func)
  def inner(*args, **kwargs):
    global _XLA_COMPUTATION_CACHE

    flat_inputs, input_tree_spec = tree_flatten((args, kwargs))
    computation_inputs = None

    # TODO: Decide what to include in the cache key.
    if use_cache and _XLA_COMPUTATION_CACHE.get(func.__name__,
                                                None) is not None:
      fn_computation, output_tree_spec, hlo_input_id_to_input_index_map, hoisted_vars = _XLA_COMPUTATION_CACHE[
          func.__name__]
      computation_inputs_size = len(hoisted_vars) + len(
          hlo_input_id_to_input_index_map)
      computation_inputs = [None] * (computation_inputs_size)
      for hlo_id, input_index in hlo_input_id_to_input_index_map.items():
        computation_inputs[hlo_id] = flat_inputs[input_index]
      for i, var in hoisted_vars.items():
        computation_inputs[i] = var
    else:
      flat_fake_inputs = [make_fake_inputs(a) for a in flat_inputs]
      fake_args, fake_kwargs = tree_unflatten(flat_fake_inputs, input_tree_spec)
      fake_outputs = func(*fake_args, **fake_kwargs)
      flat_fake_outputs, output_tree_spec = tree_flatten(fake_outputs)

      fn_ctx = torch_xla._XLAC.lowering.LoweringContext("FnComputation")
      fn_ctx.set_name_string("fn_ctx")
      fn_ctx.build(flat_fake_outputs)
      fn_hlo = fn_ctx.hlo()
      fn_computation = xb.computation_from_module_proto(
          f"xla::xb_computation_{func.__name__}", fn_hlo)

      computation_inputs, hoisted_vars, hlo_input_id_to_input_index_map = prepare_computation_inputs(
          fn_ctx, flat_fake_inputs, flat_inputs)

      if use_cache:
        _XLA_COMPUTATION_CACHE[func.__name__] = (
            fn_computation, output_tree_spec, hlo_input_id_to_input_index_map,
            hoisted_vars)

    result = torch_xla._XLAC._xla_user_computation(
        f"xla::xb_computation_{func.__name__}", computation_inputs,
        fn_computation)
    result_tree = tree_unflatten(result, output_tree_spec)
    return result_tree

  return inner
