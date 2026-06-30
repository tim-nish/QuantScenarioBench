import jax

# AD-7: float64 precision is the fixed v1 policy. Enabled exactly once here;
# Python's import system guarantees this runs before any submodule's code.
jax.config.update("jax_enable_x64", True)
