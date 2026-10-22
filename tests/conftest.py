import jax

# Tests compare against analytic and finite-difference references, which needs float64
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
