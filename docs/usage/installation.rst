===============
 Installation
===============

This is the installation guide

.. .. toctree::
..    :maxdepth: 2


-------------------
Linux
-------------------

^^^^^^^^^^^^^^^^^^^
CUDA
^^^^^^^^^^^^^^^^^^^

To install clone the repository and run::

    pip install .


To setup Julia, run::

    python -c "from quakecorrelate import setup_julia; setup_julia()"

You should now be able to import QuakeCorrelate using::

    import quakecorrelate

Ensure that if environment variables were previously set, the variable :code:`QUAKECORRELATE_GPU_ARCH` is set to :code:`CUDA`, in either a :code:`.env` file in the running directory or by setting in the path.
However QuakeCorrelate will default to using CUDA if environment variables are not set.

.. admonition:: HPC clusters

    When running on HPC/multiple node clusters, CUDA may not be able to detect the proper runtime version to use.
    By setting :code:`QUAKECORRELATE_CUDA_RUNTIME_VERSION` to the version (format :code:`<num>.<num>.<num>`) in the environment variables, the proper version can be assured.

    This has to be done before running::

        python -c "from quakecorrelate import setup_julia; setup_julia()"

    Alternatively, you can run::

        python -c "from quakecorrelate import setup_julia; setup_julia(CUDA_runtime_version='<num>.<num>.<num>')"

    Where you fill in your CUDA run time version.


-------------------
Mac
-------------------

^^^^^^^^^^^^^^^^^^^
Metal
^^^^^^^^^^^^^^^^^^^

To install clone the repository and run::

    pip install .


To setup Julia, run::

    python -c "from quakecorrelate import setup_julia; setup_julia(gpu_architecture='Metal')"

Or, with :code:`QUAKECORRELATE_GPU_ARCH` to :code:`Metal` set in either a :code:`.env` file in the running directory or by setting in the path, the following can be run::

    python -c "from quakecorrelate import setup_julia; setup_julia()"

You will also have to set this environment variable (:code:`QUAKECORRELATE_GPU_ARCH`) to :code:`Metal`,
to get QuakeCorrelate to automatically use Metal during runtime.

You should now be able to import QuakeCorrelate using::

    import quakecorrelate