===============
 Installation
===============

Here it is explained in detail how to install QuakeCorrelate for different platforms, which have been tested and developed for.
Support for other platforms can be discussed on the `Issues page <https://github.com/ylseanna/QuakeCorrelate/issues>`_.

QuakeCorrelate is a Python package that tries to use common libraries for most of its functionality such as `Obspy <https://www.github.com/obspy/Obspy>`_ and `Pandas <https://pandas.pydata.org>`_ for managing seismic data. 
For its GPU-based cross correlation functionality it uses `Squiggles.jl <https://www.github.com/gbene/Squiggles.jl>`_, run through `JuliaCall <https://juliapy.github.io/PythonCall.jl>`_.
Data is recorded in a `.zarr <https://zarr-specs.readthedocs.io>`_-based format, which enables massive data storage and parallel reading and writing.

.. warning::

    QuakeCorrelate is still in an unstable pre-release state, many aspects of the software and
    installation procedure may change unexpectedly. Be prepared to re-install and adjust to these changes.

.. admonition:: GitHub variables

    Squiggles is installed via its GitHub repository (for now) **only through SSH** access to GitHub. So it may be advisable to set the following 
    environment variables to automatically link to the locations of your GitHub SSH access keys::

        SSH_PUB_KEY_PATH=/path/to/.ssh/id_ed25519.pub
        SSH_KEY_PATH=/path/to/.ssh/id_ed25519

    Note that :code:`id_ed25519` is an example, but follows the standard format in the GitHub guide.

-------------------
Linux
-------------------

This is the most well-tested platform for this software. However GPU calculation is only supported for CUDA for now.

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

Mac support is still unfortunately somewhat limited.

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
