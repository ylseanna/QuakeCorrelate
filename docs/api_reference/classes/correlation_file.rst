^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
quakecorrelate.CorrelationFile
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: quakecorrelate.CorrelationFile
    :members:
    :inherited-members:
    :undoc-members:

==================
Usage examples
==================

~~~~~~~~~~~~~~~~~~~~~~~~~~
Indexing and iteration
~~~~~~~~~~~~~~~~~~~~~~~~~~

The correlation file class has some added functionality to enable easy access of the correlation and lag values that have been computed using Quakecorrelate.

///////////////////////////
Indexing
///////////////////////////

:code:`CorrelationFile` objects access the underlying :code:`.zarr`-object containing any correlation outputs through indexing based on the
station ID, channel ID, and array type (:code:`corr_values`, :code:`lag_values`). These will then return the :code:`.zarr`-array or group objects, which can be indexed to access specific values, e.g.::

    from quakecorrelate import CorrelationFile

    cf = CorrelationFile()

    cf["STA1"]["HHZ"].corr_values

Will return the :code:`.zarr`-array object with the calculated correlation values for station :code:`STA1`, channel :code:`HHZ`.

The groups for each station or channel have been wrapped to add some extra functionality::

    from quakecorrelate import CorrelationFile

    cf = CorrelationFile()

    cf["STA1"]

Will return the `StationGroup <http://localhost:8000/api_reference/classes/correlation_file.html#quakecorrelate.correlation_file.StationGroup>`_ object with the calculated correlation values for station :code:`STA1`.

Similarly::

    from quakecorrelate import CorrelationFile

    cf = CorrelationFile()

    cf["STA1"]["HHZ"]

Will return the `ChannelGroup <http://localhost:8000/api_reference/classes/correlation_file.html#quakecorrelate.correlation_file.ChannelGroup>`_ object with the calculated correlation values for station :code:`STA1` and channel :code:`HHZ`.


///////////////////////////
Iteration
///////////////////////////



====================
Utility classes
====================

.. autoclass:: quakecorrelate.correlation_file.StationGroup
    :members:
    :undoc-members:

.. autoclass:: quakecorrelate.correlation_file.ChannelGroup
    :members:
    :undoc-members:
