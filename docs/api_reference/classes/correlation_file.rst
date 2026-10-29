^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
quakecorrelate.CorrelationFile
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: quakecorrelate.CorrelationFile
    :members:
    :inherited-members:
    :undoc-members:

=========================
Custom behavior
=========================

The correlation file class has some added functionality to enable easy access of the correlation and lag values that have been computed using Quakecorrelate.

~~~~~~~~~~~~~~~~~~~~~~~~~~
Indexing
~~~~~~~~~~~~~~~~~~~~~~~~~~

:any:`CorrelationFile <quakecorrelate.CorrelationFile>` objects access the underlying :code:`.zarr`-object containing any correlation outputs through indexing based on the
station ID, channel ID, and array type (:code:`corr_values`, :code:`lag_values`). These will then return the :code:`.zarr`-array or group objects, which can be indexed to access specific values, e.g.::

    from quakecorrelate import CorrelationFile

    cf = CorrelationFile()

    cf["STA1"]["HHZ"].corr_values

Will return the :code:`.zarr`-array object with the calculated correlation values for station :code:`STA1`, channel :code:`HHZ`.

The groups for each station or channel have been wrapped to add some extra functionality::

    from quakecorrelate import CorrelationFile

    cf = CorrelationFile()

    cf["STA1"]

Will return the :any:`StationGroup <quakecorrelate.correlation_file.StationGroup>` object with the calculated correlation values for station :code:`STA1`.

Similarly::

    from quakecorrelate import CorrelationFile

    cf = CorrelationFile()

    cf["STA1"]["HHZ"]

Will return the :any:`ChannelGroup <quakecorrelate.correlation_file.ChannelGroup>` object with the calculated correlation values for station :code:`STA1` and channel :code:`HHZ`.

~~~~~~~~~~~~~~~~~~~~~~~~~~
Iteration
~~~~~~~~~~~~~~~~~~~~~~~~~~

Iterating over the :any:`CorrelationFile <quakecorrelate.CorrelationFile>` object will yield the :any:`StationGroup <quakecorrelate.correlation_file.StationGroup>` objects for all stations present in the file::

    from quakecorrelate import CorrelationFile

    cf = CorrelationFile()

    for stationgroup in cf:
        print(stationgroup)

Iterating over the :any:`StationGroup <quakecorrelate.correlation_file.StationGroup>`  object will yield the :any:`ChannelGroup <quakecorrelate.correlation_file.ChannelGroup>`  objects for all channels present in the station group::

    from quakecorrelate import CorrelationFile

    cf = CorrelationFile()

    for stationgroup in cf:
        for channelgroup in stationgroup:
            print(channelgroup)

=========================
Examples
=========================

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Check file content
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can check if a station is present in a correlation file by accessing the :code:`stations` attribute::

    from quakecorrelate import CorrelationFile

    cf = CorrelationFile()

    if "STA1" is in cf.stations:
        print("STA1 present!")
    else:
        print("STA1 not present!")

And similarly you can check if a channel is present in a correlation file by accessing the :code:`channels` attribute of a station group::

    from quakecorrelate import CorrelationFile

    cf = CorrelationFile()

    if "HHZ" is in cf["STA1"].channels:
        print("channel HHZ present in STA1!")
    else:
        print("channel HHZ not present in STA1!")

For a channel group the :code:`corr_values` or the :code:`lag_values` objects will be :code:`None` if they are not present::

    from quakecorrelate import CorrelationFile

    cf = CorrelationFile()

    if cf["STA1"]["HHZ"].corr_values:
        print("Correlation matrix present for channel HHZ in STA1!")
    else:
        print("Correlation matrix not present for channel HHZ in STA1!")

    if cf["STA1"]["HHZ"].lag_values:
        print("Lag matrix present for channel HHZ in STA1!")
    else:
        print("Lag matrix not present for channel HHZ in STA1!")


====================
Utility classes
====================

.. autoclass:: quakecorrelate.correlation_file.CorrelationFileMetadata
    :members:
    :undoc-members:

.. autoclass:: quakecorrelate.correlation_file.StationGroup
    :members:
    :undoc-members:

.. autoclass:: quakecorrelate.correlation_file.ChannelGroup
    :members:
    :undoc-members:
