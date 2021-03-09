Caret: A Python Interface to Asterisk
=====================================

``caret`` consists of a set of interfaces and libraries to allow programming of
Asterisk from Python. The library currently supports AGI, AMI, and the parsing
of Asterisk configuration files. The library also includes debugging facilities
for AGI.

This project provides a single library for all Asterisk based needs
of Python users and an easy transition to Python for PHP users from the PHPAGI package.


Installation
------------

Using Python
************

To install ``caret``, simply run:

.. code-block:: console

    $ pip install caret

On Debian/Ubuntu
****************
(Under development)

This will install the latest version of the library automatically.


Documentation
-------------

Documentation is still under development and parts of it are hosted on
https://ar13pit.github.io/caret/

Use Python docstrings for time being using
Python's built-in help facility::

 import caret
 help (caret)
 import caret.agi
 help (caret.agi)
 import caret.manager
 help (caret.manager)
 import caret.config
 help (caret.config)

Some notes on platforms: We now specify "platforms = 'Any'" in
``setup.py``. This means, the manager part of the package will probably
run on any platform. The agi scripts on the other hand are called
directly on the host where Asterisk is running. Since Asterisk doesn't
run on windows platforms (and probably never will) the agi part of the
package can only be run on Asterisk platforms.


Credits
-------

Thanks to Karl Putland for writing the original ``pyst`` package and
Matthew Nicholson for maintaining it for some years.

Thanks to Randall Degges for maintaining the ``pyst2`` fork and accepting
pull requests for some years.

Thanks to Arpit Aggarwal for carrying the development forward and starting the
``caret`` project.

Copyright
---------

This project has been released under GNU Lesser General Public License v3.0.



