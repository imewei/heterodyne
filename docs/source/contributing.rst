Contributing
============

Thank you for your interest in contributing to Heterodyne. This page provides
a quick overview of how to get involved. For detailed developer documentation,
see the guides linked below.

Quick Links
-----------

- Developer Guide -- development environment setup, code conventions, and
  architecture overview
- Architecture Deep Dives -- detailed design documents and ADRs

How to Contribute
-----------------

**Bug reports**

Open an issue at https://github.com/imewei/heterodyne/issues with:

- A minimal reproducing example (data file or synthetic input)
- Full traceback and error message
- Output of ``heterodyne-validate``
- Python version and OS

**Feature requests**

Open an issue describing the use case, expected behavior, and any relevant
references (papers, beamline documentation).

**Pull requests**

1. Fork the repository and create a feature branch from ``main``.
2. Install the development environment:

   .. code-block:: bash

      git clone https://github.com/<your-fork>/heterodyne.git
      cd heterodyne
      uv sync
      make dev

3. Make your changes. All contributions must:

   - Pass ``make quality`` (format, lint, type-check)
   - Include tests for new functionality
   - Update documentation if user-facing behavior changes

4. Run the test suite:

   .. code-block:: bash

      make test

5. Push your branch and open a pull request against ``main``.

Code Style
----------

- Python 3.12+ with ``from __future__ import annotations`` in all modules
- Strict type hints at API boundaries and configuration objects
- ``ruff`` for formatting and linting
- ``mypy`` for static type checking
- ``strict=True`` on all ``zip()`` calls
- No wildcard imports (``from module import *`` is prohibited)

Code of Conduct
---------------

This project follows the `Contributor Covenant v2.1
<https://www.contributor-covenant.org/version/2/1/code_of_conduct/>`_.

All participants are expected to uphold a welcoming, inclusive, and
harassment-free environment. Please report unacceptable behavior to the
project maintainers.
