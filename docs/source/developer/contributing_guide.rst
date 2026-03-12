Contributing Guide
==================

Thank you for considering a contribution to Heterodyne.  This document
describes the workflow, conventions, and quality gates that every change
must pass before it can be merged.

Development Setup
-----------------

.. code-block:: bash

   git clone https://github.com/imewei/heterodyne.git
   cd heterodyne
   make dev          # Runs uv sync and installs in editable mode

This creates a local ``.venv`` managed by **uv**.  Never install into global
or user site-packages.

Branch Naming
-------------

Use a descriptive prefix so that CI jobs and reviewers can triage at a glance:

- ``feature/<short-description>`` -- new functionality
- ``fix/<short-description>`` -- bug fixes
- ``docs/<short-description>`` -- documentation-only changes
- ``refactor/<short-description>`` -- code restructuring with no behavior change
- ``test/<short-description>`` -- test additions or improvements

Commit Conventions
------------------

Follow the `Conventional Commits <https://www.conventionalcommits.org/>`_
style:

.. code-block:: text

   feat(core): add log-space clipping for half_tr computation
   fix(nlsq): correct Jacobian norm storage in sequential strategy
   docs(config): add CMC-only template to configuration guide
   test(opt): add unit tests for CMA-ES fixes
   chore(deps): bump version and update uv.lock

The scope in parentheses should match a top-level package directory
(``core``, ``optimization``, ``config``, ``data``, ``cli``, ``viz``, etc.).

Code Style
----------

The following conventions are enforced across the codebase:

- ``from __future__ import annotations`` in **every** module.
- ``strict=True`` on **all** ``zip()`` calls.
- ``MappingProxyType`` for immutable registries, typed as
  ``Mapping[str, T]`` for MyPy compatibility.
- ``raise ... from None`` for exception translation
  (e.g., ``ValueError`` to ``KeyError``).
- ``cast()`` from ``typing`` for ``dict.get()`` returns on
  ``dict[str, Any]`` configs.
- JAX functions that return traced values use
  ``# type: ignore[no-any-return]`` where MyPy cannot infer the concrete
  type.
- No wildcard imports (``from module import *`` is prohibited).

Ruff is the single linting and formatting tool.  Run it locally before
pushing:

.. code-block:: bash

   uv run ruff check .
   uv run ruff format --check .

Pull Request Checklist
----------------------

Before requesting review, verify every item:

1. ``make quality`` passes (Ruff lint + MyPy type-check).
2. ``make test`` passes (unit tests).
3. New or changed behavior has corresponding test coverage.
4. Documentation is updated if the change affects user-facing behavior
   or configuration options.
5. Commit messages follow the conventional commit format.
6. The branch is rebased on ``main`` with no merge conflicts.

Code Review
-----------

- At least one maintainer approval is required.
- CI must be green on all matrix entries (Python 3.12, 3.13).
- Conversations must be resolved before merge.
