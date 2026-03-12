=============
Configuration
=============

Configuration management, the immutable parameter registry, parameter
space definitions with dual prior system, and shared type aliases.

Config Manager
==============

.. automodule:: heterodyne.config.manager
   :members: ConfigManager, load_xpcs_config
   :undoc-members:
   :show-inheritance:

Parameter Registry
==================

Immutable ``MappingProxyType``-based registry defining all 16 parameters
(14 physics + 2 scaling) with bounds, defaults, and prior specifications.

.. automodule:: heterodyne.config.parameter_registry
   :members: ParameterRegistry, ParameterInfo, DEFAULT_REGISTRY
   :undoc-members:
   :show-inheritance:

Parameter Manager
=================

.. automodule:: heterodyne.config.parameter_manager
   :members: ParameterManager
   :undoc-members:
   :show-inheritance:

Parameter Space
===============

.. automodule:: heterodyne.config.parameter_space
   :members: ParameterSpace, PriorDistribution
   :undoc-members:
   :show-inheritance:

Parameter Names
===============

.. automodule:: heterodyne.config.parameter_names
   :members: ALL_PARAM_NAMES, PARAM_GROUPS
   :undoc-members:
   :show-inheritance:

Type Aliases
============

.. automodule:: heterodyne.config.types
   :members:
   :undoc-members:
   :show-inheritance:
