===============
Model Hierarchy
===============

Abstract base class and concrete model implementations for heterodyne
XPCS correlation functions. The factory function :func:`create_model`
selects the appropriate model variant based on configuration.

.. automodule:: heterodyne.core.models
   :members: HeterodyneModelBase, TwoComponentModel, ReducedModel, create_model
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
