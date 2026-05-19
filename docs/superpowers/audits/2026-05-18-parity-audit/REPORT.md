# Heterodyne → Homodyne Parity Audit Report

- Homodyne SHA: `0368cbdb075fcff1908c0da2b59a1b0d37d5eeca`
- Heterodyne SHA: `6c80768c0b0ffe41254783231cfd1cc9d16f9694`
- Total gaps: **4395**

## P0 — Silent breakage — API/config/CLI/exit-code/docs-build drift (1753 gaps)

### classes (543)

| Disposition | Kind | Qualname / Path | Detail |
|---|---|---|---|
| `KEEP` | missing_method | `config.manager.ConfigManager.get_active_parameters` | homodyne method `get_active_parameters(self) -> list[str]` missing in heterodyne |
| `KEEP` | missing_method | `config.manager.ConfigManager.get_initial_parameters` | homodyne method `get_initial_parameters(self, use_midpoint_defaults: bool = True) -> dict[str, float]` missing in heterodyne |
| `KEEP` | missing_method | `config.manager.ConfigManager.get_parameter_bounds` | homodyne method `get_parameter_bounds(self, parameter_names: list[str] | None = None) -> list[dict[str, Any]]` missing in heterodyne |
| `KEEP` | missing_method | `config.manager.ConfigManager.get_target_angle_ranges` | homodyne method `get_target_angle_ranges(self) -> dict[str, Any]` missing in heterodyne |
| `KEEP` | missing_method | `config.manager.ConfigManager.is_static_mode_enabled` | homodyne method `is_static_mode_enabled(self) -> bool` missing in heterodyne |
| `KEEP` | missing_method | `config.manager.ConfigManager.load_config` | homodyne method `load_config(self) -> None` missing in heterodyne |
| `KEEP` | missing_method | `config.manager.ConfigManager.update_config` | homodyne method `update_config(self, key: str, value: Any) -> None` missing in heterodyne |
| `KEEP` | missing_method | `config.manager.ConfigManager.validate_per_angle_scaling` | homodyne method `validate_per_angle_scaling(self, n_phi: int) -> list[str]` missing in heterodyne |
| `KEEP` | missing_method | `config.parameter_manager.ParameterManager.validate_parameters` | homodyne method `validate_parameters(self, params: np.ndarray, param_names: list[str] | None = None, tolerance: float = 1e-10) -> ValidationResult` missing in heterodyne |
| `KEEP` | missing_method | `config.parameter_manager.ParameterManager.validate_physical_constraints` | homodyne method `validate_physical_constraints(self, params: dict[str, float], severity_level: str = 'warning') -> ValidationResult` missing in heterodyne |
| `KEEP` | missing_field | `config.parameter_registry.ParameterInfo.dtype` | homodyne dataclass field `dtype` missing in heterodyne |
| `KEEP` | missing_field | `config.parameter_registry.ParameterInfo.lower_bound` | homodyne dataclass field `lower_bound` missing in heterodyne |
| `KEEP` | missing_field | `config.parameter_registry.ParameterInfo.upper_bound` | homodyne dataclass field `upper_bound` missing in heterodyne |
| `KEEP` | missing_field | `config.parameter_registry.ParameterInfo.units` | homodyne dataclass field `units` missing in heterodyne |
| `KEEP` | missing_method | `config.parameter_registry.ParameterRegistry.expand_initial_values` | homodyne method `expand_initial_values(self, initial_values: dict[str, float], n_angles: int) -> dict[str, float]` missing in heterodyne |
| `KEEP` | missing_method | `config.parameter_registry.ParameterRegistry.get_all_bounds` | homodyne method `get_all_bounds(self, analysis_mode: AnalysisMode, n_angles: int = 1, include_scaling: bool = True) -> tuple[list[float], list[float]]` missing in heterodyne |
| `KEEP` | missing_method | `config.parameter_registry.ParameterRegistry.get_all_param_names` | homodyne method `get_all_param_names(self, analysis_mode: AnalysisMode, n_angles: int = 1, include_scaling: bool = True) -> list[str]` missing in heterodyne |
| `KEEP` | missing_method | `config.parameter_registry.ParameterRegistry.get_bounds` | homodyne method `get_bounds(self, name: str) -> tuple[float, float]` missing in heterodyne |
| `KEEP` | missing_method | `config.parameter_registry.ParameterRegistry.get_defaults` | homodyne method `get_defaults(self, analysis_mode: AnalysisMode, n_angles: int = 1, include_scaling: bool = True) -> list[float]` missing in heterodyne |
| `KEEP` | missing_method | `config.parameter_registry.ParameterRegistry.get_num_params` | homodyne method `get_num_params(self, analysis_mode: AnalysisMode, n_angles: int = 1, include_scaling: bool = True) -> int` missing in heterodyne |
| `KEEP` | missing_method | `config.parameter_registry.ParameterRegistry.get_param_info` | homodyne method `get_param_info(self, name: str) -> ParameterInfo` missing in heterodyne |
| `KEEP` | missing_method | `config.parameter_registry.ParameterRegistry.get_param_names` | homodyne method `get_param_names(self, analysis_mode: AnalysisMode) -> list[str]` missing in heterodyne |
| `KEEP` | missing_method | `config.parameter_registry.ParameterRegistry.scaling_names` | homodyne method `scaling_names(self) -> tuple[str, ...]` missing in heterodyne |
| `KEEP` | missing_method | `config.parameter_registry.ParameterRegistry.validate_param_values` | homodyne method `validate_param_values(self, values: dict[str, float] | list[float], analysis_mode: AnalysisMode, n_angles: int = 1, include_scaling: bool = True) -> None` missing in heterodyne |
| `KEEP` | missing_method | `config.parameter_space.ParameterSpace.clamp_to_open_interval` | homodyne method `clamp_to_open_interval(self, param_name: str, value: float, epsilon: float = 1e-06) -> float` missing in heterodyne |
| `KEEP` | missing_method | `config.parameter_space.ParameterSpace.convert_to_beta_priors` | homodyne method `convert_to_beta_priors(self) -> 'ParameterSpace'` missing in heterodyne |
| `KEEP` | missing_method | `config.parameter_space.ParameterSpace.convert_to_beta_scaled_priors` | homodyne method `convert_to_beta_scaled_priors(self) -> 'ParameterSpace'` missing in heterodyne |
| `KEEP` | missing_method | `config.parameter_space.ParameterSpace.copy` | homodyne method `copy(self) -> 'ParameterSpace'` missing in heterodyne |
| `KEEP` | missing_method | `config.parameter_space.ParameterSpace.drop_parameters` | homodyne method `drop_parameters(self, names: set[str]) -> 'ParameterSpace'` missing in heterodyne |
| `KEEP` | missing_method | `config.parameter_space.ParameterSpace.from_config` | homodyne method `from_config(cls, config_dict: dict[str, Any], analysis_mode: str | None = None) -> 'ParameterSpace'` missing in heterodyne |
| `KEEP` | missing_method | `config.parameter_space.ParameterSpace.from_defaults` | homodyne method `from_defaults(cls, analysis_mode: str = 'laminar_flow') -> 'ParameterSpace'` missing in heterodyne |
| `KEEP` | missing_method | `config.parameter_space.ParameterSpace.get_bounds` | homodyne method `get_bounds(self, param_name: str) -> tuple[float, float]` missing in heterodyne |
| `KEEP` | missing_method | `config.parameter_space.ParameterSpace.get_bounds_array` | homodyne method `get_bounds_array(self) -> tuple[np.ndarray, np.ndarray]` missing in heterodyne |
| `KEEP` | missing_method | `config.parameter_space.ParameterSpace.get_prior` | homodyne method `get_prior(self, param_name: str) -> PriorDistribution` missing in heterodyne |
| `KEEP` | missing_method | `config.parameter_space.ParameterSpace.get_prior_means` | homodyne method `get_prior_means(self) -> np.ndarray` missing in heterodyne |
| `KEEP` | missing_method | `config.parameter_space.ParameterSpace.get_single_angle_fallback_prior` | homodyne method `get_single_angle_fallback_prior(self, param_name: str) -> PriorDistribution` missing in heterodyne |
| `KEEP` | missing_method | `config.parameter_space.ParameterSpace.get_single_angle_geometry_config` | homodyne method `get_single_angle_geometry_config(self) -> dict[str, float]` missing in heterodyne |
| `KEEP` | missing_method | `config.parameter_space.ParameterSpace.validate_values` | homodyne method `validate_values(self, values: dict[str, float], tolerance: float = 1e-10) -> tuple[bool, list[str]]` missing in heterodyne |
| `KEEP` | missing_method | `config.parameter_space.ParameterSpace.with_prior_overrides` | homodyne method `with_prior_overrides(self, overrides: dict[str, PriorDistribution]) -> 'ParameterSpace'` missing in heterodyne |
| `KEEP` | missing_method | `config.parameter_space.ParameterSpace.with_single_angle_stabilization` | homodyne method `with_single_angle_stabilization(self, *, enable_beta_fallback: bool = False) -> 'ParameterSpace'` missing in heterodyne |
| `KEEP` | missing_field | `config.parameter_space.ParameterSpace.model_type` | homodyne dataclass field `model_type` missing in heterodyne |
| `KEEP` | missing_field | `config.parameter_space.ParameterSpace.parameter_names` | homodyne dataclass field `parameter_names` missing in heterodyne |
| `KEEP` | missing_field | `config.parameter_space.ParameterSpace.units` | homodyne dataclass field `units` missing in heterodyne |
| `KEEP` | missing_method | `config.parameter_space.PriorDistribution.to_numpyro_kwargs` | homodyne method `to_numpyro_kwargs(self) -> dict[str, Any]` missing in heterodyne |
| `KEEP` | missing_field | `config.parameter_space.PriorDistribution.dist_type` | homodyne dataclass field `dist_type` missing in heterodyne |
| `KEEP` | missing_field | `config.parameter_space.PriorDistribution.mu` | homodyne dataclass field `mu` missing in heterodyne |
| `KEEP` | missing_field | `config.parameter_space.PriorDistribution.sigma` | homodyne dataclass field `sigma` missing in heterodyne |
| `KEEP` | missing_field | `config.parameter_space.PriorDistribution.min_val` | homodyne dataclass field `min_val` missing in heterodyne |
| `KEEP` | missing_field | `config.parameter_space.PriorDistribution.max_val` | homodyne dataclass field `max_val` missing in heterodyne |
| `KEEP` | missing_field | `config.parameter_space.PriorDistribution.alpha` | homodyne dataclass field `alpha` missing in heterodyne |
| `KEEP` | missing_field | `config.parameter_space.PriorDistribution.beta` | homodyne dataclass field `beta` missing in heterodyne |
| `KEEP` | missing_field | `config.physics_validators.ConstraintRule.condition` | homodyne dataclass field `condition` missing in heterodyne |
| `KEEP` | missing_method | `config.physics_validators.PhysicsViolation.format` | homodyne method `format(self) -> str` missing in heterodyne |
| `KEEP` | missing_field | `config.physics_validators.PhysicsViolation.param` | homodyne dataclass field `param` missing in heterodyne |
| `KEEP` | missing_class | `config.types.BoundDict` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_class | `config.types.CMCConfig` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_class | `config.types.HomodyneConfig` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_class | `config.types.InitialParametersConfig` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_class | `config.types.NLSQValidationConfig` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_class | `config.types.ParameterSpaceConfig` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_field | `core.fitting.FitResult.p_value` | homodyne dataclass field `p_value` missing in heterodyne |
| `KEEP` | missing_field | `core.fitting.FitResult.analysis_mode` | homodyne dataclass field `analysis_mode` missing in heterodyne |
| `KEEP` | missing_method | `core.fitting.ParameterSpace.get_param_bounds` | homodyne method `get_param_bounds(self, analysis_mode: str) -> list[tuple[float, float]]` missing in heterodyne |
| `KEEP` | missing_method | `core.fitting.ParameterSpace.get_param_priors` | homodyne method `get_param_priors(self, analysis_mode: str) -> list[tuple[float, float]]` missing in heterodyne |
| `KEEP` | missing_field | `core.fitting.ParameterSpace.D0_bounds` | homodyne dataclass field `D0_bounds` missing in heterodyne |
| `KEEP` | missing_field | `core.fitting.ParameterSpace.alpha_bounds` | homodyne dataclass field `alpha_bounds` missing in heterodyne |
| `KEEP` | missing_field | `core.fitting.ParameterSpace.D_offset_bounds` | homodyne dataclass field `D_offset_bounds` missing in heterodyne |
| `KEEP` | missing_field | `core.fitting.ParameterSpace.gamma_dot_t0_bounds` | homodyne dataclass field `gamma_dot_t0_bounds` missing in heterodyne |
| `KEEP` | missing_field | `core.fitting.ParameterSpace.beta_bounds` | homodyne dataclass field `beta_bounds` missing in heterodyne |
| `KEEP` | missing_field | `core.fitting.ParameterSpace.gamma_dot_t_offset_bounds` | homodyne dataclass field `gamma_dot_t_offset_bounds` missing in heterodyne |
| `KEEP` | missing_field | `core.fitting.ParameterSpace.phi0_bounds` | homodyne dataclass field `phi0_bounds` missing in heterodyne |
| `KEEP` | missing_field | `core.fitting.ParameterSpace.D0_prior` | homodyne dataclass field `D0_prior` missing in heterodyne |
| `KEEP` | missing_field | `core.fitting.ParameterSpace.alpha_prior` | homodyne dataclass field `alpha_prior` missing in heterodyne |
| `KEEP` | missing_field | `core.fitting.ParameterSpace.D_offset_prior` | homodyne dataclass field `D_offset_prior` missing in heterodyne |
| `KEEP` | missing_field | `core.fitting.ParameterSpace.gamma_dot_t0_prior` | homodyne dataclass field `gamma_dot_t0_prior` missing in heterodyne |
| `KEEP` | missing_field | `core.fitting.ParameterSpace.beta_prior` | homodyne dataclass field `beta_prior` missing in heterodyne |
| `KEEP` | missing_field | `core.fitting.ParameterSpace.gamma_dot_t_offset_prior` | homodyne dataclass field `gamma_dot_t_offset_prior` missing in heterodyne |
| `KEEP` | missing_field | `core.fitting.ParameterSpace.phi0_prior` | homodyne dataclass field `phi0_prior` missing in heterodyne |
| `KEEP` | missing_class | `core.fitting.UnifiedHomodyneEngine` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_class | `core.homodyne_model.HomodyneModel` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_class | `core.model_mixins.BenchmarkingMixin` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_class | `core.model_mixins.GradientCapabilityMixin` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_class | `core.model_mixins.OptimizationRecommendationMixin` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_class | `core.models.CombinedModel` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_class | `core.models.DiffusionModel` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_class | `core.models.PhysicsModelBase` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_class | `core.models.ShearModel` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_field | `core.numpy_gradients.DifferentiationConfig.relative_step` | homodyne dataclass field `relative_step` missing in heterodyne |
| `KEEP` | missing_field | `core.numpy_gradients.DifferentiationConfig.min_step` | homodyne dataclass field `min_step` missing in heterodyne |
| `KEEP` | missing_field | `core.numpy_gradients.DifferentiationConfig.max_step` | homodyne dataclass field `max_step` missing in heterodyne |
| `KEEP` | missing_field | `core.numpy_gradients.DifferentiationConfig.max_iterations` | homodyne dataclass field `max_iterations` missing in heterodyne |
| `KEEP` | missing_field | `core.numpy_gradients.DifferentiationConfig.chunk_size` | homodyne dataclass field `chunk_size` missing in heterodyne |
| `KEEP` | missing_field | `core.numpy_gradients.DifferentiationConfig.complex_step_threshold` | homodyne dataclass field `complex_step_threshold` missing in heterodyne |
| `KEEP` | missing_field | `core.numpy_gradients.GradientResult.step_sizes` | homodyne dataclass field `step_sizes` missing in heterodyne |
| `KEEP` | missing_field | `core.numpy_gradients.GradientResult.function_calls` | homodyne dataclass field `function_calls` missing in heterodyne |
| `KEEP` | missing_field | `core.numpy_gradients.GradientResult.computation_time` | homodyne dataclass field `computation_time` missing in heterodyne |
| `KEEP` | missing_field | `core.numpy_gradients.GradientResult.warnings` | homodyne dataclass field `warnings` missing in heterodyne |
| `KEEP` | missing_class | `core.numpy_gradients.NumericalStabilityError` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_method | `core.physics_factors.PhysicsFactors.from_config` | homodyne method `from_config(cls, q: float, L: float, dt: float, validate: bool = True) -> 'PhysicsFactors'` missing in heterodyne |
| `KEEP` | missing_method | `core.physics_factors.PhysicsFactors.to_dict` | homodyne method `to_dict(self) -> dict` missing in heterodyne |
| `KEEP` | missing_method | `core.physics_factors.PhysicsFactors.to_tuple` | homodyne method `to_tuple(self) -> tuple[float, float]` missing in heterodyne |
| `KEEP` | missing_field | `core.physics_factors.PhysicsFactors.wavevector_q` | homodyne dataclass field `wavevector_q` missing in heterodyne |
| `KEEP` | missing_field | `core.physics_factors.PhysicsFactors.stator_rotor_gap` | homodyne dataclass field `stator_rotor_gap` missing in heterodyne |
| `KEEP` | missing_field | `core.physics_factors.PhysicsFactors.wavevector_q_squared_half_dt` | homodyne dataclass field `wavevector_q_squared_half_dt` missing in heterodyne |
| `KEEP` | missing_field | `core.physics_factors.PhysicsFactors.sinc_prefactor` | homodyne dataclass field `sinc_prefactor` missing in heterodyne |
| `KEEP` | missing_class | `data.filtering_utils.DataFilteringError` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_class | `data.filtering_utils.FilterCriteria` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_class | `data.filtering_utils.FilteringResult` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_class | `data.filtering_utils.XPCSDataFilter` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_class | `data.memory_manager.AdvancedMemoryManager` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_class | `data.memory_manager.AllocationError` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_class | `data.memory_manager.MemoryManagerError` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_class | `data.memory_manager.MemoryPool` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_class | `data.memory_manager.MemoryPressureError` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_method | `data.memory_manager.MemoryPressureMonitor.get_pressure_trend` | homodyne method `get_pressure_trend(self, window_minutes: int = 5) -> str` missing in heterodyne |
| `KEEP` | missing_method | `data.memory_manager.MemoryPressureMonitor.register_critical_callback` | homodyne method `register_critical_callback(self, callback: Callable[[MemoryStats], None]) -> None` missing in heterodyne |
| `KEEP` | missing_method | `data.memory_manager.MemoryPressureMonitor.register_recovery_callback` | homodyne method `register_recovery_callback(self, callback: Callable[[MemoryStats], None]) -> None` missing in heterodyne |
| `KEEP` | missing_method | `data.memory_manager.MemoryPressureMonitor.register_warning_callback` | homodyne method `register_warning_callback(self, callback: Callable[[MemoryStats], None]) -> None` missing in heterodyne |
| `KEEP` | missing_method | `data.memory_manager.MemoryPressureMonitor.start_monitoring` | homodyne method `start_monitoring(self) -> None` missing in heterodyne |
| `KEEP` | missing_method | `data.memory_manager.MemoryPressureMonitor.stop_monitoring` | homodyne method `stop_monitoring(self) -> None` missing in heterodyne |
| `KEEP` | missing_class | `data.memory_manager.MemoryStats` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_class | `data.optimization.AdvancedDatasetOptimizer` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_class | `data.optimization.DatasetOptimizer` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_class | `data.performance_engine.AdaptiveChunker` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_class | `data.performance_engine.CacheError` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_class | `data.performance_engine.ChunkInfo` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_class | `data.performance_engine.MemoryMapManager` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_class | `data.performance_engine.MemoryPressureError` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_class | `data.performance_engine.MultiLevelCache` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_method | `data.performance_engine.PerformanceEngine.get_performance_report` | homodyne method `get_performance_report(self) -> dict[str, Any]` missing in heterodyne |
| `KEEP` | missing_method | `data.performance_engine.PerformanceEngine.load_correlation_matrices_optimized` | homodyne method `load_correlation_matrices_optimized(self, hdf_path: str, data_keys: list[str], chunk_info: list[ChunkInfo] | None = None) -> Any` missing in heterodyne |
| `KEEP` | missing_method | `data.performance_engine.PerformanceEngine.prefetch_data` | homodyne method `prefetch_data(self, hdf_path: str, data_keys: list[str], priority: int = 5) -> Future` missing in heterodyne |
| `KEEP` | missing_method | `data.performance_engine.PerformanceEngine.shutdown` | homodyne method `shutdown(self) -> None` missing in heterodyne |
| `KEEP` | missing_class | `data.performance_engine.PerformanceEngineError` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_class | `data.performance_engine.PerformanceMetrics` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_method | `data.phi_filtering.PhiAngleFilter.filter_angles_for_optimization` | homodyne method `filter_angles_for_optimization(self, phi_angles: list[float] | np.ndarray, target_ranges: list[tuple[float, float]] | None = None, fallback_enabled: bool | None = None) -> tuple[list[int], np.ndarray]` missing in heterodyne |
| `KEEP` | missing_method | `data.phi_filtering.PhiAngleFilter.get_angle_statistics` | homodyne method `get_angle_statistics(self, phi_angles: list[float] | np.ndarray) -> dict[str, Any]` missing in heterodyne |
| `KEEP` | missing_method | `data.phi_filtering.PhiAngleFilter.validate_target_ranges` | homodyne method `validate_target_ranges(self, target_ranges: list[tuple[float, float]]) -> bool` missing in heterodyne |
| `KEEP` | missing_class | `data.preprocessing.PreprocessingConfigurationError` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_class | `data.preprocessing.PreprocessingError` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_method | `data.preprocessing.PreprocessingPipeline.load_provenance` | homodyne method `load_provenance(self, filepath: str | Path) -> PreprocessingProvenance` missing in heterodyne |
| `KEEP` | missing_method | `data.preprocessing.PreprocessingPipeline.process` | homodyne method `process(self, data: dict[str, Any]) -> PreprocessingResult` missing in heterodyne |
| `KEEP` | missing_method | `data.preprocessing.PreprocessingPipeline.save_provenance` | homodyne method `save_provenance(self, provenance: PreprocessingProvenance, filepath: str | Path) -> None` missing in heterodyne |
| `KEEP` | missing_field | `data.preprocessing.PreprocessingProvenance.transformations` | homodyne dataclass field `transformations` missing in heterodyne |
| `KEEP` | missing_field | `data.preprocessing.PreprocessingProvenance.total_duration` | homodyne dataclass field `total_duration` missing in heterodyne |
| `KEEP` | missing_field | `data.preprocessing.PreprocessingProvenance.peak_memory_usage` | homodyne dataclass field `peak_memory_usage` missing in heterodyne |
| `KEEP` | missing_field | `data.preprocessing.PreprocessingProvenance.warnings` | homodyne dataclass field `warnings` missing in heterodyne |
| `KEEP` | missing_field | `data.preprocessing.PreprocessingProvenance.errors` | homodyne dataclass field `errors` missing in heterodyne |
| `KEEP` | missing_field | `data.preprocessing.PreprocessingResult.data` | homodyne dataclass field `data` missing in heterodyne |
| `KEEP` | missing_field | `data.preprocessing.PreprocessingResult.provenance` | homodyne dataclass field `provenance` missing in heterodyne |
| `KEEP` | missing_field | `data.preprocessing.PreprocessingResult.success` | homodyne dataclass field `success` missing in heterodyne |
| `KEEP` | missing_field | `data.preprocessing.PreprocessingResult.stage_results` | homodyne dataclass field `stage_results` missing in heterodyne |
| `KEEP` | missing_field | `data.preprocessing.TransformationRecord.duration` | homodyne dataclass field `duration` missing in heterodyne |
| `KEEP` | missing_field | `data.preprocessing.TransformationRecord.input_shape` | homodyne dataclass field `input_shape` missing in heterodyne |
| `KEEP` | missing_field | `data.preprocessing.TransformationRecord.output_shape` | homodyne dataclass field `output_shape` missing in heterodyne |
| `KEEP` | missing_field | `data.preprocessing.TransformationRecord.memory_usage` | homodyne dataclass field `memory_usage` missing in heterodyne |
| `KEEP` | missing_field | `data.preprocessing.TransformationRecord.warnings` | homodyne dataclass field `warnings` missing in heterodyne |
| `KEEP` | missing_class | `data.quality_controller.DataQualityController` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_method | `data.quality_controller.QualityControlConfig.from_config_dict` | homodyne method `from_config_dict(cls, config: dict[str, Any]) -> 'QualityControlConfig'` missing in heterodyne |
| `KEEP` | missing_field | `data.quality_controller.QualityControlConfig.enabled` | homodyne dataclass field `enabled` missing in heterodyne |
| `KEEP` | missing_field | `data.quality_controller.QualityControlConfig.validation_level` | homodyne dataclass field `validation_level` missing in heterodyne |
| `KEEP` | missing_field | `data.quality_controller.QualityControlConfig.auto_repair` | homodyne dataclass field `auto_repair` missing in heterodyne |
| `KEEP` | missing_field | `data.quality_controller.QualityControlConfig.pass_threshold` | homodyne dataclass field `pass_threshold` missing in heterodyne |
| `KEEP` | missing_field | `data.quality_controller.QualityControlConfig.warn_threshold` | homodyne dataclass field `warn_threshold` missing in heterodyne |
| `KEEP` | missing_field | `data.quality_controller.QualityControlConfig.excellent_threshold` | homodyne dataclass field `excellent_threshold` missing in heterodyne |
| `KEEP` | missing_field | `data.quality_controller.QualityControlConfig.enable_raw_validation` | homodyne dataclass field `enable_raw_validation` missing in heterodyne |
| `KEEP` | missing_field | `data.quality_controller.QualityControlConfig.enable_filtering_validation` | homodyne dataclass field `enable_filtering_validation` missing in heterodyne |
| `KEEP` | missing_field | `data.quality_controller.QualityControlConfig.enable_preprocessing_validation` | homodyne dataclass field `enable_preprocessing_validation` missing in heterodyne |
| `KEEP` | missing_field | `data.quality_controller.QualityControlConfig.enable_final_validation` | homodyne dataclass field `enable_final_validation` missing in heterodyne |
| `KEEP` | missing_field | `data.quality_controller.QualityControlConfig.repair_nan_values` | homodyne dataclass field `repair_nan_values` missing in heterodyne |
| `KEEP` | missing_field | `data.quality_controller.QualityControlConfig.repair_infinite_values` | homodyne dataclass field `repair_infinite_values` missing in heterodyne |
| `KEEP` | missing_field | `data.quality_controller.QualityControlConfig.repair_negative_correlations` | homodyne dataclass field `repair_negative_correlations` missing in heterodyne |
| `KEEP` | missing_field | `data.quality_controller.QualityControlConfig.repair_scaling_issues` | homodyne dataclass field `repair_scaling_issues` missing in heterodyne |
| `KEEP` | missing_field | `data.quality_controller.QualityControlConfig.repair_format_inconsistencies` | homodyne dataclass field `repair_format_inconsistencies` missing in heterodyne |
| `KEEP` | missing_field | `data.quality_controller.QualityControlConfig.cache_validation_results` | homodyne dataclass field `cache_validation_results` missing in heterodyne |
| `KEEP` | missing_field | `data.quality_controller.QualityControlConfig.incremental_validation` | homodyne dataclass field `incremental_validation` missing in heterodyne |
| `KEEP` | missing_field | `data.quality_controller.QualityControlConfig.parallel_validation` | homodyne dataclass field `parallel_validation` missing in heterodyne |
| `KEEP` | missing_field | `data.quality_controller.QualityControlConfig.generate_reports` | homodyne dataclass field `generate_reports` missing in heterodyne |
| `KEEP` | missing_field | `data.quality_controller.QualityControlConfig.export_detailed_reports` | homodyne dataclass field `export_detailed_reports` missing in heterodyne |
| `KEEP` | missing_field | `data.quality_controller.QualityControlConfig.save_quality_history` | homodyne dataclass field `save_quality_history` missing in heterodyne |
| `KEEP` | missing_method | `data.quality_controller.QualityControlResult.get_summary` | homodyne method `get_summary(self) -> dict[str, Any]` missing in heterodyne |
| `KEEP` | missing_field | `data.quality_controller.QualityControlResult.passed` | homodyne dataclass field `passed` missing in heterodyne |
| `KEEP` | missing_field | `data.quality_controller.QualityControlResult.metrics` | homodyne dataclass field `metrics` missing in heterodyne |
| `KEEP` | missing_field | `data.quality_controller.QualityControlResult.issues` | homodyne dataclass field `issues` missing in heterodyne |
| `KEEP` | missing_field | `data.quality_controller.QualityControlResult.repairs_applied` | homodyne dataclass field `repairs_applied` missing in heterodyne |
| `KEEP` | missing_field | `data.quality_controller.QualityControlResult.processing_time` | homodyne dataclass field `processing_time` missing in heterodyne |
| `KEEP` | missing_field | `data.quality_controller.QualityControlResult.data_shape_before` | homodyne dataclass field `data_shape_before` missing in heterodyne |
| `KEEP` | missing_field | `data.quality_controller.QualityControlResult.data_shape_after` | homodyne dataclass field `data_shape_after` missing in heterodyne |
| `KEEP` | missing_field | `data.quality_controller.QualityControlResult.data_modified` | homodyne dataclass field `data_modified` missing in heterodyne |
| `KEEP` | missing_class | `data.quality_controller.QualityMetrics` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_class | `data.quality_controller.RepairStrategy` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_class | `data.types.DatasetInfo` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_class | `data.types.ProcessingStrategy` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_method | `data.validation.DataQualityReport.add_issue` | homodyne method `add_issue(self, issue: ValidationIssue) -> None` missing in heterodyne |
| `KEEP` | missing_method | `data.validation.DataQualityReport.get_summary` | homodyne method `get_summary(self) -> dict[str, Any]` missing in heterodyne |
| `KEEP` | missing_field | `data.validation.DataQualityReport.validation_level` | homodyne dataclass field `validation_level` missing in heterodyne |
| `KEEP` | missing_field | `data.validation.DataQualityReport.total_issues` | homodyne dataclass field `total_issues` missing in heterodyne |
| `KEEP` | missing_field | `data.validation.DataQualityReport.info` | homodyne dataclass field `info` missing in heterodyne |
| `KEEP` | missing_field | `data.validation.DataQualityReport.data_statistics` | homodyne dataclass field `data_statistics` missing in heterodyne |
| `KEEP` | missing_field | `data.validation.DataQualityReport.physics_checks` | homodyne dataclass field `physics_checks` missing in heterodyne |
| `KEEP` | missing_field | `data.validation.DataQualityReport.quality_score` | homodyne dataclass field `quality_score` missing in heterodyne |
| `KEEP` | missing_method | `data.validation.IncrementalValidationCache.is_valid_for_data` | homodyne method `is_valid_for_data(self, data: dict[str, Any], validation_level: str, max_age: float = 3600) -> bool` missing in heterodyne |
| `KEEP` | missing_field | `data.validation.IncrementalValidationCache.data_hash` | homodyne dataclass field `data_hash` missing in heterodyne |
| `KEEP` | missing_field | `data.validation.IncrementalValidationCache.validation_level` | homodyne dataclass field `validation_level` missing in heterodyne |
| `KEEP` | missing_field | `data.validation.IncrementalValidationCache.report` | homodyne dataclass field `report` missing in heterodyne |
| `KEEP` | missing_field | `data.validation.IncrementalValidationCache.timestamp` | homodyne dataclass field `timestamp` missing in heterodyne |
| `KEEP` | missing_field | `data.validation.IncrementalValidationCache.component_hashes` | homodyne dataclass field `component_hashes` missing in heterodyne |
| `KEEP` | missing_class | `data.xpcs_loader.XPCSConfigurationError` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_class | `data.xpcs_loader.XPCSDataFormatError` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_method | `data.xpcs_loader.XPCSDataLoader.load_experimental_data` | homodyne method `load_experimental_data(self) -> dict[str, Any]` missing in heterodyne |
| `KEEP` | missing_class | `data.xpcs_loader.XPCSDependencyError` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_field | `device.config.HardwareConfig.platform` | homodyne dataclass field `platform` missing in heterodyne |
| `KEEP` | missing_field | `device.config.HardwareConfig.num_devices` | homodyne dataclass field `num_devices` missing in heterodyne |
| `KEEP` | missing_field | `device.config.HardwareConfig.memory_per_device_gb` | homodyne dataclass field `memory_per_device_gb` missing in heterodyne |
| `KEEP` | missing_field | `device.config.HardwareConfig.num_nodes` | homodyne dataclass field `num_nodes` missing in heterodyne |
| `KEEP` | missing_field | `device.config.HardwareConfig.cores_per_node` | homodyne dataclass field `cores_per_node` missing in heterodyne |
| `KEEP` | missing_field | `device.config.HardwareConfig.total_memory_gb` | homodyne dataclass field `total_memory_gb` missing in heterodyne |
| `KEEP` | missing_field | `device.config.HardwareConfig.max_parallel_shards` | homodyne dataclass field `max_parallel_shards` missing in heterodyne |
| `KEEP` | missing_class | `optimization.batch_statistics.BatchStatistics` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_method | `optimization.checkpoint_manager.CheckpointManager.cleanup_old_checkpoints` | homodyne method `cleanup_old_checkpoints(self) -> list[Path]` missing in heterodyne |
| `KEEP` | missing_method | `optimization.checkpoint_manager.CheckpointManager.find_latest_checkpoint` | homodyne method `find_latest_checkpoint(self) -> Path | None` missing in heterodyne |
| `KEEP` | missing_method | `optimization.checkpoint_manager.CheckpointManager.load_checkpoint` | homodyne method `load_checkpoint(self, checkpoint_path: Path) -> dict` missing in heterodyne |
| `KEEP` | missing_method | `optimization.checkpoint_manager.CheckpointManager.save_checkpoint` | homodyne method `save_checkpoint(self, batch_idx: int, parameters: np.ndarray, optimizer_state: dict, loss: float, metadata: dict | None = None) -> Path` missing in heterodyne |
| `KEEP` | missing_method | `optimization.checkpoint_manager.CheckpointManager.validate_checkpoint` | homodyne method `validate_checkpoint(self, checkpoint_path: Path) -> bool` missing in heterodyne |
| `KEEP` | missing_method | `optimization.cmc.backends.base.CMCBackend.get_name` | homodyne method `get_name(self) -> str` missing in heterodyne |
| `KEEP` | missing_method | `optimization.cmc.backends.base.CMCBackend.is_available` | homodyne method `is_available(self) -> bool` missing in heterodyne |
| `KEEP` | missing_method | `optimization.cmc.backends.base.CMCBackend.run` | homodyne method `run(self, model: Callable, model_kwargs: dict[str, Any], config: CMCConfig, shards: list[PreparedData] | None = None) -> MCMCSamples` missing in heterodyne |
| `KEEP` | missing_class | `optimization.cmc.backends.multiprocessing.MultiprocessingBackend` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_class | `optimization.cmc.backends.multiprocessing.SharedDataManager` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_method | `optimization.cmc.backends.pbs.PBSBackend.get_name` | homodyne method `get_name(self) -> str` missing in heterodyne |
| `KEEP` | missing_method | `optimization.cmc.backends.pbs.PBSBackend.is_available` | homodyne method `is_available(self) -> bool` missing in heterodyne |
| `KEEP` | missing_method | `optimization.cmc.backends.pbs.PBSBackend.run` | homodyne method `run(self, model: Callable, model_kwargs: dict[str, Any], config: CMCConfig, shards: list[PreparedData] | None = None) -> MCMCSamples` missing in heterodyne |
| `KEEP` | missing_class | `optimization.cmc.backends.pjit.PjitBackend` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_class | `optimization.cmc.backends.worker_pool.WorkerPool` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_method | `optimization.cmc.config.CMCConfig.get_adaptive_sample_counts` | homodyne method `get_adaptive_sample_counts(self, shard_size: int, n_params: int = 7) -> tuple[int, int]` missing in heterodyne |
| `KEEP` | missing_method | `optimization.cmc.config.CMCConfig.get_num_shards` | homodyne method `get_num_shards(self, n_points: int, n_phi: int, n_params: int = 7) -> int` missing in heterodyne |
| `KEEP` | missing_field | `optimization.cmc.data_prep.PreparedData.data` | homodyne dataclass field `data` missing in heterodyne |
| `KEEP` | missing_field | `optimization.cmc.data_prep.PreparedData.t1` | homodyne dataclass field `t1` missing in heterodyne |
| `KEEP` | missing_field | `optimization.cmc.data_prep.PreparedData.t2` | homodyne dataclass field `t2` missing in heterodyne |
| `KEEP` | missing_field | `optimization.cmc.data_prep.PreparedData.phi` | homodyne dataclass field `phi` missing in heterodyne |
| `KEEP` | missing_field | `optimization.cmc.data_prep.PreparedData.phi_unique` | homodyne dataclass field `phi_unique` missing in heterodyne |
| `KEEP` | missing_field | `optimization.cmc.data_prep.PreparedData.phi_indices` | homodyne dataclass field `phi_indices` missing in heterodyne |
| `KEEP` | missing_field | `optimization.cmc.data_prep.PreparedData.n_total` | homodyne dataclass field `n_total` missing in heterodyne |
| `KEEP` | missing_field | `optimization.cmc.data_prep.PreparedData.n_phi` | homodyne dataclass field `n_phi` missing in heterodyne |
| `KEEP` | missing_field | `optimization.cmc.data_prep.PreparedData.noise_scale` | homodyne dataclass field `noise_scale` missing in heterodyne |
| `KEEP` | missing_field | `optimization.cmc.diagnostics.BimodalResult.stds` | homodyne dataclass field `stds` missing in heterodyne |
| `KEEP` | missing_field | `optimization.cmc.diagnostics.BimodalResult.separation` | homodyne dataclass field `separation` missing in heterodyne |
| `KEEP` | missing_field | `optimization.cmc.diagnostics.BimodalResult.relative_separation` | homodyne dataclass field `relative_separation` missing in heterodyne |
| `KEEP` | missing_field | `optimization.cmc.diagnostics.ModeCluster.samples` | homodyne dataclass field `samples` missing in heterodyne |
| `KEEP` | missing_method | `optimization.cmc.reparameterization.ReparamConfig.enable_d_total` | homodyne method `enable_d_total(self) -> bool` missing in heterodyne |
| `KEEP` | missing_method | `optimization.cmc.reparameterization.ReparamConfig.enable_log_gamma` | homodyne method `enable_log_gamma(self) -> bool` missing in heterodyne |
| `KEEP` | missing_field | `optimization.cmc.reparameterization.ReparamConfig.enable_gamma_ref` | homodyne dataclass field `enable_gamma_ref` missing in heterodyne |
| `KEEP` | missing_method | `optimization.cmc.results.CMCResult.from_mcmc_samples` | homodyne method `from_mcmc_samples(cls, mcmc_samples: MCMCSamples, stats: SamplingStats, analysis_mode: str, n_warmup: int = 500, min_ess: float | None = None) -> CMCResult` missing in heterodyne |
| `KEEP` | missing_method | `optimization.cmc.results.CMCResult.is_cmc_result` | homodyne method `is_cmc_result(self) -> bool` missing in heterodyne |
| `KEEP` | missing_method | `optimization.cmc.results.CMCResult.message` | homodyne method `message(self) -> str` missing in heterodyne |
| `KEEP` | missing_method | `optimization.cmc.results.CMCResult.success` | homodyne method `success(self) -> bool` missing in heterodyne |
| `KEEP` | missing_method | `optimization.cmc.results.CMCResult.validate_parameters` | homodyne method `validate_parameters(self, n_phi: int | None = None) -> list[str]` missing in heterodyne |
| `KEEP` | missing_field | `optimization.cmc.results.CMCResult.parameters` | homodyne dataclass field `parameters` missing in heterodyne |
| `KEEP` | missing_field | `optimization.cmc.results.CMCResult.uncertainties` | homodyne dataclass field `uncertainties` missing in heterodyne |
| `KEEP` | missing_field | `optimization.cmc.results.CMCResult.param_names` | homodyne dataclass field `param_names` missing in heterodyne |
| `KEEP` | missing_field | `optimization.cmc.results.CMCResult.inference_data` | homodyne dataclass field `inference_data` missing in heterodyne |
| `KEEP` | missing_field | `optimization.cmc.results.CMCResult.execution_time` | homodyne dataclass field `execution_time` missing in heterodyne |
| `KEEP` | missing_field | `optimization.cmc.results.CMCResult.n_chains` | homodyne dataclass field `n_chains` missing in heterodyne |
| `KEEP` | missing_field | `optimization.cmc.results.CMCResult.n_samples` | homodyne dataclass field `n_samples` missing in heterodyne |
| `KEEP` | missing_field | `optimization.cmc.results.CMCResult.n_warmup` | homodyne dataclass field `n_warmup` missing in heterodyne |
| `KEEP` | missing_field | `optimization.cmc.results.CMCResult.analysis_mode` | homodyne dataclass field `analysis_mode` missing in heterodyne |
| `KEEP` | missing_field | `optimization.cmc.results.CMCResult.covariance` | homodyne dataclass field `covariance` missing in heterodyne |
| `KEEP` | missing_field | `optimization.cmc.results.CMCResult.reduced_chi_squared` | homodyne dataclass field `reduced_chi_squared` missing in heterodyne |
| `KEEP` | missing_field | `optimization.cmc.results.CMCResult.device_info` | homodyne dataclass field `device_info` missing in heterodyne |
| `KEEP` | missing_field | `optimization.cmc.results.CMCResult.recovery_actions` | homodyne dataclass field `recovery_actions` missing in heterodyne |
| `KEEP` | missing_field | `optimization.cmc.results.CMCResult.mean_params` | homodyne dataclass field `mean_params` missing in heterodyne |
| `KEEP` | missing_field | `optimization.cmc.results.CMCResult.std_params` | homodyne dataclass field `std_params` missing in heterodyne |
| `KEEP` | missing_class | `optimization.cmc.sampler.MCMCSamples` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_method | `optimization.cmc.sampler.SamplingPlan.from_config` | homodyne method `from_config(cls, config: CMCConfig, shard_size: int, n_params: int) -> SamplingPlan` missing in heterodyne |
| `KEEP` | missing_method | `optimization.cmc.sampler.SamplingPlan.total_samples` | homodyne method `total_samples(self) -> int` missing in heterodyne |
| `KEEP` | missing_field | `optimization.cmc.sampler.SamplingPlan.n_warmup` | homodyne dataclass field `n_warmup` missing in heterodyne |
| `KEEP` | missing_field | `optimization.cmc.sampler.SamplingPlan.n_samples` | homodyne dataclass field `n_samples` missing in heterodyne |
| `KEEP` | missing_field | `optimization.cmc.sampler.SamplingPlan.n_chains` | homodyne dataclass field `n_chains` missing in heterodyne |
| `KEEP` | missing_field | `optimization.cmc.sampler.SamplingPlan.shard_size` | homodyne dataclass field `shard_size` missing in heterodyne |
| `KEEP` | missing_field | `optimization.cmc.sampler.SamplingPlan.n_params` | homodyne dataclass field `n_params` missing in heterodyne |
| `KEEP` | missing_field | `optimization.cmc.sampler.SamplingPlan.was_adapted` | homodyne dataclass field `was_adapted` missing in heterodyne |
| `KEEP` | missing_field | `optimization.cmc.sampler.SamplingStats.warmup_time` | homodyne dataclass field `warmup_time` missing in heterodyne |
| `KEEP` | missing_field | `optimization.cmc.sampler.SamplingStats.sampling_time` | homodyne dataclass field `sampling_time` missing in heterodyne |
| `KEEP` | missing_field | `optimization.cmc.sampler.SamplingStats.total_time` | homodyne dataclass field `total_time` missing in heterodyne |
| `KEEP` | missing_field | `optimization.cmc.sampler.SamplingStats.num_divergent` | homodyne dataclass field `num_divergent` missing in heterodyne |
| `KEEP` | missing_field | `optimization.cmc.sampler.SamplingStats.accept_prob` | homodyne dataclass field `accept_prob` missing in heterodyne |
| `KEEP` | missing_field | `optimization.cmc.sampler.SamplingStats.step_size` | homodyne dataclass field `step_size` missing in heterodyne |
| `KEEP` | missing_field | `optimization.cmc.sampler.SamplingStats.step_size_min` | homodyne dataclass field `step_size_min` missing in heterodyne |
| `KEEP` | missing_field | `optimization.cmc.sampler.SamplingStats.step_size_max` | homodyne dataclass field `step_size_max` missing in heterodyne |
| `KEEP` | missing_field | `optimization.cmc.sampler.SamplingStats.inverse_mass_matrix_summary` | homodyne dataclass field `inverse_mass_matrix_summary` missing in heterodyne |
| `KEEP` | missing_field | `optimization.cmc.sampler.SamplingStats.tree_depth` | homodyne dataclass field `tree_depth` missing in heterodyne |
| `KEEP` | missing_field | `optimization.cmc.sampler.SamplingStats.plan` | homodyne dataclass field `plan` missing in heterodyne |
| `KEEP` | missing_method | `optimization.cmc.scaling.ParameterScaling.to_normalized` | homodyne method `to_normalized(self, value: float) -> float` missing in heterodyne |
| `KEEP` | missing_class | `optimization.exceptions.NLSQCheckpointError` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_method | `optimization.nlsq.adapter.NLSQAdapter.fit` | homodyne method `fit(self, data: Any, config: Any, initial_params: np.ndarray | None = None, bounds: tuple[np.ndarray, np.ndarray] | None = None, analysis_mode: str = 'static_isotropic', per_angle_scaling: bool = True, diagnostics_enabled: bool = False, shear_transforms: dict[str, Any] | None = None, per_angle_scaling_initial: dict[str, list[float]] | None = None, anti_degeneracy_controller: Any | None = None) -> OptimizationResult` missing in heterodyne |
| `KEEP` | missing_class | `optimization.nlsq.adaptive_regularization.AdaptiveRegularizationConfig` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_method | `optimization.nlsq.adaptive_regularization.AdaptiveRegularizer.check_constraint_violation` | homodyne method `check_constraint_violation(self, params: np.ndarray) -> dict[str, dict]` missing in heterodyne |
| `KEEP` | missing_method | `optimization.nlsq.adaptive_regularization.AdaptiveRegularizer.compute_regularization` | homodyne method `compute_regularization(self, params: np.ndarray, mse: float, n_points: int) -> float` missing in heterodyne |
| `KEEP` | missing_method | `optimization.nlsq.adaptive_regularization.AdaptiveRegularizer.compute_regularization_gradient` | homodyne method `compute_regularization_gradient(self, params: np.ndarray, mse: float, n_points: int) -> np.ndarray` missing in heterodyne |
| `KEEP` | missing_method | `optimization.nlsq.adaptive_regularization.AdaptiveRegularizer.compute_regularization_jax` | homodyne method `compute_regularization_jax(self, params: jnp.ndarray, mse: jnp.ndarray, n_points: int) -> jnp.ndarray` missing in heterodyne |
| `KEEP` | missing_method | `optimization.nlsq.adaptive_regularization.AdaptiveRegularizer.get_diagnostics` | homodyne method `get_diagnostics(self) -> dict` missing in heterodyne |
| `KEEP` | missing_method | `optimization.nlsq.adaptive_regularization.AdaptiveRegularizer.log_summary` | homodyne method `log_summary(self, params: np.ndarray, mse: float, n_points: int) -> None` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.anti_degeneracy_controller.AntiDegeneracyConfig.shear_weighting_enable` | homodyne dataclass field `shear_weighting_enable` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.anti_degeneracy_controller.AntiDegeneracyConfig.shear_weighting_min_weight` | homodyne dataclass field `shear_weighting_min_weight` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.anti_degeneracy_controller.AntiDegeneracyConfig.shear_weighting_alpha` | homodyne dataclass field `shear_weighting_alpha` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.anti_degeneracy_controller.AntiDegeneracyConfig.shear_weighting_update_frequency` | homodyne dataclass field `shear_weighting_update_frequency` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.anti_degeneracy_controller.AntiDegeneracyConfig.shear_weighting_normalize` | homodyne dataclass field `shear_weighting_normalize` missing in heterodyne |
| `KEEP` | missing_method | `optimization.nlsq.anti_degeneracy_controller.AntiDegeneracyController.from_config` | homodyne method `from_config(cls, config_dict: dict[str, Any], n_phi: int, phi_angles: np.ndarray, n_physical: int, per_angle_scaling: bool = True, is_laminar_flow: bool = True) -> AntiDegeneracyController` missing in heterodyne |
| `KEEP` | missing_method | `optimization.nlsq.anti_degeneracy_controller.AntiDegeneracyController.get_group_variance_indices` | homodyne method `get_group_variance_indices(self) -> list[tuple[int, int]] | None` missing in heterodyne |
| `KEEP` | missing_method | `optimization.nlsq.anti_degeneracy_controller.AntiDegeneracyController.transform_params_from_constant` | homodyne method `transform_params_from_constant(self, constant_params: np.ndarray) -> np.ndarray` missing in heterodyne |
| `KEEP` | missing_method | `optimization.nlsq.anti_degeneracy_controller.AntiDegeneracyController.transform_params_from_fourier` | homodyne method `transform_params_from_fourier(self, fourier_params: np.ndarray) -> np.ndarray` missing in heterodyne |
| `KEEP` | missing_method | `optimization.nlsq.anti_degeneracy_controller.AntiDegeneracyController.transform_params_to_constant` | homodyne method `transform_params_to_constant(self, params: np.ndarray) -> np.ndarray` missing in heterodyne |
| `KEEP` | missing_method | `optimization.nlsq.anti_degeneracy_controller.AntiDegeneracyController.update_shear_phi0` | homodyne method `update_shear_phi0(self, params: np.ndarray, iteration: int = 0) -> None` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.anti_degeneracy_controller.AntiDegeneracyController.hierarchical` | homodyne dataclass field `hierarchical` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.anti_degeneracy_controller.AntiDegeneracyController.shear_weighter` | homodyne dataclass field `shear_weighter` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.anti_degeneracy_controller.AntiDegeneracyController.mapper` | homodyne dataclass field `mapper` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.cmaes_wrapper.CMAESResult.parameters` | homodyne dataclass field `parameters` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.cmaes_wrapper.CMAESResult.covariance` | homodyne dataclass field `covariance` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.cmaes_wrapper.CMAESResult.chi_squared` | homodyne dataclass field `chi_squared` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.cmaes_wrapper.CMAESResult.success` | homodyne dataclass field `success` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.cmaes_wrapper.CMAESResult.diagnostics` | homodyne dataclass field `diagnostics` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.cmaes_wrapper.CMAESResult.method_used` | homodyne dataclass field `method_used` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.cmaes_wrapper.CMAESResult.nlsq_refined` | homodyne dataclass field `nlsq_refined` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.cmaes_wrapper.CMAESResult.message` | homodyne dataclass field `message` missing in heterodyne |
| `KEEP` | missing_method | `optimization.nlsq.cmaes_wrapper.CMAESWrapper.compute_scale_ratio` | homodyne method `compute_scale_ratio(self, bounds: tuple[np.ndarray, np.ndarray]) -> float` missing in heterodyne |
| `KEEP` | missing_method | `optimization.nlsq.cmaes_wrapper.CMAESWrapper.fit` | homodyne method `fit(self, model_func: Callable, xdata: np.ndarray, ydata: np.ndarray, p0: np.ndarray, bounds: tuple[np.ndarray, np.ndarray], sigma: np.ndarray | None = None, warmstart_chi2: float | None = None) -> CMAESResult` missing in heterodyne |
| `KEEP` | missing_method | `optimization.nlsq.cmaes_wrapper.CMAESWrapper.is_available` | homodyne method `is_available(self) -> bool` missing in heterodyne |
| `KEEP` | missing_method | `optimization.nlsq.cmaes_wrapper.CMAESWrapper.should_use_cmaes` | homodyne method `should_use_cmaes(self, bounds: tuple[np.ndarray, np.ndarray], scale_threshold: float = 1000.0) -> bool` missing in heterodyne |
| `KEEP` | missing_class | `optimization.nlsq.cmaes_wrapper.CMAESWrapperConfig` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_method | `optimization.nlsq.config.HybridRecoveryConfig.get_retry_settings` | homodyne method `get_retry_settings(self, attempt: int) -> dict` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.config.HybridRecoveryConfig.log_retries` | homodyne dataclass field `log_retries` missing in heterodyne |
| `KEEP` | missing_method | `optimization.nlsq.config.NLSQConfig.from_dict` | homodyne method `from_dict(cls, config_dict: dict[str, Any]) -> NLSQConfig` missing in heterodyne |
| `KEEP` | missing_method | `optimization.nlsq.config.NLSQConfig.from_yaml` | homodyne method `from_yaml(cls, yaml_path: str) -> NLSQConfig` missing in heterodyne |
| `KEEP` | missing_method | `optimization.nlsq.config.NLSQConfig.is_valid` | homodyne method `is_valid(self) -> bool` missing in heterodyne |
| `KEEP` | missing_method | `optimization.nlsq.config.NLSQConfig.to_workflow_kwargs` | homodyne method `to_workflow_kwargs(self) -> dict[str, Any]` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.config.NLSQConfig.trust_region_scale` | homodyne dataclass field `trust_region_scale` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.config.NLSQConfig.enable_progress_bar` | homodyne dataclass field `enable_progress_bar` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.config.NLSQConfig.log_iteration_interval` | homodyne dataclass field `log_iteration_interval` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.config.NLSQConfig.enable_hybrid_streaming` | homodyne dataclass field `enable_hybrid_streaming` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.config.NLSQConfig.hybrid_normalize` | homodyne dataclass field `hybrid_normalize` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.config.NLSQConfig.hybrid_normalization_strategy` | homodyne dataclass field `hybrid_normalization_strategy` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.config.NLSQConfig.hybrid_warmup_iterations` | homodyne dataclass field `hybrid_warmup_iterations` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.config.NLSQConfig.hybrid_max_warmup_iterations` | homodyne dataclass field `hybrid_max_warmup_iterations` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.config.NLSQConfig.hybrid_warmup_learning_rate` | homodyne dataclass field `hybrid_warmup_learning_rate` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.config.NLSQConfig.hybrid_gauss_newton_max_iterations` | homodyne dataclass field `hybrid_gauss_newton_max_iterations` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.config.NLSQConfig.hybrid_gauss_newton_tol` | homodyne dataclass field `hybrid_gauss_newton_tol` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.config.NLSQConfig.hybrid_chunk_size` | homodyne dataclass field `hybrid_chunk_size` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.config.NLSQConfig.hybrid_trust_region_initial` | homodyne dataclass field `hybrid_trust_region_initial` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.config.NLSQConfig.hybrid_regularization_factor` | homodyne dataclass field `hybrid_regularization_factor` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.config.NLSQConfig.hybrid_enable_checkpoints` | homodyne dataclass field `hybrid_enable_checkpoints` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.config.NLSQConfig.hybrid_checkpoint_frequency` | homodyne dataclass field `hybrid_checkpoint_frequency` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.config.NLSQConfig.hybrid_validate_numerics` | homodyne dataclass field `hybrid_validate_numerics` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.config.NLSQConfig.hybrid_enable_warm_start_detection` | homodyne dataclass field `hybrid_enable_warm_start_detection` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.config.NLSQConfig.hybrid_warm_start_threshold` | homodyne dataclass field `hybrid_warm_start_threshold` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.config.NLSQConfig.hybrid_enable_adaptive_warmup_lr` | homodyne dataclass field `hybrid_enable_adaptive_warmup_lr` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.config.NLSQConfig.hybrid_warmup_lr_refinement` | homodyne dataclass field `hybrid_warmup_lr_refinement` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.config.NLSQConfig.hybrid_warmup_lr_careful` | homodyne dataclass field `hybrid_warmup_lr_careful` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.config.NLSQConfig.hybrid_enable_cost_guard` | homodyne dataclass field `hybrid_enable_cost_guard` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.config.NLSQConfig.hybrid_cost_increase_tolerance` | homodyne dataclass field `hybrid_cost_increase_tolerance` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.config.NLSQConfig.hybrid_enable_step_clipping` | homodyne dataclass field `hybrid_enable_step_clipping` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.config.NLSQConfig.hybrid_max_warmup_step_size` | homodyne dataclass field `hybrid_max_warmup_step_size` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.config.NLSQConfig.enable_multi_start` | homodyne dataclass field `enable_multi_start` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.config.NLSQConfig.multi_start_n_starts` | homodyne dataclass field `multi_start_n_starts` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.config.NLSQConfig.multi_start_seed` | homodyne dataclass field `multi_start_seed` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.config.NLSQConfig.multi_start_sampling_strategy` | homodyne dataclass field `multi_start_sampling_strategy` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.config.NLSQConfig.multi_start_n_workers` | homodyne dataclass field `multi_start_n_workers` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.config.NLSQConfig.multi_start_use_screening` | homodyne dataclass field `multi_start_use_screening` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.config.NLSQConfig.multi_start_screen_keep_fraction` | homodyne dataclass field `multi_start_screen_keep_fraction` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.config.NLSQConfig.multi_start_refine_top_k` | homodyne dataclass field `multi_start_refine_top_k` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.config.NLSQConfig.multi_start_refinement_ftol` | homodyne dataclass field `multi_start_refinement_ftol` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.config.NLSQConfig.multi_start_degeneracy_threshold` | homodyne dataclass field `multi_start_degeneracy_threshold` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.config.NLSQConfig.hierarchical_physical_max_iterations` | homodyne dataclass field `hierarchical_physical_max_iterations` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.config.NLSQConfig.hierarchical_per_angle_max_iterations` | homodyne dataclass field `hierarchical_per_angle_max_iterations` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.config.NLSQConfig.regularization_target_contribution` | homodyne dataclass field `regularization_target_contribution` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.config.NLSQConfig.regularization_max_cv` | homodyne dataclass field `regularization_max_cv` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.config.NLSQConfig.regularization_auto_tune_lambda` | homodyne dataclass field `regularization_auto_tune_lambda` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.config.NLSQConfig.gradient_collapse_response` | homodyne dataclass field `gradient_collapse_response` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.config.NLSQConfig.cmaes_preset` | homodyne dataclass field `cmaes_preset` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.config.NLSQConfig.cmaes_max_generations` | homodyne dataclass field `cmaes_max_generations` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.config.NLSQConfig.cmaes_popsize` | homodyne dataclass field `cmaes_popsize` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.config.NLSQConfig.cmaes_sigma` | homodyne dataclass field `cmaes_sigma` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.config.NLSQConfig.cmaes_sigma_warmstart` | homodyne dataclass field `cmaes_sigma_warmstart` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.config.NLSQConfig.cmaes_tol_fun` | homodyne dataclass field `cmaes_tol_fun` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.config.NLSQConfig.cmaes_tol_x` | homodyne dataclass field `cmaes_tol_x` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.config.NLSQConfig.cmaes_population_batch_size` | homodyne dataclass field `cmaes_population_batch_size` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.config.NLSQConfig.cmaes_data_chunk_size` | homodyne dataclass field `cmaes_data_chunk_size` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.config.NLSQConfig.cmaes_refine_with_nlsq` | homodyne dataclass field `cmaes_refine_with_nlsq` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.config.NLSQConfig.cmaes_auto_select` | homodyne dataclass field `cmaes_auto_select` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.config.NLSQConfig.cmaes_scale_threshold` | homodyne dataclass field `cmaes_scale_threshold` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.config.NLSQConfig.cmaes_memory_limit_gb` | homodyne dataclass field `cmaes_memory_limit_gb` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.config.NLSQConfig.cmaes_refinement_workflow` | homodyne dataclass field `cmaes_refinement_workflow` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.config.NLSQConfig.cmaes_refinement_ftol` | homodyne dataclass field `cmaes_refinement_ftol` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.config.NLSQConfig.cmaes_refinement_xtol` | homodyne dataclass field `cmaes_refinement_xtol` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.config.NLSQConfig.cmaes_refinement_gtol` | homodyne dataclass field `cmaes_refinement_gtol` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.config.NLSQConfig.cmaes_refinement_max_nfev` | homodyne dataclass field `cmaes_refinement_max_nfev` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.config.NLSQConfig.cmaes_refinement_loss` | homodyne dataclass field `cmaes_refinement_loss` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.config.NLSQConfig.cmaes_normalize` | homodyne dataclass field `cmaes_normalize` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.config.NLSQConfig.cmaes_normalization_epsilon` | homodyne dataclass field `cmaes_normalization_epsilon` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.config.NLSQConfig.enable_quality_validation` | homodyne dataclass field `enable_quality_validation` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.config.NLSQConfig.quality_reduced_chi_squared_threshold` | homodyne dataclass field `quality_reduced_chi_squared_threshold` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.config.NLSQConfig.quality_warn_on_max_restarts` | homodyne dataclass field `quality_warn_on_max_restarts` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.config.NLSQConfig.quality_warn_on_bounds_hit` | homodyne dataclass field `quality_warn_on_bounds_hit` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.config.NLSQConfig.quality_warn_on_convergence_failure` | homodyne dataclass field `quality_warn_on_convergence_failure` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.config.NLSQConfig.quality_bounds_tolerance` | homodyne dataclass field `quality_bounds_tolerance` missing in heterodyne |
| `KEEP` | missing_class | `optimization.nlsq.core.NLSQResult` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_class | `optimization.nlsq.data_prep.ExpandedParameters` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_class | `optimization.nlsq.data_prep.PreparedData` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_class | `optimization.nlsq.gradient_monitor.CollapseEvent` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_class | `optimization.nlsq.gradient_monitor.GradientCollapseMonitor` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_class | `optimization.nlsq.gradient_monitor.GradientMonitorConfig` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_method | `optimization.nlsq.hierarchical.HierarchicalConfig.from_dict` | homodyne method `from_dict(cls, config_dict: dict) -> HierarchicalConfig` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.hierarchical.HierarchicalConfig.enable` | homodyne dataclass field `enable` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.hierarchical.HierarchicalConfig.max_outer_iterations` | homodyne dataclass field `max_outer_iterations` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.hierarchical.HierarchicalConfig.outer_tolerance` | homodyne dataclass field `outer_tolerance` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.hierarchical.HierarchicalConfig.physical_max_iterations` | homodyne dataclass field `physical_max_iterations` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.hierarchical.HierarchicalConfig.physical_ftol` | homodyne dataclass field `physical_ftol` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.hierarchical.HierarchicalConfig.per_angle_max_iterations` | homodyne dataclass field `per_angle_max_iterations` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.hierarchical.HierarchicalConfig.per_angle_ftol` | homodyne dataclass field `per_angle_ftol` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.hierarchical.HierarchicalConfig.log_stage_transitions` | homodyne dataclass field `log_stage_transitions` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.hierarchical.HierarchicalConfig.save_intermediate_results` | homodyne dataclass field `save_intermediate_results` missing in heterodyne |
| `KEEP` | missing_class | `optimization.nlsq.hierarchical.HierarchicalOptimizer` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_field | `optimization.nlsq.hierarchical.HierarchicalResult.x` | homodyne dataclass field `x` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.hierarchical.HierarchicalResult.fun` | homodyne dataclass field `fun` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.hierarchical.HierarchicalResult.success` | homodyne dataclass field `success` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.hierarchical.HierarchicalResult.n_outer_iterations` | homodyne dataclass field `n_outer_iterations` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.hierarchical.HierarchicalResult.history` | homodyne dataclass field `history` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.hierarchical.HierarchicalResult.total_time` | homodyne dataclass field `total_time` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.hierarchical.HierarchicalResult.message` | homodyne dataclass field `message` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.memory.StrategyDecision.index_memory_gb` | homodyne dataclass field `index_memory_gb` missing in heterodyne |
| `KEEP` | missing_method | `optimization.nlsq.multistart.MultiStartConfig.from_nlsq_config` | homodyne method `from_nlsq_config(cls, nlsq_config: Any) -> MultiStartConfig` missing in heterodyne |
| `KEEP` | missing_method | `optimization.nlsq.multistart.MultiStartConfig.to_nlsq_global_config` | homodyne method `to_nlsq_global_config(self) -> Any` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.multistart.MultiStartConfig.enable` | homodyne dataclass field `enable` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.multistart.MultiStartConfig.sampling_strategy` | homodyne dataclass field `sampling_strategy` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.multistart.MultiStartConfig.custom_starts` | homodyne dataclass field `custom_starts` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.multistart.MultiStartConfig.n_workers` | homodyne dataclass field `n_workers` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.multistart.MultiStartConfig.use_screening` | homodyne dataclass field `use_screening` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.multistart.MultiStartConfig.screen_keep_fraction` | homodyne dataclass field `screen_keep_fraction` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.multistart.MultiStartConfig.refine_top_k` | homodyne dataclass field `refine_top_k` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.multistart.MultiStartConfig.refinement_ftol` | homodyne dataclass field `refinement_ftol` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.multistart.MultiStartConfig.degeneracy_threshold` | homodyne dataclass field `degeneracy_threshold` missing in heterodyne |
| `KEEP` | missing_method | `optimization.nlsq.multistart.MultiStartResult.to_optimization_result` | homodyne method `to_optimization_result(self) -> OptimizationResult` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.multistart.MultiStartResult.best` | homodyne dataclass field `best` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.multistart.MultiStartResult.all_results` | homodyne dataclass field `all_results` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.multistart.MultiStartResult.strategy_used` | homodyne dataclass field `strategy_used` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.multistart.MultiStartResult.n_unique_basins` | homodyne dataclass field `n_unique_basins` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.multistart.MultiStartResult.degeneracy_detected` | homodyne dataclass field `degeneracy_detected` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.multistart.MultiStartResult.total_wall_time` | homodyne dataclass field `total_wall_time` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.multistart.MultiStartResult.screening_costs` | homodyne dataclass field `screening_costs` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.multistart.MultiStartResult.basin_labels` | homodyne dataclass field `basin_labels` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.multistart.SingleStartResult.start_idx` | homodyne dataclass field `start_idx` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.multistart.SingleStartResult.final_params` | homodyne dataclass field `final_params` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.multistart.SingleStartResult.chi_squared` | homodyne dataclass field `chi_squared` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.multistart.SingleStartResult.reduced_chi_squared` | homodyne dataclass field `reduced_chi_squared` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.multistart.SingleStartResult.success` | homodyne dataclass field `success` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.multistart.SingleStartResult.status` | homodyne dataclass field `status` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.multistart.SingleStartResult.message` | homodyne dataclass field `message` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.multistart.SingleStartResult.n_iterations` | homodyne dataclass field `n_iterations` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.multistart.SingleStartResult.n_fev` | homodyne dataclass field `n_fev` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.multistart.SingleStartResult.hessian` | homodyne dataclass field `hessian` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.multistart.SingleStartResult.covariance` | homodyne dataclass field `covariance` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.multistart.SingleStartResult.jacobian` | homodyne dataclass field `jacobian` missing in heterodyne |
| `KEEP` | missing_class | `optimization.nlsq.parallel_accumulator.OOCComputePool` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_class | `optimization.nlsq.parallel_accumulator.OOCSharedArrays` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_method | `optimization.nlsq.parameter_index_mapper.ParameterIndexMapper.get_covariance_slice_indices` | homodyne method `get_covariance_slice_indices(self) -> tuple[slice, slice]` missing in heterodyne |
| `KEEP` | missing_method | `optimization.nlsq.parameter_index_mapper.ParameterIndexMapper.get_diagnostics` | homodyne method `get_diagnostics(self) -> dict` missing in heterodyne |
| `KEEP` | missing_method | `optimization.nlsq.parameter_index_mapper.ParameterIndexMapper.get_group_indices` | homodyne method `get_group_indices(self) -> list[tuple[int, int]]` missing in heterodyne |
| `KEEP` | missing_method | `optimization.nlsq.parameter_index_mapper.ParameterIndexMapper.get_per_angle_indices` | homodyne method `get_per_angle_indices(self) -> list[int]` missing in heterodyne |
| `KEEP` | missing_method | `optimization.nlsq.parameter_index_mapper.ParameterIndexMapper.get_physical_indices` | homodyne method `get_physical_indices(self) -> list[int]` missing in heterodyne |
| `KEEP` | missing_method | `optimization.nlsq.parameter_index_mapper.ParameterIndexMapper.mode_name` | homodyne method `mode_name(self) -> str` missing in heterodyne |
| `KEEP` | missing_method | `optimization.nlsq.parameter_index_mapper.ParameterIndexMapper.n_per_angle_total` | homodyne method `n_per_angle_total(self) -> int` missing in heterodyne |
| `KEEP` | missing_method | `optimization.nlsq.parameter_index_mapper.ParameterIndexMapper.n_per_group` | homodyne method `n_per_group(self) -> int` missing in heterodyne |
| `KEEP` | missing_method | `optimization.nlsq.parameter_index_mapper.ParameterIndexMapper.total_params` | homodyne method `total_params(self) -> int` missing in heterodyne |
| `KEEP` | missing_method | `optimization.nlsq.parameter_index_mapper.ParameterIndexMapper.use_fourier` | homodyne method `use_fourier(self) -> bool` missing in heterodyne |
| `KEEP` | missing_method | `optimization.nlsq.parameter_index_mapper.ParameterIndexMapper.validate_indices` | homodyne method `validate_indices(self, params: np.ndarray) -> bool` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.parameter_index_mapper.ParameterIndexMapper.n_phi` | homodyne dataclass field `n_phi` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.parameter_index_mapper.ParameterIndexMapper.n_physical` | homodyne dataclass field `n_physical` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.parameter_index_mapper.ParameterIndexMapper.fourier` | homodyne dataclass field `fourier` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.parameter_index_mapper.ParameterIndexMapper.use_constant` | homodyne dataclass field `use_constant` missing in heterodyne |
| `KEEP` | missing_class | `optimization.nlsq.progress.HomodyneIterationLogger` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_class | `optimization.nlsq.progress.MultiStartProgressTracker` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_class | `optimization.nlsq.progress.ProgressConfig` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_class | `optimization.nlsq.result_builder.QualityMetrics` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_class | `optimization.nlsq.result_builder.ResultBuilder` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_class | `optimization.nlsq.results.FallbackInfo` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_class | `optimization.nlsq.results.FunctionEvaluationCounter` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_class | `optimization.nlsq.results.OptimizationResult` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_class | `optimization.nlsq.results.UseSequentialOptimization` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_class | `optimization.nlsq.shear_weighting.ShearSensitivityWeighting` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_class | `optimization.nlsq.shear_weighting.ShearWeightingConfig` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_class | `optimization.nlsq.strategies.chunking.AngleDistributionStats` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_class | `optimization.nlsq.strategies.chunking.StratificationDiagnostics` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_class | `optimization.nlsq.strategies.chunking.StratifiedIndexIterator` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_field | `optimization.nlsq.strategies.executors.ExecutionResult.popt` | homodyne dataclass field `popt` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.strategies.executors.ExecutionResult.pcov` | homodyne dataclass field `pcov` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.strategies.executors.ExecutionResult.info` | homodyne dataclass field `info` missing in heterodyne |
| `KEEP` | missing_method | `optimization.nlsq.strategies.executors.LargeDatasetExecutor.execute` | homodyne method `execute(self, residual_fn: Callable[..., Any], xdata: np.ndarray, ydata: np.ndarray, initial_params: np.ndarray, bounds: tuple[np.ndarray, np.ndarray] | None, loss_name: str, x_scale_value: float | np.ndarray | str, logger: Any) -> ExecutionResult` missing in heterodyne |
| `KEEP` | missing_method | `optimization.nlsq.strategies.executors.LargeDatasetExecutor.name` | homodyne method `name(self) -> str` missing in heterodyne |
| `KEEP` | missing_method | `optimization.nlsq.strategies.executors.LargeDatasetExecutor.supports_progress` | homodyne method `supports_progress(self) -> bool` missing in heterodyne |
| `KEEP` | missing_method | `optimization.nlsq.strategies.executors.OptimizationExecutor.execute` | homodyne method `execute(self, residual_fn: Callable[..., Any], xdata: np.ndarray, ydata: np.ndarray, initial_params: np.ndarray, bounds: tuple[np.ndarray, np.ndarray] | None, loss_name: str, x_scale_value: float | np.ndarray | str, logger: Any) -> ExecutionResult` missing in heterodyne |
| `KEEP` | missing_method | `optimization.nlsq.strategies.executors.OptimizationExecutor.supports_progress` | homodyne method `supports_progress(self) -> bool` missing in heterodyne |
| `KEEP` | missing_method | `optimization.nlsq.strategies.executors.StandardExecutor.execute` | homodyne method `execute(self, residual_fn: Callable[..., Any], xdata: np.ndarray, ydata: np.ndarray, initial_params: np.ndarray, bounds: tuple[np.ndarray, np.ndarray] | None, loss_name: str, x_scale_value: float | np.ndarray | str, logger: Any) -> ExecutionResult` missing in heterodyne |
| `KEEP` | missing_method | `optimization.nlsq.strategies.executors.StandardExecutor.name` | homodyne method `name(self) -> str` missing in heterodyne |
| `KEEP` | missing_method | `optimization.nlsq.strategies.executors.StandardExecutor.supports_progress` | homodyne method `supports_progress(self) -> bool` missing in heterodyne |
| `KEEP` | missing_class | `optimization.nlsq.strategies.executors.StreamingExecutor` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_class | `optimization.nlsq.strategies.residual.StratifiedResidualFunction` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_class | `optimization.nlsq.strategies.residual_jit.StratifiedResidualFunctionJIT` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_field | `optimization.nlsq.strategies.sequential.AngleSubset.phi_indices` | homodyne dataclass field `phi_indices` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.strategies.sequential.AngleSubset.phi` | homodyne dataclass field `phi` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.strategies.sequential.AngleSubset.t1` | homodyne dataclass field `t1` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.strategies.sequential.AngleSubset.t2` | homodyne dataclass field `t2` missing in heterodyne |
| `KEEP` | missing_field | `optimization.nlsq.strategies.sequential.AngleSubset.g2_exp` | homodyne dataclass field `g2_exp` missing in heterodyne |
| `KEEP` | missing_class | `optimization.nlsq.strategies.sequential.SequentialResult` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_method | `optimization.nlsq.validation.input_validator.InputValidator.validate_all` | homodyne method `validate_all(self, xdata: np.ndarray, ydata: np.ndarray, initial_params: np.ndarray, bounds: tuple[np.ndarray, np.ndarray] | None) -> bool` missing in heterodyne |
| `KEEP` | missing_method | `optimization.nlsq.validation.input_validator.InputValidator.validation_errors` | homodyne method `validation_errors(self) -> list[str]` missing in heterodyne |
| `KEEP` | missing_class | `optimization.nlsq.validation.result_validator.ResultValidator` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_method | `optimization.nlsq.wrapper.NLSQWrapper.fit` | homodyne method `fit(self, data: Any, config: Any, initial_params: np.ndarray | None = None, bounds: tuple[np.ndarray, np.ndarray] | None = None, analysis_mode: str = 'static_isotropic', per_angle_scaling: bool = True, diagnostics_enabled: bool = False, shear_transforms: dict[str, Any] | None = None, per_angle_scaling_initial: dict[str, list[float]] | None = None) -> OptimizationResult` missing in heterodyne |
| `KEEP` | missing_method | `optimization.numerical_validation.NumericalValidator.set_bounds` | homodyne method `set_bounds(self, bounds: tuple[np.ndarray, np.ndarray]) -> None` missing in heterodyne |
| `KEEP` | missing_method | `optimization.numerical_validation.NumericalValidator.validate_gradients` | homodyne method `validate_gradients(self, gradients: Any) -> None` missing in heterodyne |
| `KEEP` | missing_method | `optimization.numerical_validation.NumericalValidator.validate_loss` | homodyne method `validate_loss(self, loss_value: Any) -> None` missing in heterodyne |
| `KEEP` | missing_method | `optimization.numerical_validation.NumericalValidator.validate_parameters` | homodyne method `validate_parameters(self, parameters: Any, bounds: tuple[np.ndarray, np.ndarray] | None = None) -> None` missing in heterodyne |
| `KEEP` | missing_class | `optimization.recovery_strategies.RecoveryStrategyApplicator` | homodyne defines this class; heterodyne does not |
| `KEEP` | missing_method | `runtime.utils.system_validator.SystemValidator.calculate_health_score` | homodyne method `calculate_health_score(self) -> int` missing in heterodyne |
| `KEEP` | missing_method | `runtime.utils.system_validator.SystemValidator.generate_report` | homodyne method `generate_report(self) -> str` missing in heterodyne |
| `KEEP` | missing_method | `runtime.utils.system_validator.SystemValidator.log` | homodyne method `log(self, message: str, level: str = 'info') -> None` missing in heterodyne |
| `KEEP` | missing_method | `runtime.utils.system_validator.SystemValidator.run_all_tests` | homodyne method `run_all_tests(self) -> dict[str, ValidationResult]` missing in heterodyne |
| `KEEP` | missing_method | `runtime.utils.system_validator.SystemValidator.run_command` | homodyne method `run_command(self, cmd: list[str], timeout: int = 30) -> tuple[bool, str, str]` missing in heterodyne |
| `KEEP` | missing_method | `runtime.utils.system_validator.SystemValidator.run_quick_tests` | homodyne method `run_quick_tests(self) -> dict[str, ValidationResult]` missing in heterodyne |
| `KEEP` | missing_method | `runtime.utils.system_validator.SystemValidator.test_config_system` | homodyne method `test_config_system(self) -> ValidationResult` missing in heterodyne |
| `KEEP` | missing_method | `runtime.utils.system_validator.SystemValidator.test_data_pipeline` | homodyne method `test_data_pipeline(self) -> ValidationResult` missing in heterodyne |
| `KEEP` | missing_method | `runtime.utils.system_validator.SystemValidator.test_dependency_versions` | homodyne method `test_dependency_versions(self) -> ValidationResult` missing in heterodyne |
| `KEEP` | missing_method | `runtime.utils.system_validator.SystemValidator.test_homodyne_installation` | homodyne method `test_homodyne_installation(self) -> ValidationResult` missing in heterodyne |
| `KEEP` | missing_method | `runtime.utils.system_validator.SystemValidator.test_integration` | homodyne method `test_integration(self) -> ValidationResult` missing in heterodyne |
| `KEEP` | missing_method | `runtime.utils.system_validator.SystemValidator.test_nlsq_integration` | homodyne method `test_nlsq_integration(self) -> ValidationResult` missing in heterodyne |
| `KEEP` | missing_field | `runtime.utils.system_validator.ValidationResult.execution_time` | homodyne dataclass field `execution_time` missing in heterodyne |
| `KEEP` | missing_field | `runtime.utils.system_validator.ValidationResult.warnings` | homodyne dataclass field `warnings` missing in heterodyne |
| `KEEP` | missing_field | `runtime.utils.system_validator.ValidationResult.error_code` | homodyne dataclass field `error_code` missing in heterodyne |
| `KEEP` | missing_method | `utils.async_io.AsyncWriter.shutdown` | homodyne method `shutdown(self) -> None` missing in heterodyne |
| `KEEP` | missing_method | `utils.async_io.AsyncWriter.submit_json` | homodyne method `submit_json(self, path: Path, data: dict[str, Any]) -> None` missing in heterodyne |
| `KEEP` | missing_method | `utils.async_io.AsyncWriter.submit_npz` | homodyne method `submit_npz(self, path: Path, data: dict[str, np.ndarray]) -> None` missing in heterodyne |
| `KEEP` | missing_method | `utils.async_io.AsyncWriter.submit_task` | homodyne method `submit_task(self, fn: Callable[..., None], *args: Any, **kwargs: Any) -> None` missing in heterodyne |
| `KEEP` | missing_method | `utils.async_io.AsyncWriter.wait_all` | homodyne method `wait_all(self, timeout: float = 60.0) -> list[Exception]` missing in heterodyne |
| `KEEP` | missing_method | `utils.logging.AnalysisSummaryLogger.log_summary` | homodyne method `log_summary(self, logger: logging.Logger | logging.LoggerAdapter) -> None` missing in heterodyne |
| `KEEP` | missing_class | `viz.datashader_backend.DatashaderRenderer` | homodyne defines this class; heterodyne does not |

### cli (26)

| Disposition | Kind | Qualname / Path | Detail |
|---|---|---|---|
| `KEEP` | missing_cli_flag | `--cmc-num-shards` | homodyne flag `--cmc-num-shards` (default=None) missing in heterodyne |
| `KEEP` | missing_cli_flag | `--cmc-plot-diagnostics` | homodyne flag `--cmc-plot-diagnostics` (default=None) missing in heterodyne |
| `KEEP` | cli_default_drift | `--config` | default homodyne="Path('./homodyne_config.yaml')" heterodyne=None |
| `KEEP` | missing_cli_flag | `--data-file` | homodyne flag `--data-file` (default=None) missing in heterodyne |
| `KEEP` | missing_cli_flag | `--dense-mass-matrix` | homodyne flag `--dense-mass-matrix` (default=None) missing in heterodyne |
| `KEEP` | missing_cli_flag | `--filter` | homodyne flag `--filter` (default='full') missing in heterodyne |
| `KEEP` | missing_cli_flag | `--force` | homodyne flag `--force` (default=None) missing in heterodyne |
| `KEEP` | missing_cli_flag | `--initial-alpha` | homodyne flag `--initial-alpha` (default=None) missing in heterodyne |
| `KEEP` | missing_cli_flag | `--initial-d-offset` | homodyne flag `--initial-d-offset` (default=None) missing in heterodyne |
| `KEEP` | missing_cli_flag | `--initial-d0` | homodyne flag `--initial-d0` (default=None) missing in heterodyne |
| `KEEP` | missing_cli_flag | `--initial-gamma-dot-offset` | homodyne flag `--initial-gamma-dot-offset` (default=None) missing in heterodyne |
| `KEEP` | missing_cli_flag | `--initial-gamma-dot-t0` | homodyne flag `--initial-gamma-dot-t0` (default=None) missing in heterodyne |
| `KEEP` | missing_cli_flag | `--laminar-flow` | homodyne flag `--laminar-flow` (default=None) missing in heterodyne |
| `KEEP` | missing_cli_flag | `--max-iterations` | homodyne flag `--max-iterations` (default=10000) missing in heterodyne |
| `KEEP` | cli_default_drift | `--mode` | default homodyne=None heterodyne="'full'" |
| `KEEP` | missing_cli_flag | `--n-chains` | homodyne flag `--n-chains` (default=None) missing in heterodyne |
| `KEEP` | missing_cli_flag | `--n-samples` | homodyne flag `--n-samples` (default=None) missing in heterodyne |
| `KEEP` | missing_cli_flag | `--n-warmup` | homodyne flag `--n-warmup` (default=None) missing in heterodyne |
| `KEEP` | missing_cli_flag | `--offset` | homodyne flag `--offset` (default=1.0) missing in heterodyne |
| `KEEP` | cli_default_drift | `--output` | default homodyne=None heterodyne="Path('heterodyne_config.yaml')" |
| `KEEP` | missing_cli_flag | `--output-dir` | homodyne flag `--output-dir` (default=Path('./homodyne_results')) missing in heterodyne |
| `KEEP` | cli_default_drift | `--output-format` | default homodyne="'yaml'" heterodyne="'both'" |
| `KEEP` | cli_default_drift | `--phi-angles` | default homodyne=None heterodyne='None' |
| `KEEP` | missing_cli_flag | `--show` | homodyne flag `--show` (default=None) missing in heterodyne |
| `KEEP` | missing_cli_flag | `--static-mode` | homodyne flag `--static-mode` (default=None) missing in heterodyne |
| `KEEP` | missing_cli_flag | `--tolerance` | homodyne flag `--tolerance` (default=1e-08) missing in heterodyne |

### configs (531)

| Disposition | Kind | Qualname / Path | Detail |
|---|---|---|---|
| `KEEP` | missing_config_key | `config.parameter_registry.ParameterInfo.dtype` | homodyne defines `config.parameter_registry.ParameterInfo.dtype`; heterodyne does not |
| `KEEP` | missing_config_key | `config.parameter_registry.ParameterInfo.lower_bound` | homodyne defines `config.parameter_registry.ParameterInfo.lower_bound`; heterodyne does not |
| `KEEP` | missing_config_key | `config.parameter_registry.ParameterInfo.units` | homodyne defines `config.parameter_registry.ParameterInfo.units`; heterodyne does not |
| `KEEP` | missing_config_key | `config.parameter_registry.ParameterInfo.upper_bound` | homodyne defines `config.parameter_registry.ParameterInfo.upper_bound`; heterodyne does not |
| `KEEP` | missing_config_key | `config.parameter_space.ParameterSpace.model_type` | homodyne defines `config.parameter_space.ParameterSpace.model_type`; heterodyne does not |
| `KEEP` | missing_config_key | `config.parameter_space.ParameterSpace.parameter_names` | homodyne defines `config.parameter_space.ParameterSpace.parameter_names`; heterodyne does not |
| `KEEP` | missing_config_key | `config.parameter_space.ParameterSpace.units` | homodyne defines `config.parameter_space.ParameterSpace.units`; heterodyne does not |
| `KEEP` | missing_config_key | `config.parameter_space.PriorDistribution.alpha` | homodyne defines `config.parameter_space.PriorDistribution.alpha`; heterodyne does not |
| `KEEP` | missing_config_key | `config.parameter_space.PriorDistribution.beta` | homodyne defines `config.parameter_space.PriorDistribution.beta`; heterodyne does not |
| `KEEP` | missing_config_key | `config.parameter_space.PriorDistribution.dist_type` | homodyne defines `config.parameter_space.PriorDistribution.dist_type`; heterodyne does not |
| `KEEP` | missing_config_key | `config.parameter_space.PriorDistribution.max_val` | homodyne defines `config.parameter_space.PriorDistribution.max_val`; heterodyne does not |
| `KEEP` | missing_config_key | `config.parameter_space.PriorDistribution.min_val` | homodyne defines `config.parameter_space.PriorDistribution.min_val`; heterodyne does not |
| `KEEP` | missing_config_key | `config.parameter_space.PriorDistribution.mu` | homodyne defines `config.parameter_space.PriorDistribution.mu`; heterodyne does not |
| `KEEP` | missing_config_key | `config.parameter_space.PriorDistribution.sigma` | homodyne defines `config.parameter_space.PriorDistribution.sigma`; heterodyne does not |
| `KEEP` | missing_config_key | `config.physics_validators.ConstraintRule.condition` | homodyne defines `config.physics_validators.ConstraintRule.condition`; heterodyne does not |
| `KEEP` | missing_config_key | `config.physics_validators.PhysicsViolation.param` | homodyne defines `config.physics_validators.PhysicsViolation.param`; heterodyne does not |
| `KEEP` | missing_config_key | `core.fitting.FitResult.analysis_mode` | homodyne defines `core.fitting.FitResult.analysis_mode`; heterodyne does not |
| `KEEP` | missing_config_key | `core.fitting.FitResult.p_value` | homodyne defines `core.fitting.FitResult.p_value`; heterodyne does not |
| `KEEP` | missing_config_key | `core.fitting.ParameterSpace.D0_bounds` | homodyne defines `core.fitting.ParameterSpace.D0_bounds`; heterodyne does not |
| `KEEP` | missing_config_key | `core.fitting.ParameterSpace.D0_prior` | homodyne defines `core.fitting.ParameterSpace.D0_prior`; heterodyne does not |
| `KEEP` | missing_config_key | `core.fitting.ParameterSpace.D_offset_bounds` | homodyne defines `core.fitting.ParameterSpace.D_offset_bounds`; heterodyne does not |
| `KEEP` | missing_config_key | `core.fitting.ParameterSpace.D_offset_prior` | homodyne defines `core.fitting.ParameterSpace.D_offset_prior`; heterodyne does not |
| `KEEP` | missing_config_key | `core.fitting.ParameterSpace.alpha_bounds` | homodyne defines `core.fitting.ParameterSpace.alpha_bounds`; heterodyne does not |
| `KEEP` | missing_config_key | `core.fitting.ParameterSpace.alpha_prior` | homodyne defines `core.fitting.ParameterSpace.alpha_prior`; heterodyne does not |
| `KEEP` | missing_config_key | `core.fitting.ParameterSpace.beta_bounds` | homodyne defines `core.fitting.ParameterSpace.beta_bounds`; heterodyne does not |
| `KEEP` | missing_config_key | `core.fitting.ParameterSpace.beta_prior` | homodyne defines `core.fitting.ParameterSpace.beta_prior`; heterodyne does not |
| `KEEP` | missing_config_key | `core.fitting.ParameterSpace.gamma_dot_t0_bounds` | homodyne defines `core.fitting.ParameterSpace.gamma_dot_t0_bounds`; heterodyne does not |
| `KEEP` | missing_config_key | `core.fitting.ParameterSpace.gamma_dot_t0_prior` | homodyne defines `core.fitting.ParameterSpace.gamma_dot_t0_prior`; heterodyne does not |
| `KEEP` | missing_config_key | `core.fitting.ParameterSpace.gamma_dot_t_offset_bounds` | homodyne defines `core.fitting.ParameterSpace.gamma_dot_t_offset_bounds`; heterodyne does not |
| `KEEP` | missing_config_key | `core.fitting.ParameterSpace.gamma_dot_t_offset_prior` | homodyne defines `core.fitting.ParameterSpace.gamma_dot_t_offset_prior`; heterodyne does not |
| `KEEP` | missing_config_key | `core.fitting.ParameterSpace.phi0_bounds` | homodyne defines `core.fitting.ParameterSpace.phi0_bounds`; heterodyne does not |
| `KEEP` | missing_config_key | `core.fitting.ParameterSpace.phi0_prior` | homodyne defines `core.fitting.ParameterSpace.phi0_prior`; heterodyne does not |
| `KEEP` | missing_config_key | `core.homodyne_model.runtime_keys` | homodyne defines `core.homodyne_model.runtime_keys`; heterodyne does not |
| `KEEP` | missing_config_key | `core.numpy_gradients.DifferentiationConfig.chunk_size` | homodyne defines `core.numpy_gradients.DifferentiationConfig.chunk_size`; heterodyne does not |
| `KEEP` | missing_config_key | `core.numpy_gradients.DifferentiationConfig.complex_step_threshold` | homodyne defines `core.numpy_gradients.DifferentiationConfig.complex_step_threshold`; heterodyne does not |
| `KEEP` | missing_config_key | `core.numpy_gradients.DifferentiationConfig.max_iterations` | homodyne defines `core.numpy_gradients.DifferentiationConfig.max_iterations`; heterodyne does not |
| `KEEP` | missing_config_key | `core.numpy_gradients.DifferentiationConfig.max_step` | homodyne defines `core.numpy_gradients.DifferentiationConfig.max_step`; heterodyne does not |
| `KEEP` | missing_config_key | `core.numpy_gradients.DifferentiationConfig.min_step` | homodyne defines `core.numpy_gradients.DifferentiationConfig.min_step`; heterodyne does not |
| `KEEP` | missing_config_key | `core.numpy_gradients.DifferentiationConfig.relative_step` | homodyne defines `core.numpy_gradients.DifferentiationConfig.relative_step`; heterodyne does not |
| `KEEP` | missing_config_key | `core.numpy_gradients.GradientResult.computation_time` | homodyne defines `core.numpy_gradients.GradientResult.computation_time`; heterodyne does not |
| `KEEP` | missing_config_key | `core.numpy_gradients.GradientResult.function_calls` | homodyne defines `core.numpy_gradients.GradientResult.function_calls`; heterodyne does not |
| `KEEP` | missing_config_key | `core.numpy_gradients.GradientResult.step_sizes` | homodyne defines `core.numpy_gradients.GradientResult.step_sizes`; heterodyne does not |
| `KEEP` | missing_config_key | `core.numpy_gradients.GradientResult.warnings` | homodyne defines `core.numpy_gradients.GradientResult.warnings`; heterodyne does not |
| `KEEP` | missing_config_key | `core.physics_factors.PhysicsFactors.sinc_prefactor` | homodyne defines `core.physics_factors.PhysicsFactors.sinc_prefactor`; heterodyne does not |
| `KEEP` | missing_config_key | `core.physics_factors.PhysicsFactors.stator_rotor_gap` | homodyne defines `core.physics_factors.PhysicsFactors.stator_rotor_gap`; heterodyne does not |
| `KEEP` | missing_config_key | `core.physics_factors.PhysicsFactors.wavevector_q` | homodyne defines `core.physics_factors.PhysicsFactors.wavevector_q`; heterodyne does not |
| `KEEP` | missing_config_key | `core.physics_factors.PhysicsFactors.wavevector_q_squared_half_dt` | homodyne defines `core.physics_factors.PhysicsFactors.wavevector_q_squared_half_dt`; heterodyne does not |
| `KEEP` | missing_config_key | `data.config.runtime_keys` | homodyne defines `data.config.runtime_keys`; heterodyne does not |
| `KEEP` | missing_config_key | `data.filtering_utils.FilteringResult.errors` | homodyne defines `data.filtering_utils.FilteringResult.errors`; heterodyne does not |
| `KEEP` | missing_config_key | `data.filtering_utils.FilteringResult.fallback_used` | homodyne defines `data.filtering_utils.FilteringResult.fallback_used`; heterodyne does not |
| `KEEP` | missing_config_key | `data.filtering_utils.FilteringResult.filter_statistics` | homodyne defines `data.filtering_utils.FilteringResult.filter_statistics`; heterodyne does not |
| `KEEP` | missing_config_key | `data.filtering_utils.FilteringResult.filters_applied` | homodyne defines `data.filtering_utils.FilteringResult.filters_applied`; heterodyne does not |
| `KEEP` | missing_config_key | `data.filtering_utils.FilteringResult.selected_indices` | homodyne defines `data.filtering_utils.FilteringResult.selected_indices`; heterodyne does not |
| `KEEP` | missing_config_key | `data.filtering_utils.FilteringResult.total_available` | homodyne defines `data.filtering_utils.FilteringResult.total_available`; heterodyne does not |
| `KEEP` | missing_config_key | `data.filtering_utils.FilteringResult.total_selected` | homodyne defines `data.filtering_utils.FilteringResult.total_selected`; heterodyne does not |
| `KEEP` | missing_config_key | `data.filtering_utils.FilteringResult.warnings` | homodyne defines `data.filtering_utils.FilteringResult.warnings`; heterodyne does not |
| `KEEP` | missing_config_key | `data.memory_manager.MemoryPool.allocated_count` | homodyne defines `data.memory_manager.MemoryPool.allocated_count`; heterodyne does not |
| `KEEP` | missing_config_key | `data.memory_manager.MemoryPool.buffer_size` | homodyne defines `data.memory_manager.MemoryPool.buffer_size`; heterodyne does not |
| `KEEP` | missing_config_key | `data.memory_manager.MemoryPool.buffers` | homodyne defines `data.memory_manager.MemoryPool.buffers`; heterodyne does not |
| `KEEP` | missing_config_key | `data.memory_manager.MemoryPool.creation_time` | homodyne defines `data.memory_manager.MemoryPool.creation_time`; heterodyne does not |
| `KEEP` | missing_config_key | `data.memory_manager.MemoryPool.hit_count` | homodyne defines `data.memory_manager.MemoryPool.hit_count`; heterodyne does not |
| `KEEP` | missing_config_key | `data.memory_manager.MemoryPool.last_access_time` | homodyne defines `data.memory_manager.MemoryPool.last_access_time`; heterodyne does not |
| `KEEP` | missing_config_key | `data.memory_manager.MemoryPool.max_buffers` | homodyne defines `data.memory_manager.MemoryPool.max_buffers`; heterodyne does not |
| `KEEP` | missing_config_key | `data.memory_manager.MemoryPool.miss_count` | homodyne defines `data.memory_manager.MemoryPool.miss_count`; heterodyne does not |
| `KEEP` | missing_config_key | `data.memory_manager.MemoryPool.pool_id` | homodyne defines `data.memory_manager.MemoryPool.pool_id`; heterodyne does not |
| `KEEP` | missing_config_key | `data.memory_manager.MemoryStats.active_pools` | homodyne defines `data.memory_manager.MemoryStats.active_pools`; heterodyne does not |
| `KEEP` | missing_config_key | `data.memory_manager.MemoryStats.allocated_pools` | homodyne defines `data.memory_manager.MemoryStats.allocated_pools`; heterodyne does not |
| `KEEP` | missing_config_key | `data.memory_manager.MemoryStats.allocation_latency_ms` | homodyne defines `data.memory_manager.MemoryStats.allocation_latency_ms`; heterodyne does not |
| `KEEP` | missing_config_key | `data.memory_manager.MemoryStats.allocation_rate` | homodyne defines `data.memory_manager.MemoryStats.allocation_rate`; heterodyne does not |
| `KEEP` | missing_config_key | `data.memory_manager.MemoryStats.available_memory_gb` | homodyne defines `data.memory_manager.MemoryStats.available_memory_gb`; heterodyne does not |
| `KEEP` | missing_config_key | `data.memory_manager.MemoryStats.deallocation_rate` | homodyne defines `data.memory_manager.MemoryStats.deallocation_rate`; heterodyne does not |
| `KEEP` | missing_config_key | `data.memory_manager.MemoryStats.fragmentation_ratio` | homodyne defines `data.memory_manager.MemoryStats.fragmentation_ratio`; heterodyne does not |
| `KEEP` | missing_config_key | `data.memory_manager.MemoryStats.gc_collections_per_min` | homodyne defines `data.memory_manager.MemoryStats.gc_collections_per_min`; heterodyne does not |
| `KEEP` | missing_config_key | `data.memory_manager.MemoryStats.memory_pressure` | homodyne defines `data.memory_manager.MemoryStats.memory_pressure`; heterodyne does not |
| `KEEP` | missing_config_key | `data.memory_manager.MemoryStats.memory_throughput_mbps` | homodyne defines `data.memory_manager.MemoryStats.memory_throughput_mbps`; heterodyne does not |
| `KEEP` | missing_config_key | `data.memory_manager.MemoryStats.page_faults_per_sec` | homodyne defines `data.memory_manager.MemoryStats.page_faults_per_sec`; heterodyne does not |
| `KEEP` | missing_config_key | `data.memory_manager.MemoryStats.pool_efficiency` | homodyne defines `data.memory_manager.MemoryStats.pool_efficiency`; heterodyne does not |
| `KEEP` | missing_config_key | `data.memory_manager.MemoryStats.pool_memory_gb` | homodyne defines `data.memory_manager.MemoryStats.pool_memory_gb`; heterodyne does not |
| `KEEP` | missing_config_key | `data.memory_manager.MemoryStats.swap_usage_gb` | homodyne defines `data.memory_manager.MemoryStats.swap_usage_gb`; heterodyne does not |
| `KEEP` | missing_config_key | `data.memory_manager.MemoryStats.total_memory_gb` | homodyne defines `data.memory_manager.MemoryStats.total_memory_gb`; heterodyne does not |
| `KEEP` | missing_config_key | `data.memory_manager.MemoryStats.used_memory_gb` | homodyne defines `data.memory_manager.MemoryStats.used_memory_gb`; heterodyne does not |
| `KEEP` | missing_config_key | `data.optimization.runtime_keys` | homodyne defines `data.optimization.runtime_keys`; heterodyne does not |
| `KEEP` | missing_config_key | `data.performance_engine.ChunkInfo.access_pattern` | homodyne defines `data.performance_engine.ChunkInfo.access_pattern`; heterodyne does not |
| `KEEP` | missing_config_key | `data.performance_engine.ChunkInfo.cache_key` | homodyne defines `data.performance_engine.ChunkInfo.cache_key`; heterodyne does not |
| `KEEP` | missing_config_key | `data.performance_engine.ChunkInfo.complexity_score` | homodyne defines `data.performance_engine.ChunkInfo.complexity_score`; heterodyne does not |
| `KEEP` | missing_config_key | `data.performance_engine.ChunkInfo.dependencies` | homodyne defines `data.performance_engine.ChunkInfo.dependencies`; heterodyne does not |
| `KEEP` | missing_config_key | `data.performance_engine.ChunkInfo.estimated_processing_time` | homodyne defines `data.performance_engine.ChunkInfo.estimated_processing_time`; heterodyne does not |
| `KEEP` | missing_config_key | `data.performance_engine.ChunkInfo.index` | homodyne defines `data.performance_engine.ChunkInfo.index`; heterodyne does not |
| `KEEP` | missing_config_key | `data.performance_engine.ChunkInfo.memory_size_mb` | homodyne defines `data.performance_engine.ChunkInfo.memory_size_mb`; heterodyne does not |
| `KEEP` | missing_config_key | `data.performance_engine.ChunkInfo.priority` | homodyne defines `data.performance_engine.ChunkInfo.priority`; heterodyne does not |
| `KEEP` | missing_config_key | `data.performance_engine.ChunkInfo.size` | homodyne defines `data.performance_engine.ChunkInfo.size`; heterodyne does not |
| `KEEP` | missing_config_key | `data.performance_engine.PerformanceMetrics.bottleneck_type` | homodyne defines `data.performance_engine.PerformanceMetrics.bottleneck_type`; heterodyne does not |
| `KEEP` | missing_config_key | `data.performance_engine.PerformanceMetrics.cache_hit_rate` | homodyne defines `data.performance_engine.PerformanceMetrics.cache_hit_rate`; heterodyne does not |
| `KEEP` | missing_config_key | `data.performance_engine.PerformanceMetrics.chunk_processing_rate` | homodyne defines `data.performance_engine.PerformanceMetrics.chunk_processing_rate`; heterodyne does not |
| `KEEP` | missing_config_key | `data.performance_engine.PerformanceMetrics.cpu_utilization` | homodyne defines `data.performance_engine.PerformanceMetrics.cpu_utilization`; heterodyne does not |
| `KEEP` | missing_config_key | `data.performance_engine.PerformanceMetrics.history_size` | homodyne defines `data.performance_engine.PerformanceMetrics.history_size`; heterodyne does not |
| `KEEP` | missing_config_key | `data.performance_engine.PerformanceMetrics.io_wait_time` | homodyne defines `data.performance_engine.PerformanceMetrics.io_wait_time`; heterodyne does not |
| `KEEP` | missing_config_key | `data.performance_engine.PerformanceMetrics.loading_speed_mbps` | homodyne defines `data.performance_engine.PerformanceMetrics.loading_speed_mbps`; heterodyne does not |
| `KEEP` | missing_config_key | `data.performance_engine.PerformanceMetrics.memory_pressure` | homodyne defines `data.performance_engine.PerformanceMetrics.memory_pressure`; heterodyne does not |
| `KEEP` | missing_config_key | `data.performance_engine.PerformanceMetrics.memory_usage_mb` | homodyne defines `data.performance_engine.PerformanceMetrics.memory_usage_mb`; heterodyne does not |
| `KEEP` | missing_config_key | `data.performance_engine.PerformanceMetrics.parallel_efficiency` | homodyne defines `data.performance_engine.PerformanceMetrics.parallel_efficiency`; heterodyne does not |
| `KEEP` | missing_config_key | `data.preprocessing.PreprocessingProvenance.errors` | homodyne defines `data.preprocessing.PreprocessingProvenance.errors`; heterodyne does not |
| `KEEP` | missing_config_key | `data.preprocessing.PreprocessingProvenance.peak_memory_usage` | homodyne defines `data.preprocessing.PreprocessingProvenance.peak_memory_usage`; heterodyne does not |
| `KEEP` | missing_config_key | `data.preprocessing.PreprocessingProvenance.total_duration` | homodyne defines `data.preprocessing.PreprocessingProvenance.total_duration`; heterodyne does not |
| `KEEP` | missing_config_key | `data.preprocessing.PreprocessingProvenance.transformations` | homodyne defines `data.preprocessing.PreprocessingProvenance.transformations`; heterodyne does not |
| `KEEP` | missing_config_key | `data.preprocessing.PreprocessingProvenance.warnings` | homodyne defines `data.preprocessing.PreprocessingProvenance.warnings`; heterodyne does not |
| `KEEP` | missing_config_key | `data.preprocessing.PreprocessingResult.data` | homodyne defines `data.preprocessing.PreprocessingResult.data`; heterodyne does not |
| `KEEP` | missing_config_key | `data.preprocessing.PreprocessingResult.provenance` | homodyne defines `data.preprocessing.PreprocessingResult.provenance`; heterodyne does not |
| `KEEP` | missing_config_key | `data.preprocessing.PreprocessingResult.stage_results` | homodyne defines `data.preprocessing.PreprocessingResult.stage_results`; heterodyne does not |
| `KEEP` | missing_config_key | `data.preprocessing.PreprocessingResult.success` | homodyne defines `data.preprocessing.PreprocessingResult.success`; heterodyne does not |
| `KEEP` | missing_config_key | `data.preprocessing.TransformationRecord.duration` | homodyne defines `data.preprocessing.TransformationRecord.duration`; heterodyne does not |
| `KEEP` | missing_config_key | `data.preprocessing.TransformationRecord.input_shape` | homodyne defines `data.preprocessing.TransformationRecord.input_shape`; heterodyne does not |
| `KEEP` | missing_config_key | `data.preprocessing.TransformationRecord.memory_usage` | homodyne defines `data.preprocessing.TransformationRecord.memory_usage`; heterodyne does not |
| `KEEP` | missing_config_key | `data.preprocessing.TransformationRecord.output_shape` | homodyne defines `data.preprocessing.TransformationRecord.output_shape`; heterodyne does not |
| `KEEP` | missing_config_key | `data.preprocessing.TransformationRecord.warnings` | homodyne defines `data.preprocessing.TransformationRecord.warnings`; heterodyne does not |
| `KEEP` | missing_config_key | `data.preprocessing.runtime_keys` | homodyne defines `data.preprocessing.runtime_keys`; heterodyne does not |
| `KEEP` | missing_config_key | `data.quality_controller.QualityControlConfig.auto_repair` | homodyne defines `data.quality_controller.QualityControlConfig.auto_repair`; heterodyne does not |
| `KEEP` | missing_config_key | `data.quality_controller.QualityControlConfig.cache_validation_results` | homodyne defines `data.quality_controller.QualityControlConfig.cache_validation_results`; heterodyne does not |
| `KEEP` | missing_config_key | `data.quality_controller.QualityControlConfig.enable_filtering_validation` | homodyne defines `data.quality_controller.QualityControlConfig.enable_filtering_validation`; heterodyne does not |
| `KEEP` | missing_config_key | `data.quality_controller.QualityControlConfig.enable_final_validation` | homodyne defines `data.quality_controller.QualityControlConfig.enable_final_validation`; heterodyne does not |
| `KEEP` | missing_config_key | `data.quality_controller.QualityControlConfig.enable_preprocessing_validation` | homodyne defines `data.quality_controller.QualityControlConfig.enable_preprocessing_validation`; heterodyne does not |
| `KEEP` | missing_config_key | `data.quality_controller.QualityControlConfig.enable_raw_validation` | homodyne defines `data.quality_controller.QualityControlConfig.enable_raw_validation`; heterodyne does not |
| `KEEP` | missing_config_key | `data.quality_controller.QualityControlConfig.enabled` | homodyne defines `data.quality_controller.QualityControlConfig.enabled`; heterodyne does not |
| `KEEP` | missing_config_key | `data.quality_controller.QualityControlConfig.excellent_threshold` | homodyne defines `data.quality_controller.QualityControlConfig.excellent_threshold`; heterodyne does not |
| `KEEP` | missing_config_key | `data.quality_controller.QualityControlConfig.export_detailed_reports` | homodyne defines `data.quality_controller.QualityControlConfig.export_detailed_reports`; heterodyne does not |
| `KEEP` | missing_config_key | `data.quality_controller.QualityControlConfig.generate_reports` | homodyne defines `data.quality_controller.QualityControlConfig.generate_reports`; heterodyne does not |
| `KEEP` | missing_config_key | `data.quality_controller.QualityControlConfig.incremental_validation` | homodyne defines `data.quality_controller.QualityControlConfig.incremental_validation`; heterodyne does not |
| `KEEP` | missing_config_key | `data.quality_controller.QualityControlConfig.parallel_validation` | homodyne defines `data.quality_controller.QualityControlConfig.parallel_validation`; heterodyne does not |
| `KEEP` | missing_config_key | `data.quality_controller.QualityControlConfig.pass_threshold` | homodyne defines `data.quality_controller.QualityControlConfig.pass_threshold`; heterodyne does not |
| `KEEP` | missing_config_key | `data.quality_controller.QualityControlConfig.repair_format_inconsistencies` | homodyne defines `data.quality_controller.QualityControlConfig.repair_format_inconsistencies`; heterodyne does not |
| `KEEP` | missing_config_key | `data.quality_controller.QualityControlConfig.repair_infinite_values` | homodyne defines `data.quality_controller.QualityControlConfig.repair_infinite_values`; heterodyne does not |
| `KEEP` | missing_config_key | `data.quality_controller.QualityControlConfig.repair_nan_values` | homodyne defines `data.quality_controller.QualityControlConfig.repair_nan_values`; heterodyne does not |
| `KEEP` | missing_config_key | `data.quality_controller.QualityControlConfig.repair_negative_correlations` | homodyne defines `data.quality_controller.QualityControlConfig.repair_negative_correlations`; heterodyne does not |
| `KEEP` | missing_config_key | `data.quality_controller.QualityControlConfig.repair_scaling_issues` | homodyne defines `data.quality_controller.QualityControlConfig.repair_scaling_issues`; heterodyne does not |
| `KEEP` | missing_config_key | `data.quality_controller.QualityControlConfig.save_quality_history` | homodyne defines `data.quality_controller.QualityControlConfig.save_quality_history`; heterodyne does not |
| `KEEP` | missing_config_key | `data.quality_controller.QualityControlConfig.validation_level` | homodyne defines `data.quality_controller.QualityControlConfig.validation_level`; heterodyne does not |
| `KEEP` | missing_config_key | `data.quality_controller.QualityControlConfig.warn_threshold` | homodyne defines `data.quality_controller.QualityControlConfig.warn_threshold`; heterodyne does not |
| `KEEP` | missing_config_key | `data.quality_controller.QualityControlResult.data_modified` | homodyne defines `data.quality_controller.QualityControlResult.data_modified`; heterodyne does not |
| `KEEP` | missing_config_key | `data.quality_controller.QualityControlResult.data_shape_after` | homodyne defines `data.quality_controller.QualityControlResult.data_shape_after`; heterodyne does not |
| `KEEP` | missing_config_key | `data.quality_controller.QualityControlResult.data_shape_before` | homodyne defines `data.quality_controller.QualityControlResult.data_shape_before`; heterodyne does not |
| `KEEP` | missing_config_key | `data.quality_controller.QualityControlResult.issues` | homodyne defines `data.quality_controller.QualityControlResult.issues`; heterodyne does not |
| `KEEP` | missing_config_key | `data.quality_controller.QualityControlResult.metrics` | homodyne defines `data.quality_controller.QualityControlResult.metrics`; heterodyne does not |
| `KEEP` | missing_config_key | `data.quality_controller.QualityControlResult.passed` | homodyne defines `data.quality_controller.QualityControlResult.passed`; heterodyne does not |
| `KEEP` | missing_config_key | `data.quality_controller.QualityControlResult.processing_time` | homodyne defines `data.quality_controller.QualityControlResult.processing_time`; heterodyne does not |
| `KEEP` | missing_config_key | `data.quality_controller.QualityControlResult.repairs_applied` | homodyne defines `data.quality_controller.QualityControlResult.repairs_applied`; heterodyne does not |
| `KEEP` | missing_config_key | `data.quality_controller.QualityMetrics.correlation_decay` | homodyne defines `data.quality_controller.QualityMetrics.correlation_decay`; heterodyne does not |
| `KEEP` | missing_config_key | `data.quality_controller.QualityMetrics.correlation_validity` | homodyne defines `data.quality_controller.QualityMetrics.correlation_validity`; heterodyne does not |
| `KEEP` | missing_config_key | `data.quality_controller.QualityMetrics.data_range_valid` | homodyne defines `data.quality_controller.QualityMetrics.data_range_valid`; heterodyne does not |
| `KEEP` | missing_config_key | `data.quality_controller.QualityMetrics.filtering_efficiency` | homodyne defines `data.quality_controller.QualityMetrics.filtering_efficiency`; heterodyne does not |
| `KEEP` | missing_config_key | `data.quality_controller.QualityMetrics.finite_fraction` | homodyne defines `data.quality_controller.QualityMetrics.finite_fraction`; heterodyne does not |
| `KEEP` | missing_config_key | `data.quality_controller.QualityMetrics.issues_detected` | homodyne defines `data.quality_controller.QualityMetrics.issues_detected`; heterodyne does not |
| `KEEP` | missing_config_key | `data.quality_controller.QualityMetrics.issues_repaired` | homodyne defines `data.quality_controller.QualityMetrics.issues_repaired`; heterodyne does not |
| `KEEP` | missing_config_key | `data.quality_controller.QualityMetrics.overall_score` | homodyne defines `data.quality_controller.QualityMetrics.overall_score`; heterodyne does not |
| `KEEP` | missing_config_key | `data.quality_controller.QualityMetrics.preprocessing_success` | homodyne defines `data.quality_controller.QualityMetrics.preprocessing_success`; heterodyne does not |
| `KEEP` | missing_config_key | `data.quality_controller.QualityMetrics.q_range_validity` | homodyne defines `data.quality_controller.QualityMetrics.q_range_validity`; heterodyne does not |
| `KEEP` | missing_config_key | `data.quality_controller.QualityMetrics.repair_success_rate` | homodyne defines `data.quality_controller.QualityMetrics.repair_success_rate`; heterodyne does not |
| `KEEP` | missing_config_key | `data.quality_controller.QualityMetrics.shape_consistency` | homodyne defines `data.quality_controller.QualityMetrics.shape_consistency`; heterodyne does not |
| `KEEP` | missing_config_key | `data.quality_controller.QualityMetrics.signal_to_noise` | homodyne defines `data.quality_controller.QualityMetrics.signal_to_noise`; heterodyne does not |
| `KEEP` | missing_config_key | `data.quality_controller.QualityMetrics.symmetry_score` | homodyne defines `data.quality_controller.QualityMetrics.symmetry_score`; heterodyne does not |
| `KEEP` | missing_config_key | `data.quality_controller.QualityMetrics.time_consistency` | homodyne defines `data.quality_controller.QualityMetrics.time_consistency`; heterodyne does not |
| `KEEP` | missing_config_key | `data.quality_controller.QualityMetrics.transformation_fidelity` | homodyne defines `data.quality_controller.QualityMetrics.transformation_fidelity`; heterodyne does not |
| `KEEP` | missing_config_key | `data.quality_controller.runtime_keys` | homodyne defines `data.quality_controller.runtime_keys`; heterodyne does not |
| `KEEP` | missing_config_key | `data.types.DatasetInfo.category` | homodyne defines `data.types.DatasetInfo.category`; heterodyne does not |
| `KEEP` | missing_config_key | `data.types.DatasetInfo.compression_ratio` | homodyne defines `data.types.DatasetInfo.compression_ratio`; heterodyne does not |
| `KEEP` | missing_config_key | `data.types.DatasetInfo.memory_usage_mb` | homodyne defines `data.types.DatasetInfo.memory_usage_mb`; heterodyne does not |
| `KEEP` | missing_config_key | `data.types.DatasetInfo.recommended_batch_size` | homodyne defines `data.types.DatasetInfo.recommended_batch_size`; heterodyne does not |
| `KEEP` | missing_config_key | `data.types.DatasetInfo.recommended_chunk_size` | homodyne defines `data.types.DatasetInfo.recommended_chunk_size`; heterodyne does not |
| `KEEP` | missing_config_key | `data.types.DatasetInfo.size` | homodyne defines `data.types.DatasetInfo.size`; heterodyne does not |
| `KEEP` | missing_config_key | `data.types.DatasetInfo.use_progressive_loading` | homodyne defines `data.types.DatasetInfo.use_progressive_loading`; heterodyne does not |
| `KEEP` | missing_config_key | `data.types.ProcessingStrategy.batch_size` | homodyne defines `data.types.ProcessingStrategy.batch_size`; heterodyne does not |
| `KEEP` | missing_config_key | `data.types.ProcessingStrategy.chunk_size` | homodyne defines `data.types.ProcessingStrategy.chunk_size`; heterodyne does not |
| `KEEP` | missing_config_key | `data.types.ProcessingStrategy.jax_config` | homodyne defines `data.types.ProcessingStrategy.jax_config`; heterodyne does not |
| `KEEP` | missing_config_key | `data.types.ProcessingStrategy.memory_limit_mb` | homodyne defines `data.types.ProcessingStrategy.memory_limit_mb`; heterodyne does not |
| `KEEP` | missing_config_key | `data.types.ProcessingStrategy.parallel_workers` | homodyne defines `data.types.ProcessingStrategy.parallel_workers`; heterodyne does not |
| `KEEP` | missing_config_key | `data.types.ProcessingStrategy.use_caching` | homodyne defines `data.types.ProcessingStrategy.use_caching`; heterodyne does not |
| `KEEP` | missing_config_key | `data.types.ProcessingStrategy.use_compression` | homodyne defines `data.types.ProcessingStrategy.use_compression`; heterodyne does not |
| `KEEP` | missing_config_key | `data.validation.DataQualityReport.data_statistics` | homodyne defines `data.validation.DataQualityReport.data_statistics`; heterodyne does not |
| `KEEP` | missing_config_key | `data.validation.DataQualityReport.info` | homodyne defines `data.validation.DataQualityReport.info`; heterodyne does not |
| `KEEP` | missing_config_key | `data.validation.DataQualityReport.physics_checks` | homodyne defines `data.validation.DataQualityReport.physics_checks`; heterodyne does not |
| `KEEP` | missing_config_key | `data.validation.DataQualityReport.quality_score` | homodyne defines `data.validation.DataQualityReport.quality_score`; heterodyne does not |
| `KEEP` | missing_config_key | `data.validation.DataQualityReport.total_issues` | homodyne defines `data.validation.DataQualityReport.total_issues`; heterodyne does not |
| `KEEP` | missing_config_key | `data.validation.DataQualityReport.validation_level` | homodyne defines `data.validation.DataQualityReport.validation_level`; heterodyne does not |
| `KEEP` | missing_config_key | `data.validation.IncrementalValidationCache.component_hashes` | homodyne defines `data.validation.IncrementalValidationCache.component_hashes`; heterodyne does not |
| `KEEP` | missing_config_key | `data.validation.IncrementalValidationCache.data_hash` | homodyne defines `data.validation.IncrementalValidationCache.data_hash`; heterodyne does not |
| `KEEP` | missing_config_key | `data.validation.IncrementalValidationCache.report` | homodyne defines `data.validation.IncrementalValidationCache.report`; heterodyne does not |
| `KEEP` | missing_config_key | `data.validation.IncrementalValidationCache.timestamp` | homodyne defines `data.validation.IncrementalValidationCache.timestamp`; heterodyne does not |
| `KEEP` | missing_config_key | `data.validation.IncrementalValidationCache.validation_level` | homodyne defines `data.validation.IncrementalValidationCache.validation_level`; heterodyne does not |
| `KEEP` | missing_config_key | `data.validation.runtime_keys` | homodyne defines `data.validation.runtime_keys`; heterodyne does not |
| `KEEP` | missing_config_key | `device.config.HardwareConfig.cores_per_node` | homodyne defines `device.config.HardwareConfig.cores_per_node`; heterodyne does not |
| `KEEP` | missing_config_key | `device.config.HardwareConfig.max_parallel_shards` | homodyne defines `device.config.HardwareConfig.max_parallel_shards`; heterodyne does not |
| `KEEP` | missing_config_key | `device.config.HardwareConfig.memory_per_device_gb` | homodyne defines `device.config.HardwareConfig.memory_per_device_gb`; heterodyne does not |
| `KEEP` | missing_config_key | `device.config.HardwareConfig.num_devices` | homodyne defines `device.config.HardwareConfig.num_devices`; heterodyne does not |
| `KEEP` | missing_config_key | `device.config.HardwareConfig.num_nodes` | homodyne defines `device.config.HardwareConfig.num_nodes`; heterodyne does not |
| `KEEP` | missing_config_key | `device.config.HardwareConfig.platform` | homodyne defines `device.config.HardwareConfig.platform`; heterodyne does not |
| `KEEP` | missing_config_key | `device.config.HardwareConfig.total_memory_gb` | homodyne defines `device.config.HardwareConfig.total_memory_gb`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.cmc.data_prep.PreparedData.data` | homodyne defines `optimization.cmc.data_prep.PreparedData.data`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.cmc.data_prep.PreparedData.n_phi` | homodyne defines `optimization.cmc.data_prep.PreparedData.n_phi`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.cmc.data_prep.PreparedData.n_total` | homodyne defines `optimization.cmc.data_prep.PreparedData.n_total`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.cmc.data_prep.PreparedData.noise_scale` | homodyne defines `optimization.cmc.data_prep.PreparedData.noise_scale`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.cmc.data_prep.PreparedData.phi` | homodyne defines `optimization.cmc.data_prep.PreparedData.phi`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.cmc.data_prep.PreparedData.phi_indices` | homodyne defines `optimization.cmc.data_prep.PreparedData.phi_indices`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.cmc.data_prep.PreparedData.phi_unique` | homodyne defines `optimization.cmc.data_prep.PreparedData.phi_unique`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.cmc.data_prep.PreparedData.t1` | homodyne defines `optimization.cmc.data_prep.PreparedData.t1`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.cmc.data_prep.PreparedData.t2` | homodyne defines `optimization.cmc.data_prep.PreparedData.t2`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.cmc.diagnostics.BimodalResult.relative_separation` | homodyne defines `optimization.cmc.diagnostics.BimodalResult.relative_separation`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.cmc.diagnostics.BimodalResult.separation` | homodyne defines `optimization.cmc.diagnostics.BimodalResult.separation`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.cmc.diagnostics.BimodalResult.stds` | homodyne defines `optimization.cmc.diagnostics.BimodalResult.stds`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.cmc.diagnostics.ModeCluster.samples` | homodyne defines `optimization.cmc.diagnostics.ModeCluster.samples`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.cmc.reparameterization.ReparamConfig.enable_gamma_ref` | homodyne defines `optimization.cmc.reparameterization.ReparamConfig.enable_gamma_ref`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.cmc.results.CMCResult.analysis_mode` | homodyne defines `optimization.cmc.results.CMCResult.analysis_mode`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.cmc.results.CMCResult.covariance` | homodyne defines `optimization.cmc.results.CMCResult.covariance`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.cmc.results.CMCResult.device_info` | homodyne defines `optimization.cmc.results.CMCResult.device_info`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.cmc.results.CMCResult.execution_time` | homodyne defines `optimization.cmc.results.CMCResult.execution_time`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.cmc.results.CMCResult.inference_data` | homodyne defines `optimization.cmc.results.CMCResult.inference_data`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.cmc.results.CMCResult.mean_params` | homodyne defines `optimization.cmc.results.CMCResult.mean_params`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.cmc.results.CMCResult.n_chains` | homodyne defines `optimization.cmc.results.CMCResult.n_chains`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.cmc.results.CMCResult.n_samples` | homodyne defines `optimization.cmc.results.CMCResult.n_samples`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.cmc.results.CMCResult.n_warmup` | homodyne defines `optimization.cmc.results.CMCResult.n_warmup`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.cmc.results.CMCResult.param_names` | homodyne defines `optimization.cmc.results.CMCResult.param_names`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.cmc.results.CMCResult.parameters` | homodyne defines `optimization.cmc.results.CMCResult.parameters`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.cmc.results.CMCResult.recovery_actions` | homodyne defines `optimization.cmc.results.CMCResult.recovery_actions`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.cmc.results.CMCResult.reduced_chi_squared` | homodyne defines `optimization.cmc.results.CMCResult.reduced_chi_squared`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.cmc.results.CMCResult.std_params` | homodyne defines `optimization.cmc.results.CMCResult.std_params`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.cmc.results.CMCResult.uncertainties` | homodyne defines `optimization.cmc.results.CMCResult.uncertainties`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.cmc.sampler.MCMCSamples.bimodal_consensus` | homodyne defines `optimization.cmc.sampler.MCMCSamples.bimodal_consensus`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.cmc.sampler.MCMCSamples.extra_fields` | homodyne defines `optimization.cmc.sampler.MCMCSamples.extra_fields`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.cmc.sampler.MCMCSamples.n_chains` | homodyne defines `optimization.cmc.sampler.MCMCSamples.n_chains`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.cmc.sampler.MCMCSamples.n_samples` | homodyne defines `optimization.cmc.sampler.MCMCSamples.n_samples`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.cmc.sampler.MCMCSamples.num_shards` | homodyne defines `optimization.cmc.sampler.MCMCSamples.num_shards`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.cmc.sampler.MCMCSamples.param_names` | homodyne defines `optimization.cmc.sampler.MCMCSamples.param_names`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.cmc.sampler.MCMCSamples.samples` | homodyne defines `optimization.cmc.sampler.MCMCSamples.samples`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.cmc.sampler.MCMCSamples.shard_adapted_n_warmup` | homodyne defines `optimization.cmc.sampler.MCMCSamples.shard_adapted_n_warmup`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.cmc.sampler.SamplingPlan.n_chains` | homodyne defines `optimization.cmc.sampler.SamplingPlan.n_chains`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.cmc.sampler.SamplingPlan.n_params` | homodyne defines `optimization.cmc.sampler.SamplingPlan.n_params`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.cmc.sampler.SamplingPlan.n_samples` | homodyne defines `optimization.cmc.sampler.SamplingPlan.n_samples`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.cmc.sampler.SamplingPlan.n_warmup` | homodyne defines `optimization.cmc.sampler.SamplingPlan.n_warmup`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.cmc.sampler.SamplingPlan.shard_size` | homodyne defines `optimization.cmc.sampler.SamplingPlan.shard_size`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.cmc.sampler.SamplingPlan.was_adapted` | homodyne defines `optimization.cmc.sampler.SamplingPlan.was_adapted`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.cmc.sampler.SamplingStats.accept_prob` | homodyne defines `optimization.cmc.sampler.SamplingStats.accept_prob`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.cmc.sampler.SamplingStats.inverse_mass_matrix_summary` | homodyne defines `optimization.cmc.sampler.SamplingStats.inverse_mass_matrix_summary`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.cmc.sampler.SamplingStats.num_divergent` | homodyne defines `optimization.cmc.sampler.SamplingStats.num_divergent`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.cmc.sampler.SamplingStats.plan` | homodyne defines `optimization.cmc.sampler.SamplingStats.plan`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.cmc.sampler.SamplingStats.sampling_time` | homodyne defines `optimization.cmc.sampler.SamplingStats.sampling_time`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.cmc.sampler.SamplingStats.step_size` | homodyne defines `optimization.cmc.sampler.SamplingStats.step_size`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.cmc.sampler.SamplingStats.step_size_max` | homodyne defines `optimization.cmc.sampler.SamplingStats.step_size_max`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.cmc.sampler.SamplingStats.step_size_min` | homodyne defines `optimization.cmc.sampler.SamplingStats.step_size_min`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.cmc.sampler.SamplingStats.total_time` | homodyne defines `optimization.cmc.sampler.SamplingStats.total_time`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.cmc.sampler.SamplingStats.tree_depth` | homodyne defines `optimization.cmc.sampler.SamplingStats.tree_depth`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.cmc.sampler.SamplingStats.warmup_time` | homodyne defines `optimization.cmc.sampler.SamplingStats.warmup_time`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.adaptive_regularization.AdaptiveRegularizationConfig.auto_tune_lambda` | homodyne defines `optimization.nlsq.adaptive_regularization.AdaptiveRegularizationConfig.auto_tune_lambda`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.adaptive_regularization.AdaptiveRegularizationConfig.enable` | homodyne defines `optimization.nlsq.adaptive_regularization.AdaptiveRegularizationConfig.enable`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.adaptive_regularization.AdaptiveRegularizationConfig.group_indices` | homodyne defines `optimization.nlsq.adaptive_regularization.AdaptiveRegularizationConfig.group_indices`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.adaptive_regularization.AdaptiveRegularizationConfig.lambda_base` | homodyne defines `optimization.nlsq.adaptive_regularization.AdaptiveRegularizationConfig.lambda_base`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.adaptive_regularization.AdaptiveRegularizationConfig.max_cv` | homodyne defines `optimization.nlsq.adaptive_regularization.AdaptiveRegularizationConfig.max_cv`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.adaptive_regularization.AdaptiveRegularizationConfig.mode` | homodyne defines `optimization.nlsq.adaptive_regularization.AdaptiveRegularizationConfig.mode`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.adaptive_regularization.AdaptiveRegularizationConfig.target_contribution` | homodyne defines `optimization.nlsq.adaptive_regularization.AdaptiveRegularizationConfig.target_contribution`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.adaptive_regularization.AdaptiveRegularizationConfig.target_cv` | homodyne defines `optimization.nlsq.adaptive_regularization.AdaptiveRegularizationConfig.target_cv`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.anti_degeneracy_controller.AntiDegeneracyConfig.shear_weighting_alpha` | homodyne defines `optimization.nlsq.anti_degeneracy_controller.AntiDegeneracyConfig.shear_weighting_alpha`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.anti_degeneracy_controller.AntiDegeneracyConfig.shear_weighting_enable` | homodyne defines `optimization.nlsq.anti_degeneracy_controller.AntiDegeneracyConfig.shear_weighting_enable`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.anti_degeneracy_controller.AntiDegeneracyConfig.shear_weighting_min_weight` | homodyne defines `optimization.nlsq.anti_degeneracy_controller.AntiDegeneracyConfig.shear_weighting_min_weight`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.anti_degeneracy_controller.AntiDegeneracyConfig.shear_weighting_normalize` | homodyne defines `optimization.nlsq.anti_degeneracy_controller.AntiDegeneracyConfig.shear_weighting_normalize`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.anti_degeneracy_controller.AntiDegeneracyConfig.shear_weighting_update_frequency` | homodyne defines `optimization.nlsq.anti_degeneracy_controller.AntiDegeneracyConfig.shear_weighting_update_frequency`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.anti_degeneracy_controller.AntiDegeneracyController.hierarchical` | homodyne defines `optimization.nlsq.anti_degeneracy_controller.AntiDegeneracyController.hierarchical`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.anti_degeneracy_controller.AntiDegeneracyController.mapper` | homodyne defines `optimization.nlsq.anti_degeneracy_controller.AntiDegeneracyController.mapper`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.anti_degeneracy_controller.AntiDegeneracyController.shear_weighter` | homodyne defines `optimization.nlsq.anti_degeneracy_controller.AntiDegeneracyController.shear_weighter`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.cmaes_wrapper.CMAESResult.chi_squared` | homodyne defines `optimization.nlsq.cmaes_wrapper.CMAESResult.chi_squared`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.cmaes_wrapper.CMAESResult.covariance` | homodyne defines `optimization.nlsq.cmaes_wrapper.CMAESResult.covariance`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.cmaes_wrapper.CMAESResult.diagnostics` | homodyne defines `optimization.nlsq.cmaes_wrapper.CMAESResult.diagnostics`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.cmaes_wrapper.CMAESResult.message` | homodyne defines `optimization.nlsq.cmaes_wrapper.CMAESResult.message`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.cmaes_wrapper.CMAESResult.method_used` | homodyne defines `optimization.nlsq.cmaes_wrapper.CMAESResult.method_used`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.cmaes_wrapper.CMAESResult.nlsq_refined` | homodyne defines `optimization.nlsq.cmaes_wrapper.CMAESResult.nlsq_refined`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.cmaes_wrapper.CMAESResult.parameters` | homodyne defines `optimization.nlsq.cmaes_wrapper.CMAESResult.parameters`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.cmaes_wrapper.CMAESResult.success` | homodyne defines `optimization.nlsq.cmaes_wrapper.CMAESResult.success`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.cmaes_wrapper.CMAESWrapperConfig.auto_memory` | homodyne defines `optimization.nlsq.cmaes_wrapper.CMAESWrapperConfig.auto_memory`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.cmaes_wrapper.CMAESWrapperConfig.data_chunk_size` | homodyne defines `optimization.nlsq.cmaes_wrapper.CMAESWrapperConfig.data_chunk_size`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.cmaes_wrapper.CMAESWrapperConfig.max_generations` | homodyne defines `optimization.nlsq.cmaes_wrapper.CMAESWrapperConfig.max_generations`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.cmaes_wrapper.CMAESWrapperConfig.max_restarts` | homodyne defines `optimization.nlsq.cmaes_wrapper.CMAESWrapperConfig.max_restarts`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.cmaes_wrapper.CMAESWrapperConfig.memory_limit_gb` | homodyne defines `optimization.nlsq.cmaes_wrapper.CMAESWrapperConfig.memory_limit_gb`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.cmaes_wrapper.CMAESWrapperConfig.normalization_epsilon` | homodyne defines `optimization.nlsq.cmaes_wrapper.CMAESWrapperConfig.normalization_epsilon`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.cmaes_wrapper.CMAESWrapperConfig.normalize` | homodyne defines `optimization.nlsq.cmaes_wrapper.CMAESWrapperConfig.normalize`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.cmaes_wrapper.CMAESWrapperConfig.popsize` | homodyne defines `optimization.nlsq.cmaes_wrapper.CMAESWrapperConfig.popsize`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.cmaes_wrapper.CMAESWrapperConfig.population_batch_size` | homodyne defines `optimization.nlsq.cmaes_wrapper.CMAESWrapperConfig.population_batch_size`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.cmaes_wrapper.CMAESWrapperConfig.preset` | homodyne defines `optimization.nlsq.cmaes_wrapper.CMAESWrapperConfig.preset`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.cmaes_wrapper.CMAESWrapperConfig.refine_with_nlsq` | homodyne defines `optimization.nlsq.cmaes_wrapper.CMAESWrapperConfig.refine_with_nlsq`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.cmaes_wrapper.CMAESWrapperConfig.refinement_ftol` | homodyne defines `optimization.nlsq.cmaes_wrapper.CMAESWrapperConfig.refinement_ftol`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.cmaes_wrapper.CMAESWrapperConfig.refinement_gtol` | homodyne defines `optimization.nlsq.cmaes_wrapper.CMAESWrapperConfig.refinement_gtol`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.cmaes_wrapper.CMAESWrapperConfig.refinement_loss` | homodyne defines `optimization.nlsq.cmaes_wrapper.CMAESWrapperConfig.refinement_loss`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.cmaes_wrapper.CMAESWrapperConfig.refinement_max_nfev` | homodyne defines `optimization.nlsq.cmaes_wrapper.CMAESWrapperConfig.refinement_max_nfev`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.cmaes_wrapper.CMAESWrapperConfig.refinement_workflow` | homodyne defines `optimization.nlsq.cmaes_wrapper.CMAESWrapperConfig.refinement_workflow`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.cmaes_wrapper.CMAESWrapperConfig.refinement_xtol` | homodyne defines `optimization.nlsq.cmaes_wrapper.CMAESWrapperConfig.refinement_xtol`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.cmaes_wrapper.CMAESWrapperConfig.restart_strategy` | homodyne defines `optimization.nlsq.cmaes_wrapper.CMAESWrapperConfig.restart_strategy`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.cmaes_wrapper.CMAESWrapperConfig.sigma` | homodyne defines `optimization.nlsq.cmaes_wrapper.CMAESWrapperConfig.sigma`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.cmaes_wrapper.CMAESWrapperConfig.sigma_warmstart` | homodyne defines `optimization.nlsq.cmaes_wrapper.CMAESWrapperConfig.sigma_warmstart`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.cmaes_wrapper.CMAESWrapperConfig.tol_fun` | homodyne defines `optimization.nlsq.cmaes_wrapper.CMAESWrapperConfig.tol_fun`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.cmaes_wrapper.CMAESWrapperConfig.tol_x` | homodyne defines `optimization.nlsq.cmaes_wrapper.CMAESWrapperConfig.tol_x`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.config.HybridRecoveryConfig.log_retries` | homodyne defines `optimization.nlsq.config.HybridRecoveryConfig.log_retries`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.config.NLSQConfig.cmaes_auto_select` | homodyne defines `optimization.nlsq.config.NLSQConfig.cmaes_auto_select`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.config.NLSQConfig.cmaes_data_chunk_size` | homodyne defines `optimization.nlsq.config.NLSQConfig.cmaes_data_chunk_size`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.config.NLSQConfig.cmaes_max_generations` | homodyne defines `optimization.nlsq.config.NLSQConfig.cmaes_max_generations`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.config.NLSQConfig.cmaes_memory_limit_gb` | homodyne defines `optimization.nlsq.config.NLSQConfig.cmaes_memory_limit_gb`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.config.NLSQConfig.cmaes_normalization_epsilon` | homodyne defines `optimization.nlsq.config.NLSQConfig.cmaes_normalization_epsilon`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.config.NLSQConfig.cmaes_normalize` | homodyne defines `optimization.nlsq.config.NLSQConfig.cmaes_normalize`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.config.NLSQConfig.cmaes_popsize` | homodyne defines `optimization.nlsq.config.NLSQConfig.cmaes_popsize`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.config.NLSQConfig.cmaes_population_batch_size` | homodyne defines `optimization.nlsq.config.NLSQConfig.cmaes_population_batch_size`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.config.NLSQConfig.cmaes_preset` | homodyne defines `optimization.nlsq.config.NLSQConfig.cmaes_preset`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.config.NLSQConfig.cmaes_refine_with_nlsq` | homodyne defines `optimization.nlsq.config.NLSQConfig.cmaes_refine_with_nlsq`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.config.NLSQConfig.cmaes_refinement_ftol` | homodyne defines `optimization.nlsq.config.NLSQConfig.cmaes_refinement_ftol`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.config.NLSQConfig.cmaes_refinement_gtol` | homodyne defines `optimization.nlsq.config.NLSQConfig.cmaes_refinement_gtol`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.config.NLSQConfig.cmaes_refinement_loss` | homodyne defines `optimization.nlsq.config.NLSQConfig.cmaes_refinement_loss`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.config.NLSQConfig.cmaes_refinement_max_nfev` | homodyne defines `optimization.nlsq.config.NLSQConfig.cmaes_refinement_max_nfev`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.config.NLSQConfig.cmaes_refinement_workflow` | homodyne defines `optimization.nlsq.config.NLSQConfig.cmaes_refinement_workflow`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.config.NLSQConfig.cmaes_refinement_xtol` | homodyne defines `optimization.nlsq.config.NLSQConfig.cmaes_refinement_xtol`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.config.NLSQConfig.cmaes_scale_threshold` | homodyne defines `optimization.nlsq.config.NLSQConfig.cmaes_scale_threshold`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.config.NLSQConfig.cmaes_sigma` | homodyne defines `optimization.nlsq.config.NLSQConfig.cmaes_sigma`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.config.NLSQConfig.cmaes_sigma_warmstart` | homodyne defines `optimization.nlsq.config.NLSQConfig.cmaes_sigma_warmstart`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.config.NLSQConfig.cmaes_tol_fun` | homodyne defines `optimization.nlsq.config.NLSQConfig.cmaes_tol_fun`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.config.NLSQConfig.cmaes_tol_x` | homodyne defines `optimization.nlsq.config.NLSQConfig.cmaes_tol_x`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.config.NLSQConfig.enable_hybrid_streaming` | homodyne defines `optimization.nlsq.config.NLSQConfig.enable_hybrid_streaming`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.config.NLSQConfig.enable_multi_start` | homodyne defines `optimization.nlsq.config.NLSQConfig.enable_multi_start`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.config.NLSQConfig.enable_progress_bar` | homodyne defines `optimization.nlsq.config.NLSQConfig.enable_progress_bar`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.config.NLSQConfig.enable_quality_validation` | homodyne defines `optimization.nlsq.config.NLSQConfig.enable_quality_validation`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.config.NLSQConfig.gradient_collapse_response` | homodyne defines `optimization.nlsq.config.NLSQConfig.gradient_collapse_response`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.config.NLSQConfig.hierarchical_per_angle_max_iterations` | homodyne defines `optimization.nlsq.config.NLSQConfig.hierarchical_per_angle_max_iterations`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.config.NLSQConfig.hierarchical_physical_max_iterations` | homodyne defines `optimization.nlsq.config.NLSQConfig.hierarchical_physical_max_iterations`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.config.NLSQConfig.hybrid_checkpoint_frequency` | homodyne defines `optimization.nlsq.config.NLSQConfig.hybrid_checkpoint_frequency`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.config.NLSQConfig.hybrid_chunk_size` | homodyne defines `optimization.nlsq.config.NLSQConfig.hybrid_chunk_size`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.config.NLSQConfig.hybrid_cost_increase_tolerance` | homodyne defines `optimization.nlsq.config.NLSQConfig.hybrid_cost_increase_tolerance`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.config.NLSQConfig.hybrid_enable_adaptive_warmup_lr` | homodyne defines `optimization.nlsq.config.NLSQConfig.hybrid_enable_adaptive_warmup_lr`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.config.NLSQConfig.hybrid_enable_checkpoints` | homodyne defines `optimization.nlsq.config.NLSQConfig.hybrid_enable_checkpoints`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.config.NLSQConfig.hybrid_enable_cost_guard` | homodyne defines `optimization.nlsq.config.NLSQConfig.hybrid_enable_cost_guard`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.config.NLSQConfig.hybrid_enable_step_clipping` | homodyne defines `optimization.nlsq.config.NLSQConfig.hybrid_enable_step_clipping`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.config.NLSQConfig.hybrid_enable_warm_start_detection` | homodyne defines `optimization.nlsq.config.NLSQConfig.hybrid_enable_warm_start_detection`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.config.NLSQConfig.hybrid_gauss_newton_max_iterations` | homodyne defines `optimization.nlsq.config.NLSQConfig.hybrid_gauss_newton_max_iterations`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.config.NLSQConfig.hybrid_gauss_newton_tol` | homodyne defines `optimization.nlsq.config.NLSQConfig.hybrid_gauss_newton_tol`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.config.NLSQConfig.hybrid_max_warmup_iterations` | homodyne defines `optimization.nlsq.config.NLSQConfig.hybrid_max_warmup_iterations`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.config.NLSQConfig.hybrid_max_warmup_step_size` | homodyne defines `optimization.nlsq.config.NLSQConfig.hybrid_max_warmup_step_size`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.config.NLSQConfig.hybrid_normalization_strategy` | homodyne defines `optimization.nlsq.config.NLSQConfig.hybrid_normalization_strategy`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.config.NLSQConfig.hybrid_normalize` | homodyne defines `optimization.nlsq.config.NLSQConfig.hybrid_normalize`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.config.NLSQConfig.hybrid_regularization_factor` | homodyne defines `optimization.nlsq.config.NLSQConfig.hybrid_regularization_factor`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.config.NLSQConfig.hybrid_trust_region_initial` | homodyne defines `optimization.nlsq.config.NLSQConfig.hybrid_trust_region_initial`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.config.NLSQConfig.hybrid_validate_numerics` | homodyne defines `optimization.nlsq.config.NLSQConfig.hybrid_validate_numerics`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.config.NLSQConfig.hybrid_warm_start_threshold` | homodyne defines `optimization.nlsq.config.NLSQConfig.hybrid_warm_start_threshold`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.config.NLSQConfig.hybrid_warmup_iterations` | homodyne defines `optimization.nlsq.config.NLSQConfig.hybrid_warmup_iterations`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.config.NLSQConfig.hybrid_warmup_learning_rate` | homodyne defines `optimization.nlsq.config.NLSQConfig.hybrid_warmup_learning_rate`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.config.NLSQConfig.hybrid_warmup_lr_careful` | homodyne defines `optimization.nlsq.config.NLSQConfig.hybrid_warmup_lr_careful`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.config.NLSQConfig.hybrid_warmup_lr_refinement` | homodyne defines `optimization.nlsq.config.NLSQConfig.hybrid_warmup_lr_refinement`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.config.NLSQConfig.log_iteration_interval` | homodyne defines `optimization.nlsq.config.NLSQConfig.log_iteration_interval`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.config.NLSQConfig.multi_start_degeneracy_threshold` | homodyne defines `optimization.nlsq.config.NLSQConfig.multi_start_degeneracy_threshold`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.config.NLSQConfig.multi_start_n_starts` | homodyne defines `optimization.nlsq.config.NLSQConfig.multi_start_n_starts`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.config.NLSQConfig.multi_start_n_workers` | homodyne defines `optimization.nlsq.config.NLSQConfig.multi_start_n_workers`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.config.NLSQConfig.multi_start_refine_top_k` | homodyne defines `optimization.nlsq.config.NLSQConfig.multi_start_refine_top_k`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.config.NLSQConfig.multi_start_refinement_ftol` | homodyne defines `optimization.nlsq.config.NLSQConfig.multi_start_refinement_ftol`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.config.NLSQConfig.multi_start_sampling_strategy` | homodyne defines `optimization.nlsq.config.NLSQConfig.multi_start_sampling_strategy`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.config.NLSQConfig.multi_start_screen_keep_fraction` | homodyne defines `optimization.nlsq.config.NLSQConfig.multi_start_screen_keep_fraction`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.config.NLSQConfig.multi_start_seed` | homodyne defines `optimization.nlsq.config.NLSQConfig.multi_start_seed`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.config.NLSQConfig.multi_start_use_screening` | homodyne defines `optimization.nlsq.config.NLSQConfig.multi_start_use_screening`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.config.NLSQConfig.quality_bounds_tolerance` | homodyne defines `optimization.nlsq.config.NLSQConfig.quality_bounds_tolerance`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.config.NLSQConfig.quality_reduced_chi_squared_threshold` | homodyne defines `optimization.nlsq.config.NLSQConfig.quality_reduced_chi_squared_threshold`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.config.NLSQConfig.quality_warn_on_bounds_hit` | homodyne defines `optimization.nlsq.config.NLSQConfig.quality_warn_on_bounds_hit`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.config.NLSQConfig.quality_warn_on_convergence_failure` | homodyne defines `optimization.nlsq.config.NLSQConfig.quality_warn_on_convergence_failure`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.config.NLSQConfig.quality_warn_on_max_restarts` | homodyne defines `optimization.nlsq.config.NLSQConfig.quality_warn_on_max_restarts`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.config.NLSQConfig.regularization_auto_tune_lambda` | homodyne defines `optimization.nlsq.config.NLSQConfig.regularization_auto_tune_lambda`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.config.NLSQConfig.regularization_max_cv` | homodyne defines `optimization.nlsq.config.NLSQConfig.regularization_max_cv`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.config.NLSQConfig.regularization_target_contribution` | homodyne defines `optimization.nlsq.config.NLSQConfig.regularization_target_contribution`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.config.NLSQConfig.trust_region_scale` | homodyne defines `optimization.nlsq.config.NLSQConfig.trust_region_scale`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.data_prep.ExpandedParameters.bounds` | homodyne defines `optimization.nlsq.data_prep.ExpandedParameters.bounds`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.data_prep.ExpandedParameters.n_angles` | homodyne defines `optimization.nlsq.data_prep.ExpandedParameters.n_angles`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.data_prep.ExpandedParameters.n_params` | homodyne defines `optimization.nlsq.data_prep.ExpandedParameters.n_params`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.data_prep.ExpandedParameters.n_physical` | homodyne defines `optimization.nlsq.data_prep.ExpandedParameters.n_physical`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.data_prep.ExpandedParameters.params` | homodyne defines `optimization.nlsq.data_prep.ExpandedParameters.params`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.data_prep.PreparedData.n_data` | homodyne defines `optimization.nlsq.data_prep.PreparedData.n_data`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.data_prep.PreparedData.n_phi` | homodyne defines `optimization.nlsq.data_prep.PreparedData.n_phi`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.data_prep.PreparedData.phi_unique` | homodyne defines `optimization.nlsq.data_prep.PreparedData.phi_unique`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.data_prep.PreparedData.xdata` | homodyne defines `optimization.nlsq.data_prep.PreparedData.xdata`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.data_prep.PreparedData.ydata` | homodyne defines `optimization.nlsq.data_prep.PreparedData.ydata`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.gradient_monitor.CollapseEvent.iteration` | homodyne defines `optimization.nlsq.gradient_monitor.CollapseEvent.iteration`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.gradient_monitor.CollapseEvent.per_angle_grad_norm` | homodyne defines `optimization.nlsq.gradient_monitor.CollapseEvent.per_angle_grad_norm`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.gradient_monitor.CollapseEvent.physical_grad_norm` | homodyne defines `optimization.nlsq.gradient_monitor.CollapseEvent.physical_grad_norm`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.gradient_monitor.CollapseEvent.ratio` | homodyne defines `optimization.nlsq.gradient_monitor.CollapseEvent.ratio`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.gradient_monitor.CollapseEvent.response_mode` | homodyne defines `optimization.nlsq.gradient_monitor.CollapseEvent.response_mode`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.gradient_monitor.GradientMonitorConfig.check_interval` | homodyne defines `optimization.nlsq.gradient_monitor.GradientMonitorConfig.check_interval`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.gradient_monitor.GradientMonitorConfig.consecutive_triggers` | homodyne defines `optimization.nlsq.gradient_monitor.GradientMonitorConfig.consecutive_triggers`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.gradient_monitor.GradientMonitorConfig.enable` | homodyne defines `optimization.nlsq.gradient_monitor.GradientMonitorConfig.enable`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.gradient_monitor.GradientMonitorConfig.lambda_multiplier_on_collapse` | homodyne defines `optimization.nlsq.gradient_monitor.GradientMonitorConfig.lambda_multiplier_on_collapse`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.gradient_monitor.GradientMonitorConfig.ratio_threshold` | homodyne defines `optimization.nlsq.gradient_monitor.GradientMonitorConfig.ratio_threshold`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.gradient_monitor.GradientMonitorConfig.reset_per_angle_to_mean` | homodyne defines `optimization.nlsq.gradient_monitor.GradientMonitorConfig.reset_per_angle_to_mean`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.gradient_monitor.GradientMonitorConfig.response_mode` | homodyne defines `optimization.nlsq.gradient_monitor.GradientMonitorConfig.response_mode`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.gradient_monitor.GradientMonitorConfig.watch_consecutive_triggers` | homodyne defines `optimization.nlsq.gradient_monitor.GradientMonitorConfig.watch_consecutive_triggers`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.gradient_monitor.GradientMonitorConfig.watch_min_iteration` | homodyne defines `optimization.nlsq.gradient_monitor.GradientMonitorConfig.watch_min_iteration`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.gradient_monitor.GradientMonitorConfig.watch_parameters` | homodyne defines `optimization.nlsq.gradient_monitor.GradientMonitorConfig.watch_parameters`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.gradient_monitor.GradientMonitorConfig.watch_threshold` | homodyne defines `optimization.nlsq.gradient_monitor.GradientMonitorConfig.watch_threshold`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.hierarchical.HierarchicalConfig.enable` | homodyne defines `optimization.nlsq.hierarchical.HierarchicalConfig.enable`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.hierarchical.HierarchicalConfig.log_stage_transitions` | homodyne defines `optimization.nlsq.hierarchical.HierarchicalConfig.log_stage_transitions`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.hierarchical.HierarchicalConfig.max_outer_iterations` | homodyne defines `optimization.nlsq.hierarchical.HierarchicalConfig.max_outer_iterations`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.hierarchical.HierarchicalConfig.outer_tolerance` | homodyne defines `optimization.nlsq.hierarchical.HierarchicalConfig.outer_tolerance`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.hierarchical.HierarchicalConfig.per_angle_ftol` | homodyne defines `optimization.nlsq.hierarchical.HierarchicalConfig.per_angle_ftol`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.hierarchical.HierarchicalConfig.per_angle_max_iterations` | homodyne defines `optimization.nlsq.hierarchical.HierarchicalConfig.per_angle_max_iterations`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.hierarchical.HierarchicalConfig.physical_ftol` | homodyne defines `optimization.nlsq.hierarchical.HierarchicalConfig.physical_ftol`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.hierarchical.HierarchicalConfig.physical_max_iterations` | homodyne defines `optimization.nlsq.hierarchical.HierarchicalConfig.physical_max_iterations`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.hierarchical.HierarchicalConfig.save_intermediate_results` | homodyne defines `optimization.nlsq.hierarchical.HierarchicalConfig.save_intermediate_results`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.hierarchical.HierarchicalResult.fun` | homodyne defines `optimization.nlsq.hierarchical.HierarchicalResult.fun`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.hierarchical.HierarchicalResult.history` | homodyne defines `optimization.nlsq.hierarchical.HierarchicalResult.history`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.hierarchical.HierarchicalResult.message` | homodyne defines `optimization.nlsq.hierarchical.HierarchicalResult.message`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.hierarchical.HierarchicalResult.n_outer_iterations` | homodyne defines `optimization.nlsq.hierarchical.HierarchicalResult.n_outer_iterations`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.hierarchical.HierarchicalResult.success` | homodyne defines `optimization.nlsq.hierarchical.HierarchicalResult.success`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.hierarchical.HierarchicalResult.total_time` | homodyne defines `optimization.nlsq.hierarchical.HierarchicalResult.total_time`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.hierarchical.HierarchicalResult.x` | homodyne defines `optimization.nlsq.hierarchical.HierarchicalResult.x`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.memory.StrategyDecision.index_memory_gb` | homodyne defines `optimization.nlsq.memory.StrategyDecision.index_memory_gb`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.multistart.MultiStartConfig.custom_starts` | homodyne defines `optimization.nlsq.multistart.MultiStartConfig.custom_starts`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.multistart.MultiStartConfig.degeneracy_threshold` | homodyne defines `optimization.nlsq.multistart.MultiStartConfig.degeneracy_threshold`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.multistart.MultiStartConfig.enable` | homodyne defines `optimization.nlsq.multistart.MultiStartConfig.enable`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.multistart.MultiStartConfig.n_workers` | homodyne defines `optimization.nlsq.multistart.MultiStartConfig.n_workers`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.multistart.MultiStartConfig.refine_top_k` | homodyne defines `optimization.nlsq.multistart.MultiStartConfig.refine_top_k`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.multistart.MultiStartConfig.refinement_ftol` | homodyne defines `optimization.nlsq.multistart.MultiStartConfig.refinement_ftol`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.multistart.MultiStartConfig.sampling_strategy` | homodyne defines `optimization.nlsq.multistart.MultiStartConfig.sampling_strategy`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.multistart.MultiStartConfig.screen_keep_fraction` | homodyne defines `optimization.nlsq.multistart.MultiStartConfig.screen_keep_fraction`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.multistart.MultiStartConfig.use_screening` | homodyne defines `optimization.nlsq.multistart.MultiStartConfig.use_screening`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.multistart.MultiStartResult.all_results` | homodyne defines `optimization.nlsq.multistart.MultiStartResult.all_results`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.multistart.MultiStartResult.basin_labels` | homodyne defines `optimization.nlsq.multistart.MultiStartResult.basin_labels`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.multistart.MultiStartResult.best` | homodyne defines `optimization.nlsq.multistart.MultiStartResult.best`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.multistart.MultiStartResult.degeneracy_detected` | homodyne defines `optimization.nlsq.multistart.MultiStartResult.degeneracy_detected`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.multistart.MultiStartResult.n_unique_basins` | homodyne defines `optimization.nlsq.multistart.MultiStartResult.n_unique_basins`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.multistart.MultiStartResult.screening_costs` | homodyne defines `optimization.nlsq.multistart.MultiStartResult.screening_costs`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.multistart.MultiStartResult.strategy_used` | homodyne defines `optimization.nlsq.multistart.MultiStartResult.strategy_used`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.multistart.MultiStartResult.total_wall_time` | homodyne defines `optimization.nlsq.multistart.MultiStartResult.total_wall_time`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.multistart.SingleStartResult.chi_squared` | homodyne defines `optimization.nlsq.multistart.SingleStartResult.chi_squared`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.multistart.SingleStartResult.covariance` | homodyne defines `optimization.nlsq.multistart.SingleStartResult.covariance`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.multistart.SingleStartResult.final_params` | homodyne defines `optimization.nlsq.multistart.SingleStartResult.final_params`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.multistart.SingleStartResult.hessian` | homodyne defines `optimization.nlsq.multistart.SingleStartResult.hessian`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.multistart.SingleStartResult.jacobian` | homodyne defines `optimization.nlsq.multistart.SingleStartResult.jacobian`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.multistart.SingleStartResult.message` | homodyne defines `optimization.nlsq.multistart.SingleStartResult.message`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.multistart.SingleStartResult.n_fev` | homodyne defines `optimization.nlsq.multistart.SingleStartResult.n_fev`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.multistart.SingleStartResult.n_iterations` | homodyne defines `optimization.nlsq.multistart.SingleStartResult.n_iterations`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.multistart.SingleStartResult.reduced_chi_squared` | homodyne defines `optimization.nlsq.multistart.SingleStartResult.reduced_chi_squared`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.multistart.SingleStartResult.start_idx` | homodyne defines `optimization.nlsq.multistart.SingleStartResult.start_idx`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.multistart.SingleStartResult.status` | homodyne defines `optimization.nlsq.multistart.SingleStartResult.status`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.multistart.SingleStartResult.success` | homodyne defines `optimization.nlsq.multistart.SingleStartResult.success`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.parameter_index_mapper.ParameterIndexMapper.fourier` | homodyne defines `optimization.nlsq.parameter_index_mapper.ParameterIndexMapper.fourier`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.parameter_index_mapper.ParameterIndexMapper.n_phi` | homodyne defines `optimization.nlsq.parameter_index_mapper.ParameterIndexMapper.n_phi`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.parameter_index_mapper.ParameterIndexMapper.n_physical` | homodyne defines `optimization.nlsq.parameter_index_mapper.ParameterIndexMapper.n_physical`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.parameter_index_mapper.ParameterIndexMapper.use_constant` | homodyne defines `optimization.nlsq.parameter_index_mapper.ParameterIndexMapper.use_constant`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.progress.ProgressConfig.description` | homodyne defines `optimization.nlsq.progress.ProgressConfig.description`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.progress.ProgressConfig.enable_progress_bar` | homodyne defines `optimization.nlsq.progress.ProgressConfig.enable_progress_bar`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.progress.ProgressConfig.log_interval` | homodyne defines `optimization.nlsq.progress.ProgressConfig.log_interval`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.progress.ProgressConfig.max_nfev` | homodyne defines `optimization.nlsq.progress.ProgressConfig.max_nfev`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.progress.ProgressConfig.verbose` | homodyne defines `optimization.nlsq.progress.ProgressConfig.verbose`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.result_builder.QualityMetrics.chi_squared` | homodyne defines `optimization.nlsq.result_builder.QualityMetrics.chi_squared`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.result_builder.QualityMetrics.n_at_bounds` | homodyne defines `optimization.nlsq.result_builder.QualityMetrics.n_at_bounds`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.result_builder.QualityMetrics.quality_flag` | homodyne defines `optimization.nlsq.result_builder.QualityMetrics.quality_flag`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.result_builder.QualityMetrics.reduced_chi_squared` | homodyne defines `optimization.nlsq.result_builder.QualityMetrics.reduced_chi_squared`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.result_builder.ResultBuilder.covariance` | homodyne defines `optimization.nlsq.result_builder.ResultBuilder.covariance`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.result_builder.ResultBuilder.info` | homodyne defines `optimization.nlsq.result_builder.ResultBuilder.info`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.result_builder.ResultBuilder.n_data` | homodyne defines `optimization.nlsq.result_builder.ResultBuilder.n_data`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.result_builder.ResultBuilder.nlsq_diagnostics` | homodyne defines `optimization.nlsq.result_builder.ResultBuilder.nlsq_diagnostics`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.result_builder.ResultBuilder.parameters` | homodyne defines `optimization.nlsq.result_builder.ResultBuilder.parameters`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.result_builder.ResultBuilder.recovery_actions` | homodyne defines `optimization.nlsq.result_builder.ResultBuilder.recovery_actions`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.result_builder.ResultBuilder.start_time` | homodyne defines `optimization.nlsq.result_builder.ResultBuilder.start_time`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.result_builder.ResultBuilder.stratification_diagnostics` | homodyne defines `optimization.nlsq.result_builder.ResultBuilder.stratification_diagnostics`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.results.FallbackInfo.adapter_error` | homodyne defines `optimization.nlsq.results.FallbackInfo.adapter_error`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.results.FallbackInfo.adapter_used` | homodyne defines `optimization.nlsq.results.FallbackInfo.adapter_used`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.results.FallbackInfo.fallback_occurred` | homodyne defines `optimization.nlsq.results.FallbackInfo.fallback_occurred`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.results.FallbackInfo.wrapper_error` | homodyne defines `optimization.nlsq.results.FallbackInfo.wrapper_error`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.results.FunctionEvaluationCounter.count` | homodyne defines `optimization.nlsq.results.FunctionEvaluationCounter.count`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.results.FunctionEvaluationCounter.fn` | homodyne defines `optimization.nlsq.results.FunctionEvaluationCounter.fn`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.results.OptimizationResult.chi_squared` | homodyne defines `optimization.nlsq.results.OptimizationResult.chi_squared`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.results.OptimizationResult.convergence_status` | homodyne defines `optimization.nlsq.results.OptimizationResult.convergence_status`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.results.OptimizationResult.covariance` | homodyne defines `optimization.nlsq.results.OptimizationResult.covariance`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.results.OptimizationResult.device_info` | homodyne defines `optimization.nlsq.results.OptimizationResult.device_info`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.results.OptimizationResult.execution_time` | homodyne defines `optimization.nlsq.results.OptimizationResult.execution_time`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.results.OptimizationResult.iterations` | homodyne defines `optimization.nlsq.results.OptimizationResult.iterations`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.results.OptimizationResult.nlsq_diagnostics` | homodyne defines `optimization.nlsq.results.OptimizationResult.nlsq_diagnostics`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.results.OptimizationResult.parameters` | homodyne defines `optimization.nlsq.results.OptimizationResult.parameters`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.results.OptimizationResult.quality_flag` | homodyne defines `optimization.nlsq.results.OptimizationResult.quality_flag`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.results.OptimizationResult.recovery_actions` | homodyne defines `optimization.nlsq.results.OptimizationResult.recovery_actions`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.results.OptimizationResult.reduced_chi_squared` | homodyne defines `optimization.nlsq.results.OptimizationResult.reduced_chi_squared`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.results.OptimizationResult.sigma_is_default` | homodyne defines `optimization.nlsq.results.OptimizationResult.sigma_is_default`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.results.OptimizationResult.stratification_diagnostics` | homodyne defines `optimization.nlsq.results.OptimizationResult.stratification_diagnostics`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.results.OptimizationResult.streaming_diagnostics` | homodyne defines `optimization.nlsq.results.OptimizationResult.streaming_diagnostics`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.results.OptimizationResult.uncertainties` | homodyne defines `optimization.nlsq.results.OptimizationResult.uncertainties`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.results.UseSequentialOptimization.data` | homodyne defines `optimization.nlsq.results.UseSequentialOptimization.data`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.results.UseSequentialOptimization.reason` | homodyne defines `optimization.nlsq.results.UseSequentialOptimization.reason`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.shear_weighting.ShearWeightingConfig.alpha` | homodyne defines `optimization.nlsq.shear_weighting.ShearWeightingConfig.alpha`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.shear_weighting.ShearWeightingConfig.enable` | homodyne defines `optimization.nlsq.shear_weighting.ShearWeightingConfig.enable`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.shear_weighting.ShearWeightingConfig.initial_phi0` | homodyne defines `optimization.nlsq.shear_weighting.ShearWeightingConfig.initial_phi0`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.shear_weighting.ShearWeightingConfig.min_weight` | homodyne defines `optimization.nlsq.shear_weighting.ShearWeightingConfig.min_weight`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.shear_weighting.ShearWeightingConfig.normalize` | homodyne defines `optimization.nlsq.shear_weighting.ShearWeightingConfig.normalize`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.shear_weighting.ShearWeightingConfig.update_frequency` | homodyne defines `optimization.nlsq.shear_weighting.ShearWeightingConfig.update_frequency`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.shear_weighting.runtime_keys` | homodyne defines `optimization.nlsq.shear_weighting.runtime_keys`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.strategies.chunking.AngleDistributionStats.counts` | homodyne defines `optimization.nlsq.strategies.chunking.AngleDistributionStats.counts`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.strategies.chunking.AngleDistributionStats.fractions` | homodyne defines `optimization.nlsq.strategies.chunking.AngleDistributionStats.fractions`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.strategies.chunking.AngleDistributionStats.imbalance_ratio` | homodyne defines `optimization.nlsq.strategies.chunking.AngleDistributionStats.imbalance_ratio`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.strategies.chunking.AngleDistributionStats.is_balanced` | homodyne defines `optimization.nlsq.strategies.chunking.AngleDistributionStats.is_balanced`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.strategies.chunking.AngleDistributionStats.max_angle` | homodyne defines `optimization.nlsq.strategies.chunking.AngleDistributionStats.max_angle`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.strategies.chunking.AngleDistributionStats.min_angle` | homodyne defines `optimization.nlsq.strategies.chunking.AngleDistributionStats.min_angle`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.strategies.chunking.AngleDistributionStats.n_angles` | homodyne defines `optimization.nlsq.strategies.chunking.AngleDistributionStats.n_angles`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.strategies.chunking.AngleDistributionStats.unique_angles` | homodyne defines `optimization.nlsq.strategies.chunking.AngleDistributionStats.unique_angles`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.strategies.chunking.StratificationDiagnostics.angle_coverage` | homodyne defines `optimization.nlsq.strategies.chunking.StratificationDiagnostics.angle_coverage`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.strategies.chunking.StratificationDiagnostics.angles_per_chunk` | homodyne defines `optimization.nlsq.strategies.chunking.StratificationDiagnostics.angles_per_chunk`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.strategies.chunking.StratificationDiagnostics.chunk_balance` | homodyne defines `optimization.nlsq.strategies.chunking.StratificationDiagnostics.chunk_balance`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.strategies.chunking.StratificationDiagnostics.chunk_sizes` | homodyne defines `optimization.nlsq.strategies.chunking.StratificationDiagnostics.chunk_sizes`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.strategies.chunking.StratificationDiagnostics.execution_time_ms` | homodyne defines `optimization.nlsq.strategies.chunking.StratificationDiagnostics.execution_time_ms`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.strategies.chunking.StratificationDiagnostics.memory_efficiency` | homodyne defines `optimization.nlsq.strategies.chunking.StratificationDiagnostics.memory_efficiency`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.strategies.chunking.StratificationDiagnostics.memory_overhead_mb` | homodyne defines `optimization.nlsq.strategies.chunking.StratificationDiagnostics.memory_overhead_mb`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.strategies.chunking.StratificationDiagnostics.n_chunks` | homodyne defines `optimization.nlsq.strategies.chunking.StratificationDiagnostics.n_chunks`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.strategies.chunking.StratificationDiagnostics.throughput_points_per_sec` | homodyne defines `optimization.nlsq.strategies.chunking.StratificationDiagnostics.throughput_points_per_sec`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.strategies.chunking.StratificationDiagnostics.use_index_based` | homodyne defines `optimization.nlsq.strategies.chunking.StratificationDiagnostics.use_index_based`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.strategies.chunking.StratifiedIndexIterator.chunk_sizes` | homodyne defines `optimization.nlsq.strategies.chunking.StratifiedIndexIterator.chunk_sizes`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.strategies.chunking.StratifiedIndexIterator.indices` | homodyne defines `optimization.nlsq.strategies.chunking.StratifiedIndexIterator.indices`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.strategies.executors.ExecutionResult.info` | homodyne defines `optimization.nlsq.strategies.executors.ExecutionResult.info`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.strategies.executors.ExecutionResult.pcov` | homodyne defines `optimization.nlsq.strategies.executors.ExecutionResult.pcov`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.strategies.executors.ExecutionResult.popt` | homodyne defines `optimization.nlsq.strategies.executors.ExecutionResult.popt`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.strategies.sequential.AngleSubset.g2_exp` | homodyne defines `optimization.nlsq.strategies.sequential.AngleSubset.g2_exp`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.strategies.sequential.AngleSubset.phi` | homodyne defines `optimization.nlsq.strategies.sequential.AngleSubset.phi`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.strategies.sequential.AngleSubset.phi_indices` | homodyne defines `optimization.nlsq.strategies.sequential.AngleSubset.phi_indices`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.strategies.sequential.AngleSubset.t1` | homodyne defines `optimization.nlsq.strategies.sequential.AngleSubset.t1`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.strategies.sequential.AngleSubset.t2` | homodyne defines `optimization.nlsq.strategies.sequential.AngleSubset.t2`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.strategies.sequential.SequentialResult.combined_covariance` | homodyne defines `optimization.nlsq.strategies.sequential.SequentialResult.combined_covariance`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.strategies.sequential.SequentialResult.combined_parameters` | homodyne defines `optimization.nlsq.strategies.sequential.SequentialResult.combined_parameters`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.strategies.sequential.SequentialResult.final_jacobian_norms` | homodyne defines `optimization.nlsq.strategies.sequential.SequentialResult.final_jacobian_norms`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.strategies.sequential.SequentialResult.initial_jacobian_norms` | homodyne defines `optimization.nlsq.strategies.sequential.SequentialResult.initial_jacobian_norms`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.strategies.sequential.SequentialResult.n_angles_failed` | homodyne defines `optimization.nlsq.strategies.sequential.SequentialResult.n_angles_failed`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.strategies.sequential.SequentialResult.n_angles_optimized` | homodyne defines `optimization.nlsq.strategies.sequential.SequentialResult.n_angles_optimized`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.strategies.sequential.SequentialResult.per_angle_results` | homodyne defines `optimization.nlsq.strategies.sequential.SequentialResult.per_angle_results`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.strategies.sequential.SequentialResult.success_rate` | homodyne defines `optimization.nlsq.strategies.sequential.SequentialResult.success_rate`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.strategies.sequential.SequentialResult.total_cost` | homodyne defines `optimization.nlsq.strategies.sequential.SequentialResult.total_cost`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.strategies.stratified_ls.runtime_keys` | homodyne defines `optimization.nlsq.strategies.stratified_ls.runtime_keys`; heterodyne does not |
| `KEEP` | missing_config_key | `optimization.nlsq.transforms.runtime_keys` | homodyne defines `optimization.nlsq.transforms.runtime_keys`; heterodyne does not |
| `KEEP` | missing_config_key | `runtime.utils.system_validator.ValidationResult.error_code` | homodyne defines `runtime.utils.system_validator.ValidationResult.error_code`; heterodyne does not |
| `KEEP` | missing_config_key | `runtime.utils.system_validator.ValidationResult.execution_time` | homodyne defines `runtime.utils.system_validator.ValidationResult.execution_time`; heterodyne does not |
| `KEEP` | missing_config_key | `runtime.utils.system_validator.ValidationResult.warnings` | homodyne defines `runtime.utils.system_validator.ValidationResult.warnings`; heterodyne does not |
| `KEEP` | missing_config_key | `viz.nlsq_plots.runtime_keys` | homodyne defines `viz.nlsq_plots.runtime_keys`; heterodyne does not |

### docs (48)

| Disposition | Kind | Qualname / Path | Detail |
|---|---|---|---|
| `KEEP` | broken_autodoc_target | `api/cli.rst` | expected autodoc target `heterodyne.cli.args_parser` not present in heterodyne docs page |
| `KEEP` | broken_autodoc_target | `api/cli.rst` | expected autodoc target `heterodyne.cli.config_handling` not present in heterodyne docs page |
| `KEEP` | broken_autodoc_target | `api/cli.rst` | expected autodoc target `heterodyne.cli.plot_dispatch` not present in heterodyne docs page |
| `KEEP` | broken_autodoc_target | `api/cli.rst` | expected autodoc target `heterodyne.cli.result_saving` not present in heterodyne docs page |
| `KEEP` | broken_autodoc_target | `api/cmc.rst` | expected autodoc target `heterodyne.optimization.cmc.config.CMCConfig` not present in heterodyne docs page |
| `KEEP` | broken_autodoc_target | `api/cmc.rst` | expected autodoc target `heterodyne.optimization.cmc.core.fit_mcmc_jax` not present in heterodyne docs page |
| `KEEP` | broken_autodoc_target | `api/cmc.rst` | expected autodoc target `heterodyne.optimization.cmc.results.CMCResult` not present in heterodyne docs page |
| `KEEP` | broken_autodoc_target | `api/config.rst` | expected autodoc target `heterodyne.config.manager.ConfigManager` not present in heterodyne docs page |
| `KEEP` | broken_autodoc_target | `api/config.rst` | expected autodoc target `heterodyne.config.parameter_registry.ParameterInfo` not present in heterodyne docs page |
| `KEEP` | broken_autodoc_target | `api/config.rst` | expected autodoc target `heterodyne.config.parameter_registry.ParameterRegistry` not present in heterodyne docs page |
| `KEEP` | broken_autodoc_target | `api/config.rst` | expected autodoc target `heterodyne.config.parameter_registry.get_all_param_names` not present in heterodyne docs page |
| `KEEP` | broken_autodoc_target | `api/config.rst` | expected autodoc target `heterodyne.config.parameter_registry.get_bounds` not present in heterodyne docs page |
| `KEEP` | broken_autodoc_target | `api/config.rst` | expected autodoc target `heterodyne.config.parameter_registry.get_defaults` not present in heterodyne docs page |
| `KEEP` | broken_autodoc_target | `api/config.rst` | expected autodoc target `heterodyne.config.parameter_registry.get_param_names` not present in heterodyne docs page |
| `KEEP` | broken_autodoc_target | `api/config.rst` | expected autodoc target `heterodyne.config.parameter_registry.get_registry` not present in heterodyne docs page |
| `KEEP` | broken_autodoc_target | `api/config.rst` | expected autodoc target `heterodyne.config.parameter_space.ParameterSpace` not present in heterodyne docs page |
| `KEEP` | broken_autodoc_target | `api/core.rst` | expected autodoc target `heterodyne.core.scaling_utils` not present in heterodyne docs page |
| `KEEP` | broken_autodoc_target | `api/data.rst` | expected autodoc target `heterodyne.data.xpcs_loader.XPCSDataLoader` not present in heterodyne docs page |
| `KEEP` | broken_autodoc_target | `api/device.rst` | expected autodoc target `heterodyne.device` not present in heterodyne docs page |
| `KEEP` | broken_autodoc_target | `api/device.rst` | expected autodoc target `heterodyne.device.benchmark_device_performance` not present in heterodyne docs page |
| `KEEP` | broken_autodoc_target | `api/device.rst` | expected autodoc target `heterodyne.device.configure_optimal_device` not present in heterodyne docs page |
| `KEEP` | broken_autodoc_target | `api/device.rst` | expected autodoc target `heterodyne.device.get_device_status` not present in heterodyne docs page |
| `KEEP` | broken_autodoc_target | `api/io.rst` | expected autodoc target `heterodyne.io` not present in heterodyne docs page |
| `KEEP` | broken_autodoc_target | `api/jax_backend.rst` | expected autodoc target `heterodyne.core.jax_backend.batch_chi_squared` not present in heterodyne docs page |
| `KEEP` | broken_autodoc_target | `api/jax_backend.rst` | expected autodoc target `heterodyne.core.jax_backend.clear_meshgrid_cache` not present in heterodyne docs page |
| `KEEP` | broken_autodoc_target | `api/jax_backend.rst` | expected autodoc target `heterodyne.core.jax_backend.compute_chi_squared` not present in heterodyne docs page |
| `KEEP` | broken_autodoc_target | `api/jax_backend.rst` | expected autodoc target `heterodyne.core.jax_backend.compute_g1_diffusion` not present in heterodyne docs page |
| `KEEP` | broken_autodoc_target | `api/jax_backend.rst` | expected autodoc target `heterodyne.core.jax_backend.compute_g1_shear` not present in heterodyne docs page |
| `KEEP` | broken_autodoc_target | `api/jax_backend.rst` | expected autodoc target `heterodyne.core.jax_backend.compute_g1_total` not present in heterodyne docs page |
| `KEEP` | broken_autodoc_target | `api/jax_backend.rst` | expected autodoc target `heterodyne.core.jax_backend.compute_g2_scaled` not present in heterodyne docs page |
| `KEEP` | broken_autodoc_target | `api/jax_backend.rst` | expected autodoc target `heterodyne.core.jax_backend.compute_g2_scaled_with_factors` not present in heterodyne docs page |
| `KEEP` | broken_autodoc_target | `api/jax_backend.rst` | expected autodoc target `heterodyne.core.jax_backend.get_cache_stats` not present in heterodyne docs page |
| `KEEP` | broken_autodoc_target | `api/jax_backend.rst` | expected autodoc target `heterodyne.core.jax_backend.get_cached_meshgrid` not present in heterodyne docs page |
| `KEEP` | broken_autodoc_target | `api/jax_backend.rst` | expected autodoc target `heterodyne.core.jax_backend.get_device_info` not present in heterodyne docs page |
| `KEEP` | broken_autodoc_target | `api/jax_backend.rst` | expected autodoc target `heterodyne.core.jax_backend.get_performance_summary` not present in heterodyne docs page |
| `KEEP` | broken_autodoc_target | `api/jax_backend.rst` | expected autodoc target `heterodyne.core.jax_backend.reset_cache_stats` not present in heterodyne docs page |
| `KEEP` | broken_autodoc_target | `api/jax_backend.rst` | expected autodoc target `heterodyne.core.jax_backend.validate_backend` not present in heterodyne docs page |
| `KEEP` | broken_autodoc_target | `api/jax_backend.rst` | expected autodoc target `heterodyne.core.jax_backend.vectorized_g2_computation` not present in heterodyne docs page |
| `KEEP` | broken_autodoc_target | `api/models.rst` | expected autodoc target `heterodyne.core.models.CombinedModel` not present in heterodyne docs page |
| `KEEP` | broken_autodoc_target | `api/models.rst` | expected autodoc target `heterodyne.core.models.DiffusionModel` not present in heterodyne docs page |
| `KEEP` | broken_autodoc_target | `api/models.rst` | expected autodoc target `heterodyne.core.models.PhysicsModelBase` not present in heterodyne docs page |
| `KEEP` | broken_autodoc_target | `api/models.rst` | expected autodoc target `heterodyne.core.models.ShearModel` not present in heterodyne docs page |
| `KEEP` | broken_autodoc_target | `api/models.rst` | expected autodoc target `heterodyne.core.models.create_model` not present in heterodyne docs page |
| `KEEP` | broken_autodoc_target | `api/utils.rst` | expected autodoc target `heterodyne.device.cpu` not present in heterodyne docs page |
| `KEEP` | broken_autodoc_target | `api/utils.rst` | expected autodoc target `heterodyne.device.cpu.detect_cpu_info` not present in heterodyne docs page |
| `KEEP` | broken_autodoc_target | `api/utils.rst` | expected autodoc target `heterodyne.utils.async_io` not present in heterodyne docs page |
| `KEEP` | broken_autodoc_target | `api/viz.rst` | expected autodoc target `heterodyne.viz` not present in heterodyne docs page |
| `KEEP` | broken_autodoc_target | `api/viz.rst` | expected autodoc target `heterodyne.viz.datashader_backend` not present in heterodyne docs page |

### exports (13)

| Disposition | Kind | Qualname / Path | Detail |
|---|---|---|---|
| `KEEP` | missing_export | `` | missing from __all__: ['ParameterSpace', 'ScaledFittingEngine', 'TheoryEngine', 'cli_main', 'compute_g2_scaled', 'configure_optimal_device', 'fit_mcmc_jax', 'get_device_status', 'get_optimization_info', 'get_package_info'] |
| `KEEP` | missing_export | `cli` | missing from __all__: ['validate_args'] |
| `KEEP` | missing_export | `config` | missing from __all__: ['BoundDict', 'ConstraintRule', 'ConstraintSeverity', 'InitialParametersConfig', 'PHYSICS_CONSTRAINTS', 'ParameterSpaceConfig', 'PhysicsViolation', 'get_all_param_names', 'get_bounds', 'get_defaults', 'get_param_names', 'get_registry', 'validate_all_parameters', 'validate_cross_parameter_constraints', 'validate_single_parameter'] |
| `KEEP` | missing_export | `core` | missing from __all__: ['CombinedModel', 'DiffusionModel', 'HomodyneModel', 'PhysicsFactors', 'PhysicsModelBase', 'ShearModel', 'TheoryEngine', 'compute_chi2_theory', 'compute_g1_diffusion', 'compute_g1_shear', 'compute_g2_scaled', 'compute_g2_scaled_with_factors', 'compute_g2_theory', 'create_physics_factors_from_config_dict', 'gradient_g2', 'hessian_g2', 'jax_available', 'parameter_bounds', 'validate_parameters'] |
| `KEEP` | missing_export | `data` | missing from __all__: ['XPCSConfigurationError', 'XPCSDataFormatError', 'XPCSDependencyError', 'get_data_module_info', 'load_xpcs_config'] |
| `KEEP` | missing_export | `io` | missing from __all__: ['create_mcmc_analysis_dict', 'create_mcmc_diagnostics_dict', 'create_mcmc_parameters_dict'] |
| `KEEP` | missing_export | `optimization` | missing from __all__: ['MCMCResult', 'MCMC_AVAILABLE', 'MultiStartConfig', 'MultiStartResult', 'NLSQWrapper', 'NLSQ_AVAILABLE', 'OPTIMIZATION_STATUS', 'OptimizationResult', 'StratificationDiagnostics', 'StratifiedResidualFunction', 'StratifiedResidualFunctionJIT', 'cmc', 'create_angle_stratified_data', 'create_angle_stratified_indices', 'create_stratified_residual_function', 'fit_mcmc_jax', 'fit_nlsq_multistart', 'nlsq', 'optimize_per_angle_sequential', 'should_use_stratification'] |
| `KEEP` | missing_export | `optimization.cmc.backends` | missing from __all__: ['CMCBackend'] |
| `KEEP` | missing_export | `optimization.nlsq` | missing from __all__: ['AdapterConfig', 'AdaptiveHybridStreamingOptimizer', 'CMAESConfig', 'CMAESDiagnostics', 'CMAESOptimizer', 'CMAESWrapper', 'CMAESWrapperConfig', 'CMAES_PRESETS', 'CurveFit', 'CurveFitResult', 'DEFAULT_MEMORY_FRACTION', 'ExecutionResult', 'ExpandedParameters', 'FALLBACK_THRESHOLD_GB', 'FunctionEvaluationCounter', 'GlobalOptimizationConfig', 'HybridRecoveryConfig', 'HybridStreamingConfig', 'JAC_SAMPLE_SIZE', 'JAX_AVAILABLE', 'LargeDatasetExecutor', 'MethodSelector', 'MultiStartConfig', 'MultiStartOrchestrator', 'MultiStartResult', 'NLSQDatasetSizeTier', 'NLSQMemoryManager', 'NLSQOptimizationRecovery', 'NLSQ_AVAILABLE', 'NLSQ_CACHING_AVAILABLE', 'NLSQ_CMAES_AVAILABLE', 'NLSQ_CURVEFIT_AVAILABLE', 'NLSQ_GLOBAL_OPT_AVAILABLE', 'NLSQ_GOAL_AVAILABLE', 'NLSQ_RESULT_AVAILABLE', 'NLSQ_STABILITY_AVAILABLE', 'NLSQ_STREAMING_AVAILABLE', 'NLSQ_WORKFLOW_AVAILABLE', 'NumericalStabilityGuard', 'OptimizationExecutor', 'OptimizationGoal', 'OptimizationResult', 'ParameterIndexMapper', 'PreparedData', 'QualityMetrics', 'ResultBuilder', 'SingleStartResult', 'StandardExecutor', 'StrategyDecision', 'StratificationDiagnostics', 'StratifiedResidualFunction', 'StratifiedResidualFunctionJIT', 'StreamingExecutor', 'WorkflowSelector', 'WorkflowTier', '_get_param_names', 'analyze_angle_distribution', 'auto_configure_cmaes_memory', 'build_parameter_labels', 'build_parameter_labels_utils', 'classify_parameter_status', 'classify_parameter_status_utils', 'clear_model_cache', 'compute_consistent_per_angle_init', 'compute_default_popsize', 'compute_jacobian_stats', 'compute_quality_metrics', 'compute_stratification_diagnostics', 'compute_theoretical_fits', 'compute_uncertainties', 'convert_bounds_to_nlsq_format', 'create_angle_stratified_data', 'create_angle_stratified_indices', 'create_stratified_residual_function', 'curve_fit', 'detect_degeneracy', 'detect_total_system_memory', 'determine_convergence_status', 'estimate_cmaes_memory_gb', 'estimate_peak_memory_gb', 'estimate_stratification_memory', 'expand_per_angle_parameters', 'extract_parameters_from_result', 'fit_nlsq_cmaes', 'fit_nlsq_multistart', 'format_diagnostics_report', 'generate_random_starts', 'get_adapter', 'get_adaptive_memory_threshold', 'get_cache_stats', 'get_executor', 'get_memory_manager', 'get_or_create_model', 'get_physical_param_count', 'include_custom_starts', 'is_adapter_available', 'is_evosax_available', 'normalize_analysis_mode', 'normalize_nlsq_result', 'optimize_per_angle_sequential', 'run_multistart_nlsq', 'sample_xdata', 'screen_starts', 'should_use_stratification', 'validate_bounds', 'validate_initial_params', 'validate_n_starts_for_lhs'] |
| `KEEP` | missing_export | `optimization.nlsq.strategies` | missing from __all__: ['ExecutionResult', 'JAC_SAMPLE_SIZE', 'LargeDatasetExecutor', 'OptimizationExecutor', 'StandardExecutor', 'StratificationDiagnostics', 'StratifiedResidualFunction', 'StratifiedResidualFunctionJIT', 'StreamingExecutor', 'analyze_angle_distribution', 'compute_stratification_diagnostics', 'create_angle_stratified_data', 'create_angle_stratified_indices', 'create_stratified_residual_function', 'estimate_stratification_memory', 'format_diagnostics_report', 'get_executor', 'optimize_per_angle_sequential', 'should_use_stratification'] |
| `KEEP` | missing_export | `optimization.nlsq.validation` | missing from __all__: ['validate_array_dimensions', 'validate_bounds_consistency', 'validate_covariance', 'validate_initial_params', 'validate_no_nan_inf', 'validate_optimized_params', 'validate_result_consistency'] |
| `KEEP` | missing_export | `utils` | missing from __all__: ['PathValidationError', 'get_safe_output_dir', 'validate_plot_save_path', 'validate_save_path'] |
| `KEEP` | missing_export | `viz` | missing from __all__: ['DatashaderRenderer', 'generate_and_plot_fitted_simulations', 'generate_nlsq_plots', 'plot_c2_comparison_fast', 'plot_c2_heatmap_fast', 'plot_experimental_data', 'plot_fit_comparison', 'plot_posterior_comparison', 'plot_simulated_data', 'plot_trace_plots'] |

### logs_errors (1)

| Disposition | Kind | Qualname / Path | Detail |
|---|---|---|---|
| `KEEP` | exit_code_drift | `<package>` | homodyne exit codes=[0, 1, 130] heterodyne=[] |

### signatures (591)

| Disposition | Kind | Qualname / Path | Detail |
|---|---|---|---|
| `KEEP` | missing_in_heterodyne | `__init__.get_package_info` | homodyne has `get_package_info() -> dict`; heterodyne missing |
| `KEEP` | changed | `cli.args_parser.validate_args` | homodyne: `validate_args(args: argparse.Namespace) -> bool`<br>  heterodyne: `validate_args(args: argparse.Namespace) -> list[str]` |
| `KEEP` | changed | `cli.commands.dispatch_command` | homodyne: `dispatch_command(args: argparse.Namespace) -> dict[str, Any]`<br>  heterodyne: `dispatch_command(args: argparse.Namespace) -> int` |
| `KEEP` | missing_in_heterodyne | `cli.commands.normalize_angle_to_symmetric_range` | homodyne has `normalize_angle_to_symmetric_range(angle: float | NDArray[np.floating[Any]]) -> float | NDArray[np.floating[Any]]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `cli.config_generator.create_parser` | homodyne has `create_parser() -> argparse.ArgumentParser`; heterodyne missing |
| `KEEP` | changed | `cli.config_generator.generate_config` | homodyne: `generate_config(mode: str, output_path: Path, force: bool = False, filter_mode: str = 'full') -> dict[str, Any]`<br>  heterodyne: `generate_config(output_path: Path | str, data_path: str | None = None, q: float | None = None, dt: float | None = None, time_length: int | None = None, overwrite: bool = False, mode: str = 'full') -> Path` |
| `KEEP` | changed | `cli.config_generator.get_template_path` | homodyne: `get_template_path(mode: str) -> Path`<br>  heterodyne: `get_template_path() -> Path` |
| `KEEP` | changed | `cli.config_generator.main` | homodyne: `main() -> int`<br>  heterodyne: `main() -> None` |
| `KEEP` | changed | `cli.config_generator.validate_config` | homodyne: `validate_config(config_path: Path) -> bool`<br>  heterodyne: `validate_config(path: Path | str) -> bool` |
| `KEEP` | changed | `cli.main.main` | homodyne: `main() -> None`<br>  heterodyne: `main(argv: list[str] | None = None) -> int` |
| `KEEP` | changed | `cli.main.main_hexp` | homodyne: `main_hexp() -> None`<br>  heterodyne: `main_hexp() -> int` |
| `KEEP` | changed | `cli.main.main_hsim` | homodyne: `main_hsim() -> None`<br>  heterodyne: `main_hsim() -> int` |
| `KEEP` | missing_in_heterodyne | `cli.optimization_runner.load_nlsq_result_from_file` | homodyne has `load_nlsq_result_from_file(nlsq_result_path: Path) -> dict[str, Any] | None`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `cli.result_saving.save_mcmc_results` | homodyne has `save_mcmc_results(result: Any, data: dict[str, Any], config: Any, output_dir: Path) -> None`; heterodyne missing |
| `KEEP` | changed | `cli.result_saving.save_nlsq_results` | homodyne: `save_nlsq_results(result: Any, data: dict[str, Any], config: Any, output_dir: Path) -> None`<br>  heterodyne: `save_nlsq_results(results: list[NLSQResult], output_dir: Path, phi_angles: list[float], c2_exp: np.ndarray | None = None) -> list[Path]` |
| `KEEP` | missing_in_heterodyne | `cli.xla_config.detect_optimal_devices` | homodyne has `detect_optimal_devices()`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `cli.xla_config.set_mode` | homodyne has `set_mode(mode: str) -> bool`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `cli.xla_config.show_config` | homodyne has `show_config()`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `config.manager.ConfigManager.get_active_parameters` | homodyne has `get_active_parameters(self) -> list[str]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `config.manager.ConfigManager.get_initial_parameters` | homodyne has `get_initial_parameters(self, use_midpoint_defaults: bool = True) -> dict[str, float]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `config.manager.ConfigManager.get_parameter_bounds` | homodyne has `get_parameter_bounds(self, parameter_names: list[str] | None = None) -> list[dict[str, Any]]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `config.manager.ConfigManager.get_target_angle_ranges` | homodyne has `get_target_angle_ranges(self) -> dict[str, Any]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `config.manager.ConfigManager.is_static_mode_enabled` | homodyne has `is_static_mode_enabled(self) -> bool`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `config.manager.ConfigManager.load_config` | homodyne has `load_config(self) -> None`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `config.manager.ConfigManager.update_config` | homodyne has `update_config(self, key: str, value: Any) -> None`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `config.manager.ConfigManager.validate_per_angle_scaling` | homodyne has `validate_per_angle_scaling(self, n_phi: int) -> list[str]`; heterodyne missing |
| `KEEP` | changed | `config.manager.load_xpcs_config` | homodyne: `load_xpcs_config(config_path: str) -> dict[str, Any]`<br>  heterodyne: `load_xpcs_config(path: Path | str) -> ConfigManager` |
| `KEEP` | missing_in_heterodyne | `config.parameter_manager.ParameterManager.validate_parameters` | homodyne has `validate_parameters(self, params: np.ndarray, param_names: list[str] | None = None, tolerance: float = 1e-10) -> ValidationResult`; heterodyne missing |
| `KEEP` | changed | `config.parameter_manager.ParameterManager.validate_physical_constraints` | homodyne: `validate_physical_constraints(self, params: dict[str, float], severity_level: str = 'warning') -> ValidationResult`<br>  heterodyne: `validate_physical_constraints(self, params: dict[str, float] | np.ndarray | None = None, severity_level: str = 'warning') -> ValidationResult` |
| `KEEP` | missing_in_heterodyne | `config.parameter_names.get_num_parameters` | homodyne has `get_num_parameters(analysis_mode: AnalysisMode) -> int`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `config.parameter_names.get_parameter_description` | homodyne has `get_parameter_description(param_name: str) -> str`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `config.parameter_names.get_parameter_names` | homodyne has `get_parameter_names(analysis_mode: AnalysisMode) -> list[str]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `config.parameter_names.get_physical_param_names` | homodyne has `get_physical_param_names(analysis_mode: AnalysisMode) -> list[str]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `config.parameter_names.validate_parameter_names` | homodyne has `validate_parameter_names(param_names: list[str], analysis_mode: AnalysisMode, strict: bool = True) -> None`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `config.parameter_names.verify_samples_dict` | homodyne has `verify_samples_dict(samples_dict: dict, analysis_mode: AnalysisMode) -> None`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `config.parameter_registry.ParameterRegistry.expand_initial_values` | homodyne has `expand_initial_values(self, initial_values: dict[str, float], n_angles: int) -> dict[str, float]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `config.parameter_registry.ParameterRegistry.get_all_bounds` | homodyne has `get_all_bounds(self, analysis_mode: AnalysisMode, n_angles: int = 1, include_scaling: bool = True) -> tuple[list[float], list[float]]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `config.parameter_registry.ParameterRegistry.get_all_param_names` | homodyne has `get_all_param_names(self, analysis_mode: AnalysisMode, n_angles: int = 1, include_scaling: bool = True) -> list[str]`; heterodyne missing |
| `KEEP` | changed | `config.parameter_registry.ParameterRegistry.get_bounds` | homodyne: `get_bounds(self, name: str) -> tuple[float, float]`<br>  heterodyne: `get_bounds(self) -> tuple[list[float], list[float]]` |
| `KEEP` | changed | `config.parameter_registry.ParameterRegistry.get_defaults` | homodyne: `get_defaults(self, analysis_mode: AnalysisMode, n_angles: int = 1, include_scaling: bool = True) -> list[float]`<br>  heterodyne: `get_defaults(self) -> dict[str, float]` |
| `KEEP` | missing_in_heterodyne | `config.parameter_registry.ParameterRegistry.get_num_params` | homodyne has `get_num_params(self, analysis_mode: AnalysisMode, n_angles: int = 1, include_scaling: bool = True) -> int`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `config.parameter_registry.ParameterRegistry.get_param_info` | homodyne has `get_param_info(self, name: str) -> ParameterInfo`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `config.parameter_registry.ParameterRegistry.get_param_names` | homodyne has `get_param_names(self, analysis_mode: AnalysisMode) -> list[str]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `config.parameter_registry.ParameterRegistry.scaling_names` | homodyne has `scaling_names(self) -> tuple[str, ...]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `config.parameter_registry.ParameterRegistry.validate_param_values` | homodyne has `validate_param_values(self, values: dict[str, float] | list[float], analysis_mode: AnalysisMode, n_angles: int = 1, include_scaling: bool = True) -> None`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `config.parameter_registry.get_all_param_names` | homodyne has `get_all_param_names(analysis_mode: AnalysisMode, n_angles: int = 1, include_scaling: bool = True) -> list[str]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `config.parameter_registry.get_bounds` | homodyne has `get_bounds(name: str) -> tuple[float, float]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `config.parameter_registry.get_defaults` | homodyne has `get_defaults(analysis_mode: AnalysisMode, n_angles: int = 1, include_scaling: bool = True) -> list[float]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `config.parameter_registry.get_param_names` | homodyne has `get_param_names(analysis_mode: AnalysisMode) -> list[str]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `config.parameter_registry.get_registry` | homodyne has `get_registry() -> ParameterRegistry`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `config.parameter_space.ParameterSpace.clamp_to_open_interval` | homodyne has `clamp_to_open_interval(self, param_name: str, value: float, epsilon: float = 1e-06) -> float`; heterodyne missing |
| `KEEP` | changed | `config.parameter_space.ParameterSpace.convert_to_beta_priors` | homodyne: `convert_to_beta_priors(self) -> 'ParameterSpace'`<br>  heterodyne: `convert_to_beta_priors(self) -> None` |
| `KEEP` | missing_in_heterodyne | `config.parameter_space.ParameterSpace.convert_to_beta_scaled_priors` | homodyne has `convert_to_beta_scaled_priors(self) -> 'ParameterSpace'`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `config.parameter_space.ParameterSpace.copy` | homodyne has `copy(self) -> 'ParameterSpace'`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `config.parameter_space.ParameterSpace.drop_parameters` | homodyne has `drop_parameters(self, names: set[str]) -> 'ParameterSpace'`; heterodyne missing |
| `KEEP` | changed | `config.parameter_space.ParameterSpace.from_config` | homodyne: `from_config(cls, config_dict: dict[str, Any], analysis_mode: str | None = None) -> 'ParameterSpace'`<br>  heterodyne: `from_config(cls, config: dict[str, Any]) -> ParameterSpace` |
| `KEEP` | missing_in_heterodyne | `config.parameter_space.ParameterSpace.from_defaults` | homodyne has `from_defaults(cls, analysis_mode: str = 'laminar_flow') -> 'ParameterSpace'`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `config.parameter_space.ParameterSpace.get_bounds` | homodyne has `get_bounds(self, param_name: str) -> tuple[float, float]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `config.parameter_space.ParameterSpace.get_bounds_array` | homodyne has `get_bounds_array(self) -> tuple[np.ndarray, np.ndarray]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `config.parameter_space.ParameterSpace.get_prior` | homodyne has `get_prior(self, param_name: str) -> PriorDistribution`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `config.parameter_space.ParameterSpace.get_prior_means` | homodyne has `get_prior_means(self) -> np.ndarray`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `config.parameter_space.ParameterSpace.get_single_angle_fallback_prior` | homodyne has `get_single_angle_fallback_prior(self, param_name: str) -> PriorDistribution`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `config.parameter_space.ParameterSpace.get_single_angle_geometry_config` | homodyne has `get_single_angle_geometry_config(self) -> dict[str, float]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `config.parameter_space.ParameterSpace.validate_values` | homodyne has `validate_values(self, values: dict[str, float], tolerance: float = 1e-10) -> tuple[bool, list[str]]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `config.parameter_space.ParameterSpace.with_prior_overrides` | homodyne has `with_prior_overrides(self, overrides: dict[str, PriorDistribution]) -> 'ParameterSpace'`; heterodyne missing |
| `KEEP` | changed | `config.parameter_space.ParameterSpace.with_single_angle_stabilization` | homodyne: `with_single_angle_stabilization(self, *, enable_beta_fallback: bool = False) -> 'ParameterSpace'`<br>  heterodyne: `with_single_angle_stabilization(self) -> ParameterSpace` |
| `KEEP` | missing_in_heterodyne | `config.parameter_space.PriorDistribution.to_numpyro_kwargs` | homodyne has `to_numpyro_kwargs(self) -> dict[str, Any]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `config.physics_validators.PhysicsViolation.format` | homodyne has `format(self) -> str`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `config.physics_validators.get_constraint_summary` | homodyne has `get_constraint_summary() -> dict[str, Any]`; heterodyne missing |
| `KEEP` | changed | `config.physics_validators.validate_all_parameters` | homodyne: `validate_all_parameters(params: dict[str, float], min_severity: str = ConstraintSeverity.WARNING) -> list[PhysicsViolation]`<br>  heterodyne: `validate_all_parameters(params: dict[str, float], min_severity: ConstraintSeverity = ConstraintSeverity.INFO) -> list[PhysicsViolation]` |
| `KEEP` | changed | `config.physics_validators.validate_cross_parameter_constraints` | homodyne: `validate_cross_parameter_constraints(params: dict[str, float], min_severity: str = ConstraintSeverity.WARNING) -> list[PhysicsViolation]`<br>  heterodyne: `validate_cross_parameter_constraints(params: dict[str, float], min_severity: ConstraintSeverity = ConstraintSeverity.INFO) -> list[PhysicsViolation]` |
| `KEEP` | changed | `config.physics_validators.validate_single_parameter` | homodyne: `validate_single_parameter(param: str, value: float, min_severity: str = ConstraintSeverity.WARNING) -> list[PhysicsViolation]`<br>  heterodyne: `validate_single_parameter(param: str, value: float, min_severity: ConstraintSeverity = ConstraintSeverity.INFO) -> list[PhysicsViolation]` |
| `KEEP` | changed | `core.diagonal_correction.apply_diagonal_correction` | homodyne: `apply_diagonal_correction(c2_mat: ArrayLike, method: Method = 'basic', backend: Backend = 'auto', **config: Any) -> np.ndarray | jnp.ndarray`<br>  heterodyne: `apply_diagonal_correction(c2: Any, width: int = 1, method: str = 'basic', backend: str | None = None, **config: Any) -> Any` |
| `KEEP` | changed | `core.diagonal_correction.apply_diagonal_correction_batch` | homodyne: `apply_diagonal_correction_batch(c2_matrices: ArrayLike, method: Method = 'basic', backend: Backend = 'auto', **config: Any) -> np.ndarray | jnp.ndarray`<br>  heterodyne: `apply_diagonal_correction_batch(c2_batch: Any, width: int = 1, method: str = 'basic', backend: str | None = None, **config: Any) -> Any` |
| `KEEP` | changed | `core.fitting.ParameterSpace.get_param_bounds` | homodyne: `get_param_bounds(self, analysis_mode: str) -> list[tuple[float, float]]`<br>  heterodyne: `get_param_bounds(self) -> list[tuple[float, float]]` |
| `KEEP` | changed | `core.fitting.ParameterSpace.get_param_priors` | homodyne: `get_param_priors(self, analysis_mode: str) -> list[tuple[float, float]]`<br>  heterodyne: `get_param_priors(self) -> list[tuple[float, float]]` |
| `KEEP` | missing_in_heterodyne | `core.fitting.UnifiedHomodyneEngine.compute_likelihood` | homodyne has `compute_likelihood(self, params: np.ndarray, contrast: float, offset: float, data: np.ndarray, sigma: np.ndarray, t1: np.ndarray, t2: np.ndarray, phi: np.ndarray, q: float, L: float, dt: float | None = None) -> float`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `core.fitting.UnifiedHomodyneEngine.detect_dataset_size` | homodyne has `detect_dataset_size(self, data: np.ndarray) -> str`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `core.fitting.UnifiedHomodyneEngine.estimate_scaling_parameters` | homodyne has `estimate_scaling_parameters(self, data: np.ndarray, theory: np.ndarray, validate_bounds: bool = True) -> tuple[float, float]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `core.fitting.UnifiedHomodyneEngine.get_parameter_info` | homodyne has `get_parameter_info(self) -> dict[str, Any]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `core.fitting.UnifiedHomodyneEngine.validate_inputs` | homodyne has `validate_inputs(self, data: np.ndarray, sigma: np.ndarray | None, t1: np.ndarray, t2: np.ndarray, phi: np.ndarray, q: float, L: float) -> None`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `core.homodyne_model.HomodyneModel.compute_c2` | homodyne has `compute_c2(self, params: np.ndarray, phi_angles: np.ndarray, contrast: float = 0.5, offset: float = 1.0) -> np.ndarray`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `core.homodyne_model.HomodyneModel.compute_c2_single_angle` | homodyne has `compute_c2_single_angle(self, params: np.ndarray, phi: float, contrast: float = 0.5, offset: float = 1.0) -> np.ndarray`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `core.homodyne_model.HomodyneModel.config_summary` | homodyne has `config_summary(self) -> dict`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `core.homodyne_model.HomodyneModel.plot_simulated_data` | homodyne has `plot_simulated_data(self, params: np.ndarray, phi_angles: np.ndarray, output_dir: str = './simulated_data', contrast: float = 0.5, offset: float = 1.0, generate_plots: bool = True) -> tuple[np.ndarray, Path]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `core.model_mixins.BenchmarkingMixin.benchmark_gradient_performance` | homodyne has `benchmark_gradient_performance(self: _PhysicsModelProtocol, test_params: jnp.ndarray | None = None) -> dict[str, Any]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `core.model_mixins.BenchmarkingMixin.validate_gradient_accuracy` | homodyne has `validate_gradient_accuracy(self: _PhysicsModelProtocol, test_params: jnp.ndarray | None = None, tolerance: float = 1e-06) -> dict[str, Any]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `core.model_mixins.GradientCapabilityMixin.get_best_gradient_method` | homodyne has `get_best_gradient_method(self) -> str`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `core.model_mixins.GradientCapabilityMixin.get_gradient_capabilities` | homodyne has `get_gradient_capabilities(self) -> dict[str, Any]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `core.model_mixins.GradientCapabilityMixin.get_gradient_function` | homodyne has `get_gradient_function(self) -> Callable`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `core.model_mixins.GradientCapabilityMixin.get_hessian_function` | homodyne has `get_hessian_function(self) -> Callable`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `core.model_mixins.GradientCapabilityMixin.supports_gradients` | homodyne has `supports_gradients(self) -> bool`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `core.model_mixins.OptimizationRecommendationMixin.get_model_info` | homodyne has `get_model_info(self: _PhysicsModelProtocol) -> dict`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `core.model_mixins.OptimizationRecommendationMixin.get_optimization_recommendations` | homodyne has `get_optimization_recommendations(self: _PhysicsModelProtocol) -> list[str]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `core.models.CombinedModel.compute_chi_squared` | homodyne has `compute_chi_squared(self, params: jnp.ndarray, data: jnp.ndarray, sigma: jnp.ndarray, t1: jnp.ndarray, t2: jnp.ndarray, phi: jnp.ndarray, q: float, L: float, contrast: float, offset: float) -> float`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `core.models.CombinedModel.compute_g1` | homodyne has `compute_g1(self, params: jnp.ndarray, t1: jnp.ndarray, t2: jnp.ndarray, phi: jnp.ndarray, q: float, L: float, dt: float | None = None) -> jnp.ndarray`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `core.models.CombinedModel.compute_g1_batch` | homodyne has `compute_g1_batch(self, params: jnp.ndarray, t1_batch: jnp.ndarray, t2_batch: jnp.ndarray, phi_batch: jnp.ndarray, q: float, L: float, dt: float | None = None) -> jnp.ndarray`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `core.models.CombinedModel.compute_g2` | homodyne has `compute_g2(self, params: jnp.ndarray, t1: jnp.ndarray, t2: jnp.ndarray, phi: jnp.ndarray, q: float, L: float, contrast: float, offset: float, dt: float) -> jnp.ndarray`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `core.models.CombinedModel.get_default_parameters` | homodyne has `get_default_parameters(self) -> jnp.ndarray`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `core.models.CombinedModel.get_parameter_bounds` | homodyne has `get_parameter_bounds(self) -> list[tuple[float, float]]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `core.models.DiffusionModel.compute_g1` | homodyne has `compute_g1(self, params: jnp.ndarray, t1: jnp.ndarray, t2: jnp.ndarray, phi: jnp.ndarray, q: float, L: float, dt: float | None = None) -> jnp.ndarray`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `core.models.DiffusionModel.get_default_parameters` | homodyne has `get_default_parameters(self) -> jnp.ndarray`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `core.models.DiffusionModel.get_parameter_bounds` | homodyne has `get_parameter_bounds(self) -> list[tuple[float, float]]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `core.models.PhysicsModelBase.compute_g1` | homodyne has `compute_g1(self, params: jnp.ndarray, t1: jnp.ndarray, t2: jnp.ndarray, phi: jnp.ndarray, q: float, L: float, dt: float | None = None) -> jnp.ndarray`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `core.models.PhysicsModelBase.get_default_parameters` | homodyne has `get_default_parameters(self) -> jnp.ndarray`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `core.models.PhysicsModelBase.get_parameter_bounds` | homodyne has `get_parameter_bounds(self) -> list[tuple[float, float]]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `core.models.PhysicsModelBase.get_parameter_dict` | homodyne has `get_parameter_dict(self, params: jnp.ndarray) -> dict[str, float]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `core.models.PhysicsModelBase.validate_parameters` | homodyne has `validate_parameters(self, params: jnp.ndarray) -> bool`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `core.models.ShearModel.compute_g1` | homodyne has `compute_g1(self, params: jnp.ndarray, t1: jnp.ndarray, t2: jnp.ndarray, phi: jnp.ndarray, q: float, L: float, dt: float | None = None) -> jnp.ndarray`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `core.models.ShearModel.get_default_parameters` | homodyne has `get_default_parameters(self) -> jnp.ndarray`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `core.models.ShearModel.get_parameter_bounds` | homodyne has `get_parameter_bounds(self) -> list[tuple[float, float]]`; heterodyne missing |
| `KEEP` | changed | `core.models.create_model` | homodyne: `create_model(analysis_mode: str) -> CombinedModel`<br>  heterodyne: `create_model(mode: str) -> HeterodyneModelBase` |
| `KEEP` | missing_in_heterodyne | `core.models.get_available_models` | homodyne has `get_available_models() -> list[str]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `core.numpy_gradients.numpy_gradient` | homodyne has `numpy_gradient(func: Callable, argnums: int | list[int] = 0, config: DifferentiationConfig | None = None) -> Callable`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `core.numpy_gradients.numpy_hessian` | homodyne has `numpy_hessian(func: Callable, argnums: int | list[int] = 0, config: DifferentiationConfig | None = None) -> Callable`; heterodyne missing |
| `KEEP` | changed | `core.numpy_gradients.validate_gradient_accuracy` | homodyne: `validate_gradient_accuracy(func: Callable, x: np.ndarray, analytical_grad: np.ndarray | None = None, tolerance: float = 1e-06) -> dict[str, Any]`<br>  heterodyne: `validate_gradient_accuracy(analytic_grad: np.ndarray, numerical_grad: np.ndarray, rtol: float = 0.0001, atol: float = 1e-06) -> GradientResult` |
| `KEEP` | missing_in_heterodyne | `core.physics_factors.PhysicsFactors.from_config` | homodyne has `from_config(cls, q: float, L: float, dt: float, validate: bool = True) -> 'PhysicsFactors'`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `core.physics_factors.PhysicsFactors.to_dict` | homodyne has `to_dict(self) -> dict`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `core.physics_factors.PhysicsFactors.to_tuple` | homodyne has `to_tuple(self) -> tuple[float, float]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `core.physics_factors.create_physics_factors_from_config_dict` | homodyne has `create_physics_factors_from_config_dict(config: dict) -> PhysicsFactors`; heterodyne missing |
| `KEEP` | changed | `core.scaling_utils.compute_averaged_scaling` | homodyne: `compute_averaged_scaling(c2_data: np.ndarray, t1: np.ndarray, t2: np.ndarray, phi_indices: np.ndarray, n_phi: int, contrast_bounds: tuple[float, float], offset_bounds: tuple[float, float], log: Logger | LoggerAdapter[Logger] | None = None) -> tuple[float, float, np.ndarray, np.ndarray]`<br>  heterodyne: `compute_averaged_scaling(c2_data: np.ndarray, t1: np.ndarray, t2: np.ndarray, phi_indices: np.ndarray, n_phi: int, contrast_bounds: tuple[float, float], offset_bounds: tuple[float, float], log: logging.Logger | logging.LoggerAdapter[logging.Logger] | None = None) -> tuple[float, float, np.ndarray, np.ndarray]` |
| `KEEP` | changed | `core.scaling_utils.estimate_per_angle_scaling` | homodyne: `estimate_per_angle_scaling(c2_data: np.ndarray, t1: np.ndarray, t2: np.ndarray, phi_indices: np.ndarray, n_phi: int, contrast_bounds: tuple[float, float], offset_bounds: tuple[float, float], log: Logger | LoggerAdapter[Logger] | None = None) -> dict[str, float]`<br>  heterodyne: `estimate_per_angle_scaling(c2_data: np.ndarray, t1: np.ndarray, t2: np.ndarray, phi_indices: np.ndarray, n_phi: int, contrast_bounds: tuple[float, float], offset_bounds: tuple[float, float], log: logging.Logger | logging.LoggerAdapter[logging.Logger] | None = None) -> dict[str, float]` |
| `KEEP` | missing_in_heterodyne | `data.__init__.get_data_module_info` | homodyne has `get_data_module_info() -> dict`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.angle_filtering.angle_in_range` | homodyne has `angle_in_range(angle: float, min_angle: float, max_angle: float) -> bool`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.angle_filtering.apply_angle_filtering` | homodyne has `apply_angle_filtering(phi_angles: np.ndarray, c2_exp: np.ndarray, config: dict[str, Any]) -> tuple[list[int], np.ndarray, np.ndarray]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.angle_filtering.apply_angle_filtering_for_optimization` | homodyne has `apply_angle_filtering_for_optimization(data: dict[str, Any], config: Any) -> dict[str, Any]`; heterodyne missing |
| `KEEP` | changed | `data.config.apply_config_defaults` | homodyne: `apply_config_defaults(config: dict[str, Any], schema: dict[str, Any] | None = None) -> dict[str, Any]`<br>  heterodyne: `apply_config_defaults(config: dict[str, Any]) -> dict[str, Any]` |
| `KEEP` | changed | `data.config.create_example_yaml_config` | homodyne: `create_example_yaml_config(output_path: str | Path, data_folder: str = '/path/to/data', data_file: str = 'experiment.hdf') -> None`<br>  heterodyne: `create_example_yaml_config(path: Path | str) -> None` |
| `KEEP` | changed | `data.config.load_json_config` | homodyne: `load_json_config(config_path: str | Path) -> dict[str, Any]`<br>  heterodyne: `load_json_config(path: Path | str) -> dict[str, Any]` |
| `KEEP` | changed | `data.config.load_yaml_config` | homodyne: `load_yaml_config(config_path: str | Path) -> dict[str, Any]`<br>  heterodyne: `load_yaml_config(path: Path | str) -> dict[str, Any]` |
| `KEEP` | changed | `data.config.migrate_json_to_yaml_config` | homodyne: `migrate_json_to_yaml_config(json_config: dict[str, Any], yaml_output_path: str | Path | None = None) -> dict[str, Any]`<br>  heterodyne: `migrate_json_to_yaml_config(json_path: Path | str, yaml_path: Path | str) -> None` |
| `KEEP` | changed | `data.config.save_yaml_config` | homodyne: `save_yaml_config(config: dict[str, Any], output_path: str | Path) -> None`<br>  heterodyne: `save_yaml_config(config: dict[str, Any], path: Path | str) -> None` |
| `KEEP` | changed | `data.config.validate_config_schema` | homodyne: `validate_config_schema(config: dict[str, Any], schema: dict[str, Any] | None = None) -> ConfigValidationResult`<br>  heterodyne: `validate_config_schema(config: dict[str, Any]) -> ConfigValidationResult` |
| `KEEP` | missing_in_heterodyne | `data.filtering_utils.XPCSDataFilter.apply_filtering` | homodyne has `apply_filtering(self, dqlist: np.ndarray, dphilist: np.ndarray, correlation_matrices: list[np.ndarray] | None = None) -> FilteringResult`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.filtering_utils.apply_data_filtering` | homodyne has `apply_data_filtering(dqlist: np.ndarray, dphilist: np.ndarray, config: dict[str, Any], correlation_matrices: list[np.ndarray] | None = None) -> np.ndarray | None`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.memory_manager.AdvancedMemoryManager.cleanup_virtual_memory` | homodyne has `cleanup_virtual_memory(self) -> None`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.memory_manager.AdvancedMemoryManager.get_memory_stats` | homodyne has `get_memory_stats(self) -> dict[str, Any]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.memory_manager.AdvancedMemoryManager.managed_allocation` | homodyne has `managed_allocation(self, size: int, dtype: np.dtype = np.float64, pool_enabled: bool = True)`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.memory_manager.AdvancedMemoryManager.optimize_for_workload` | homodyne has `optimize_for_workload(self, workload_type: str, dataset_size_gb: float) -> None`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.memory_manager.AdvancedMemoryManager.shutdown` | homodyne has `shutdown(self) -> None`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.memory_manager.MemoryPool.get_buffer` | homodyne has `get_buffer(self) -> np.ndarray | None`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.memory_manager.MemoryPool.hit_rate` | homodyne has `hit_rate(self) -> float`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.memory_manager.MemoryPool.memory_usage_mb` | homodyne has `memory_usage_mb(self) -> float`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.memory_manager.MemoryPool.return_buffer` | homodyne has `return_buffer(self, buffer: np.ndarray) -> None`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.memory_manager.MemoryPressureMonitor.get_pressure_trend` | homodyne has `get_pressure_trend(self, window_minutes: int = 5) -> str`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.memory_manager.MemoryPressureMonitor.register_critical_callback` | homodyne has `register_critical_callback(self, callback: Callable[[MemoryStats], None]) -> None`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.memory_manager.MemoryPressureMonitor.register_recovery_callback` | homodyne has `register_recovery_callback(self, callback: Callable[[MemoryStats], None]) -> None`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.memory_manager.MemoryPressureMonitor.register_warning_callback` | homodyne has `register_warning_callback(self, callback: Callable[[MemoryStats], None]) -> None`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.memory_manager.MemoryPressureMonitor.start_monitoring` | homodyne has `start_monitoring(self) -> None`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.memory_manager.MemoryPressureMonitor.stop_monitoring` | homodyne has `stop_monitoring(self) -> None`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.memory_manager.MemoryStats.get_pressure_level` | homodyne has `get_pressure_level(self) -> str`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.memory_manager.MemoryStats.update_system_stats` | homodyne has `update_system_stats(self) -> None`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.optimization.AdvancedDatasetOptimizer.cleanup` | homodyne has `cleanup(self) -> None`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.optimization.AdvancedDatasetOptimizer.get_optimization_statistics` | homodyne has `get_optimization_statistics(self) -> dict[str, Any]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.optimization.AdvancedDatasetOptimizer.optimize_massive_dataset` | homodyne has `optimize_massive_dataset(self, data: np.ndarray, sigma: np.ndarray, t1: np.ndarray, t2: np.ndarray, phi: np.ndarray, hdf_path: str | None = None, method: str = 'nlsq', **kwargs: Any) -> dict[str, Any]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.optimization.DatasetOptimizer.analyze_dataset` | homodyne has `analyze_dataset(self, data: np.ndarray, sigma: np.ndarray | None = None) -> DatasetInfo`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.optimization.DatasetOptimizer.create_chunked_iterator` | homodyne has `create_chunked_iterator(self, data: np.ndarray, sigma: np.ndarray, t1: np.ndarray, t2: np.ndarray, phi: np.ndarray, chunk_size: int) -> Iterator[tuple[np.ndarray, ...]]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.optimization.DatasetOptimizer.estimate_processing_time` | homodyne has `estimate_processing_time(self, dataset_info: DatasetInfo, method: str = 'nlsq') -> dict[str, float]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.optimization.DatasetOptimizer.get_processing_strategy` | homodyne has `get_processing_strategy(self, dataset_info: DatasetInfo, method: str = 'nlsq') -> ProcessingStrategy`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.optimization.DatasetOptimizer.optimize_for_cmc` | homodyne has `optimize_for_cmc(self, data: np.ndarray, sigma: np.ndarray, t1: np.ndarray, t2: np.ndarray, phi: np.ndarray, **kwargs: Any) -> dict[str, Any]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.optimization.DatasetOptimizer.optimize_for_nlsq` | homodyne has `optimize_for_nlsq(self, data: np.ndarray, sigma: np.ndarray, t1: np.ndarray, t2: np.ndarray, phi: np.ndarray, **kwargs: Any) -> dict[str, Any]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.optimization.create_advanced_dataset_optimizer` | homodyne has `create_advanced_dataset_optimizer(config: dict[str, Any] | None = None, **kwargs: Any) -> AdvancedDatasetOptimizer`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.optimization.create_dataset_optimizer` | homodyne has `create_dataset_optimizer(**kwargs: Any) -> DatasetOptimizer`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.optimization.optimize_for_method` | homodyne has `optimize_for_method(data: np.ndarray, sigma: np.ndarray, t1: np.ndarray, t2: np.ndarray, phi: np.ndarray, method: str = 'nlsq', **kwargs: Any) -> dict[str, Any]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.optimization.optimize_for_method_advanced` | homodyne has `optimize_for_method_advanced(data: np.ndarray, sigma: np.ndarray, t1: np.ndarray, t2: np.ndarray, phi: np.ndarray, method: str = 'nlsq', hdf_path: str | None = None, config: dict[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.performance_engine.AdaptiveChunker.calculate_optimal_chunk_size` | homodyne has `calculate_optimal_chunk_size(self, total_size: int, data_complexity: float = 1.0, available_memory_mb: float | None = None) -> int`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.performance_engine.AdaptiveChunker.create_chunk_plan` | homodyne has `create_chunk_plan(self, total_size: int, chunk_size: int) -> list[ChunkInfo]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.performance_engine.AdaptiveChunker.update_performance_feedback` | homodyne has `update_performance_feedback(self, chunk_info: ChunkInfo, actual_processing_time: float, success: bool = True) -> None`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.performance_engine.MemoryMapManager.close_all` | homodyne has `close_all(self) -> None`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.performance_engine.MemoryMapManager.open_memory_mapped_hdf5` | homodyne has `open_memory_mapped_hdf5(self, file_path: str, mode: str = 'r') -> Iterator[Any]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.performance_engine.MultiLevelCache.get` | homodyne has `get(self, key: str) -> Any | None`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.performance_engine.MultiLevelCache.get_cache_stats` | homodyne has `get_cache_stats(self) -> dict[str, Any]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.performance_engine.MultiLevelCache.put` | homodyne has `put(self, key: str, item: Any, priority: int = 5) -> None`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.performance_engine.PerformanceEngine.get_performance_report` | homodyne has `get_performance_report(self) -> dict[str, Any]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.performance_engine.PerformanceEngine.load_correlation_matrices_optimized` | homodyne has `load_correlation_matrices_optimized(self, hdf_path: str, data_keys: list[str], chunk_info: list[ChunkInfo] | None = None) -> Any`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.performance_engine.PerformanceEngine.prefetch_data` | homodyne has `prefetch_data(self, hdf_path: str, data_keys: list[str], priority: int = 5) -> Future`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.performance_engine.PerformanceEngine.shutdown` | homodyne has `shutdown(self) -> None`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.performance_engine.PerformanceMetrics.get_trend` | homodyne has `get_trend(self, metric: str, window: int = 10) -> float`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.performance_engine.PerformanceMetrics.update` | homodyne has `update(self, **kwargs: Any) -> None`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.phi_filtering.PhiAngleFilter.filter_angles_for_optimization` | homodyne has `filter_angles_for_optimization(self, phi_angles: list[float] | np.ndarray, target_ranges: list[tuple[float, float]] | None = None, fallback_enabled: bool | None = None) -> tuple[list[int], np.ndarray]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.phi_filtering.PhiAngleFilter.get_angle_statistics` | homodyne has `get_angle_statistics(self, phi_angles: list[float] | np.ndarray) -> dict[str, Any]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.phi_filtering.PhiAngleFilter.validate_target_ranges` | homodyne has `validate_target_ranges(self, target_ranges: list[tuple[float, float]]) -> bool`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.phi_filtering.create_anisotropic_ranges` | homodyne has `create_anisotropic_ranges() -> list[tuple[float, float]]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.phi_filtering.create_isotropic_ranges` | homodyne has `create_isotropic_ranges() -> list[tuple[float, float]]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.phi_filtering.filter_phi_angles` | homodyne has `filter_phi_angles(phi_angles: list[float] | np.ndarray, config: dict[str, Any] | None = None, target_ranges: list[tuple[float, float]] | None = None, fallback_enabled: bool | None = None) -> tuple[list[int], np.ndarray]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.preprocessing.PreprocessingPipeline.load_provenance` | homodyne has `load_provenance(self, filepath: str | Path) -> PreprocessingProvenance`; heterodyne missing |
| `KEEP` | changed | `data.preprocessing.PreprocessingPipeline.process` | homodyne: `process(self, data: dict[str, Any]) -> PreprocessingResult`<br>  heterodyne: `process(self, c2: np.ndarray) -> PreprocessingResult` |
| `KEEP` | missing_in_heterodyne | `data.preprocessing.PreprocessingPipeline.save_provenance` | homodyne has `save_provenance(self, provenance: PreprocessingProvenance, filepath: str | Path) -> None`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.preprocessing.create_default_preprocessing_config` | homodyne has `create_default_preprocessing_config() -> dict[str, Any]`; heterodyne missing |
| `KEEP` | changed | `data.preprocessing.preprocess_xpcs_data` | homodyne: `preprocess_xpcs_data(data: dict[str, Any], config: dict[str, Any] | None = None) -> PreprocessingResult`<br>  heterodyne: `preprocess_xpcs_data(c2: np.ndarray, normalize_method: NormalizationMethod = NormalizationMethod.DIAGONAL, noise_reduction: NoiseReductionMethod = NoiseReductionMethod.NONE, remove_outliers: bool = True, symmetrize: bool = True, baseline_correction: bool = False, **kwargs: Any) -> PreprocessingResult` |
| `KEEP` | missing_in_heterodyne | `data.quality_controller.DataQualityController.clear_cache` | homodyne has `clear_cache(self) -> None`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.quality_controller.DataQualityController.generate_quality_report` | homodyne has `generate_quality_report(self, results: list[QualityControlResult], output_path: str | None = None) -> dict[str, Any]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.quality_controller.DataQualityController.get_performance_stats` | homodyne has `get_performance_stats(self) -> dict[str, Any]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.quality_controller.DataQualityController.get_quality_history` | homodyne has `get_quality_history(self) -> list[dict[str, Any]]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.quality_controller.DataQualityController.validate_data_stage` | homodyne has `validate_data_stage(self, data: dict[str, Any], stage: QualityControlStage, previous_result: QualityControlResult | None = None) -> QualityControlResult`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.quality_controller.QualityControlConfig.from_config_dict` | homodyne has `from_config_dict(cls, config: dict[str, Any]) -> 'QualityControlConfig'`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.quality_controller.QualityControlResult.get_summary` | homodyne has `get_summary(self) -> dict[str, Any]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.quality_controller.QualityMetrics.to_dict` | homodyne has `to_dict(self) -> dict[str, Any]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.quality_controller.create_quality_controller` | homodyne has `create_quality_controller(config: dict[str, Any]) -> DataQualityController`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.quality_controller.validate_data_with_quality_control` | homodyne has `validate_data_with_quality_control(data: dict[str, Any], config: dict[str, Any], stage: QualityControlStage = QualityControlStage.FINAL_DATA) -> QualityControlResult`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.validation.DataQualityReport.add_issue` | homodyne has `add_issue(self, issue: ValidationIssue) -> None`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.validation.DataQualityReport.get_summary` | homodyne has `get_summary(self) -> dict[str, Any]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.validation.IncrementalValidationCache.is_valid_for_data` | homodyne has `is_valid_for_data(self, data: dict[str, Any], validation_level: str, max_age: float = 3600) -> bool`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.validation.clear_validation_cache` | homodyne has `clear_validation_cache() -> None`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.validation.get_cache_stats` | homodyne has `get_cache_stats() -> dict[str, Any]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.validation.validate_data_component` | homodyne has `validate_data_component(data: dict[str, Any], component_name: str, validation_level: str = 'basic', config: dict[str, Any] | None = None) -> DataQualityReport`; heterodyne missing |
| `KEEP` | changed | `data.validation.validate_xpcs_data` | homodyne: `validate_xpcs_data(data: dict[str, Any], config: dict[str, Any] | None = None, validation_level: str = 'basic') -> DataQualityReport`<br>  heterodyne: `validate_xpcs_data(data: XPCSData, expected_shape: tuple[int, ...] | None = None, min_value: float | None = None, max_value: float | None = None, check_symmetry: bool = True, check_nans: bool = True) -> DataQualityReport` |
| `KEEP` | changed | `data.validation.validate_xpcs_data_incremental` | homodyne: `validate_xpcs_data_incremental(data: dict[str, Any], config: dict[str, Any] | None = None, validation_level: str = 'basic', previous_report: DataQualityReport | None = None, force_revalidate: bool = False) -> DataQualityReport`<br>  heterodyne: `validate_xpcs_data_incremental(data: XPCSData, level: ValidationLevel = ValidationLevel.FULL, cache: IncrementalValidationCache | None = None) -> DataQualityReport` |
| `KEEP` | missing_in_heterodyne | `data.validators.validate_by_rules` | homodyne has `validate_by_rules(config: dict[str, Any], section: str, rules: dict[str, dict[str, Any]] | None = None) -> list[str]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.validators.validate_enum_value` | homodyne has `validate_enum_value(value: str | None, field_name: str, allowed_values: list[str], *, default: str | None = None) -> list[str]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.validators.validate_file_path` | homodyne has `validate_file_path(folder: str | None, filename: str | None, *, check_folder: bool = True, check_file: bool = True) -> list[str]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.validators.validate_frame_range` | homodyne has `validate_frame_range(start_frame: int | None, end_frame: int | None, *, min_frame: int = 1) -> list[str]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.validators.validate_numeric_range` | homodyne has `validate_numeric_range(range_dict: dict[str, Any] | None, field_name: str, *, require_positive: bool = False, value_bounds: tuple[float, float] | None = None, allow_wrapped: bool = False) -> list[str]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.validators.validate_positive_value` | homodyne has `validate_positive_value(value: float | int | None, field_name: str, *, allow_zero: bool = False) -> list[str]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.xpcs_loader.XPCSDataLoader.load_experimental_data` | homodyne has `load_experimental_data(self) -> dict[str, Any]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `data.xpcs_loader.load_xpcs_config` | homodyne has `load_xpcs_config(config_path: str | Path) -> dict[str, Any]`; heterodyne missing |
| `KEEP` | changed | `data.xpcs_loader.load_xpcs_data` | homodyne: `load_xpcs_data(config_path: str | dict | None = None, config_dict: dict | None = None) -> dict[str, Any]`<br>  heterodyne: `load_xpcs_data(file_path: Path | str, c2_key: str = 'c2', time_key: str = 't', format: str | None = None, use_cache: bool = False, frame_range: tuple[int, int] | None = None, select_q: float | None = None, q_tolerance: float | None = None, cache_dir: Path | None = None, cache_template: str | None = None, template_vars: dict[str, str] | None = None, cache_compression: bool = True) -> XPCSData` |
| `KEEP` | missing_in_heterodyne | `device.__init__.benchmark_device_performance` | homodyne has `benchmark_device_performance(test_size: int = 5000) -> dict[str, Any]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `device.__init__.configure_optimal_device` | homodyne has `configure_optimal_device(cpu_threads: int | None = None) -> dict[str, Any]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `device.__init__.get_device_status` | homodyne has `get_device_status() -> dict[str, Any]`; heterodyne missing |
| `KEEP` | changed | `device.cpu.benchmark_cpu_performance` | homodyne: `benchmark_cpu_performance(test_size: int = 10000, num_iterations: int = 5) -> dict[str, float]`<br>  heterodyne: `benchmark_cpu_performance(cpu_info: CPUInfo | None = None, matrix_size: int = 1000) -> dict[str, float]` |
| `KEEP` | changed | `device.cpu.configure_cpu_hpc` | homodyne: `configure_cpu_hpc(num_threads: int | None = None, enable_hyperthreading: bool = False, numa_policy: str = 'auto', memory_optimization: str = 'standard', enable_onednn: bool = False) -> dict[str, Any]`<br>  heterodyne: `configure_cpu_hpc(cpu_info: CPUInfo | None = None, use_physical_cores_only: bool = True, numa_aware: bool = True) -> dict[str, str]` |
| `KEEP` | missing_in_heterodyne | `device.cpu.configure_cpu_threading` | homodyne has `configure_cpu_threading(num_threads: int | None = None) -> dict[str, Any]`; heterodyne missing |
| `KEEP` | changed | `device.cpu.detect_cpu_info` | homodyne: `detect_cpu_info() -> dict[str, Any]`<br>  heterodyne: `detect_cpu_info() -> CPUInfo` |
| `KEEP` | changed | `device.cpu.get_optimal_batch_size` | homodyne: `get_optimal_batch_size(data_size: int, available_memory_gb: float | None = None, target_memory_usage: float = 0.7) -> int`<br>  heterodyne: `get_optimal_batch_size(cpu_info: CPUInfo | None = None, data_size: int = 1000, element_bytes: int = 8) -> int` |
| `KEEP` | changed | `io.json_utils.json_safe` | homodyne: `json_safe(value: Any) -> Any`<br>  heterodyne: `json_safe(obj: Any) -> Any` |
| `KEEP` | changed | `io.json_utils.json_serializer` | homodyne: `json_serializer(obj: Any) -> Any`<br>  heterodyne: `json_serializer(obj: Any) -> str` |
| `KEEP` | missing_in_heterodyne | `io.mcmc_writers.create_mcmc_analysis_dict` | homodyne has `create_mcmc_analysis_dict(result: Any, data: dict[str, Any], method_name: str) -> dict`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `io.mcmc_writers.create_mcmc_diagnostics_dict` | homodyne has `create_mcmc_diagnostics_dict(result: Any) -> dict`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `io.mcmc_writers.create_mcmc_parameters_dict` | homodyne has `create_mcmc_parameters_dict(result: Any) -> dict`; heterodyne missing |
| `KEEP` | changed | `io.nlsq_writers.save_nlsq_json_files` | homodyne: `save_nlsq_json_files(param_dict: dict[str, Any], analysis_dict: dict[str, Any], convergence_dict: dict[str, Any], output_dir: Path) -> None`<br>  heterodyne: `save_nlsq_json_files(result: NLSQResult, output_dir: Path | str, prefix: str = 'nlsq') -> dict[str, Path]` |
| `KEEP` | changed | `io.nlsq_writers.save_nlsq_npz_file` | homodyne: `save_nlsq_npz_file(phi_angles: np.ndarray, c2_exp: np.ndarray, c2_raw: np.ndarray, c2_scaled: np.ndarray, c2_solver: np.ndarray | None, per_angle_scaling: np.ndarray, per_angle_scaling_solver: np.ndarray, residuals: np.ndarray, residuals_norm: np.ndarray, t1: np.ndarray, t2: np.ndarray, q: float, output_dir: Path) -> None`<br>  heterodyne: `save_nlsq_npz_file(result: NLSQResult, output_path: Path | str, include_residuals: bool = True, include_jacobian: bool = False, c2_exp: np.ndarray | None = None) -> Path` |
| `KEEP` | missing_in_heterodyne | `optimization.__init__.get_optimization_info` | homodyne has `get_optimization_info() -> dict[str, Any]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.batch_statistics.BatchStatistics.get_average_iterations` | homodyne has `get_average_iterations(self) -> float`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.batch_statistics.BatchStatistics.get_average_loss` | homodyne has `get_average_loss(self) -> float`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.batch_statistics.BatchStatistics.get_statistics` | homodyne has `get_statistics(self) -> dict[str, Any]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.batch_statistics.BatchStatistics.get_success_rate` | homodyne has `get_success_rate(self) -> float`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.batch_statistics.BatchStatistics.record_batch` | homodyne has `record_batch(self, batch_idx: int, success: bool, loss: float, iterations: int, recovery_actions: list[str], error_type: str | None = None) -> None`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.checkpoint_manager.CheckpointManager.cleanup_old_checkpoints` | homodyne has `cleanup_old_checkpoints(self) -> list[Path]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.checkpoint_manager.CheckpointManager.find_latest_checkpoint` | homodyne has `find_latest_checkpoint(self) -> Path | None`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.checkpoint_manager.CheckpointManager.load_checkpoint` | homodyne has `load_checkpoint(self, checkpoint_path: Path) -> dict`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.checkpoint_manager.CheckpointManager.save_checkpoint` | homodyne has `save_checkpoint(self, batch_idx: int, parameters: np.ndarray, optimizer_state: dict, loss: float, metadata: dict | None = None) -> Path`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.checkpoint_manager.CheckpointManager.validate_checkpoint` | homodyne has `validate_checkpoint(self, checkpoint_path: Path) -> bool`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.cmc.backends.base.CMCBackend.get_name` | homodyne has `get_name(self) -> str`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.cmc.backends.base.CMCBackend.is_available` | homodyne has `is_available(self) -> bool`; heterodyne missing |
| `KEEP` | changed | `optimization.cmc.backends.base.CMCBackend.run` | homodyne: `run(self, model: Callable, model_kwargs: dict[str, Any], config: CMCConfig, shards: list[PreparedData] | None = None) -> MCMCSamples`<br>  heterodyne: `run(self, model: Callable[..., Any], config: CMCConfig, rng_key: jnp.ndarray, init_params: dict[str, jnp.ndarray] | None = None) -> dict[str, Any]` |
| `KEEP` | changed | `optimization.cmc.backends.base.combine_shard_samples` | homodyne: `combine_shard_samples(shard_samples: list[MCMCSamples], method: str = 'weighted_gaussian', chunk_size: int = 500) -> MCMCSamples`<br>  heterodyne: `combine_shard_samples(shard_samples: list[dict[str, np.ndarray]], *, method: str = 'consensus_mc', chunk_size: int = 500, seed: int = 42) -> dict[str, np.ndarray]` |
| `KEEP` | changed | `optimization.cmc.backends.base.combine_shard_samples_bimodal` | homodyne: `combine_shard_samples_bimodal(shard_samples: list[MCMCSamples], cluster_assignments: tuple[list[int], list[int]], bimodal_detections: list[dict[str, Any]], modal_params: list[str], co_occurrence: dict[str, Any], method: str = 'consensus_mc', chunk_seed: int = 0) -> tuple[MCMCSamples, BimodalConsensusResult]`<br>  heterodyne: `combine_shard_samples_bimodal(shard_samples: list[dict[str, np.ndarray]], *, cluster_param: str | None = None, method: str = 'consensus_mc', seed: int = 42) -> dict[str, dict[str, np.ndarray]]` |
| `KEEP` | changed | `optimization.cmc.backends.base.select_backend` | homodyne: `select_backend(config: CMCConfig) -> CMCBackend`<br>  heterodyne: `select_backend(config: CMCConfig) -> MCMCBackend` |
| `KEEP` | missing_in_heterodyne | `optimization.cmc.backends.multiprocessing.MultiprocessingBackend.get_name` | homodyne has `get_name(self) -> str`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.cmc.backends.multiprocessing.MultiprocessingBackend.is_available` | homodyne has `is_available(self) -> bool`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.cmc.backends.multiprocessing.MultiprocessingBackend.run` | homodyne has `run(self, model: Callable, model_kwargs: dict[str, Any], config: CMCConfig, shards: list[PreparedData] | None = None, initial_values: dict[str, float] | None = None, parameter_space: ParameterSpace | None = None, analysis_mode: str = 'static', progress_bar: bool = True) -> MCMCSamples`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.cmc.backends.multiprocessing.SharedDataManager.cleanup` | homodyne has `cleanup(self) -> None`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.cmc.backends.multiprocessing.SharedDataManager.create_shared_array` | homodyne has `create_shared_array(self, name: str, array: np.ndarray) -> dict[str, Any]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.cmc.backends.multiprocessing.SharedDataManager.create_shared_bytes` | homodyne has `create_shared_bytes(self, name: str, data: bytes) -> dict[str, Any]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.cmc.backends.multiprocessing.SharedDataManager.create_shared_dict` | homodyne has `create_shared_dict(self, name: str, d: dict) -> dict[str, Any]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.cmc.backends.multiprocessing.SharedDataManager.create_shared_shard_arrays` | homodyne has `create_shared_shard_arrays(self, shard_data_list: list[dict[str, Any]]) -> list[dict[str, Any]]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.cmc.backends.pbs.PBSBackend.get_name` | homodyne has `get_name(self) -> str`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.cmc.backends.pbs.PBSBackend.is_available` | homodyne has `is_available(self) -> bool`; heterodyne missing |
| `KEEP` | changed | `optimization.cmc.backends.pbs.PBSBackend.run` | homodyne: `run(self, model: Callable, model_kwargs: dict[str, Any], config: CMCConfig, shards: list[PreparedData] | None = None) -> MCMCSamples`<br>  heterodyne: `run(self, model: Callable[..., Any], config: CMCConfig, rng_key: jnp.ndarray, init_params: dict[str, jnp.ndarray] | None = None) -> dict[str, Any]` |
| `KEEP` | missing_in_heterodyne | `optimization.cmc.backends.pjit.PjitBackend.get_name` | homodyne has `get_name(self) -> str`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.cmc.backends.pjit.PjitBackend.is_available` | homodyne has `is_available(self) -> bool`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.cmc.backends.pjit.PjitBackend.run` | homodyne has `run(self, model: Callable, model_kwargs: dict[str, Any], config: CMCConfig, shards: list[PreparedData] | None = None, *, initial_values: dict[str, float] | None = None, parameter_space: Any | None = None, analysis_mode: str | None = None, progress_bar: bool = True) -> MCMCSamples`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.cmc.backends.worker_pool.WorkerPool.get_result` | homodyne has `get_result(self, timeout: float = 300.0) -> dict[str, Any]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.cmc.backends.worker_pool.WorkerPool.is_alive` | homodyne has `is_alive(self) -> bool`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.cmc.backends.worker_pool.WorkerPool.n_workers` | homodyne has `n_workers(self) -> int`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.cmc.backends.worker_pool.WorkerPool.result_queue` | homodyne has `result_queue(self) -> multiprocessing.Queue`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.cmc.backends.worker_pool.WorkerPool.results_pending` | homodyne has `results_pending(self) -> bool`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.cmc.backends.worker_pool.WorkerPool.shutdown` | homodyne has `shutdown(self, timeout: float = 10.0) -> None`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.cmc.backends.worker_pool.WorkerPool.submit` | homodyne has `submit(self, task: dict[str, Any]) -> None`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.cmc.backends.worker_pool.should_use_pool` | homodyne has `should_use_pool(n_shards: int, n_workers: int) -> bool`; heterodyne missing |
| `KEEP` | changed | `optimization.cmc.config.CMCConfig.get_adaptive_sample_counts` | homodyne: `get_adaptive_sample_counts(self, shard_size: int, n_params: int = 7) -> tuple[int, int]`<br>  heterodyne: `get_adaptive_sample_counts(self, shard_size: int, n_params: int = _N_PARAMS_HETERODYNE) -> tuple[int, int]` |
| `KEEP` | changed | `optimization.cmc.config.CMCConfig.get_num_shards` | homodyne: `get_num_shards(self, n_points: int, n_phi: int, n_params: int = 7) -> int`<br>  heterodyne: `get_num_shards(self, n_points: int, n_phi: int, n_params: int = _N_PARAMS_HETERODYNE) -> int` |
| `KEEP` | missing_in_heterodyne | `optimization.cmc.core.fit_mcmc_jax` | homodyne has `fit_mcmc_jax(data: np.ndarray, t1: np.ndarray, t2: np.ndarray, phi: np.ndarray, q: float, L: float, analysis_mode: str, method: str = 'mcmc', cmc_config: dict[str, Any] | None = None, initial_values: dict[str, float] | None = None, parameter_space: ParameterSpace | None = None, dt: float | None = None, output_dir: Path | str | None = None, progress_bar: bool = True, run_id: str | None = None, nlsq_result: dict | None = None, **kwargs) -> CMCResult`; heterodyne missing |
| `KEEP` | changed | `optimization.cmc.core.run_cmc_analysis` | homodyne: `run_cmc_analysis(data: np.ndarray, t1: np.ndarray, t2: np.ndarray, phi: np.ndarray, q: float, L: float, analysis_mode: str, config: CMCConfig, parameter_space: ParameterSpace, initial_values: dict[str, float] | None = None, dt: float | None = None) -> CMCResult`<br>  heterodyne: `run_cmc_analysis(model: HeterodyneModel, c2_data: np.ndarray | jnp.ndarray, config: CMCConfig | None = None, **kwargs: Any) -> CMCResult` |
| `KEEP` | missing_in_heterodyne | `optimization.cmc.data_prep.compute_data_statistics` | homodyne has `compute_data_statistics(data: np.ndarray) -> dict[str, float]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.cmc.data_prep.create_xdata_dict` | homodyne has `create_xdata_dict(prepared: PreparedData, q: float, L: float, dt: float, analysis_mode: str) -> dict[str, Any]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.cmc.data_prep.estimate_noise_scale` | homodyne has `estimate_noise_scale(data: np.ndarray) -> float`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.cmc.data_prep.extract_phi_info` | homodyne has `extract_phi_info(phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.cmc.data_prep.prepare_mcmc_data` | homodyne has `prepare_mcmc_data(data: np.ndarray, t1: np.ndarray, t2: np.ndarray, phi: np.ndarray, filter_diagonal: bool = True) -> PreparedData`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.cmc.data_prep.shard_data_angle_balanced` | homodyne has `shard_data_angle_balanced(prepared: PreparedData, num_shards: int | None = None, max_points_per_shard: int | None = None, max_shards: int = 500, min_angle_coverage: float = 0.8, seed: int = 42) -> list[PreparedData]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.cmc.data_prep.shard_data_random` | homodyne has `shard_data_random(prepared: PreparedData, num_shards: int | None = None, max_points_per_shard: int | None = None, max_shards: int = 100, seed: int = 42) -> list[PreparedData]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.cmc.data_prep.shard_data_stratified` | homodyne has `shard_data_stratified(prepared: PreparedData, num_shards: int | None = None, max_points_per_shard: int | None = None, max_shards_per_angle: int = 100, seed: int = 42) -> list[PreparedData]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.cmc.data_prep.validate_pooled_data` | homodyne has `validate_pooled_data(data: np.ndarray, t1: np.ndarray, t2: np.ndarray, phi: np.ndarray) -> None`; heterodyne missing |
| `KEEP` | changed | `optimization.cmc.diagnostics.check_shard_bimodality` | homodyne: `check_shard_bimodality(samples: dict[str, np.ndarray], params_to_check: list[str] | None = None) -> dict[str, BimodalResult]`<br>  heterodyne: `check_shard_bimodality(shard_samples: dict[int, dict[str, np.ndarray]], bic_threshold: float = 10.0, min_weight: float = 0.0, min_separation: float = 0.0) -> dict[str, list[BimodalResult]]` |
| `KEEP` | changed | `optimization.cmc.diagnostics.cluster_shard_modes` | homodyne: `cluster_shard_modes(bimodal_detections: list[dict[str, Any]], successful_samples: list[Any], bimodal_summary: dict[str, Any], param_bounds: dict[str, tuple[float, float]]) -> tuple[list[int], list[int]]`<br>  heterodyne: `cluster_shard_modes(bimodal_detections: dict[str, list[BimodalResult]], shard_samples: dict[int, dict[str, np.ndarray]], param_bounds: dict[str, tuple[float, float]] | None = None) -> tuple[list[int], list[int]]` |
| `KEEP` | changed | `optimization.cmc.diagnostics.compute_ess` | homodyne: `compute_ess(samples: dict[str, np.ndarray]) -> tuple[dict[str, float], dict[str, float]]`<br>  heterodyne: `compute_ess(samples: np.ndarray) -> float` |
| `KEEP` | changed | `optimization.cmc.diagnostics.compute_nlsq_comparison_metrics` | homodyne: `compute_nlsq_comparison_metrics(cmc_mean: float, cmc_std: float, nlsq_value: float, nlsq_std: float | None = None) -> dict[str, float]`<br>  heterodyne: `compute_nlsq_comparison_metrics(posterior_samples: dict[str, np.ndarray], nlsq_values: dict[str, float]) -> dict[str, dict[str, float]]` |
| `KEEP` | changed | `optimization.cmc.diagnostics.compute_posterior_contraction` | homodyne: `compute_posterior_contraction(posterior_std: float, prior_std: float) -> float`<br>  heterodyne: `compute_posterior_contraction(result: CMCResult, prior_std: dict[str, float]) -> dict[str, float]` |
| `KEEP` | changed | `optimization.cmc.diagnostics.compute_precision_analysis` | homodyne: `compute_precision_analysis(cmc_result: dict[str, dict], nlsq_result: dict[str, float] | None = None, nlsq_uncertainties: dict[str, float] | None = None, prior_stds: dict[str, float] | None = None) -> dict[str, dict[str, float]]`<br>  heterodyne: `compute_precision_analysis(posterior_samples: dict[str, np.ndarray]) -> dict[str, dict[str, float]]` |
| `KEEP` | changed | `optimization.cmc.diagnostics.compute_r_hat` | homodyne: `compute_r_hat(samples: dict[str, np.ndarray]) -> dict[str, float]`<br>  heterodyne: `compute_r_hat(samples: np.ndarray) -> float` |
| `KEEP` | changed | `optimization.cmc.diagnostics.detect_bimodal` | homodyne: `detect_bimodal(samples: np.ndarray, min_weight: float = 0.2, min_relative_separation: float = 0.5) -> BimodalResult`<br>  heterodyne: `detect_bimodal(samples: np.ndarray, param_name: str, bic_threshold: float = 10.0, min_weight: float = 0.0, min_separation: float = 0.0) -> BimodalResult` |
| `KEEP` | changed | `optimization.cmc.diagnostics.summarize_cross_shard_bimodality` | homodyne: `summarize_cross_shard_bimodality(bimodal_detections: list[dict[str, Any]], n_shards: int, consensus_means: dict[str, float] | None = None, significance_threshold: float = 0.05) -> dict[str, Any]`<br>  heterodyne: `summarize_cross_shard_bimodality(bimodal_detections: dict[str, list[BimodalResult]], n_shards: int, consensus_means: dict[str, float] | None = None, significance_threshold: float = 0.05) -> dict[str, Any]` |
| `KEEP` | changed | `optimization.cmc.io.samples_to_arviz` | homodyne: `samples_to_arviz(samples_data: dict[str, Any])`<br>  heterodyne: `samples_to_arviz(samples_data: dict[str, Any]) -> Any` |
| `KEEP` | changed | `optimization.cmc.model.get_model_param_count` | homodyne: `get_model_param_count(n_phi: int, analysis_mode: str, per_angle_mode: str = 'individual') -> int`<br>  heterodyne: `get_model_param_count(n_phi: int, per_angle_mode: str = 'individual') -> int` |
| `KEEP` | missing_in_heterodyne | `optimization.cmc.model.get_xpcs_model` | homodyne has `get_xpcs_model(per_angle_mode: str = 'individual', use_reparameterization: bool = False)`; heterodyne missing |
| `KEEP` | changed | `optimization.cmc.model.validate_model_output` | homodyne: `validate_model_output(c2_theory: jnp.ndarray, params: jnp.ndarray) -> bool | jnp.ndarray`<br>  heterodyne: `validate_model_output(c2_theory: jnp.ndarray, params: jnp.ndarray) -> bool` |
| `KEEP` | missing_in_heterodyne | `optimization.cmc.model.xpcs_model_averaged` | homodyne has `xpcs_model_averaged(data: jnp.ndarray, t1: jnp.ndarray, t2: jnp.ndarray, phi_unique: jnp.ndarray, phi_indices: jnp.ndarray, q: float, L: float, dt: float, analysis_mode: str, parameter_space: ParameterSpace, n_phi: int, time_grid: jnp.ndarray | None = None, noise_scale: float = 0.1, fixed_contrast: jnp.ndarray | None = None, fixed_offset: jnp.ndarray | None = None, nlsq_prior_config: dict | None = None, num_shards: int = 1, shard_grid: ShardGrid | None = None, **kwargs) -> None`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.cmc.model.xpcs_model_constant` | homodyne has `xpcs_model_constant(data: jnp.ndarray, t1: jnp.ndarray, t2: jnp.ndarray, phi_unique: jnp.ndarray, phi_indices: jnp.ndarray, q: float, L: float, dt: float, analysis_mode: str, parameter_space: ParameterSpace, n_phi: int, time_grid: jnp.ndarray | None = None, noise_scale: float = 0.1, fixed_contrast: jnp.ndarray | None = None, fixed_offset: jnp.ndarray | None = None, num_shards: int = 1, shard_grid: ShardGrid | None = None, **kwargs) -> None`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.cmc.model.xpcs_model_constant_averaged` | homodyne has `xpcs_model_constant_averaged(data: jnp.ndarray, t1: jnp.ndarray, t2: jnp.ndarray, phi_unique: jnp.ndarray, phi_indices: jnp.ndarray, q: float, L: float, dt: float, analysis_mode: str, parameter_space: ParameterSpace, n_phi: int, time_grid: jnp.ndarray | None = None, noise_scale: float = 0.1, fixed_contrast: jnp.ndarray | None = None, fixed_offset: jnp.ndarray | None = None, nlsq_prior_config: dict | None = None, num_shards: int = 1, shard_grid: ShardGrid | None = None, **kwargs) -> None`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.cmc.model.xpcs_model_reparameterized` | homodyne has `xpcs_model_reparameterized(data: jnp.ndarray, t1: jnp.ndarray, t2: jnp.ndarray, phi_unique: jnp.ndarray, phi_indices: jnp.ndarray, q: float, L: float, dt: float, analysis_mode: str, parameter_space: ParameterSpace, n_phi: int, time_grid: jnp.ndarray | None = None, noise_scale: float = 0.1, fixed_contrast: jnp.ndarray | None = None, fixed_offset: jnp.ndarray | None = None, reparam_config: ReparamConfig | None = None, nlsq_prior_config: dict | None = None, num_shards: int = 1, t_ref: float = 1.0, shard_grid: ShardGrid | None = None, **kwargs) -> None`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.cmc.model.xpcs_model_scaled` | homodyne has `xpcs_model_scaled(data: jnp.ndarray, t1: jnp.ndarray, t2: jnp.ndarray, phi_unique: jnp.ndarray, phi_indices: jnp.ndarray, q: float, L: float, dt: float, analysis_mode: str, parameter_space: ParameterSpace, n_phi: int, time_grid: jnp.ndarray | None = None, noise_scale: float = 0.1, num_shards: int = 1, shard_grid: ShardGrid | None = None, **kwargs) -> None`; heterodyne missing |
| `KEEP` | changed | `optimization.cmc.plotting.generate_diagnostic_plots` | homodyne: `generate_diagnostic_plots(result: CMCResult, output_dir: Path, figsize: tuple[int, int] = DEFAULT_FIGSIZE, dpi: int = DEFAULT_DPI, param_subset: list[str] | None = None) -> list[Path]`<br>  heterodyne: `generate_diagnostic_plots(idata: object, output_dir: Path, var_names: list[str] | None = None, dpi: int = DEFAULT_DPI) -> dict[str, Path]` |
| `KEEP` | changed | `optimization.cmc.plotting.plot_autocorr` | homodyne: `plot_autocorr(idata: az.InferenceData, output_dir: Path, var_names: list[str] | None = None, figsize: tuple[int, int] = DEFAULT_FIGSIZE, dpi: int = DEFAULT_DPI) -> Path`<br>  heterodyne: `plot_autocorr(idata: object, output_dir: Path, var_names: list[str] | None = None, figsize: tuple[int, int] = DEFAULT_FIGSIZE, dpi: int = DEFAULT_DPI) -> Path` |
| `KEEP` | changed | `optimization.cmc.plotting.plot_energy` | homodyne: `plot_energy(idata: az.InferenceData, output_dir: Path, figsize: tuple[int, int] = (10, 6), dpi: int = DEFAULT_DPI) -> Path`<br>  heterodyne: `plot_energy(idata: object, output_dir: Path, figsize: tuple[int, int] = (10, 6), dpi: int = DEFAULT_DPI) -> Path` |
| `KEEP` | changed | `optimization.cmc.plotting.plot_ess` | homodyne: `plot_ess(idata: az.InferenceData, output_dir: Path, var_names: list[str] | None = None, figsize: tuple[int, int] = (10, 6), dpi: int = DEFAULT_DPI) -> Path`<br>  heterodyne: `plot_ess(idata: object, output_dir: Path, var_names: list[str] | None = None, figsize: tuple[int, int] = (10, 6), dpi: int = DEFAULT_DPI) -> Path` |
| `KEEP` | changed | `optimization.cmc.plotting.plot_forest` | homodyne: `plot_forest(idata: az.InferenceData, output_dir: Path, var_names: list[str] | None = None, figsize: tuple[int, int] = DEFAULT_FIGSIZE, dpi: int = DEFAULT_DPI) -> Path`<br>  heterodyne: `plot_forest(idata: object, output_dir: Path, var_names: list[str] | None = None, figsize: tuple[int, int] = DEFAULT_FIGSIZE, dpi: int = DEFAULT_DPI) -> Path` |
| `KEEP` | missing_in_heterodyne | `optimization.cmc.plotting.plot_pair` | homodyne has `plot_pair(idata: az.InferenceData, output_dir: Path, var_names: list[str] | None = None, figsize: tuple[int, int] = DEFAULT_FIGSIZE, dpi: int = DEFAULT_DPI) -> Path`; heterodyne missing |
| `KEEP` | changed | `optimization.cmc.plotting.plot_rank` | homodyne: `plot_rank(idata: az.InferenceData, output_dir: Path, var_names: list[str] | None = None, figsize: tuple[int, int] = DEFAULT_FIGSIZE, dpi: int = DEFAULT_DPI) -> Path`<br>  heterodyne: `plot_rank(idata: object, output_dir: Path, var_names: list[str] | None = None, figsize: tuple[int, int] = DEFAULT_FIGSIZE, dpi: int = DEFAULT_DPI) -> Path` |
| `KEEP` | missing_in_heterodyne | `optimization.cmc.plotting.plot_trace` | homodyne has `plot_trace(idata: az.InferenceData, output_dir: Path, var_names: list[str] | None = None, figsize: tuple[int, int] = DEFAULT_FIGSIZE, dpi: int = DEFAULT_DPI) -> Path`; heterodyne missing |
| `KEEP` | changed | `optimization.cmc.priors.build_init_values_dict` | homodyne: `build_init_values_dict(n_phi: int, analysis_mode: str, initial_values: dict[str, float] | None, parameter_space: ParameterSpace, *, c2_data: np.ndarray | None = None, t1: np.ndarray | None = None, t2: np.ndarray | None = None, phi_indices: np.ndarray | None = None, per_angle_mode: str = 'individual') -> dict[str, float]`<br>  heterodyne: `build_init_values_dict(nlsq_values: dict[str, float] | None = None, vary_flags: dict[str, bool] | None = None, fallback: str = 'prior_mean') -> dict[str, float]` |
| `KEEP` | missing_in_heterodyne | `optimization.cmc.priors.build_nlsq_informed_prior` | homodyne has `build_nlsq_informed_prior(param_name: str, nlsq_value: float, nlsq_std: float | None, bounds: tuple[float, float], width_factor: float = 2.0) -> dist.Distribution`; heterodyne missing |
| `KEEP` | changed | `optimization.cmc.priors.build_nlsq_informed_priors` | homodyne: `build_nlsq_informed_priors(nlsq_result: dict[str, float], nlsq_uncertainties: dict[str, float] | None, parameter_space: ParameterSpace, analysis_mode: str, n_phi: int, width_factor: float = 2.0) -> dict[str, dist.Distribution]`<br>  heterodyne: `build_nlsq_informed_priors(nlsq_result: NLSQResult, param_space: ParameterSpace, width_factor: float = 2.0) -> dict[str, dist.Distribution]` |
| `KEEP` | missing_in_heterodyne | `optimization.cmc.priors.build_prior` | homodyne has `build_prior(param_name: str, parameter_space: ParameterSpace) -> dist.Distribution`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.cmc.priors.build_prior_from_spec` | homodyne has `build_prior_from_spec(prior_spec: PriorDistribution) -> dist.Distribution`; heterodyne missing |
| `KEEP` | changed | `optimization.cmc.priors.estimate_per_angle_scaling` | homodyne: `estimate_per_angle_scaling(c2_data: np.ndarray, t1: np.ndarray, t2: np.ndarray, phi_indices: np.ndarray, n_phi: int, contrast_bounds: tuple[float, float], offset_bounds: tuple[float, float]) -> dict[str, float]`<br>  heterodyne: `estimate_per_angle_scaling(data_dict: dict[str, Any], angle_keys: list[str] | None = None) -> dict[str, tuple[float, float]]` |
| `KEEP` | changed | `optimization.cmc.priors.extract_nlsq_values_for_cmc` | homodyne: `extract_nlsq_values_for_cmc(nlsq_result: dict | Any) -> tuple[dict[str, float], dict[str, float] | None]`<br>  heterodyne: `extract_nlsq_values_for_cmc(nlsq_result: NLSQResult) -> tuple[dict[str, float], dict[str, float] | None]` |
| `KEEP` | missing_in_heterodyne | `optimization.cmc.priors.get_init_value` | homodyne has `get_init_value(param_name: str, initial_values: dict[str, float] | None, parameter_space: ParameterSpace) -> float`; heterodyne missing |
| `KEEP` | changed | `optimization.cmc.priors.get_param_names_in_order` | homodyne: `get_param_names_in_order(n_phi: int, analysis_mode: str, per_angle_mode: str = 'individual') -> list[str]`<br>  heterodyne: `get_param_names_in_order(vary_flags: dict[str, bool] | None = None) -> list[str]` |
| `KEEP` | changed | `optimization.cmc.priors.validate_initial_value_bounds` | homodyne: `validate_initial_value_bounds(param_name: str, value: float, parameter_space: ParameterSpace) -> tuple[float, bool]`<br>  heterodyne: `validate_initial_value_bounds(init_values: dict[str, float], param_specs: dict[str, Any] | None = None) -> dict[str, list[str]]` |
| `KEEP` | missing_in_heterodyne | `optimization.cmc.reparameterization.ReparamConfig.enable_d_total` | homodyne has `enable_d_total(self) -> bool`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.cmc.reparameterization.ReparamConfig.enable_log_gamma` | homodyne has `enable_log_gamma(self) -> bool`; heterodyne missing |
| `KEEP` | changed | `optimization.cmc.reparameterization.compute_t_ref` | homodyne: `compute_t_ref(dt: float, t_max: float, *, fallback_value: float | None = None) -> float`<br>  heterodyne: `compute_t_ref(dt: float, t_max: float, fallback_value: float | None = None) -> float` |
| `KEEP` | changed | `optimization.cmc.reparameterization.transform_nlsq_to_reparam_space` | homodyne: `transform_nlsq_to_reparam_space(nlsq_values: dict[str, float], nlsq_uncertainties: dict[str, float] | None, t_ref: float) -> tuple[dict[str, float], dict[str, float]]`<br>  heterodyne: `transform_nlsq_to_reparam_space(nlsq_values: dict[str, float], nlsq_uncertainties: dict[str, float], t_ref: float, config: ReparamConfig | None = None) -> tuple[dict[str, float], dict[str, float]]` |
| `KEEP` | missing_in_heterodyne | `optimization.cmc.results.CMCResult.from_mcmc_samples` | homodyne has `from_mcmc_samples(cls, mcmc_samples: MCMCSamples, stats: SamplingStats, analysis_mode: str, n_warmup: int = 500, min_ess: float | None = None) -> CMCResult`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.cmc.results.CMCResult.is_cmc_result` | homodyne has `is_cmc_result(self) -> bool`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.cmc.results.CMCResult.message` | homodyne has `message(self) -> str`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.cmc.results.CMCResult.success` | homodyne has `success(self) -> bool`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.cmc.results.CMCResult.validate_parameters` | homodyne has `validate_parameters(self, n_phi: int | None = None) -> list[str]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.cmc.results.compute_fitted_c2` | homodyne has `compute_fitted_c2(result: CMCResult, t1: np.ndarray, t2: np.ndarray, phi: np.ndarray, q: float, L: float, dt: float, analysis_mode: str, fixed_contrasts: np.ndarray | None = None, fixed_offsets: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.cmc.results.create_inference_data` | homodyne has `create_inference_data(mcmc_samples: MCMCSamples) -> az.InferenceData`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.cmc.results.samples_dict_from_array` | homodyne has `samples_dict_from_array(samples_array: np.ndarray, param_names: list[str]) -> dict[str, np.ndarray]`; heterodyne missing |
| `KEEP` | changed | `optimization.cmc.sampler.SamplingPlan.from_config` | homodyne: `from_config(cls, config: CMCConfig, shard_size: int, n_params: int) -> SamplingPlan`<br>  heterodyne: `from_config(cls, config: CMCConfig, n_data: int | None = None, n_params: int | None = None) -> SamplingPlan` |
| `KEEP` | missing_in_heterodyne | `optimization.cmc.sampler.SamplingPlan.total_samples` | homodyne has `total_samples(self) -> int`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.cmc.sampler.create_init_strategy` | homodyne has `create_init_strategy(initial_values: dict[str, float] | None, param_names: list[str], use_init_to_value: bool = True, z_space_values: dict[str, float] | None = None) -> Callable`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.cmc.sampler.run_nuts_sampling` | homodyne has `run_nuts_sampling(model: Callable, model_kwargs: dict[str, Any], config: CMCConfig, initial_values: dict[str, float] | None, parameter_space: ParameterSpace, n_phi: int, analysis_mode: str, rng_key: jax.random.PRNGKey | None = None, progress_bar: bool = True, per_angle_mode: str = 'individual') -> tuple[MCMCSamples, SamplingStats]`; heterodyne missing |
| `KEEP` | changed | `optimization.cmc.sampler.run_nuts_with_retry` | homodyne: `run_nuts_with_retry(model: Callable, model_kwargs: dict[str, Any], config: CMCConfig, initial_values: dict[str, float] | None, parameter_space: ParameterSpace, n_phi: int, analysis_mode: str, max_retries: int = 3, rng_key: jax.random.PRNGKey | None = None, per_angle_mode: str = 'individual') -> tuple[MCMCSamples, SamplingStats]`<br>  heterodyne: `run_nuts_with_retry(sampler: NUTSSampler, model_fn: Any, model_kwargs: dict[str, Any], max_retries: int = 3, target_accept_increment: float = 0.05, *, step_size_factor: float | None = None) -> tuple[dict[str, Any], SamplingStats]` |
| `KEEP` | changed | `optimization.cmc.scaling.ParameterScaling.to_normalized` | homodyne: `to_normalized(self, value: float) -> float`<br>  heterodyne: `to_normalized(self, value: float | jnp.ndarray) -> float | jnp.ndarray` |
| `KEEP` | changed | `optimization.cmc.scaling.compute_scaling_factors` | homodyne: `compute_scaling_factors(parameter_space: ParameterSpace, n_phi: int, analysis_mode: str) -> dict[str, ParameterScaling]`<br>  heterodyne: `compute_scaling_factors(space: ParameterSpace, nlsq_values: dict[str, float] | None = None, nlsq_uncertainties: dict[str, float] | None = None, width_factor: float = 2.0) -> dict[str, ParameterScaling]` |
| `KEEP` | missing_in_heterodyne | `optimization.cmc.scaling.sample_scaled_parameter` | homodyne has `sample_scaled_parameter(name: str, scaling: ParameterScaling, initial_z: float | None = None, prior_scale: float = 1.0) -> jnp.ndarray`; heterodyne missing |
| `KEEP` | changed | `optimization.cmc.scaling.transform_initial_values_to_z` | homodyne: `transform_initial_values_to_z(initial_values: dict[str, float] | None, scalings: dict[str, ParameterScaling]) -> dict[str, float]`<br>  heterodyne: `transform_initial_values_to_z(initial_values: dict[str, float], scalings: dict[str, ParameterScaling]) -> dict[str, float]` |
| `KEEP` | changed | `optimization.gradient_diagnostics.compute_gradient_norms` | homodyne: `compute_gradient_norms(parameters: dict[str, float], data: Any, config: Any, analysis_mode: str) -> dict[str, float]`<br>  heterodyne: `compute_gradient_norms(residual_fn: Any, param_array: jnp.ndarray, param_names: list[str]) -> dict[str, float]` |
| `KEEP` | changed | `optimization.gradient_diagnostics.compute_optimal_x_scale` | homodyne: `compute_optimal_x_scale(parameters: dict[str, float], data: Any, config: Any, analysis_mode: str, baseline_params: list[str] | None = None, safety_factor: float = 1.0, min_scale: float = 1e-08, max_scale: float = 100.0) -> dict[str, float]`<br>  heterodyne: `compute_optimal_x_scale(gradient_norms: dict[str, float], baseline_params: list[str] | None = None, safety_factor: float = 1.0, min_scale: float = 1e-08, max_scale: float = 100.0) -> dict[str, float]` |
| `KEEP` | changed | `optimization.gradient_diagnostics.diagnose_gradient_imbalance` | homodyne: `diagnose_gradient_imbalance(parameters: dict[str, float], data: Any, config: Any, analysis_mode: str, threshold: float = 10.0) -> dict[str, Any]`<br>  heterodyne: `diagnose_gradient_imbalance(gradient_norms: dict[str, float], threshold: float = 10.0) -> dict[str, Any]` |
| `KEEP` | missing_in_heterodyne | `optimization.gradient_diagnostics.print_gradient_report` | homodyne has `print_gradient_report(parameters: dict[str, float], data: Any, config: Any, analysis_mode: str) -> None`; heterodyne missing |
| `KEEP` | changed | `optimization.nlsq.adapter.NLSQAdapter.fit` | homodyne: `fit(self, data: Any, config: Any, initial_params: np.ndarray | None = None, bounds: tuple[np.ndarray, np.ndarray] | None = None, analysis_mode: str = 'static_isotropic', per_angle_scaling: bool = True, diagnostics_enabled: bool = False, shear_transforms: dict[str, Any] | None = None, per_angle_scaling_initial: dict[str, list[float]] | None = None, anti_degeneracy_controller: Any | None = None) -> OptimizationResult`<br>  heterodyne: `fit(self, residual_fn: Callable[[np.ndarray], np.ndarray], initial_params: np.ndarray, bounds: tuple[np.ndarray, np.ndarray], config: NLSQConfig, jacobian_fn: Callable[[np.ndarray], np.ndarray] | None = None, *, analysis_mode: str = 'full', per_angle_scaling: bool = True, diagnostics_enabled: bool = False, per_angle_scaling_initial: dict[str, list[float]] | None = None, anti_degeneracy_controller: Any | None = None) -> NLSQResult` |
| `KEEP` | changed | `optimization.nlsq.adapter.get_or_create_model` | homodyne: `get_or_create_model(analysis_mode: str, phi_angles: np.ndarray, q: float, per_angle_scaling: bool = True, config: dict[str, Any] | None = None, enable_jit: bool = True) -> tuple[Any, Callable[[np.ndarray, Any], np.ndarray], bool]`<br>  heterodyne: `get_or_create_model(analysis_mode: str, phi_angles: np.ndarray, q: float, per_angle_scaling: bool = True, config: dict[str, Any] | None = None, enable_jit: bool = True) -> tuple[Any, Callable[..., Any] | None, bool]` |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.adaptive_regularization.AdaptiveRegularizationConfig.from_dict` | homodyne has `from_dict(cls, config_dict: dict) -> AdaptiveRegularizationConfig`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.adaptive_regularization.AdaptiveRegularizer.check_constraint_violation` | homodyne has `check_constraint_violation(self, params: np.ndarray) -> dict[str, dict]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.adaptive_regularization.AdaptiveRegularizer.compute_regularization` | homodyne has `compute_regularization(self, params: np.ndarray, mse: float, n_points: int) -> float`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.adaptive_regularization.AdaptiveRegularizer.compute_regularization_gradient` | homodyne has `compute_regularization_gradient(self, params: np.ndarray, mse: float, n_points: int) -> np.ndarray`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.adaptive_regularization.AdaptiveRegularizer.compute_regularization_jax` | homodyne has `compute_regularization_jax(self, params: jnp.ndarray, mse: jnp.ndarray, n_points: int) -> jnp.ndarray`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.adaptive_regularization.AdaptiveRegularizer.get_diagnostics` | homodyne has `get_diagnostics(self) -> dict`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.adaptive_regularization.AdaptiveRegularizer.log_summary` | homodyne has `log_summary(self, params: np.ndarray, mse: float, n_points: int) -> None`; heterodyne missing |
| `KEEP` | changed | `optimization.nlsq.anti_degeneracy_controller.AntiDegeneracyController.from_config` | homodyne: `from_config(cls, config_dict: dict[str, Any], n_phi: int, phi_angles: np.ndarray, n_physical: int, per_angle_scaling: bool = True, is_laminar_flow: bool = True) -> AntiDegeneracyController`<br>  heterodyne: `from_config(cls, config_dict: dict[str, Any], n_phi: int, phi_angles: np.ndarray, n_physical: int = 14, per_angle_scaling: bool = True) -> AntiDegeneracyController` |
| `KEEP` | changed | `optimization.nlsq.anti_degeneracy_controller.AntiDegeneracyController.get_group_variance_indices` | homodyne: `get_group_variance_indices(self) -> list[tuple[int, int]] | None`<br>  heterodyne: `get_group_variance_indices(self) -> list[tuple[int, int]]` |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.anti_degeneracy_controller.AntiDegeneracyController.transform_params_from_constant` | homodyne has `transform_params_from_constant(self, constant_params: np.ndarray) -> np.ndarray`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.anti_degeneracy_controller.AntiDegeneracyController.transform_params_from_fourier` | homodyne has `transform_params_from_fourier(self, fourier_params: np.ndarray) -> np.ndarray`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.anti_degeneracy_controller.AntiDegeneracyController.transform_params_to_constant` | homodyne has `transform_params_to_constant(self, params: np.ndarray) -> np.ndarray`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.anti_degeneracy_controller.AntiDegeneracyController.update_shear_phi0` | homodyne has `update_shear_phi0(self, params: np.ndarray, iteration: int = 0) -> None`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.cmaes_wrapper.CMAESWrapper.compute_scale_ratio` | homodyne has `compute_scale_ratio(self, bounds: tuple[np.ndarray, np.ndarray]) -> float`; heterodyne missing |
| `KEEP` | changed | `optimization.nlsq.cmaes_wrapper.CMAESWrapper.fit` | homodyne: `fit(self, model_func: Callable, xdata: np.ndarray, ydata: np.ndarray, p0: np.ndarray, bounds: tuple[np.ndarray, np.ndarray], sigma: np.ndarray | None = None, warmstart_chi2: float | None = None) -> CMAESResult`<br>  heterodyne: `fit(self, objective_fn: Callable[[np.ndarray], float], x0: np.ndarray, bounds: tuple[np.ndarray, np.ndarray], *, residual_fn: Callable[[np.ndarray], np.ndarray] | None = None, n_data: int | None = None, parameter_names: list[str] | None = None, metadata: dict[str, Any] | None = None) -> NLSQResult` |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.cmaes_wrapper.CMAESWrapper.is_available` | homodyne has `is_available(self) -> bool`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.cmaes_wrapper.CMAESWrapper.should_use_cmaes` | homodyne has `should_use_cmaes(self, bounds: tuple[np.ndarray, np.ndarray], scale_threshold: float = 1000.0) -> bool`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.cmaes_wrapper.CMAESWrapperConfig.from_nlsq_config` | homodyne has `from_nlsq_config(cls, config: NLSQConfig) -> CMAESWrapperConfig`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.cmaes_wrapper.CMAESWrapperConfig.to_cmaes_config` | homodyne has `to_cmaes_config(self, n_params: int, *, sigma_override: float | None = None) -> Any`; heterodyne missing |
| `KEEP` | changed | `optimization.nlsq.cmaes_wrapper.fit_with_cmaes` | homodyne: `fit_with_cmaes(model_func: Callable, xdata: np.ndarray, ydata: np.ndarray, p0: np.ndarray, bounds: tuple[np.ndarray, np.ndarray], sigma: np.ndarray | None = None, config: CMAESWrapperConfig | None = None) -> CMAESResult`<br>  heterodyne: `fit_with_cmaes(objective_fn: Callable[[np.ndarray], float], initial_params: np.ndarray, bounds: tuple[np.ndarray, np.ndarray], parameter_names: list[str] | None = None, *, config: CMAESConfig | None = None, residual_fn: Callable[[np.ndarray], np.ndarray] | None = None, n_data: int | None = None, anti_degeneracy: bool = False, metadata: dict[str, Any] | None = None) -> NLSQResult` |
| `KEEP` | changed | `optimization.nlsq.config.HybridRecoveryConfig.get_retry_settings` | homodyne: `get_retry_settings(self, attempt: int) -> dict`<br>  heterodyne: `get_retry_settings(self, attempt: int) -> dict[str, float]` |
| `KEEP` | changed | `optimization.nlsq.config.NLSQConfig.from_dict` | homodyne: `from_dict(cls, config_dict: dict[str, Any]) -> NLSQConfig`<br>  heterodyne: `from_dict(cls, config: dict[str, Any]) -> NLSQConfig` |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.config.NLSQConfig.from_yaml` | homodyne has `from_yaml(cls, yaml_path: str) -> NLSQConfig`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.config.NLSQConfig.is_valid` | homodyne has `is_valid(self) -> bool`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.config.NLSQConfig.to_workflow_kwargs` | homodyne has `to_workflow_kwargs(self) -> dict[str, Any]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.core.fit_nlsq_cmaes` | homodyne has `fit_nlsq_cmaes(data: dict[str, Any], config: ConfigManager, initial_params: dict[str, float] | None = None, per_angle_scaling: bool = True) -> OptimizationResult`; heterodyne missing |
| `KEEP` | changed | `optimization.nlsq.core.fit_nlsq_jax` | homodyne: `fit_nlsq_jax(data: dict[str, Any], config: ConfigManager, initial_params: dict[str, float] | None = None, per_angle_scaling: bool = True, use_adapter: bool = False, _skip_global_selection: bool = False) -> OptimizationResult`<br>  heterodyne: `fit_nlsq_jax(model: HeterodyneModel, c2_data: np.ndarray | jnp.ndarray, phi_angle: float = 0.0, config: NLSQConfig | None = None, weights: np.ndarray | jnp.ndarray | None = None, use_nlsq_library: bool = True, *, _skip_global_selection: bool = False) -> NLSQResult` |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.core.fit_nlsq_multistart` | homodyne has `fit_nlsq_multistart(data: dict[str, Any], config: ConfigManager, initial_params: dict[str, float] | None = None, per_angle_scaling: bool = True) -> MultiStartResult`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.data_prep.build_parameter_labels` | homodyne has `build_parameter_labels(per_angle_scaling: bool, n_phi: int, physical_param_names: list[str]) -> list[str]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.data_prep.classify_parameter_status` | homodyne has `classify_parameter_status(values: np.ndarray, lower: np.ndarray | None, upper: np.ndarray | None, atol: float = 1e-09) -> list[str]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.data_prep.convert_bounds_to_nlsq_format` | homodyne has `convert_bounds_to_nlsq_format(bounds: tuple[np.ndarray, np.ndarray] | tuple[list, list] | None) -> tuple[np.ndarray, np.ndarray] | None`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.data_prep.expand_per_angle_parameters` | homodyne has `expand_per_angle_parameters(compact_params: np.ndarray, compact_bounds: tuple[np.ndarray, np.ndarray] | None, n_angles: int, n_physical: int, logger: Any = None) -> ExpandedParameters`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.data_prep.validate_bounds` | homodyne has `validate_bounds(bounds: tuple[np.ndarray, np.ndarray] | None, n_params: int, logger: Any = None) -> tuple[np.ndarray, np.ndarray] | None`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.data_prep.validate_initial_params` | homodyne has `validate_initial_params(params: np.ndarray, bounds: tuple[np.ndarray, np.ndarray] | None, logger: Any = None) -> np.ndarray`; heterodyne missing |
| `KEEP` | changed | `optimization.nlsq.fallback_chain.execute_optimization_with_fallback` | homodyne: `execute_optimization_with_fallback(strategy: OptimizationStrategy, wrapped_residual_fn: Callable[..., np.ndarray], xdata: np.ndarray, ydata: np.ndarray, validated_params: np.ndarray, nlsq_bounds: tuple[np.ndarray, np.ndarray] | None, loss_name: str, x_scale_value: float | str, config: Any, start_time: float, log: logging.Logger | logging.LoggerAdapter[logging.Logger], enable_recovery: bool, execute_with_recovery_fn: Callable, fit_with_hybrid_streaming_fn: Callable, streaming_available: bool, curve_fit_fn: Callable, curve_fit_large_fn: Callable, fast_mode: bool = False) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any], list[str], str]`<br>  heterodyne: `execute_optimization_with_fallback(model: HeterodyneModel, c2_data: np.ndarray, phi_angle: float, config: NLSQConfig, *, start_strategy: OptimizationStrategy | None = None, weights: np.ndarray | None = None) -> NLSQResult` |
| `KEEP` | changed | `optimization.nlsq.fallback_chain.get_fallback_strategy` | homodyne: `get_fallback_strategy(current_strategy: OptimizationStrategy) -> OptimizationStrategy | None`<br>  heterodyne: `get_fallback_strategy(current: OptimizationStrategy, error: Exception | None = None) -> OptimizationStrategy | None` |
| `KEEP` | changed | `optimization.nlsq.fallback_chain.handle_nlsq_result` | homodyne: `handle_nlsq_result(result: Any, strategy: OptimizationStrategy) -> tuple[np.ndarray, np.ndarray, dict]`<br>  heterodyne: `handle_nlsq_result(raw_result: Any) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]` |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.fit_computation.compute_g2_batch` | homodyne has `compute_g2_batch(physical_params: jnp.ndarray, t1: jnp.ndarray, t2: jnp.ndarray, phi_angles: jnp.ndarray, q: float, L: float, dt: float, contrast: float = 1.0, offset: float = 1.0) -> jnp.ndarray`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.fit_computation.compute_g2_batch_with_per_angle_scaling` | homodyne has `compute_g2_batch_with_per_angle_scaling(physical_params: jnp.ndarray, t1: jnp.ndarray, t2: jnp.ndarray, phi_angles: jnp.ndarray, q: float, L: float, dt: float, contrasts: jnp.ndarray, offsets: jnp.ndarray) -> jnp.ndarray`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.fourier_reparam.create_fourier_model_wrapper` | homodyne has `create_fourier_model_wrapper(model_fn: Callable[[np.ndarray, np.ndarray], np.ndarray], fourier: FourierReparameterizer, n_physical: int) -> Callable[[np.ndarray, np.ndarray], np.ndarray]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.gradient_monitor.GradientCollapseMonitor.check` | homodyne has `check(self, gradients: np.ndarray, iteration: int, params: np.ndarray | None = None, loss: float | None = None) -> str`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.gradient_monitor.GradientCollapseMonitor.compute_reset_params` | homodyne has `compute_reset_params(self, params: np.ndarray, n_phi: int) -> np.ndarray`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.gradient_monitor.GradientCollapseMonitor.get_diagnostics` | homodyne has `get_diagnostics(self) -> dict`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.gradient_monitor.GradientCollapseMonitor.get_response` | homodyne has `get_response(self) -> dict | None`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.gradient_monitor.GradientCollapseMonitor.log_summary` | homodyne has `log_summary(self) -> None`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.gradient_monitor.GradientCollapseMonitor.reset` | homodyne has `reset(self) -> None`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.gradient_monitor.GradientMonitorConfig.from_dict` | homodyne has `from_dict(cls, config_dict: dict) -> GradientMonitorConfig`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.gradient_monitor.create_gradient_function_with_monitoring` | homodyne has `create_gradient_function_with_monitoring(grad_fn: Callable[[np.ndarray], np.ndarray], monitor: GradientCollapseMonitor) -> Callable[[np.ndarray], np.ndarray]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.hierarchical.HierarchicalConfig.from_dict` | homodyne has `from_dict(cls, config_dict: dict) -> HierarchicalConfig`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.hierarchical.HierarchicalOptimizer.fit` | homodyne has `fit(self, loss_fn: Callable[[np.ndarray], float], grad_fn: Callable[[np.ndarray], np.ndarray] | None, p0: np.ndarray, bounds: tuple[np.ndarray, np.ndarray], outer_iteration_callback: Callable[[np.ndarray, int], None] | None = None) -> HierarchicalResult`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.hierarchical.HierarchicalOptimizer.get_diagnostics` | homodyne has `get_diagnostics(self) -> dict`; heterodyne missing |
| `KEEP` | changed | `optimization.nlsq.jacobian.analyze_parameter_sensitivity` | homodyne: `analyze_parameter_sensitivity(residual_fn: Callable[..., Any], x_subset: np.ndarray, params: np.ndarray, param_names: list[str]) -> dict[str, float]`<br>  heterodyne: `analyze_parameter_sensitivity(jacobian: np.ndarray, param_names: list[str]) -> dict[str, float]` |
| `KEEP` | changed | `optimization.nlsq.jacobian.compute_jacobian_condition_number` | homodyne: `compute_jacobian_condition_number(residual_fn: Callable[..., Any], x_subset: np.ndarray, params: np.ndarray) -> float | None`<br>  heterodyne: `compute_jacobian_condition_number(jacobian: np.ndarray) -> float` |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.jacobian.compute_jacobian_stats` | homodyne has `compute_jacobian_stats(residual_fn: Callable[..., Any], x_subset: np.ndarray, params: np.ndarray, scaling_factor: float) -> tuple[np.ndarray | None, np.ndarray | None]`; heterodyne missing |
| `KEEP` | changed | `optimization.nlsq.jacobian.estimate_gradient_noise` | homodyne: `estimate_gradient_noise(residual_fn: Callable[..., Any], x_subset: np.ndarray, params: np.ndarray, n_samples: int = 5, perturbation: float = 1e-06, seed: int = 42) -> float | None`<br>  heterodyne: `estimate_gradient_noise(jacobian: np.ndarray, residuals: np.ndarray) -> dict[str, float]` |
| `KEEP` | changed | `optimization.nlsq.memory.estimate_peak_memory_gb` | homodyne: `estimate_peak_memory_gb(n_points: int, n_params: int, bytes_per_element: int = 8, jacobian_overhead: float = 6.5) -> float`<br>  heterodyne: `estimate_peak_memory_gb(n_points: int, n_params: int, *, bytes_per_element: int = 8, jacobian_overhead: float = _JACOBIAN_OVERHEAD) -> float` |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.memory.get_adaptive_memory_threshold` | homodyne has `get_adaptive_memory_threshold(memory_fraction: float | None = None) -> tuple[float, dict[str, Any]]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.multistart.MultiStartConfig.from_nlsq_config` | homodyne has `from_nlsq_config(cls, nlsq_config: Any) -> MultiStartConfig`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.multistart.MultiStartConfig.to_nlsq_global_config` | homodyne has `to_nlsq_global_config(self) -> Any`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.multistart.MultiStartResult.to_optimization_result` | homodyne has `to_optimization_result(self) -> OptimizationResult`; heterodyne missing |
| `KEEP` | changed | `optimization.nlsq.multistart.check_zero_volume_bounds` | homodyne: `check_zero_volume_bounds(bounds: NDArray[np.float64]) -> bool`<br>  heterodyne: `check_zero_volume_bounds(bounds_lower: np.ndarray, bounds_upper: np.ndarray) -> list[int]` |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.multistart.detect_degeneracy` | homodyne has `detect_degeneracy(results: list[SingleStartResult], chi_sq_threshold: float = 0.1, param_threshold: float = 0.2) -> tuple[bool, int, NDArray[np.int64] | None]`; heterodyne missing |
| `KEEP` | changed | `optimization.nlsq.multistart.generate_lhs_starts` | homodyne: `generate_lhs_starts(bounds: NDArray[np.float64], n_starts: int, seed: int = 42) -> NDArray[np.float64]`<br>  heterodyne: `generate_lhs_starts(n_starts: int, bounds_lower: np.ndarray, bounds_upper: np.ndarray, seed: int = 42) -> np.ndarray` |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.multistart.generate_random_starts` | homodyne has `generate_random_starts(bounds: NDArray[np.float64], n_starts: int, seed: int = 42) -> NDArray[np.float64]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.multistart.get_n_workers` | homodyne has `get_n_workers(config: MultiStartConfig, n_starts: int) -> int`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.multistart.include_custom_starts` | homodyne has `include_custom_starts(generated_starts: NDArray[np.float64], custom_starts: list[list[float]] | NDArray[np.float64] | None, bounds: NDArray[np.float64]) -> NDArray[np.float64]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.multistart.run_multistart_nlsq` | homodyne has `run_multistart_nlsq(data: dict[str, Any], bounds: NDArray[np.float64], config: MultiStartConfig, single_fit_func: Callable[[dict[str, Any], NDArray[np.float64]], SingleStartResult], cost_func: Callable[[NDArray[np.float64]], float] | None = None, custom_starts: list[list[float]] | NDArray[np.float64] | None = None) -> MultiStartResult`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.multistart.screen_starts` | homodyne has `screen_starts(cost_func: Callable[[NDArray[np.float64]], float], starts: NDArray[np.float64], keep_fraction: float = 0.5, min_keep: int = 3, n_workers: int = 0) -> tuple[NDArray[np.float64], NDArray[np.float64]]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.multistart.validate_n_starts_for_lhs` | homodyne has `validate_n_starts_for_lhs(n_starts: int, n_params: int, warn: bool = True) -> int`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.parallel_accumulator.OOCComputePool.compute_accumulators` | homodyne has `compute_accumulators(self, params: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, float]]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.parallel_accumulator.OOCComputePool.compute_chi2` | homodyne has `compute_chi2(self, params: np.ndarray, stride: int = 1) -> float`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.parallel_accumulator.OOCComputePool.shutdown` | homodyne has `shutdown(self) -> None`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.parallel_accumulator.OOCSharedArrays.cleanup` | homodyne has `cleanup(self) -> None`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.parallel_accumulator.OOCSharedArrays.get_refs` | homodyne has `get_refs(self) -> dict[str, dict[str, Any]]`; heterodyne missing |
| `KEEP` | changed | `optimization.nlsq.parallel_accumulator.accumulate_chunks_parallel` | homodyne: `accumulate_chunks_parallel(chunks: list[tuple[np.ndarray, np.ndarray, float]], n_workers: int = 4) -> tuple[np.ndarray, np.ndarray, float, int]`<br>  heterodyne: `accumulate_chunks_parallel(chunks: list[tuple[np.ndarray, np.ndarray]], residual_fn: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]], params: np.ndarray, n_workers: int = 2) -> GaussNewtonAccumulation` |
| `KEEP` | changed | `optimization.nlsq.parallel_accumulator.accumulate_chunks_sequential` | homodyne: `accumulate_chunks_sequential(chunks: list[tuple[np.ndarray, np.ndarray, float]]) -> tuple[np.ndarray, np.ndarray, float, int]`<br>  heterodyne: `accumulate_chunks_sequential(chunks: list[tuple[np.ndarray, np.ndarray]], residual_fn: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]], params: np.ndarray) -> GaussNewtonAccumulation` |
| `KEEP` | changed | `optimization.nlsq.parallel_accumulator.create_ooc_kernels` | homodyne: `create_ooc_kernels(per_angle_scaling: bool, n_phi: int, phi_unique: Any, t1_unique_global: Any, t2_unique_global: Any, n_t1: int, n_t2: int, q_val: float, L_val: float, dt_val: float) -> tuple[Callable, Callable]`<br>  heterodyne: `create_ooc_kernels(n_params: int = 14) -> tuple[Callable[..., Any], Callable[..., Any]]` |
| `KEEP` | changed | `optimization.nlsq.parallel_accumulator.should_use_parallel_accumulation` | homodyne: `should_use_parallel_accumulation(n_chunks: int) -> bool`<br>  heterodyne: `should_use_parallel_accumulation(n_chunks: int, threshold: int = 10) -> bool` |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.parallel_accumulator.should_use_parallel_compute` | homodyne has `should_use_parallel_compute(n_chunks: int) -> bool`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.parameter_index_mapper.ParameterIndexMapper.get_covariance_slice_indices` | homodyne has `get_covariance_slice_indices(self) -> tuple[slice, slice]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.parameter_index_mapper.ParameterIndexMapper.get_diagnostics` | homodyne has `get_diagnostics(self) -> dict`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.parameter_index_mapper.ParameterIndexMapper.get_group_indices` | homodyne has `get_group_indices(self) -> list[tuple[int, int]]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.parameter_index_mapper.ParameterIndexMapper.get_per_angle_indices` | homodyne has `get_per_angle_indices(self) -> list[int]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.parameter_index_mapper.ParameterIndexMapper.get_physical_indices` | homodyne has `get_physical_indices(self) -> list[int]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.parameter_index_mapper.ParameterIndexMapper.mode_name` | homodyne has `mode_name(self) -> str`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.parameter_index_mapper.ParameterIndexMapper.n_per_angle_total` | homodyne has `n_per_angle_total(self) -> int`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.parameter_index_mapper.ParameterIndexMapper.n_per_group` | homodyne has `n_per_group(self) -> int`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.parameter_index_mapper.ParameterIndexMapper.total_params` | homodyne has `total_params(self) -> int`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.parameter_index_mapper.ParameterIndexMapper.use_fourier` | homodyne has `use_fourier(self) -> bool`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.parameter_index_mapper.ParameterIndexMapper.validate_indices` | homodyne has `validate_indices(self, params: np.ndarray) -> bool`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.parameter_utils.build_parameter_labels` | homodyne has `build_parameter_labels(per_angle_scaling: bool, n_phi: int, physical_param_names: list[str]) -> list[str]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.parameter_utils.classify_parameter_status` | homodyne has `classify_parameter_status(values: np.ndarray, lower: np.ndarray | None, upper: np.ndarray | None, atol: float = 1e-09) -> list[str]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.parameter_utils.compute_consistent_per_angle_init` | homodyne has `compute_consistent_per_angle_init(stratified_data: Any, physical_params: np.ndarray, physical_param_names: list[str], default_contrast: float = 0.5, default_offset: float = 1.0, logger: Any = None) -> tuple[np.ndarray, np.ndarray]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.parameter_utils.compute_jacobian_stats` | homodyne has `compute_jacobian_stats(residual_fn: Callable[..., Any], x_subset: np.ndarray, params: np.ndarray, scaling_factor: float) -> tuple[np.ndarray | None, np.ndarray | None]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.parameter_utils.compute_quantile_per_angle_scaling` | homodyne has `compute_quantile_per_angle_scaling(stratified_data: Any, contrast_bounds: tuple[float, float] = (0.0, 1.0), offset_bounds: tuple[float, float] = (0.5, 1.5), lag_floor_quantile: float = 0.8, lag_ceiling_quantile: float = 0.2, value_quantile_low: float = 0.1, value_quantile_high: float = 0.9, logger: Any = None) -> tuple[np.ndarray, np.ndarray]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.parameter_utils.sample_xdata` | homodyne has `sample_xdata(xdata: np.ndarray, max_points: int) -> np.ndarray`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.progress.HomodyneIterationLogger.close` | homodyne has `close(self) -> None`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.progress.MultiStartProgressTracker.close` | homodyne has `close(self) -> None`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.progress.MultiStartProgressTracker.update` | homodyne has `update(self, start_idx: int, success: bool, chi_squared: float, message: str = '', wall_time: float | None = None) -> None`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.progress.ProgressConfig.from_nlsq_config` | homodyne has `from_nlsq_config(cls, nlsq_config: NLSQConfig, max_nfev: int | None = None, description: str = 'NLSQ Fitting') -> ProgressConfig`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.progress.create_progress_callback` | homodyne has `create_progress_callback(config: ProgressConfig | None = None, enable_progress_bar: bool = True, verbose: int = 1, log_interval: int = 10, max_nfev: int = 1000, description: str = 'NLSQ Fitting') -> tuple[CallbackBase | None, HomodyneIterationLogger | None]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.progress.create_streaming_progress_callback` | homodyne has `create_streaming_progress_callback(n_total_points: int, batch_size: int, max_epochs: int, enable_progress_bar: bool = True, verbose: int = 1) -> Callable[[int, np.ndarray, float], bool] | None`; heterodyne missing |
| `KEEP` | changed | `optimization.nlsq.recovery.diagnose_error` | homodyne: `diagnose_error(error: Exception, params: np.ndarray, bounds: tuple[np.ndarray, np.ndarray] | None, attempt: int) -> dict[str, Any]`<br>  heterodyne: `diagnose_error(error: Exception) -> ErrorDiagnosis` |
| `KEEP` | changed | `optimization.nlsq.recovery.execute_with_recovery` | homodyne: `execute_with_recovery(residual_fn: Callable[[np.ndarray], np.ndarray], xdata: np.ndarray, ydata: np.ndarray, initial_params: np.ndarray, bounds: tuple[np.ndarray, np.ndarray] | None, strategy: OptimizationStrategy, log: logging.Logger | logging.LoggerAdapter[logging.Logger], loss_name: str, x_scale_value: float | str | np.ndarray, handle_nlsq_result_fn: Callable, curve_fit_fn: Callable, curve_fit_large_fn: Callable) -> tuple[np.ndarray, np.ndarray, dict, list[str], str]`<br>  heterodyne: `execute_with_recovery(fit_fn: Callable[[np.ndarray, tuple[np.ndarray, np.ndarray], NLSQConfig], NLSQResult], initial_params: np.ndarray, bounds: tuple[np.ndarray, np.ndarray], config: NLSQConfig, *, max_retries: int = 3, perturb_scale: float = 0.1, rng_seed: int = 42) -> NLSQResult` |
| `KEEP` | changed | `optimization.nlsq.recovery.safe_uncertainties_from_pcov` | homodyne: `safe_uncertainties_from_pcov(pcov: np.ndarray, n_params: int) -> np.ndarray`<br>  heterodyne: `safe_uncertainties_from_pcov(pcov: np.ndarray | None, n_params: int = 14) -> np.ndarray` |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.result_builder.ResultBuilder.build` | homodyne has `build(self, residual_fn: Any = None, xdata: np.ndarray | None = None) -> dict[str, Any]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.result_builder.ResultBuilder.with_covariance` | homodyne has `with_covariance(self, cov: np.ndarray) -> ResultBuilder`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.result_builder.ResultBuilder.with_data_size` | homodyne has `with_data_size(self, n_data: int) -> ResultBuilder`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.result_builder.ResultBuilder.with_fourier_covariance_transform` | homodyne has `with_fourier_covariance_transform(self, fourier_reparameterizer: Any, n_phi: int, n_physical: int) -> ResultBuilder`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.result_builder.ResultBuilder.with_info` | homodyne has `with_info(self, info: dict[str, Any]) -> ResultBuilder`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.result_builder.ResultBuilder.with_nlsq_diagnostics` | homodyne has `with_nlsq_diagnostics(self, diags: dict[str, Any]) -> ResultBuilder`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.result_builder.ResultBuilder.with_parameters` | homodyne has `with_parameters(self, params: np.ndarray) -> ResultBuilder`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.result_builder.ResultBuilder.with_recovery_actions` | homodyne has `with_recovery_actions(self, actions: list[str]) -> ResultBuilder`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.result_builder.ResultBuilder.with_start_time` | homodyne has `with_start_time(self, start_time: float) -> ResultBuilder`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.result_builder.ResultBuilder.with_stratification_diagnostics` | homodyne has `with_stratification_diagnostics(self, diags: Any) -> ResultBuilder`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.result_builder.compute_quality_metrics` | homodyne has `compute_quality_metrics(residuals: np.ndarray, n_data: int, n_params: int, parameter_status: list[str] | None = None) -> QualityMetrics`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.result_builder.compute_uncertainties` | homodyne has `compute_uncertainties(covariance: np.ndarray) -> np.ndarray`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.result_builder.determine_convergence_status` | homodyne has `determine_convergence_status(info: dict[str, Any], quality_metrics: QualityMetrics) -> str`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.result_builder.normalize_nlsq_result` | homodyne has `normalize_nlsq_result(result: Any, strategy_name: str = 'unknown', logger: Any = None) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.results.FallbackInfo.to_dict` | homodyne has `to_dict(self) -> dict[str, Any]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.results.OptimizationResult.message` | homodyne has `message(self) -> str`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.results.OptimizationResult.success` | homodyne has `success(self) -> bool`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.shear_weighting.ShearSensitivityWeighting.apply_weights_to_loss` | homodyne has `apply_weights_to_loss(self, residuals: Array, phi_indices: Array) -> Array`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.shear_weighting.ShearSensitivityWeighting.compute_weighted_mse` | homodyne has `compute_weighted_mse(self, residuals: Array, phi_indices: Array) -> Array`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.shear_weighting.ShearSensitivityWeighting.get_diagnostics` | homodyne has `get_diagnostics(self) -> dict`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.shear_weighting.ShearSensitivityWeighting.get_weights` | homodyne has `get_weights(self, phi0_current: float | None = None) -> np.ndarray`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.shear_weighting.ShearSensitivityWeighting.get_weights_jax` | homodyne has `get_weights_jax(self) -> Array`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.shear_weighting.ShearSensitivityWeighting.phi0_current` | homodyne has `phi0_current(self) -> float`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.shear_weighting.ShearSensitivityWeighting.update_phi0` | homodyne has `update_phi0(self, params: np.ndarray, iteration: int = 0) -> None`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.shear_weighting.ShearWeightingConfig.from_config` | homodyne has `from_config(cls, config: Mapping) -> ShearWeightingConfig`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.shear_weighting.create_shear_weighting` | homodyne has `create_shear_weighting(phi_angles: np.ndarray, n_physical: int, config: Mapping | None = None, physical_param_names: list[str] | None = None) -> ShearSensitivityWeighting | None`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.strategies.chunking.analyze_angle_distribution` | homodyne has `analyze_angle_distribution(phi: jnp.ndarray | np.ndarray) -> AngleDistributionStats`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.strategies.chunking.calculate_adaptive_chunk_size` | homodyne has `calculate_adaptive_chunk_size(total_points: int, n_params: int, n_angles: int, available_memory_gb: float | None = None, safety_factor: float = 5.0, min_chunk_size: int = 10000, max_chunk_size: int = 500000) -> int`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.strategies.chunking.compute_stratification_diagnostics` | homodyne has `compute_stratification_diagnostics(phi_original: np.ndarray, phi_stratified: np.ndarray, execution_time_ms: float, use_index_based: bool = False, target_chunk_size: int = 100000, chunk_sizes: list[int] | None = None) -> StratificationDiagnostics`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.strategies.chunking.create_angle_stratified_data` | homodyne has `create_angle_stratified_data(phi: jnp.ndarray, t1: jnp.ndarray, t2: jnp.ndarray, g2_exp: jnp.ndarray, target_chunk_size: int = 100000) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, list[int]]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.strategies.chunking.create_angle_stratified_indices` | homodyne has `create_angle_stratified_indices(phi: jnp.ndarray | np.ndarray, target_chunk_size: int = 100000) -> tuple[np.ndarray, list[int]]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.strategies.chunking.estimate_nlsq_optimization_memory` | homodyne has `estimate_nlsq_optimization_memory(n_points: int, n_params: int, n_features: int = 4, dtype_bytes: int = 8) -> dict[str, Any]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.strategies.chunking.estimate_stratification_memory` | homodyne has `estimate_stratification_memory(n_points: int, n_features: int = 4, use_index_based: bool = False, estimated_expansion: float = 1.0) -> dict[str, Any]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.strategies.chunking.format_diagnostics_report` | homodyne has `format_diagnostics_report(diagnostics: StratificationDiagnostics) -> str`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.strategies.chunking.get_stratified_chunk_iterator` | homodyne has `get_stratified_chunk_iterator(phi: jnp.ndarray | np.ndarray, target_chunk_size: int = 100000) -> StratifiedIndexIterator`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.strategies.chunking.should_use_stratification` | homodyne has `should_use_stratification(n_points: int, n_angles: int, per_angle_scaling: bool, imbalance_ratio: float) -> tuple[bool, str]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.strategies.executors.LargeDatasetExecutor.execute` | homodyne has `execute(self, residual_fn: Callable[..., Any], xdata: np.ndarray, ydata: np.ndarray, initial_params: np.ndarray, bounds: tuple[np.ndarray, np.ndarray] | None, loss_name: str, x_scale_value: float | np.ndarray | str, logger: Any) -> ExecutionResult`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.strategies.executors.LargeDatasetExecutor.name` | homodyne has `name(self) -> str`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.strategies.executors.LargeDatasetExecutor.supports_progress` | homodyne has `supports_progress(self) -> bool`; heterodyne missing |
| `KEEP` | changed | `optimization.nlsq.strategies.executors.OptimizationExecutor.execute` | homodyne: `execute(self, residual_fn: Callable[..., Any], xdata: np.ndarray, ydata: np.ndarray, initial_params: np.ndarray, bounds: tuple[np.ndarray, np.ndarray] | None, loss_name: str, x_scale_value: float | np.ndarray | str, logger: Any) -> ExecutionResult`<br>  heterodyne: `execute(self, model: HeterodyneModel, c2_data: np.ndarray, phi_angle: float, config: NLSQConfig, weights: np.ndarray | None = None) -> ExecutionResult` |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.strategies.executors.OptimizationExecutor.supports_progress` | homodyne has `supports_progress(self) -> bool`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.strategies.executors.StandardExecutor.execute` | homodyne has `execute(self, residual_fn: Callable[..., Any], xdata: np.ndarray, ydata: np.ndarray, initial_params: np.ndarray, bounds: tuple[np.ndarray, np.ndarray] | None, loss_name: str, x_scale_value: float | np.ndarray | str, logger: Any) -> ExecutionResult`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.strategies.executors.StandardExecutor.name` | homodyne has `name(self) -> str`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.strategies.executors.StandardExecutor.supports_progress` | homodyne has `supports_progress(self) -> bool`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.strategies.executors.StreamingExecutor.execute` | homodyne has `execute(self, residual_fn: Callable[..., Any], xdata: np.ndarray, ydata: np.ndarray, initial_params: np.ndarray, bounds: tuple[np.ndarray, np.ndarray] | None, loss_name: str, x_scale_value: float | np.ndarray | str, logger: Any) -> ExecutionResult`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.strategies.executors.StreamingExecutor.name` | homodyne has `name(self) -> str`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.strategies.executors.StreamingExecutor.supports_progress` | homodyne has `supports_progress(self) -> bool`; heterodyne missing |
| `KEEP` | changed | `optimization.nlsq.strategies.executors.get_executor` | homodyne: `get_executor(strategy_name: str, checkpoint_config: dict[str, Any] | None = None) -> OptimizationExecutor`<br>  heterodyne: `get_executor(strategy_name: str | None = None, n_data: int = 0, **kwargs: Any) -> OptimizationExecutor` |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.strategies.hybrid_streaming.estimate_memory_for_stratified_ls` | homodyne has `estimate_memory_for_stratified_ls(n_points: int, n_params: int, n_chunks: int) -> float`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.strategies.hybrid_streaming.fit_with_hybrid_streaming_optimizer` | homodyne has `fit_with_hybrid_streaming_optimizer(residual_fn: Any, xdata: np.ndarray, ydata: np.ndarray, initial_params: np.ndarray, bounds: tuple[np.ndarray, np.ndarray] | None, logger: Any, nlsq_config: Any = None, fast_mode: bool = False) -> tuple[np.ndarray, np.ndarray, dict]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.strategies.hybrid_streaming.fit_with_stratified_hybrid_streaming` | homodyne has `fit_with_stratified_hybrid_streaming(stratified_data: Any, per_angle_scaling: bool, physical_param_names: list[str], initial_params: np.ndarray, bounds: tuple[np.ndarray, np.ndarray] | None, logger: Any, hybrid_config: dict | None = None, anti_degeneracy_config: dict | None = None) -> tuple[np.ndarray, np.ndarray, dict]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.strategies.hybrid_streaming.fit_with_streaming_optimizer_deprecated` | homodyne has `fit_with_streaming_optimizer_deprecated(residual_fn: Any, xdata: np.ndarray, ydata: np.ndarray, initial_params: np.ndarray, bounds: tuple[np.ndarray, np.ndarray] | None, logger: Any, checkpoint_config: dict | None = None) -> tuple[np.ndarray, np.ndarray, dict]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.strategies.hybrid_streaming.fit_with_streaming_optimizer_stratified_deprecated` | homodyne has `fit_with_streaming_optimizer_stratified_deprecated(stratified_data: Any, per_angle_scaling: bool, physical_param_names: list[str], initial_params: np.ndarray, bounds: tuple[np.ndarray, np.ndarray] | None, logger: Any, streaming_config: dict | None = None) -> tuple[np.ndarray, np.ndarray, dict]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.strategies.hybrid_streaming.should_use_streaming` | homodyne has `should_use_streaming(n_points: int, n_params: int, n_chunks: int, memory_threshold_gb: float | None = None, memory_fraction: float | None = None) -> tuple[bool, float, str]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.strategies.out_of_core.fit_with_out_of_core_accumulation` | homodyne has `fit_with_out_of_core_accumulation(stratified_data: Any, data: Any, per_angle_scaling: bool, physical_param_names: list[str], initial_params: np.ndarray, bounds: tuple[np.ndarray, np.ndarray] | None, log: logging.Logger | logging.LoggerAdapter[logging.Logger], config: Any, fast_chi2_mode: bool = False, anti_degeneracy_config: dict | None = None) -> tuple[np.ndarray, np.ndarray, dict]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.strategies.residual.StratifiedResidualFunction.get_diagnostics` | homodyne has `get_diagnostics(self) -> dict[str, Any]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.strategies.residual.StratifiedResidualFunction.jax_residual` | homodyne has `jax_residual(self, params: jnp.ndarray) -> jnp.ndarray`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.strategies.residual.StratifiedResidualFunction.log_diagnostics` | homodyne has `log_diagnostics(self) -> None`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.strategies.residual.StratifiedResidualFunction.validate_chunk_structure` | homodyne has `validate_chunk_structure(self) -> bool`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.strategies.residual.create_stratified_residual_function` | homodyne has `create_stratified_residual_function(stratified_data: Any, per_angle_scaling: bool, physical_param_names: list[str], logger: logging.Logger | None = None, validate: bool = True) -> StratifiedResidualFunction`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.strategies.residual_jit.StratifiedResidualFunctionJIT.get_diagnostics` | homodyne has `get_diagnostics(self) -> dict`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.strategies.residual_jit.StratifiedResidualFunctionJIT.log_diagnostics` | homodyne has `log_diagnostics(self) -> None`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.strategies.residual_jit.StratifiedResidualFunctionJIT.validate_chunk_structure` | homodyne has `validate_chunk_structure(self) -> bool`; heterodyne missing |
| `KEEP` | changed | `optimization.nlsq.strategies.sequential.combine_angle_results` | homodyne: `combine_angle_results(per_angle_results: list[dict[str, Any]], weighting: str = 'inverse_variance') -> tuple[np.ndarray, np.ndarray, float]`<br>  heterodyne: `combine_angle_results(per_angle_results: list[StrategyResult], weighting: str = 'inverse_variance') -> tuple[np.ndarray, np.ndarray, float]` |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.strategies.sequential.optimize_per_angle_sequential` | homodyne has `optimize_per_angle_sequential(phi: np.ndarray, t1: np.ndarray, t2: np.ndarray, g2_exp: np.ndarray, residual_func: callable, initial_params: np.ndarray, bounds: tuple[np.ndarray, np.ndarray], weighting: str = 'inverse_variance', min_success_rate: float = 0.5, parameter_names: Sequence[str] | None = None, **optimizer_kwargs) -> SequentialResult`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.strategies.sequential.optimize_single_angle` | homodyne has `optimize_single_angle(subset: AngleSubset, residual_func: Callable, initial_params: np.ndarray, bounds: tuple[np.ndarray, np.ndarray], **optimizer_kwargs) -> dict[str, Any]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.strategies.sequential.split_data_by_angle` | homodyne has `split_data_by_angle(phi: np.ndarray, t1: np.ndarray, t2: np.ndarray, g2_exp: np.ndarray, min_points_per_angle: int = 10) -> list[AngleSubset]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.strategies.stratified_ls.create_stratified_chunks` | homodyne has `create_stratified_chunks(stratified_data: Any, target_chunk_size: int = 100000) -> Any`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.strategies.stratified_ls.fit_with_stratified_least_squares` | homodyne has `fit_with_stratified_least_squares(stratified_data: Any, per_angle_scaling: bool, physical_param_names: list[str], initial_params: np.ndarray, bounds: tuple[np.ndarray, np.ndarray] | None, log: logging.Logger | logging.LoggerAdapter[logging.Logger], target_chunk_size: int = 100000, anti_degeneracy_config: dict | None = None, nlsq_config_dict: dict | None = None) -> tuple[np.ndarray, np.ndarray, dict]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.transforms.adjust_covariance_for_transforms` | homodyne has `adjust_covariance_for_transforms(covariance: np.ndarray, transformed_params: np.ndarray, physical_params: np.ndarray, state: dict[str, Any] | None) -> np.ndarray`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.transforms.apply_forward_shear_transforms_to_bounds` | homodyne has `apply_forward_shear_transforms_to_bounds(bounds: tuple[np.ndarray, np.ndarray] | None, state: dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.transforms.apply_forward_shear_transforms_to_vector` | homodyne has `apply_forward_shear_transforms_to_vector(params: np.ndarray, index_map: dict[str, int], transform_cfg: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.transforms.apply_inverse_shear_transforms_to_vector` | homodyne has `apply_inverse_shear_transforms_to_vector(params: np.ndarray, state: dict[str, Any] | None) -> np.ndarray`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.transforms.build_per_parameter_x_scale` | homodyne has `build_per_parameter_x_scale(per_angle_scaling: bool, n_angles: int, physical_param_names: list[str], analysis_mode: str, override_map: dict[str, float]) -> np.ndarray | None`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.transforms.build_physical_index_map` | homodyne has `build_physical_index_map(per_angle_scaling: bool, n_angles: int, physical_param_names: list[str]) -> dict[str, int]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.transforms.format_x_scale_for_log` | homodyne has `format_x_scale_for_log(value: Any) -> str`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.transforms.normalize_param_key` | homodyne has `normalize_param_key(name: str | None) -> str`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.transforms.normalize_x_scale_map` | homodyne has `normalize_x_scale_map(raw_map: Any) -> dict[str, float]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.transforms.parse_shear_transform_config` | homodyne has `parse_shear_transform_config(config: Any | None) -> dict[str, Any]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.transforms.wrap_model_function_with_transforms` | homodyne has `wrap_model_function_with_transforms(model_fn: Any, state: dict[str, Any] | None) -> Any`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.transforms.wrap_stratified_function_with_transforms` | homodyne has `wrap_stratified_function_with_transforms(residual_fn: Any, state: dict[str, Any] | None) -> Any`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.validation.input_validator.InputValidator.validate_all` | homodyne has `validate_all(self, xdata: np.ndarray, ydata: np.ndarray, initial_params: np.ndarray, bounds: tuple[np.ndarray, np.ndarray] | None) -> bool`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.validation.input_validator.InputValidator.validation_errors` | homodyne has `validation_errors(self) -> list[str]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.validation.input_validator.validate_array_dimensions` | homodyne has `validate_array_dimensions(xdata: np.ndarray, ydata: np.ndarray) -> bool`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.validation.input_validator.validate_bounds_consistency` | homodyne has `validate_bounds_consistency(bounds: tuple[np.ndarray, np.ndarray], initial_params: np.ndarray) -> bool`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.validation.input_validator.validate_initial_params` | homodyne has `validate_initial_params(initial_params: np.ndarray, bounds: tuple[np.ndarray, np.ndarray] | None) -> bool`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.validation.input_validator.validate_no_nan_inf` | homodyne has `validate_no_nan_inf(arr: np.ndarray, name: str, iteration: int | None = None, context: dict[str, Any] | None = None) -> bool`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.validation.result_validator.ResultValidator.validate_all` | homodyne has `validate_all(self, params: np.ndarray, covariance: np.ndarray | None, bounds: tuple[np.ndarray, np.ndarray] | None, chi_squared: float | None = None) -> bool`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.validation.result_validator.ResultValidator.validation_warnings` | homodyne has `validation_warnings(self) -> list[str]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.validation.result_validator.validate_covariance` | homodyne has `validate_covariance(covariance: np.ndarray, n_params: int) -> bool`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.validation.result_validator.validate_optimized_params` | homodyne has `validate_optimized_params(params: np.ndarray, bounds: tuple[np.ndarray, np.ndarray] | None, tolerance: float = 1e-10) -> bool`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.validation.result_validator.validate_result_consistency` | homodyne has `validate_result_consistency(params: np.ndarray, chi_squared: float) -> bool`; heterodyne missing |
| `KEEP` | changed | `optimization.nlsq.wrapper.NLSQWrapper.fit` | homodyne: `fit(self, data: Any, config: Any, initial_params: np.ndarray | None = None, bounds: tuple[np.ndarray, np.ndarray] | None = None, analysis_mode: str = 'static_isotropic', per_angle_scaling: bool = True, diagnostics_enabled: bool = False, shear_transforms: dict[str, Any] | None = None, per_angle_scaling_initial: dict[str, list[float]] | None = None) -> OptimizationResult`<br>  heterodyne: `fit(self, data: Any, config: Any, initial_params: np.ndarray | None = None, bounds: tuple[np.ndarray, np.ndarray] | None = None, per_angle_scaling: bool = True, diagnostics_enabled: bool = False, per_angle_scaling_initial: dict[str, list[float]] | None = None) -> NLSQResult` |
| `KEEP` | missing_in_heterodyne | `optimization.nlsq.wrapper.create_multistart_warmup_func` | homodyne has `create_multistart_warmup_func(model_func: Callable[..., np.ndarray], xdata: np.ndarray, ydata: np.ndarray, bounds: tuple[np.ndarray, np.ndarray] | None = None, warmup_learning_rate: float = 0.001, normalize: bool = True, chunk_size: int = 50000) -> Callable[[dict[str, Any], np.ndarray, int], Any]`; heterodyne missing |
| `KEEP` | changed | `optimization.numerical_validation.NumericalValidator.set_bounds` | homodyne: `set_bounds(self, bounds: tuple[np.ndarray, np.ndarray]) -> None`<br>  heterodyne: `set_bounds(self, bounds: dict[str, tuple[float, float]]) -> None` |
| `KEEP` | changed | `optimization.numerical_validation.NumericalValidator.validate_gradients` | homodyne: `validate_gradients(self, gradients: Any) -> None`<br>  heterodyne: `validate_gradients(self, gradients: jnp.ndarray | np.ndarray) -> None` |
| `KEEP` | changed | `optimization.numerical_validation.NumericalValidator.validate_loss` | homodyne: `validate_loss(self, loss_value: Any) -> None`<br>  heterodyne: `validate_loss(self, loss_value: jnp.ndarray | float) -> None` |
| `KEEP` | changed | `optimization.numerical_validation.NumericalValidator.validate_parameters` | homodyne: `validate_parameters(self, parameters: Any, bounds: tuple[np.ndarray, np.ndarray] | None = None) -> None`<br>  heterodyne: `validate_parameters(self, parameters: jnp.ndarray | np.ndarray | dict[str, Any], bounds: dict[str, tuple[float, float]] | None = None) -> None` |
| `KEEP` | missing_in_heterodyne | `optimization.recovery_strategies.RecoveryStrategyApplicator.get_recovery_strategy` | homodyne has `get_recovery_strategy(self, error: Exception, params: np.ndarray, attempt: int, bounds: tuple[np.ndarray, np.ndarray] | None = None) -> tuple[str, np.ndarray] | None`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `optimization.recovery_strategies.RecoveryStrategyApplicator.should_retry` | homodyne has `should_retry(self, attempt: int) -> bool`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `runtime.utils.system_validator.SystemValidator.calculate_health_score` | homodyne has `calculate_health_score(self) -> int`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `runtime.utils.system_validator.SystemValidator.generate_report` | homodyne has `generate_report(self) -> str`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `runtime.utils.system_validator.SystemValidator.log` | homodyne has `log(self, message: str, level: str = 'info') -> None`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `runtime.utils.system_validator.SystemValidator.run_all_tests` | homodyne has `run_all_tests(self) -> dict[str, ValidationResult]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `runtime.utils.system_validator.SystemValidator.run_command` | homodyne has `run_command(self, cmd: list[str], timeout: int = 30) -> tuple[bool, str, str]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `runtime.utils.system_validator.SystemValidator.run_quick_tests` | homodyne has `run_quick_tests(self) -> dict[str, ValidationResult]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `runtime.utils.system_validator.SystemValidator.test_config_system` | homodyne has `test_config_system(self) -> ValidationResult`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `runtime.utils.system_validator.SystemValidator.test_data_pipeline` | homodyne has `test_data_pipeline(self) -> ValidationResult`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `runtime.utils.system_validator.SystemValidator.test_dependency_versions` | homodyne has `test_dependency_versions(self) -> ValidationResult`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `runtime.utils.system_validator.SystemValidator.test_homodyne_installation` | homodyne has `test_homodyne_installation(self) -> ValidationResult`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `runtime.utils.system_validator.SystemValidator.test_integration` | homodyne has `test_integration(self) -> ValidationResult`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `runtime.utils.system_validator.SystemValidator.test_nlsq_integration` | homodyne has `test_nlsq_integration(self) -> ValidationResult`; heterodyne missing |
| `KEEP` | changed | `runtime.utils.system_validator.main` | homodyne: `main() -> None`<br>  heterodyne: `main() -> int` |
| `KEEP` | changed | `utils.async_io.AsyncWriter.shutdown` | homodyne: `shutdown(self) -> None`<br>  heterodyne: `shutdown(self, wait: bool = True) -> None` |
| `KEEP` | missing_in_heterodyne | `utils.async_io.AsyncWriter.submit_json` | homodyne has `submit_json(self, path: Path, data: dict[str, Any]) -> None`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `utils.async_io.AsyncWriter.submit_npz` | homodyne has `submit_npz(self, path: Path, data: dict[str, np.ndarray]) -> None`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `utils.async_io.AsyncWriter.submit_task` | homodyne has `submit_task(self, fn: Callable[..., None], *args: Any, **kwargs: Any) -> None`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `utils.async_io.AsyncWriter.wait_all` | homodyne has `wait_all(self, timeout: float = 60.0) -> list[Exception]`; heterodyne missing |
| `KEEP` | changed | `utils.logging.AnalysisSummaryLogger.log_summary` | homodyne: `log_summary(self, logger: logging.Logger | logging.LoggerAdapter) -> None`<br>  heterodyne: `log_summary(self, logger: logging.Logger | logging.LoggerAdapter[Any]) -> None` |
| `KEEP` | changed | `utils.logging.configure_logging` | homodyne: `configure_logging(logging_config: Mapping[str, Any] | None, *, verbose: bool = False, quiet: bool = False, output_dir: Path | str | None = None, run_id: str | None = None) -> Path | None`<br>  heterodyne: `configure_logging(level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = 'INFO', log_file: Path | str | None = None, format_string: str | None = None) -> None` |
| `KEEP` | missing_in_heterodyne | `utils.path_validation.get_safe_output_dir` | homodyne has `get_safe_output_dir(output_dir: str | Path | None = None, default_subdir: str = 'homodyne_output') -> Path`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `utils.path_validation.validate_plot_save_path` | homodyne has `validate_plot_save_path(path: str | Path | None, *, require_parent_exists: bool = True) -> Path | None`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `utils.path_validation.validate_save_path` | homodyne has `validate_save_path(path: str | Path | None, *, allowed_extensions: tuple[str, ...] | None = None, require_parent_exists: bool = True, allow_absolute: bool = True, base_dir: Path | None = None) -> Path | None`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `viz.datashader_backend.DatashaderRenderer.rasterize_heatmap` | homodyne has `rasterize_heatmap(self, data: np.ndarray, x_coords: np.ndarray, y_coords: np.ndarray, cmap: str = 'jet', vmin: float | None = None, vmax: float | None = None) -> Image.Image`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `viz.datashader_backend.plot_c2_comparison_fast` | homodyne has `plot_c2_comparison_fast(c2_exp: np.ndarray, c2_fit: np.ndarray, residuals: np.ndarray, t1: np.ndarray, t2: np.ndarray, output_path: Path, phi_angle: float, width: int = 800, height: int = 800, *, vmin: float | None = None, vmax: float | None = None, adaptive: bool = True, percentile_min: float = 1.0, percentile_max: float = 99.0) -> None`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `viz.datashader_backend.plot_c2_heatmap_fast` | homodyne has `plot_c2_heatmap_fast(c2_data: np.ndarray, t1: np.ndarray, t2: np.ndarray, output_path: Path, title: str = '', phi_angle: float | None = None, cmap: str = 'jet', width: int = 800, height: int = 800, *, vmin: float | None = None, vmax: float | None = None, adaptive: bool = False, percentile_min: float = 1.0, percentile_max: float = 99.0) -> None`; heterodyne missing |
| `KEEP` | changed | `viz.diagnostics.compute_diagonal_overlay_stats` | homodyne: `compute_diagonal_overlay_stats(c2_exp: np.ndarray, c2_solver: np.ndarray, c2_posthoc: np.ndarray, *, phi_index: int = 0) -> DiagonalOverlayResult`<br>  heterodyne: `compute_diagonal_overlay_stats(c2_exp: np.ndarray, c2_solver: np.ndarray | None, c2_posthoc: np.ndarray, *, phi_index: int = 0) -> DiagonalOverlayResult` |
| `KEEP` | changed | `viz.mcmc_arviz.plot_arviz_pair` | homodyne: `plot_arviz_pair(result: Any, var_names: list[str] | None = None, figsize: tuple[float, float] | None = None, show: bool = False, save_path: str | Path | None = None, dpi: int = 150, **kwargs: Any) -> Figure`<br>  heterodyne: `plot_arviz_pair(result: CMCResult, var_names: list[str] | None = None, save_path: Path | str | None = None) -> Figure | None` |
| `KEEP` | changed | `viz.mcmc_arviz.plot_arviz_posterior` | homodyne: `plot_arviz_posterior(result: Any, var_names: list[str] | None = None, hdi_prob: float = 0.95, figsize: tuple[float, float] | None = None, show: bool = False, save_path: str | Path | None = None, dpi: int = 150, **kwargs: Any) -> Figure`<br>  heterodyne: `plot_arviz_posterior(result: CMCResult, var_names: list[str] | None = None, hdi_prob: float = 0.95, save_path: Path | str | None = None) -> Figure | None` |
| `KEEP` | changed | `viz.mcmc_arviz.plot_arviz_trace` | homodyne: `plot_arviz_trace(result: Any, var_names: list[str] | None = None, figsize: tuple[float, float] | None = None, show: bool = False, save_path: str | Path | None = None, dpi: int = 150, **kwargs: Any) -> Figure`<br>  heterodyne: `plot_arviz_trace(result: CMCResult, var_names: list[str] | None = None, save_path: Path | str | None = None) -> Figure | None` |
| `KEEP` | missing_in_heterodyne | `viz.mcmc_comparison.plot_posterior_comparison` | homodyne has `plot_posterior_comparison(result: Any, param_indices: list[int] | None = None, figsize: tuple[float, float] | None = None, bins: int = 30, show: bool = False, save_path: str | Path | None = None, dpi: int = 150) -> Figure`; heterodyne missing |
| `KEEP` | changed | `viz.mcmc_dashboard.plot_cmc_summary_dashboard` | homodyne: `plot_cmc_summary_dashboard(result: Any, figsize: tuple[float, float] = (16, 12), show: bool = False, save_path: str | Path | None = None, dpi: int = 150) -> Figure`<br>  heterodyne: `plot_cmc_summary_dashboard(result: CMCResult, figsize: tuple[float, float] = (16, 12), save_path: str | Path | None = None, dpi: int = 150) -> Figure` |
| `KEEP` | missing_in_heterodyne | `viz.mcmc_diagnostics.compute_bfmi` | homodyne has `compute_bfmi(energy: np.ndarray, *, per_chain: bool = False) -> float | np.ndarray`; heterodyne missing |
| `KEEP` | changed | `viz.mcmc_diagnostics.plot_convergence_diagnostics` | homodyne: `plot_convergence_diagnostics(result: Any, metrics: list[str] | None = None, figsize: tuple[float, float] | None = None, rhat_threshold: float = 1.1, ess_threshold: float = 400.0, show: bool = False, save_path: str | Path | None = None, dpi: int = 150) -> Figure`<br>  heterodyne: `plot_convergence_diagnostics(result: CMCResult, save_path: Path | str | None = None) -> Figure` |
| `KEEP` | changed | `viz.mcmc_diagnostics.plot_kl_divergence_matrix` | homodyne: `plot_kl_divergence_matrix(result: Any, figsize: tuple[float, float] = (8, 7), cmap: str = 'coolwarm', threshold: float = 2.0, show: bool = False, save_path: str | Path | None = None, dpi: int = 150) -> Figure`<br>  heterodyne: `plot_kl_divergence_matrix(result: CMCResult, save_path: Path | str | None = None) -> Figure` |
| `KEEP` | missing_in_heterodyne | `viz.mcmc_diagnostics.plot_trace_plots` | homodyne has `plot_trace_plots(result: Any, param_names: list[str] | None = None, max_params: int = 9, figsize: tuple[float, float] | None = None, show: bool = False, save_path: str | Path | None = None, dpi: int = 150) -> Figure`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `viz.mcmc_report.generate_mcmc_diagnostic_report` | homodyne has `generate_mcmc_diagnostic_report(result: Any, output_dir: str | Path, prefix: str = 'mcmc', include_heatmaps: bool = True, dpi: int = 150) -> dict[str, Path]`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `viz.mcmc_report.print_mcmc_summary` | homodyne has `print_mcmc_summary(result: Any) -> None`; heterodyne missing |
| `KEEP` | missing_in_heterodyne | `viz.validation.validate_plot_arrays` | homodyne has `validate_plot_arrays(*arrays: np.ndarray, names: list[str] | None = None) -> bool`; heterodyne missing |

## P1 — Structural drift — files added/missing/renamed, docs pages missing (1167 gaps)

### cli (26)

| Disposition | Kind | Qualname / Path | Detail |
|---|---|---|---|
| `KEEP` | extra_cli_flag | `--data` | heterodyne-only CLI flag |
| `KEEP` | extra_cli_flag | `--debug` | heterodyne-only CLI flag |
| `KEEP` | extra_cli_flag | `--dt` | heterodyne-only CLI flag |
| `KEEP` | extra_cli_flag | `--info` | heterodyne-only CLI flag |
| `KEEP` | extra_cli_flag | `--initial-D-offset-ref` | heterodyne-only CLI flag |
| `KEEP` | extra_cli_flag | `--initial-D-offset-sample` | heterodyne-only CLI flag |
| `KEEP` | extra_cli_flag | `--initial-D0-ref` | heterodyne-only CLI flag |
| `KEEP` | extra_cli_flag | `--initial-D0-sample` | heterodyne-only CLI flag |
| `KEEP` | extra_cli_flag | `--initial-alpha-ref` | heterodyne-only CLI flag |
| `KEEP` | extra_cli_flag | `--initial-alpha-sample` | heterodyne-only CLI flag |
| `KEEP` | extra_cli_flag | `--initial-f0` | heterodyne-only CLI flag |
| `KEEP` | extra_cli_flag | `--initial-v-offset` | heterodyne-only CLI flag |
| `KEEP` | extra_cli_flag | `--initial-v0` | heterodyne-only CLI flag |
| `KEEP` | extra_cli_flag | `--multistart` | heterodyne-only CLI flag |
| `KEEP` | extra_cli_flag | `--multistart-n` | heterodyne-only CLI flag |
| `KEEP` | extra_cli_flag | `--no-plot` | heterodyne-only CLI flag |
| `KEEP` | extra_cli_flag | `--no-x64` | heterodyne-only CLI flag |
| `KEEP` | extra_cli_flag | `--num-chains` | heterodyne-only CLI flag |
| `KEEP` | extra_cli_flag | `--num-samples` | heterodyne-only CLI flag |
| `KEEP` | extra_cli_flag | `--offset-sim` | heterodyne-only CLI flag |
| `KEEP` | extra_cli_flag | `--overwrite` | heterodyne-only CLI flag |
| `KEEP` | extra_cli_flag | `--phi` | heterodyne-only CLI flag |
| `KEEP` | extra_cli_flag | `--plot` | heterodyne-only CLI flag |
| `KEEP` | extra_cli_flag | `--q` | heterodyne-only CLI flag |
| `KEEP` | extra_cli_flag | `--show-template` | heterodyne-only CLI flag |
| `KEEP` | extra_cli_flag | `--time-length` | heterodyne-only CLI flag |

### configs (419)

| Disposition | Kind | Qualname / Path | Detail |
|---|---|---|---|
| `KEEP` | extra_config_key | `config.parameter_manager.ParameterManager.space` | heterodyne-only config key |
| `KEEP` | extra_config_key | `config.parameter_registry.ParameterInfo.group` | heterodyne-only config key |
| `KEEP` | extra_config_key | `config.parameter_registry.ParameterInfo.max_bound` | heterodyne-only config key |
| `KEEP` | extra_config_key | `config.parameter_registry.ParameterInfo.min_bound` | heterodyne-only config key |
| `KEEP` | extra_config_key | `config.parameter_registry.ParameterInfo.unit` | heterodyne-only config key |
| `KEEP` | extra_config_key | `config.parameter_registry.ParameterInfo.vary_default` | heterodyne-only config key |
| `KEEP` | extra_config_key | `config.parameter_space.ParameterSpace.values` | heterodyne-only config key |
| `KEEP` | extra_config_key | `config.parameter_space.ParameterSpace.vary` | heterodyne-only config key |
| `KEEP` | extra_config_key | `config.parameter_space.PriorDistribution.params` | heterodyne-only config key |
| `KEEP` | extra_config_key | `config.parameter_space.PriorDistribution.prior_type` | heterodyne-only config key |
| `KEEP` | extra_config_key | `config.parameter_space.runtime_keys` | heterodyne-only config key |
| `KEEP` | extra_config_key | `config.physics_validators.ConstraintRule.check` | heterodyne-only config key |
| `KEEP` | extra_config_key | `config.physics_validators.PhysicsViolation.parameter` | heterodyne-only config key |
| `KEEP` | extra_config_key | `config.physics_validators.ValidationResult.errors` | heterodyne-only config key |
| `KEEP` | extra_config_key | `config.physics_validators.ValidationResult.info` | heterodyne-only config key |
| `KEEP` | extra_config_key | `config.physics_validators.ValidationResult.is_valid` | heterodyne-only config key |
| `KEEP` | extra_config_key | `config.physics_validators.ValidationResult.warnings` | heterodyne-only config key |
| `KEEP` | extra_config_key | `core.backend_api.BackendConfig.backend` | heterodyne-only config key |
| `KEEP` | extra_config_key | `core.backend_api.BackendConfig.device` | heterodyne-only config key |
| `KEEP` | extra_config_key | `core.backend_api.BackendConfig.precision` | heterodyne-only config key |
| `KEEP` | extra_config_key | `core.heterodyne_model.HeterodyneModel.param_manager` | heterodyne-only config key |
| `KEEP` | extra_config_key | `core.heterodyne_model.HeterodyneModel.scaling` | heterodyne-only config key |
| `KEEP` | extra_config_key | `core.heterodyne_model.runtime_keys` | heterodyne-only config key |
| `KEEP` | extra_config_key | `core.numpy_gradients.GradientResult.elapsed_seconds` | heterodyne-only config key |
| `KEEP` | extra_config_key | `core.numpy_gradients.GradientResult.n_function_evals` | heterodyne-only config key |
| `KEEP` | extra_config_key | `core.physics_factors.CachedMatrices.mean_time` | heterodyne-only config key |
| `KEEP` | extra_config_key | `core.physics_factors.CachedMatrices.time_diff` | heterodyne-only config key |
| `KEEP` | extra_config_key | `core.physics_factors.CachedMatrices.tril_indices` | heterodyne-only config key |
| `KEEP` | extra_config_key | `core.physics_factors.CachedMatrices.triu_indices` | heterodyne-only config key |
| `KEEP` | extra_config_key | `core.physics_factors.PhysicsFactors.n_times` | heterodyne-only config key |
| `KEEP` | extra_config_key | `core.physics_factors.PhysicsFactors.phi_angle` | heterodyne-only config key |
| `KEEP` | extra_config_key | `core.physics_factors.PhysicsFactors.q` | heterodyne-only config key |
| `KEEP` | extra_config_key | `core.physics_factors.PhysicsFactors.q_squared` | heterodyne-only config key |
| `KEEP` | extra_config_key | `core.physics_factors.PhysicsFactors.t` | heterodyne-only config key |
| `KEEP` | extra_config_key | `core.scaling_utils.PerAngleScaling.contrast` | heterodyne-only config key |
| `KEEP` | extra_config_key | `core.scaling_utils.PerAngleScaling.n_angles` | heterodyne-only config key |
| `KEEP` | extra_config_key | `core.scaling_utils.PerAngleScaling.offset` | heterodyne-only config key |
| `KEEP` | extra_config_key | `core.scaling_utils.PerAngleScaling.vary_contrast` | heterodyne-only config key |
| `KEEP` | extra_config_key | `core.scaling_utils.PerAngleScaling.vary_offset` | heterodyne-only config key |
| `KEEP` | extra_config_key | `core.scaling_utils.ScalingConfig.initial_contrast` | heterodyne-only config key |
| `KEEP` | extra_config_key | `core.scaling_utils.ScalingConfig.initial_offset` | heterodyne-only config key |
| `KEEP` | extra_config_key | `core.scaling_utils.ScalingConfig.mode` | heterodyne-only config key |
| `KEEP` | extra_config_key | `core.scaling_utils.ScalingConfig.n_angles` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.config.DataConfig.angle_range` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.config.DataConfig.diagonal_width` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.config.DataConfig.file_path` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.config.DataConfig.format` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.config.DataConfig.normalize` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.config.DataConfig.outlier_sigma` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.config.DataConfig.q_range` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.config.DataConfig.remove_outliers` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.config.DataConfig.time_range` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.memory_manager.ChunkInfo.end` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.memory_manager.ChunkInfo.priority` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.memory_manager.ChunkInfo.size_bytes` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.memory_manager.ChunkInfo.start` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.memory_manager.MemoryBudget.allocated_bytes` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.memory_manager.MemoryBudget.peak_bytes` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.memory_manager.MemoryBudget.total_bytes` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.optimization.DatasetSizeCategory.category` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.optimization.DatasetSizeCategory.n_elements` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.optimization.DatasetSizeCategory.recommended_chunk_size` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.optimization.DatasetSizeCategory.size_bytes` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.optimization.DatasetSizeCategory.use_compression` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.optimization.DatasetSizeCategory.use_memory_mapping` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.optimization.DatasetSizeCategory.use_progressive_loading` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.optimization.SubsamplingConfig.max_points` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.optimization.SubsamplingConfig.method` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.optimization.SubsamplingConfig.seed` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.performance_engine.CacheEntry.access_count` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.performance_engine.CacheEntry.data` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.performance_engine.CacheEntry.size_bytes` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.performance_engine.CacheEntry.timestamp` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.performance_engine.TieredCacheConfig.compression` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.performance_engine.TieredCacheConfig.disk_cache_dir` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.performance_engine.TieredCacheConfig.disk_max_bytes` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.performance_engine.TieredCacheConfig.memory_max_bytes` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.phi_filtering.PhiFilterResult.c2` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.phi_filtering.PhiFilterResult.n_angles` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.phi_filtering.PhiFilterResult.phi_angles` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.phi_filtering.PhiFilterResult.selected_indices` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.preprocessing.PreprocessingProvenance.created_at` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.preprocessing.PreprocessingProvenance.records` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.preprocessing.PreprocessingProvenance.source_file` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.preprocessing.PreprocessingResult.applied_steps` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.preprocessing.PreprocessingResult.c2` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.preprocessing.PreprocessingResult.statistics` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.preprocessing.TransformationRecord.input_hash` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.preprocessing.TransformationRecord.output_hash` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.quality_controller.QualityControlConfig.auto_repair_nans` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.quality_controller.QualityControlConfig.auto_repair_outliers` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.quality_controller.QualityControlConfig.min_time_points` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.quality_controller.QualityControlConfig.nan_threshold` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.quality_controller.QualityControlConfig.outlier_sigma` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.quality_controller.QualityControlConfig.report_format` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.quality_controller.QualityControlConfig.snr_threshold` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.quality_controller.QualityControlConfig.symmetry_threshold` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.quality_controller.QualityControlConfig.value_range_max` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.quality_controller.QualityControlResult.auto_corrections` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.quality_controller.QualityControlResult.metadata` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.quality_controller.QualityControlResult.quality_score` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.quality_controller.QualityControlResult.report` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.quality_controller.QualityMetric.level` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.quality_controller.QualityMetric.message` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.quality_controller.QualityMetric.name` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.quality_controller.QualityMetric.threshold` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.quality_controller.QualityMetric.value` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.quality_controller.QualityReport.metrics` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.quality_controller.QualityReport.overall_level` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.quality_controller.QualityReport.recommendations` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.types.FilterResult.data` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.types.FilterResult.mask` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.types.FilterResult.n_removed` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.types.FilterResult.reason` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.validation.DataQualityReport.statistics` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.xpcs_loader.XPCSData.c2` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.xpcs_loader.XPCSData.metadata` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.xpcs_loader.XPCSData.phi_angles` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.xpcs_loader.XPCSData.q` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.xpcs_loader.XPCSData.q_values` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.xpcs_loader.XPCSData.t1` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.xpcs_loader.XPCSData.t2` | heterodyne-only config key |
| `KEEP` | extra_config_key | `data.xpcs_loader.XPCSData.uncertainties` | heterodyne-only config key |
| `KEEP` | extra_config_key | `device.config.HardwareConfig.available_cores` | heterodyne-only config key |
| `KEEP` | extra_config_key | `device.config.HardwareConfig.cpu_info` | heterodyne-only config key |
| `KEEP` | extra_config_key | `device.config.HardwareConfig.max_parallel_chains` | heterodyne-only config key |
| `KEEP` | extra_config_key | `device.config.HardwareConfig.memory_gb` | heterodyne-only config key |
| `KEEP` | extra_config_key | `device.config.HardwareConfig.recommended_chains` | heterodyne-only config key |
| `KEEP` | extra_config_key | `device.cpu.CPUInfo.architecture` | heterodyne-only config key |
| `KEEP` | extra_config_key | `device.cpu.CPUInfo.cache_sizes` | heterodyne-only config key |
| `KEEP` | extra_config_key | `device.cpu.CPUInfo.has_avx` | heterodyne-only config key |
| `KEEP` | extra_config_key | `device.cpu.CPUInfo.has_avx2` | heterodyne-only config key |
| `KEEP` | extra_config_key | `device.cpu.CPUInfo.has_avx512` | heterodyne-only config key |
| `KEEP` | extra_config_key | `device.cpu.CPUInfo.logical_cores` | heterodyne-only config key |
| `KEEP` | extra_config_key | `device.cpu.CPUInfo.model_name` | heterodyne-only config key |
| `KEEP` | extra_config_key | `device.cpu.CPUInfo.numa_nodes` | heterodyne-only config key |
| `KEEP` | extra_config_key | `device.cpu.CPUInfo.physical_cores` | heterodyne-only config key |
| `KEEP` | extra_config_key | `device.cpu.CPUInfo.vendor` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.batch_statistics.BatchResult.convergence_count` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.batch_statistics.BatchResult.mean_chi2` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.batch_statistics.BatchResult.overall_success_rate` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.batch_statistics.BatchResult.statistics` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.batch_statistics.BatchResult.total_count` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.batch_statistics.FitStatistics.mean` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.batch_statistics.FitStatistics.median` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.batch_statistics.FitStatistics.n_fits` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.batch_statistics.FitStatistics.param_name` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.batch_statistics.FitStatistics.q25` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.batch_statistics.FitStatistics.q75` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.batch_statistics.FitStatistics.std` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.checkpoint_manager.CheckpointData.checksum` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.checkpoint_manager.CheckpointData.cost` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.checkpoint_manager.CheckpointData.iteration` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.checkpoint_manager.CheckpointData.metadata` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.checkpoint_manager.CheckpointData.parameters` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.checkpoint_manager.CheckpointData.timestamp` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.checkpoint_manager.CheckpointData.version` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.backends.base.BackendCapabilities.max_parallel_shards` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.backends.base.BackendCapabilities.supports_parallel_chains` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.backends.base.BackendCapabilities.supports_sharding` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.backends.base.ShardPosterior.covariance` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.backends.base.ShardPosterior.mean` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.backends.base.ShardPosterior.n_samples` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.backends.base.ShardPosterior.shard_id` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.backends.multiprocessing_backend.ArraySpec.allow_none` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.backends.multiprocessing_backend.ArraySpec.description` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.backends.multiprocessing_backend.ArraySpec.expected_dtype` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.backends.pbs.PBSConfig.cleanup_on_success` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.backends.pbs.PBSConfig.extra_pbs_directives` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.backends.pbs.PBSConfig.max_retries` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.backends.pbs.PBSConfig.memory` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.backends.pbs.PBSConfig.nodes` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.backends.pbs.PBSConfig.poll_interval` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.backends.pbs.PBSConfig.ppn` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.backends.pbs.PBSConfig.python_executable` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.backends.pbs.PBSConfig.queue` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.backends.pbs.PBSConfig.walltime` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.backends.pbs.PBSConfig.working_dir` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.backends.pbs.ShardResult.error_message` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.backends.pbs.ShardResult.job_id` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.backends.pbs.ShardResult.samples` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.backends.pbs.ShardResult.shard_id` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.backends.pbs.ShardResult.success` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.config.CMCConfig.allow_degenerate_warmstart` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.config.CMCConfig.fast_warmup` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.config.CMCConfig.init_strategy` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.config.CMCConfig.min_bfmi` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.config.CMCConfig.use_log_space_priors` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.config.CMCConfig.use_nlsq_warmstart` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.config.CMCConfig.use_reparam` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.data_prep.PreparedData.c2_data` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.data_prep.PreparedData.dt` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.data_prep.PreparedData.metadata` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.data_prep.PreparedData.n_angles` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.data_prep.PreparedData.n_times` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.data_prep.PreparedData.phi_angles` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.data_prep.PreparedData.q` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.data_prep.PreparedData.time_array` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.data_prep.PreparedData.weights` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.data_prep.runtime_keys` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.diagnostics.BimodalResult.bic_bimodal` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.diagnostics.BimodalResult.bic_unimodal` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.diagnostics.BimodalResult.delta_bic` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.diagnostics.BimodalResult.param_name` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.diagnostics.ConvergenceReport.bfmi_passed` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.diagnostics.ConvergenceReport.ess_passed` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.diagnostics.ConvergenceReport.messages` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.diagnostics.ConvergenceReport.passed` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.diagnostics.ConvergenceReport.r_hat_passed` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.diagnostics.DivergenceReport.divergence_rate` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.diagnostics.DivergenceReport.messages` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.diagnostics.DivergenceReport.n_divergent` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.diagnostics.DivergenceReport.n_total` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.diagnostics.DivergenceReport.severity` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.reparameterization.ReparamConfig.enable_d_sample` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.reparameterization.ReparamConfig.enable_v_ref` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.results.CMCResult.bfmi` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.results.CMCResult.convergence_passed` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.results.CMCResult.credible_intervals` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.results.CMCResult.map_estimate` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.results.CMCResult.metadata` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.results.CMCResult.num_chains` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.results.CMCResult.num_samples` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.results.CMCResult.num_warmup` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.results.CMCResult.parameter_names` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.results.CMCResult.posterior_mean` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.results.CMCResult.posterior_std` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.results.CMCResult.wall_time_seconds` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.sampler.AdaptiveSamplingPlan.base_plan` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.sampler.AdaptiveSamplingPlan.n_params` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.sampler.AdaptiveSamplingPlan.shard_size` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.sampler.SamplingPlan.adapt_step_size` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.sampler.SamplingPlan.chain_method` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.sampler.SamplingPlan.dense_mass` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.sampler.SamplingPlan.fast_warmup` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.sampler.SamplingPlan.max_tree_depth` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.sampler.SamplingPlan.num_chains` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.sampler.SamplingPlan.num_samples` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.sampler.SamplingPlan.num_warmup` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.sampler.SamplingPlan.seed` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.sampler.SamplingPlan.target_accept` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.sampler.SamplingStats.divergence_rate` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.sampler.SamplingStats.max_tree_depth_fraction` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.sampler.SamplingStats.mean_accept_prob` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.sampler.SamplingStats.num_divergences` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.sampler.SamplingStats.num_samples` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.sampler.SamplingStats.num_warmup` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.cmc.sampler.SamplingStats.wall_time_seconds` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.gradient_diagnostics.GradientHealth.is_healthy` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.gradient_diagnostics.GradientHealth.issues` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.gradient_diagnostics.GradientHealth.metrics` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.adapter.CachedModel.fitter` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.adapter.CachedModel.last_accessed` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.adapter.ModelCacheKey.callable_scope` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.adapter.ModelCacheKey.n_data` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.adapter.ModelCacheKey.n_params` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.adapter.ModelCacheKey.scaling_mode` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.adaptive_regularization.RegularizationConfig.adaptation_rate` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.adaptive_regularization.RegularizationConfig.lambda_init` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.adaptive_regularization.RegularizationConfig.lambda_max` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.adaptive_regularization.RegularizationConfig.lambda_min` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.anti_degeneracy_controller.DegeneracyCheck.affected_params` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.anti_degeneracy_controller.DegeneracyCheck.is_degenerate` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.anti_degeneracy_controller.DegeneracyCheck.message` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.anti_degeneracy_controller.DegeneracyCheck.suggested_action` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.cmaes_wrapper.CMAESConfig.diagonal_filtering` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.cmaes_wrapper.CMAESConfig.max_restarts` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.cmaes_wrapper.CMAESConfig.maxiter` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.cmaes_wrapper.CMAESConfig.popsize` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.cmaes_wrapper.CMAESConfig.restart_strategy` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.cmaes_wrapper.CMAESConfig.seed` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.cmaes_wrapper.CMAESConfig.sigma0` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.cmaes_wrapper.CMAESConfig.tolfun` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.cmaes_wrapper.CMAESConfig.tolx` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.cmaes_wrapper.CMAESResult.best_cost` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.cmaes_wrapper.CMAESResult.best_params` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.cmaes_wrapper.CMAESResult.converged` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.cmaes_wrapper.CMAESResult.final_sigma` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.cmaes_wrapper.CMAESResult.history` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.cmaes_wrapper.CMAESResult.n_evaluations` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.cmaes_wrapper.CMAESResult.n_iterations` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.config.HybridRecoveryConfig.perturb_scale` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.config.NLSQConfig.analysis_mode` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.config.NLSQConfig.chunk_size` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.config.NLSQConfig.cmaes_anti_degeneracy` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.config.NLSQConfig.cmaes_diagonal_filtering` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.config.NLSQConfig.cmaes_max_iterations` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.config.NLSQConfig.cmaes_population_size` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.config.NLSQConfig.cmaes_sigma0` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.config.NLSQConfig.cmaes_tolfun` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.config.NLSQConfig.cmaes_tolx` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.config.NLSQConfig.diff_step` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.config.NLSQConfig.enable_anti_degeneracy` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.config.NLSQConfig.hierarchical_inner_tolerance` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.config.NLSQConfig.hybrid_convergence_threshold` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.config.NLSQConfig.hybrid_convergence_window` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.config.NLSQConfig.hybrid_enable` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.config.NLSQConfig.hybrid_lbfgs_memory` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.config.NLSQConfig.hybrid_max_phases` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.config.NLSQConfig.hybrid_method` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.config.NLSQConfig.hybrid_normalization` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.config.NLSQConfig.hybrid_warmup_fraction` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.config.NLSQConfig.loss_scale` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.config.NLSQConfig.loss_weights` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.config.NLSQConfig.max_nfev` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.config.NLSQConfig.method` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.config.NLSQConfig.multistart` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.config.NLSQConfig.multistart_n` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.config.NLSQConfig.n_params` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.config.NLSQConfig.nlsq_memory_fallback_gb` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.config.NLSQConfig.nlsq_memory_fraction` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.config.NLSQConfig.nlsq_rescale_data` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.config.NLSQConfig.nlsq_stability` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.config.NLSQConfig.nlsq_x_scale` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.config.NLSQConfig.recovery_config` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.config.NLSQConfig.refine_top_k` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.config.NLSQConfig.sampling_strategy` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.config.NLSQConfig.screen_keep_fraction` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.config.NLSQConfig.step_bound` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.config.NLSQConfig.tolerance` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.config.NLSQConfig.tr_solver` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.config.NLSQConfig.use_jac` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.config.NLSQConfig.use_nlsq_library` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.config.NLSQConfig.validation` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.config.NLSQValidationConfig.chi2_fail_high` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.config.NLSQValidationConfig.chi2_warn_high` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.config.NLSQValidationConfig.chi2_warn_low` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.config.NLSQValidationConfig.correlation_warn` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.config.NLSQValidationConfig.max_relative_uncertainty` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.config.runtime_keys` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.gradient_monitor.GradientSnapshot.gradient_norm` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.gradient_monitor.GradientSnapshot.iteration` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.gradient_monitor.GradientSnapshot.max_gradient` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.gradient_monitor.GradientSnapshot.parameter_gradients` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.hierarchical.HierarchicalConfig.per_stage_config` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.hierarchical.HierarchicalConfig.skip_failed_stages` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.hierarchical.HierarchicalConfig.stages` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.hierarchical.HierarchicalResult.best_cost` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.hierarchical.HierarchicalResult.best_params` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.hierarchical.HierarchicalResult.converged` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.hierarchical.HierarchicalResult.n_stages_completed` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.hierarchical.HierarchicalResult.stage_results` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.hierarchical.HierarchicalResult.total_iterations` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.multistart.MultiStartConfig.max_data_points_for_parallel` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.multistart.MultiStartConfig.max_workers` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.multistart.MultiStartConfig.parallel` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.multistart.MultiStartConfig.worker_timeout` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.multistart.MultiStartResult.all_starts` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.multistart.MultiStartResult.best_result` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.multistart.MultiStartResult.n_total` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.multistart.MultiStartResult.wall_time_total` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.multistart.SingleStartResult.result` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.multistart.SingleStartResult.start_index` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.parallel_accumulator.GaussNewtonAccumulation.JtJ` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.parallel_accumulator.GaussNewtonAccumulation.Jtf` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.parallel_accumulator.GaussNewtonAccumulation.cost` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.parallel_accumulator.GaussNewtonAccumulation.n_data` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.progress.ProgressRecord.cost` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.progress.ProgressRecord.cost_change` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.progress.ProgressRecord.gradient_norm` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.progress.ProgressRecord.iteration` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.progress.ProgressRecord.step_norm` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.progress.ProgressRecord.wall_time` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.recovery.ErrorDiagnosis.category` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.recovery.ErrorDiagnosis.message` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.recovery.ErrorDiagnosis.recoverable` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.recovery.ErrorDiagnosis.suggested_action` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.results.NLSQResult.convergence_reason` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.results.NLSQResult.covariance` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.results.NLSQResult.final_cost` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.results.NLSQResult.fitted_correlation` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.results.NLSQResult.jacobian` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.results.NLSQResult.message` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.results.NLSQResult.metadata` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.results.NLSQResult.n_function_evals` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.results.NLSQResult.n_iterations` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.results.NLSQResult.parameter_names` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.results.NLSQResult.parameters` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.results.NLSQResult.reduced_chi_squared` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.results.NLSQResult.residuals` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.results.NLSQResult.success` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.results.NLSQResult.uncertainties` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.results.NLSQResult.wall_time_seconds` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.strategies.base.StrategyResult.metadata` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.strategies.base.StrategyResult.n_chunks` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.strategies.base.StrategyResult.peak_memory_mb` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.strategies.base.StrategyResult.result` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.strategies.base.StrategyResult.strategy_name` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.strategies.executors.ExecutionResult.executor_name` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.strategies.executors.ExecutionResult.strategy_result` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.strategies.executors.ExecutionResult.wall_time` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.strategies.sequential.AngleSubset.angle_index` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.strategies.sequential.AngleSubset.c2_data` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.strategies.sequential.AngleSubset.weights` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.strategies.sequential.MultiAngleResult.n_angles_failed` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.strategies.sequential.MultiAngleResult.n_angles_success` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.strategies.sequential.MultiAngleResult.n_angles_total` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.strategies.sequential.MultiAngleResult.per_angle_results` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.strategies.sequential.MultiAngleResult.phi_angles` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.strategies.sequential.MultiAngleResult.success_rate` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.validation.result.ValidationIssue.message` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.validation.result.ValidationIssue.metric_name` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.validation.result.ValidationIssue.metric_value` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.validation.result.ValidationIssue.severity` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.validation.result.ValidationReport.is_valid` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.validation.result.ValidationReport.issues` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.validation.result_validator.ValidationReport.errors` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.validation.result_validator.ValidationReport.metrics` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.validation.result_validator.ValidationReport.passed` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.nlsq.validation.result_validator.ValidationReport.warnings` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.recovery_strategies.RecoveryPlan.action` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.recovery_strategies.RecoveryPlan.message` | heterodyne-only config key |
| `KEEP` | extra_config_key | `optimization.recovery_strategies.RecoveryPlan.modified_config` | heterodyne-only config key |
| `KEEP` | extra_config_key | `runtime.utils.system_validator.runtime_keys` | heterodyne-only config key |
| `KEEP` | extra_config_key | `viz.mcmc_report.ReportConfig.ci_level` | heterodyne-only config key |
| `KEEP` | extra_config_key | `viz.mcmc_report.ReportConfig.float_precision` | heterodyne-only config key |
| `KEEP` | extra_config_key | `viz.mcmc_report.ReportConfig.include_correlation` | heterodyne-only config key |
| `KEEP` | extra_config_key | `viz.mcmc_report.ReportConfig.include_diagnostics` | heterodyne-only config key |
| `KEEP` | extra_config_key | `viz.mcmc_report.ReportConfig.include_timing` | heterodyne-only config key |

### docs (27)

| Disposition | Kind | Qualname / Path | Detail |
|---|---|---|---|
| `KEEP` | missing_doc | `api/cmc_backends.rst` | homodyne has docs page api/cmc_backends.rst; heterodyne does not |
| `KEEP` | missing_doc | `api/cmc_reparameterization.rst` | homodyne has docs page api/cmc_reparameterization.rst; heterodyne does not |
| `KEEP` | missing_doc | `api/cmc_sampler.rst` | homodyne has docs page api/cmc_sampler.rst; heterodyne does not |
| `KEEP` | missing_doc | `api/homodyne_model.rst` | homodyne has docs page api/homodyne_model.rst; heterodyne does not |
| `KEEP` | missing_doc | `api/nlsq_adapter.rst` | homodyne has docs page api/nlsq_adapter.rst; heterodyne does not |
| `KEEP` | missing_doc | `api/nlsq_wrapper.rst` | homodyne has docs page api/nlsq_wrapper.rst; heterodyne does not |
| `KEEP` | missing_doc | `api/optimization_guide.rst` | homodyne has docs page api/optimization_guide.rst; heterodyne does not |
| `KEEP` | missing_doc | `api/theory_engine.rst` | homodyne has docs page api/theory_engine.rst; heterodyne does not |
| `KEEP` | missing_doc | `architecture/homodyne-architecture-overview.md` | homodyne has docs page architecture/homodyne-architecture-overview.md; heterodyne does not |
| `KEEP` | missing_doc | `developer/adrs/adr_001_jax_cpu_only.rst` | homodyne has docs page developer/adrs/adr_001_jax_cpu_only.rst; heterodyne does not |
| `KEEP` | missing_doc | `developer/adrs/adr_002_nlsq_cmc_split.rst` | homodyne has docs page developer/adrs/adr_002_nlsq_cmc_split.rst; heterodyne does not |
| `KEEP` | missing_doc | `developer/adrs/adr_003_anti_degeneracy.rst` | homodyne has docs page developer/adrs/adr_003_anti_degeneracy.rst; heterodyne does not |
| `KEEP` | missing_doc | `developer/adrs/adr_004_consensus_monte_carlo.rst` | homodyne has docs page developer/adrs/adr_004_consensus_monte_carlo.rst; heterodyne does not |
| `KEEP` | missing_doc | `developer/adrs/adr_005_per_angle_scaling.rst` | homodyne has docs page developer/adrs/adr_005_per_angle_scaling.rst; heterodyne does not |
| `KEEP` | missing_doc | `developer/adrs/adr_006_no_consumer_gpu.rst` | homodyne has docs page developer/adrs/adr_006_no_consumer_gpu.rst; heterodyne does not |
| `KEEP` | missing_doc | `developer/adrs/index.rst` | homodyne has docs page developer/adrs/index.rst; heterodyne does not |
| `KEEP` | missing_doc | `theory/anti_degeneracy.rst` | homodyne has docs page theory/anti_degeneracy.rst; heterodyne does not |
| `KEEP` | missing_doc | `theory/anti_degeneracy_defense.rst` | homodyne has docs page theory/anti_degeneracy_defense.rst; heterodyne does not |
| `KEEP` | missing_doc | `theory/homodyne_scattering.rst` | homodyne has docs page theory/homodyne_scattering.rst; heterodyne does not |
| `KEEP` | missing_doc | `theory/theoretical_framework.rst` | homodyne has docs page theory/theoretical_framework.rst; heterodyne does not |
| `KEEP` | missing_doc | `theory/yielding_dynamics.rst` | homodyne has docs page theory/yielding_dynamics.rst; heterodyne does not |
| `KEEP` | missing_doc | `user_guide/01_fundamentals/analysis_modes.rst` | homodyne has docs page user_guide/01_fundamentals/analysis_modes.rst; heterodyne does not |
| `KEEP` | missing_doc | `user_guide/01_fundamentals/homodyne_overview.rst` | homodyne has docs page user_guide/01_fundamentals/homodyne_overview.rst; heterodyne does not |
| `KEEP` | missing_doc | `user_guide/02_data_and_fitting/model_selection.rst` | homodyne has docs page user_guide/02_data_and_fitting/model_selection.rst; heterodyne does not |
| `KEEP` | missing_doc | `user_guide/03_advanced_topics/laminar_flow.rst` | homodyne has docs page user_guide/03_advanced_topics/laminar_flow.rst; heterodyne does not |
| `KEEP` | missing_doc | `user_guide/03_advanced_topics/streaming_mode.rst` | homodyne has docs page user_guide/03_advanced_topics/streaming_mode.rst; heterodyne does not |
| `KEEP` | missing_doc | `user_guide/04_practical_guides/batch_processing.rst` | homodyne has docs page user_guide/04_practical_guides/batch_processing.rst; heterodyne does not |

### exports (17)

| Disposition | Kind | Qualname / Path | Detail |
|---|---|---|---|
| `KEEP` | extra_export | `` | heterodyne-only in __all__: ['CMCConfig', 'CMCResult', 'HAS_VIZ', 'HeterodyneModel', 'NLSQConfig', 'NLSQResult', 'ParameterManager', 'TwoComponentModel', '__version_tuple__', 'fit_cmc_jax', 'get_device_config', 'load_xpcs_config'] |
| `KEEP` | extra_export | `cli` | heterodyne-only in __all__: ['apply_cli_overrides', 'config_main', 'configure_xla', 'dispatch_plots', 'load_and_merge_config', 'load_and_validate_data', 'resolve_phi_angles', 'run_cmc', 'run_nlsq'] |
| `KEEP` | extra_export | `config` | heterodyne-only in __all__: ['ALL_PARAM_NAMES', 'ANGLE_PARAMS', 'AnalysisConfig', 'AnalyzerParametersConfig', 'CMCBackendConfig', 'CMCCombinationConfig', 'CMCInitializationConfig', 'CMCOptimizationConfig', 'CMCPerShardMCMCConfig', 'CMCShardingConfig', 'CMCValidationConfig', 'FRACTION_PARAMS', 'HardwareConfig', 'HeterodyneConfig', 'HmcConfig', 'LoggingConfig', 'MetadataConfig', 'NLSQOptimizationConfig', 'OptimizationConfig', 'OutputConfig', 'PARAMETER_NAME_MAPPING', 'PARAM_GROUPS', 'ParameterConfig', 'ParameterGroupConfig', 'PhiFilteringConfig', 'REFERENCE_PARAMS', 'SAMPLE_PARAMS', 'ScatteringConfig', 'SequentialConfig', 'StratificationConfig', 'StreamingConfig', 'TemporalConfig', 'VELOCITY_PARAMS'] |
| `KEEP` | extra_export | `core` | heterodyne-only in __all__: ['DatasetSize', 'HeterodyneModel', 'HeterodyneModelBase', 'PARAMETER_BOUNDS', 'PerAngleScaling', 'ScalingConfig', 'ShardGrid', 'TwoComponentModel', 'UnifiedHeterodyneEngine', 'apply_diagonal_correction', 'compute_c2_elementwise', 'compute_c2_heterodyne', 'compute_diagonal_mask', 'compute_flat_residuals', 'compute_fraction', 'compute_g1_transport', 'compute_log_likelihood', 'compute_log_likelihood_elementwise', 'compute_nlsq_jacobian', 'compute_posterior_predictive', 'compute_sharded_log_likelihood', 'compute_sharded_log_likelihood_elementwise', 'compute_time_integral_matrix', 'compute_transport_coefficient', 'compute_transport_integral_matrix', 'compute_transport_integral_matrix_theory', 'compute_transport_rate', 'compute_velocity_integral_matrix', 'compute_velocity_rate', 'compute_weights_excluding_diagonal', 'create_physics_factors', 'create_time_integral_matrix', 'estimate_diagonal_excess', 'make_residual_fn', 'make_varying_residual_fn', 'precompute_shard_grid', 'precompute_shard_grid_from_matrix', 'prepare_shards_elementwise', 'safe_divide', 'safe_exp', 'safe_log', 'safe_power', 'safe_sinc', 'safe_sqrt', 'smooth_abs', 'solve_least_squares_chunked_jax', 'solve_least_squares_general_jax', 'solve_least_squares_jax', 'symmetrize', 'trapezoid_cumsum'] |
| `KEEP` | extra_export | `data` | heterodyne-only in __all__: ['AdaptiveChunker', 'AngleRange', 'ChunkInfo', 'DataConfig', 'DataQualityReport', 'DataSlice', 'DatasetSizeCategory', 'FilterResult', 'MemoryManager', 'MemoryMapManager', 'MemoryPressureLevel', 'MemoryPressureMonitor', 'PerformanceEngine', 'PhiAngleFilter', 'PreprocessingPipeline', 'QRange', 'QualityController', 'QualityLevel', 'QualityMetric', 'QualityReport', 'TieredCache', 'TieredCacheConfig', 'apply_q_range_filter', 'apply_sigma_clip', 'apply_time_window', 'categorize_dataset', 'compute_angle_quality', 'compute_data_mask', 'create_loading_plan', 'filter_by_angle_range', 'filter_by_phi', 'find_nearest_angle', 'preprocess_correlation', 'process_chunks_parallel', 'select_angles', 'select_optimal_wavevector', 'validate_correlation_shape', 'validate_no_nan', 'validate_q_range', 'validate_time_arrays', 'validate_weights', 'validate_xpcs_data'] |
| `KEEP` | extra_export | `device` | heterodyne-only in __all__: ['CMCBackend', 'CPUInfo', 'ClusterType', 'HardwareConfig', 'benchmark_cpu_performance', 'configure_cpu_hpc', 'configure_jax_cpu', 'configure_optimal_device', 'detect_cluster_type', 'detect_cpu_info', 'detect_hardware', 'get_available_memory', 'get_backend_name', 'get_device_status', 'get_jax_cpu_flags', 'get_optimal_batch_size'] |
| `KEEP` | extra_export | `io` | heterodyne-only in __all__: ['save_mcmc_diagnostics', 'save_mcmc_results'] |
| `KEEP` | extra_export | `optimization` | heterodyne-only in __all__: ['BoundsError', 'ConvergenceError', 'DegeneracyError', 'FitQualityConfig', 'FitQualityReport', 'NLSQAdapter', 'NLSQConfig', 'NumericalError', 'OptimizationError', 'ValidationError', 'fit_cmc_jax', 'validate_fit_quality'] |
| `KEEP` | extra_export | `optimization.cmc` | heterodyne-only in __all__: ['BimodalConsensusResult', 'BimodalResult', 'DIVERGENCE_RATE_CRITICAL', 'DIVERGENCE_RATE_HIGH', 'DIVERGENCE_RATE_TARGET', 'ModeCluster', 'ParameterScaling', 'ParameterStats', 'ReparamConfig', 'SamplingStats', 'build_init_values_dict', 'check_shard_bimodality', 'cluster_shard_modes', 'compute_nlsq_comparison_metrics', 'compute_precision_analysis', 'compute_t_ref', 'detect_bimodal', 'estimate_per_angle_scaling', 'extract_nlsq_values_for_cmc', 'fit_cmc_jax', 'fit_cmc_sharded', 'get_heterodyne_model', 'get_model_param_count', 'get_param_names_in_order', 'run_cmc_analysis', 'run_nuts_with_retry', 'summarize_cross_shard_bimodality', 'summarize_diagnostics', 'validate_convergence', 'validate_initial_value_bounds', 'validate_model_output'] |
| `KEEP` | extra_export | `optimization.cmc.backends` | heterodyne-only in __all__: ['CPUBackend', 'MCMCBackend', 'PBSConfig', 'PersistentWorkerPool', 'ShardPosterior', 'WorkerPoolBackend', 'combine_shard_samples', 'combine_shard_samples_bimodal', 'consensus_mc', 'robust_consensus_mc', 'should_use_persistent_pool'] |
| `KEEP` | extra_export | `optimization.nlsq` | heterodyne-only in __all__: ['BoundsValidator', 'CMAES_AVAILABLE', 'ChunkedStrategy', 'ConvergenceValidator', 'DegeneracyCheck', 'FitQualityConfig', 'FitQualityReport', 'FitQualityValidator', 'FittingStrategy', 'GradientCollapseDetector', 'HierarchicalResult', 'InputValidator', 'JITStrategy', 'MultiStartOptimizer', 'NLSQValidationConfig', 'ParameterTransform', 'ResidualStrategy', 'ResultValidator', 'SequentialStrategy', 'StrategyResult', 'TimedContext', 'ValidationIssue', 'ValidationReport', 'ValidationSeverity', 'adjust_covariance_for_bounds', 'analyze_parameter_sensitivity', 'build_failed_result', 'build_result_from_arrays', 'build_result_from_scipy', 'classify_fit_quality', 'compute_adaptive_cmaes_params', 'compute_degrees_of_freedom', 'compute_effective_lambda', 'compute_jacobian_condition_number', 'compute_weights', 'denormalize_from_unit_cube', 'detect_hierarchical_trigger', 'estimate_gradient_noise', 'fit_nlsq_multi_phi', 'flatten_upper_triangle', 'normalize_to_unit_cube', 'prepare_fit_data', 'select_strategy', 'suggest_regularization', 'unflatten_upper_triangle', 'validate_fit_quality'] |
| `KEEP` | extra_export | `optimization.nlsq.strategies` | heterodyne-only in __all__: ['ChunkedStrategy', 'FittingStrategy', 'HybridStreamingStrategy', 'JITStrategy', 'OutOfCoreStrategy', 'ResidualJITStrategy', 'ResidualStrategy', 'SequentialStrategy', 'StrategyResult', 'StratifiedLSStrategy', 'select_strategy'] |
| `KEEP` | extra_export | `optimization.nlsq.validation` | heterodyne-only in __all__: ['BoundsValidator', 'ConvergenceValidator', 'FitQualityValidator', 'ValidationIssue', 'ValidationReport', 'ValidationSeverity', 'classify_fit_quality'] |
| `KEEP` | extra_export | `runtime` | heterodyne-only in __all__: ['Severity'] |
| `KEEP` | extra_export | `runtime.utils` | heterodyne-only in __all__: ['Severity'] |
| `KEEP` | extra_export | `utils` | heterodyne-only in __all__: ['ensure_directory', 'resolve_path', 'validate_file_exists', 'validate_output_path'] |
| `KEEP` | extra_export | `viz` | heterodyne-only in __all__: ['ReportConfig', 'generate_report', 'plot_adaptation_summary', 'plot_arviz_pair', 'plot_arviz_posterior', 'plot_arviz_trace', 'plot_corner', 'plot_correlation', 'plot_diagonal_decay', 'plot_divergence_scatter', 'plot_ess_evolution', 'plot_g1_components', 'plot_multi_angle_comparison', 'plot_nlsq_fit', 'plot_nlsq_vs_cmc', 'plot_parameter_uncertainties', 'plot_phi_dependence', 'plot_posterior', 'plot_residual_map', 'plot_trace', 'to_inference_data'] |

### file_inventory (46)

| Disposition | Kind | Qualname / Path | Detail |
|---|---|---|---|
| `KEEP` | missing_py_file | `core.homodyne_model` | homodyne has `core.homodyne_model.py`; heterodyne does not |
| `KEEP` | missing_py_file | `optimization.cmc.backends.multiprocessing` | homodyne has `optimization.cmc.backends.multiprocessing.py`; heterodyne does not |
| `KEEP` | missing_py_file | `optimization.cmc.backends.pjit` | homodyne has `optimization.cmc.backends.pjit.py`; heterodyne does not |
| `KEEP` | missing_py_file | `optimization.nlsq.shear_weighting` | homodyne has `optimization.nlsq.shear_weighting.py`; heterodyne does not |
| `KEEP` | missing_py_file | `optimization.nlsq.strategies.chunking` | homodyne has `optimization.nlsq.strategies.chunking.py`; heterodyne does not |
| `KEEP` | extra_py_file | `core.backend_api` | heterodyne-only file (candidate for absorb-then-delete) |
| `KEEP` | extra_py_file | `core.heterodyne_model` | heterodyne-only file (candidate for absorb-then-delete) |
| `KEEP` | extra_py_file | `core.physics_kernel` | heterodyne-only file (candidate for absorb-then-delete) |
| `KEEP` | extra_py_file | `optimization.cmc.backends.cpu_backend` | heterodyne-only file (candidate for absorb-then-delete) |
| `KEEP` | extra_py_file | `optimization.cmc.backends.multiprocessing_backend` | heterodyne-only file (candidate for absorb-then-delete) |
| `KEEP` | extra_py_file | `optimization.cmc.backends.pjit_backend` | heterodyne-only file (candidate for absorb-then-delete) |
| `KEEP` | extra_py_file | `optimization.cmc.prior_builder` | heterodyne-only file (candidate for absorb-then-delete) |
| `KEEP` | extra_py_file | `optimization.cmc.warmstart` | heterodyne-only file (candidate for absorb-then-delete) |
| `KEEP` | extra_py_file | `optimization.nlsq.strategies.base` | heterodyne-only file (candidate for absorb-then-delete) |
| `KEEP` | extra_py_file | `optimization.nlsq.strategies.chunked` | heterodyne-only file (candidate for absorb-then-delete) |
| `KEEP` | extra_py_file | `optimization.nlsq.strategies.jit_strategy` | heterodyne-only file (candidate for absorb-then-delete) |
| `KEEP` | extra_py_file | `optimization.nlsq.validation.bounds` | heterodyne-only file (candidate for absorb-then-delete) |
| `KEEP` | extra_py_file | `optimization.nlsq.validation.convergence` | heterodyne-only file (candidate for absorb-then-delete) |
| `KEEP` | extra_py_file | `optimization.nlsq.validation.result` | heterodyne-only file (candidate for absorb-then-delete) |
| `KEEP` | missing_doc_file | `api/cmc_backends.rst` | homodyne has docs file `api/cmc_backends.rst`; heterodyne does not |
| `KEEP` | missing_doc_file | `api/cmc_reparameterization.rst` | homodyne has docs file `api/cmc_reparameterization.rst`; heterodyne does not |
| `KEEP` | missing_doc_file | `api/cmc_sampler.rst` | homodyne has docs file `api/cmc_sampler.rst`; heterodyne does not |
| `KEEP` | missing_doc_file | `api/homodyne_model.rst` | homodyne has docs file `api/homodyne_model.rst`; heterodyne does not |
| `KEEP` | missing_doc_file | `api/nlsq_adapter.rst` | homodyne has docs file `api/nlsq_adapter.rst`; heterodyne does not |
| `KEEP` | missing_doc_file | `api/nlsq_wrapper.rst` | homodyne has docs file `api/nlsq_wrapper.rst`; heterodyne does not |
| `KEEP` | missing_doc_file | `api/optimization_guide.rst` | homodyne has docs file `api/optimization_guide.rst`; heterodyne does not |
| `KEEP` | missing_doc_file | `api/theory_engine.rst` | homodyne has docs file `api/theory_engine.rst`; heterodyne does not |
| `KEEP` | missing_doc_file | `architecture/homodyne-architecture-overview.md` | homodyne has docs file `architecture/homodyne-architecture-overview.md`; heterodyne does not |
| `KEEP` | missing_doc_file | `developer/adrs/adr_001_jax_cpu_only.rst` | homodyne has docs file `developer/adrs/adr_001_jax_cpu_only.rst`; heterodyne does not |
| `KEEP` | missing_doc_file | `developer/adrs/adr_002_nlsq_cmc_split.rst` | homodyne has docs file `developer/adrs/adr_002_nlsq_cmc_split.rst`; heterodyne does not |
| `KEEP` | missing_doc_file | `developer/adrs/adr_003_anti_degeneracy.rst` | homodyne has docs file `developer/adrs/adr_003_anti_degeneracy.rst`; heterodyne does not |
| `KEEP` | missing_doc_file | `developer/adrs/adr_004_consensus_monte_carlo.rst` | homodyne has docs file `developer/adrs/adr_004_consensus_monte_carlo.rst`; heterodyne does not |
| `KEEP` | missing_doc_file | `developer/adrs/adr_005_per_angle_scaling.rst` | homodyne has docs file `developer/adrs/adr_005_per_angle_scaling.rst`; heterodyne does not |
| `KEEP` | missing_doc_file | `developer/adrs/adr_006_no_consumer_gpu.rst` | homodyne has docs file `developer/adrs/adr_006_no_consumer_gpu.rst`; heterodyne does not |
| `KEEP` | missing_doc_file | `developer/adrs/index.rst` | homodyne has docs file `developer/adrs/index.rst`; heterodyne does not |
| `KEEP` | missing_doc_file | `theory/anti_degeneracy.rst` | homodyne has docs file `theory/anti_degeneracy.rst`; heterodyne does not |
| `KEEP` | missing_doc_file | `theory/anti_degeneracy_defense.rst` | homodyne has docs file `theory/anti_degeneracy_defense.rst`; heterodyne does not |
| `KEEP` | missing_doc_file | `theory/homodyne_scattering.rst` | homodyne has docs file `theory/homodyne_scattering.rst`; heterodyne does not |
| `KEEP` | missing_doc_file | `theory/theoretical_framework.rst` | homodyne has docs file `theory/theoretical_framework.rst`; heterodyne does not |
| `KEEP` | missing_doc_file | `theory/yielding_dynamics.rst` | homodyne has docs file `theory/yielding_dynamics.rst`; heterodyne does not |
| `KEEP` | missing_doc_file | `user_guide/01_fundamentals/analysis_modes.rst` | homodyne has docs file `user_guide/01_fundamentals/analysis_modes.rst`; heterodyne does not |
| `KEEP` | missing_doc_file | `user_guide/01_fundamentals/homodyne_overview.rst` | homodyne has docs file `user_guide/01_fundamentals/homodyne_overview.rst`; heterodyne does not |
| `KEEP` | missing_doc_file | `user_guide/02_data_and_fitting/model_selection.rst` | homodyne has docs file `user_guide/02_data_and_fitting/model_selection.rst`; heterodyne does not |
| `KEEP` | missing_doc_file | `user_guide/03_advanced_topics/laminar_flow.rst` | homodyne has docs file `user_guide/03_advanced_topics/laminar_flow.rst`; heterodyne does not |
| `KEEP` | missing_doc_file | `user_guide/03_advanced_topics/streaming_mode.rst` | homodyne has docs file `user_guide/03_advanced_topics/streaming_mode.rst`; heterodyne does not |
| `KEEP` | missing_doc_file | `user_guide/04_practical_guides/batch_processing.rst` | homodyne has docs file `user_guide/04_practical_guides/batch_processing.rst`; heterodyne does not |

### signatures (632)

| Disposition | Kind | Qualname / Path | Detail |
|---|---|---|---|
| `KEEP` | extra_in_heterodyne | `__init__.get_device_config` | heterodyne-only: `get_device_config() -> types.ModuleType` |
| `KEEP` | extra_in_heterodyne | `cli.config_handling.apply_cli_overrides` | heterodyne-only: `apply_cli_overrides(config_manager: ConfigManager, args: argparse.Namespace) -> None` |
| `KEEP` | extra_in_heterodyne | `cli.config_handling.load_and_merge_config` | heterodyne-only: `load_and_merge_config(yaml_path: Path | str, cli_args: argparse.Namespace) -> ConfigManager` |
| `KEEP` | extra_in_heterodyne | `cli.data_pipeline.load_and_validate_data` | heterodyne-only: `load_and_validate_data(config_manager: ConfigManager) -> XPCSData` |
| `KEEP` | extra_in_heterodyne | `cli.data_pipeline.prepare_cmc_data` | heterodyne-only: `prepare_cmc_data(data: Any, phi_angles: list[float]) -> dict[str, Any]` |
| `KEEP` | extra_in_heterodyne | `cli.data_pipeline.resolve_phi_angles` | heterodyne-only: `resolve_phi_angles(args: argparse.Namespace, config_manager: ConfigManager, data_phi_angles: np.ndarray | None = None) -> list[float]` |
| `KEEP` | extra_in_heterodyne | `cli.optimization_runner.resolve_nlsq_warmstart` | heterodyne-only: `resolve_nlsq_warmstart(args: argparse.Namespace, output_dir: Path) -> NLSQResult | None` |
| `KEEP` | extra_in_heterodyne | `cli.optimization_runner.run_cmc` | heterodyne-only: `run_cmc(model: HeterodyneModel, c2_data: np.ndarray, phi_angles: list[float], config_manager: ConfigManager, args: argparse.Namespace, output_dir: Path, nlsq_results: list[NLSQResult] | None = None, summary: AnalysisSummaryLogger | None = None, data_phi_angles: np.ndarray | None = None) -> list[CMCResult]` |
| `KEEP` | extra_in_heterodyne | `cli.optimization_runner.run_nlsq` | heterodyne-only: `run_nlsq(model: HeterodyneModel, c2_data: np.ndarray, phi_angles: list[float], config_manager: ConfigManager, args: argparse.Namespace, output_dir: Path, summary: AnalysisSummaryLogger | None = None, data_phi_angles: np.ndarray | None = None) -> list[NLSQResult]` |
| `KEEP` | extra_in_heterodyne | `cli.plot_dispatch.dispatch_plots` | heterodyne-only: `dispatch_plots(model: HeterodyneModel, c2_data: np.ndarray, nlsq_results: list[NLSQResult] | None = None, cmc_results: list[CMCResult] | None = None, output_dir: Path | None = None, mode: str = 'both', phi_angles: list[float] | None = None, data_dict: dict[str, Any] | None = None) -> None` |
| `KEEP` | extra_in_heterodyne | `cli.plot_dispatch.handle_plotting` | heterodyne-only: `handle_plotting(args: Any, result: Any, data: dict[str, Any], config: dict[str, Any] | None = None) -> None` |
| `KEEP` | extra_in_heterodyne | `cli.result_saving.save_cmc_results` | heterodyne-only: `save_cmc_results(results: list[CMCResult], output_dir: Path, phi_angles: list[float]) -> list[Path]` |
| `KEEP` | extra_in_heterodyne | `cli.result_saving.save_results` | heterodyne-only: `save_results(method: str, nlsq_results: list[NLSQResult] | None, cmc_results: list[CMCResult] | None, output_dir: Path, phi_angles: list[float] | None = None, model: Any | None = None) -> dict[str, list[Path]]` |
| `KEEP` | extra_in_heterodyne | `cli.result_saving.save_summary_manifest` | heterodyne-only: `save_summary_manifest(nlsq_paths: list[Path], cmc_paths: list[Path], output_dir: Path) -> Path` |
| `KEEP` | extra_in_heterodyne | `cli.xla_config.auto_configure` | heterodyne-only: `auto_configure() -> dict[str, str]` |
| `KEEP` | extra_in_heterodyne | `cli.xla_config.configure_xla` | heterodyne-only: `configure_xla(num_threads: int | None = None, disable_jit: bool = False, enable_x64: bool = True) -> dict[str, str]` |
| `KEEP` | extra_in_heterodyne | `cli.xla_config.get_cpu_info` | heterodyne-only: `get_cpu_info() -> dict[str, int | str]` |
| `KEEP` | extra_in_heterodyne | `config.manager.ConfigManager.cache_compression` | heterodyne-only: `cache_compression(self) -> bool` |
| `KEEP` | extra_in_heterodyne | `config.manager.ConfigManager.cache_file_path` | heterodyne-only: `cache_file_path(self) -> Path | None` |
| `KEEP` | extra_in_heterodyne | `config.manager.ConfigManager.cache_filename_template` | heterodyne-only: `cache_filename_template(self) -> str | None` |
| `KEEP` | extra_in_heterodyne | `config.manager.ConfigManager.cmc_config` | heterodyne-only: `cmc_config(self) -> dict[str, Any]` |
| `KEEP` | extra_in_heterodyne | `config.manager.ConfigManager.data_file_path` | heterodyne-only: `data_file_path(self) -> Path` |
| `KEEP` | extra_in_heterodyne | `config.manager.ConfigManager.data_folder_path` | heterodyne-only: `data_folder_path(self) -> Path | None` |
| `KEEP` | extra_in_heterodyne | `config.manager.ConfigManager.dt` | heterodyne-only: `dt(self) -> float` |
| `KEEP` | extra_in_heterodyne | `config.manager.ConfigManager.end_frame` | heterodyne-only: `end_frame(self) -> int` |
| `KEEP` | extra_in_heterodyne | `config.manager.ConfigManager.file_format` | heterodyne-only: `file_format(self) -> str` |
| `KEEP` | extra_in_heterodyne | `config.manager.ConfigManager.from_dict` | heterodyne-only: `from_dict(cls, config: dict[str, Any]) -> ConfigManager` |
| `KEEP` | extra_in_heterodyne | `config.manager.ConfigManager.from_json` | heterodyne-only: `from_json(cls, path: Path | str) -> ConfigManager` |
| `KEEP` | extra_in_heterodyne | `config.manager.ConfigManager.from_yaml` | heterodyne-only: `from_yaml(cls, path: Path | str) -> ConfigManager` |
| `KEEP` | extra_in_heterodyne | `config.manager.ConfigManager.get_parameter_value` | heterodyne-only: `get_parameter_value(self, group: str, name: str) -> float` |
| `KEEP` | extra_in_heterodyne | `config.manager.ConfigManager.get_parameter_vary` | heterodyne-only: `get_parameter_vary(self, group: str, name: str) -> bool` |
| `KEEP` | extra_in_heterodyne | `config.manager.ConfigManager.nlsq_config` | heterodyne-only: `nlsq_config(self) -> dict[str, Any]` |
| `KEEP` | extra_in_heterodyne | `config.manager.ConfigManager.optimization_method` | heterodyne-only: `optimization_method(self) -> str` |
| `KEEP` | extra_in_heterodyne | `config.manager.ConfigManager.output_dir` | heterodyne-only: `output_dir(self) -> Path` |
| `KEEP` | extra_in_heterodyne | `config.manager.ConfigManager.parameters_config` | heterodyne-only: `parameters_config(self) -> dict[str, Any]` |
| `KEEP` | extra_in_heterodyne | `config.manager.ConfigManager.phi_angles` | heterodyne-only: `phi_angles(self) -> list[float] | None` |
| `KEEP` | extra_in_heterodyne | `config.manager.ConfigManager.raw_config` | heterodyne-only: `raw_config(self) -> dict[str, Any]` |
| `KEEP` | extra_in_heterodyne | `config.manager.ConfigManager.start_frame` | heterodyne-only: `start_frame(self) -> int` |
| `KEEP` | extra_in_heterodyne | `config.manager.ConfigManager.stator_rotor_gap` | heterodyne-only: `stator_rotor_gap(self) -> float | None` |
| `KEEP` | extra_in_heterodyne | `config.manager.ConfigManager.t_start` | heterodyne-only: `t_start(self) -> int` |
| `KEEP` | extra_in_heterodyne | `config.manager.ConfigManager.time_length` | heterodyne-only: `time_length(self) -> int` |
| `KEEP` | extra_in_heterodyne | `config.manager.ConfigManager.to_yaml` | heterodyne-only: `to_yaml(self, path: Path | str) -> None` |
| `KEEP` | extra_in_heterodyne | `config.manager.ConfigManager.update_optimization_config` | heterodyne-only: `update_optimization_config(self, section: str, key: str, value: Any) -> None` |
| `KEEP` | extra_in_heterodyne | `config.manager.ConfigManager.wavevector_q` | heterodyne-only: `wavevector_q(self) -> float` |
| `KEEP` | extra_in_heterodyne | `config.parameter_manager.ParameterManager.expand_varying_to_full` | heterodyne-only: `expand_varying_to_full(self, varying_params: np.ndarray | jnp.ndarray) -> np.ndarray` |
| `KEEP` | extra_in_heterodyne | `config.parameter_manager.ParameterManager.extract_varying` | heterodyne-only: `extract_varying(self, full_params: np.ndarray | jnp.ndarray) -> np.ndarray` |
| `KEEP` | extra_in_heterodyne | `config.parameter_manager.ParameterManager.fixed_indices` | heterodyne-only: `fixed_indices(self) -> list[int]` |
| `KEEP` | extra_in_heterodyne | `config.parameter_manager.ParameterManager.from_config` | heterodyne-only: `from_config(cls, config: dict[str, Any]) -> ParameterManager` |
| `KEEP` | extra_in_heterodyne | `config.parameter_manager.ParameterManager.get_bounds` | heterodyne-only: `get_bounds(self) -> tuple[np.ndarray, np.ndarray]` |
| `KEEP` | extra_in_heterodyne | `config.parameter_manager.ParameterManager.get_full_values` | heterodyne-only: `get_full_values(self) -> np.ndarray` |
| `KEEP` | extra_in_heterodyne | `config.parameter_manager.ParameterManager.get_group_values` | heterodyne-only: `get_group_values(self, group: str) -> dict[str, float]` |
| `KEEP` | extra_in_heterodyne | `config.parameter_manager.ParameterManager.get_initial_values` | heterodyne-only: `get_initial_values(self) -> np.ndarray` |
| `KEEP` | extra_in_heterodyne | `config.parameter_manager.ParameterManager.get_parameter_dict` | heterodyne-only: `get_parameter_dict(self) -> dict[str, float]` |
| `KEEP` | extra_in_heterodyne | `config.parameter_manager.ParameterManager.n_params` | heterodyne-only: `n_params(self) -> int` |
| `KEEP` | extra_in_heterodyne | `config.parameter_manager.ParameterManager.n_varying` | heterodyne-only: `n_varying(self) -> int` |
| `KEEP` | extra_in_heterodyne | `config.parameter_manager.ParameterManager.set_bounds` | heterodyne-only: `set_bounds(self, name: str, lower: float, upper: float) -> None` |
| `KEEP` | extra_in_heterodyne | `config.parameter_manager.ParameterManager.set_vary` | heterodyne-only: `set_vary(self, name: str, vary: bool) -> None` |
| `KEEP` | extra_in_heterodyne | `config.parameter_manager.ParameterManager.update_values` | heterodyne-only: `update_values(self, params: np.ndarray | dict[str, float]) -> None` |
| `KEEP` | extra_in_heterodyne | `config.parameter_manager.ParameterManager.validate_physics` | heterodyne-only: `validate_physics(self, params: np.ndarray | None = None) -> list[str]` |
| `KEEP` | extra_in_heterodyne | `config.parameter_manager.ParameterManager.varying_indices` | heterodyne-only: `varying_indices(self) -> list[int]` |
| `KEEP` | extra_in_heterodyne | `config.parameter_manager.ParameterManager.varying_names` | heterodyne-only: `varying_names(self) -> list[str]` |
| `KEEP` | extra_in_heterodyne | `config.parameter_names.get_group_indices` | heterodyne-only: `get_group_indices(group: str) -> tuple[int, ...]` |
| `KEEP` | extra_in_heterodyne | `config.parameter_names.get_param_index` | heterodyne-only: `get_param_index(name: str) -> int` |
| `KEEP` | extra_in_heterodyne | `config.parameter_registry.ParameterInfo.clip_value` | heterodyne-only: `clip_value(self, value: float) -> float` |
| `KEEP` | extra_in_heterodyne | `config.parameter_registry.ParameterInfo.validate_value` | heterodyne-only: `validate_value(self, value: float) -> bool` |
| `KEEP` | extra_in_heterodyne | `config.parameter_registry.ParameterRegistry.get_group` | heterodyne-only: `get_group(self, group_name: str) -> list[ParameterInfo]` |
| `KEEP` | extra_in_heterodyne | `config.parameter_registry.ParameterRegistry.get_log_space_names` | heterodyne-only: `get_log_space_names(self) -> list[str]` |
| `KEEP` | extra_in_heterodyne | `config.parameter_registry.ParameterRegistry.get_scaling_names` | heterodyne-only: `get_scaling_names(self) -> list[str]` |
| `KEEP` | extra_in_heterodyne | `config.parameter_registry.ParameterRegistry.get_varying_indices` | heterodyne-only: `get_varying_indices(self, vary_flags: dict[str, bool]) -> list[int]` |
| `KEEP` | extra_in_heterodyne | `config.parameter_space.ParameterSpace.array_to_dict` | heterodyne-only: `array_to_dict(self, arr: np.ndarray | jnp.ndarray) -> dict[str, float]` |
| `KEEP` | extra_in_heterodyne | `config.parameter_space.ParameterSpace.fixed_names` | heterodyne-only: `fixed_names(self) -> list[str]` |
| `KEEP` | extra_in_heterodyne | `config.parameter_space.ParameterSpace.get_bounds_arrays` | heterodyne-only: `get_bounds_arrays(self) -> tuple[np.ndarray, np.ndarray]` |
| `KEEP` | extra_in_heterodyne | `config.parameter_space.ParameterSpace.get_initial_array` | heterodyne-only: `get_initial_array(self) -> np.ndarray` |
| `KEEP` | extra_in_heterodyne | `config.parameter_space.ParameterSpace.get_vary_mask` | heterodyne-only: `get_vary_mask(self) -> np.ndarray` |
| `KEEP` | extra_in_heterodyne | `config.parameter_space.ParameterSpace.n_total` | heterodyne-only: `n_total(self) -> int` |
| `KEEP` | extra_in_heterodyne | `config.parameter_space.ParameterSpace.n_varying` | heterodyne-only: `n_varying(self) -> int` |
| `KEEP` | extra_in_heterodyne | `config.parameter_space.ParameterSpace.scaling_values` | heterodyne-only: `scaling_values(self) -> dict[str, float]` |
| `KEEP` | extra_in_heterodyne | `config.parameter_space.ParameterSpace.to_config` | heterodyne-only: `to_config(self) -> dict[str, Any]` |
| `KEEP` | extra_in_heterodyne | `config.parameter_space.ParameterSpace.update_from_dict` | heterodyne-only: `update_from_dict(self, params: dict[str, float]) -> None` |
| `KEEP` | extra_in_heterodyne | `config.parameter_space.ParameterSpace.validate` | heterodyne-only: `validate(self) -> list[str]` |
| `KEEP` | extra_in_heterodyne | `config.parameter_space.ParameterSpace.varying_names` | heterodyne-only: `varying_names(self) -> list[str]` |
| `KEEP` | extra_in_heterodyne | `config.parameter_space.ParameterSpace.varying_physics_names` | heterodyne-only: `varying_physics_names(self) -> list[str]` |
| `KEEP` | extra_in_heterodyne | `config.parameter_space.PriorDistribution.beta_scaled` | heterodyne-only: `beta_scaled(cls, low: float, high: float, concentration1: float, concentration2: float) -> PriorDistribution` |
| `KEEP` | extra_in_heterodyne | `config.parameter_space.PriorDistribution.halfnormal` | heterodyne-only: `halfnormal(cls, scale: float) -> PriorDistribution` |
| `KEEP` | extra_in_heterodyne | `config.parameter_space.PriorDistribution.lognormal` | heterodyne-only: `lognormal(cls, loc: float, scale: float) -> PriorDistribution` |
| `KEEP` | extra_in_heterodyne | `config.parameter_space.PriorDistribution.normal` | heterodyne-only: `normal(cls, loc: float, scale: float) -> PriorDistribution` |
| `KEEP` | extra_in_heterodyne | `config.parameter_space.PriorDistribution.to_numpyro` | heterodyne-only: `to_numpyro(self, name: str) -> Any` |
| `KEEP` | extra_in_heterodyne | `config.parameter_space.PriorDistribution.truncated_normal` | heterodyne-only: `truncated_normal(cls, loc: float, scale: float, low: float, high: float) -> PriorDistribution` |
| `KEEP` | extra_in_heterodyne | `config.parameter_space.PriorDistribution.uniform` | heterodyne-only: `uniform(cls, low: float, high: float) -> PriorDistribution` |
| `KEEP` | extra_in_heterodyne | `config.parameter_space.clamp_to_open_interval` | heterodyne-only: `clamp_to_open_interval(value: float, low: float, high: float, epsilon: float = 1e-06) -> float` |
| `KEEP` | extra_in_heterodyne | `config.physics_validators.validate_correlation_inputs` | heterodyne-only: `validate_correlation_inputs(t1: np.ndarray, t2: np.ndarray, c2_data: np.ndarray) -> ValidationResult` |
| `KEEP` | extra_in_heterodyne | `config.physics_validators.validate_parameters` | heterodyne-only: `validate_parameters(params: np.ndarray | dict[str, float]) -> ValidationResult` |
| `KEEP` | extra_in_heterodyne | `config.physics_validators.validate_time_integral_safety` | heterodyne-only: `validate_time_integral_safety(alpha: float, t_min: float, t_max: float) -> ValidationResult` |
| `KEEP` | extra_in_heterodyne | `core.backend_api.ensure_array` | heterodyne-only: `ensure_array(x: Any, backend: Backend | None = None) -> Any` |
| `KEEP` | extra_in_heterodyne | `core.backend_api.get_array_module` | heterodyne-only: `get_array_module(backend: Backend | None = None) -> ModuleType` |
| `KEEP` | extra_in_heterodyne | `core.backend_api.get_current_backend` | heterodyne-only: `get_current_backend() -> Backend` |
| `KEEP` | extra_in_heterodyne | `core.backend_api.set_backend` | heterodyne-only: `set_backend(backend: Backend) -> None` |
| `KEEP` | extra_in_heterodyne | `core.diagonal_correction.compute_diagonal_mask` | heterodyne-only: `compute_diagonal_mask(n_times: int, width: int = 1) -> jnp.ndarray` |
| `KEEP` | extra_in_heterodyne | `core.diagonal_correction.compute_weights_excluding_diagonal` | heterodyne-only: `compute_weights_excluding_diagonal(shape: tuple[int, int], width: int = 1) -> jnp.ndarray` |
| `KEEP` | extra_in_heterodyne | `core.diagonal_correction.estimate_diagonal_excess` | heterodyne-only: `estimate_diagonal_excess(c2: jnp.ndarray, width: int = 1) -> dict[str, Any]` |
| `KEEP` | extra_in_heterodyne | `core.fitting.UnifiedHeterodyneEngine.compute_likelihood` | heterodyne-only: `compute_likelihood(self, params: np.ndarray, contrast: float, offset: float, data: np.ndarray, sigma: np.ndarray, t: np.ndarray, phi: float, q: float, dt: float | None = None) -> float` |
| `KEEP` | extra_in_heterodyne | `core.fitting.UnifiedHeterodyneEngine.detect_dataset_size` | heterodyne-only: `detect_dataset_size(self, data: np.ndarray) -> str` |
| `KEEP` | extra_in_heterodyne | `core.fitting.UnifiedHeterodyneEngine.estimate_scaling_parameters` | heterodyne-only: `estimate_scaling_parameters(self, data: np.ndarray, theory: np.ndarray, validate_bounds: bool = True) -> tuple[float, float]` |
| `KEEP` | extra_in_heterodyne | `core.fitting.UnifiedHeterodyneEngine.get_parameter_info` | heterodyne-only: `get_parameter_info(self) -> dict[str, Any]` |
| `KEEP` | extra_in_heterodyne | `core.fitting.UnifiedHeterodyneEngine.validate_inputs` | heterodyne-only: `validate_inputs(self, data: np.ndarray, sigma: np.ndarray | None, t: np.ndarray, phi: np.ndarray | float, q: float) -> None` |
| `KEEP` | extra_in_heterodyne | `core.heterodyne_model.HeterodyneModel.compute_correlation` | heterodyne-only: `compute_correlation(self, phi_angle: float = 0.0, params: np.ndarray | None = None, contrast: float | None = None, offset: float | None = None, angle_idx: int = 0) -> jnp.ndarray` |
| `KEEP` | extra_in_heterodyne | `core.heterodyne_model.HeterodyneModel.compute_fraction` | heterodyne-only: `compute_fraction(self, params: np.ndarray | None = None) -> jnp.ndarray` |
| `KEEP` | extra_in_heterodyne | `core.heterodyne_model.HeterodyneModel.compute_g1_reference` | heterodyne-only: `compute_g1_reference(self, params: np.ndarray | None = None) -> jnp.ndarray` |
| `KEEP` | extra_in_heterodyne | `core.heterodyne_model.HeterodyneModel.compute_g1_sample` | heterodyne-only: `compute_g1_sample(self, params: np.ndarray | None = None) -> jnp.ndarray` |
| `KEEP` | extra_in_heterodyne | `core.heterodyne_model.HeterodyneModel.compute_residuals` | heterodyne-only: `compute_residuals(self, c2_data: np.ndarray | jnp.ndarray, phi_angle: float = 0.0, params: np.ndarray | None = None, weights: np.ndarray | jnp.ndarray | None = None, contrast: float | None = None, offset: float | None = None, angle_idx: int = 0) -> jnp.ndarray` |
| `KEEP` | extra_in_heterodyne | `core.heterodyne_model.HeterodyneModel.create_residual_function` | heterodyne-only: `create_residual_function(self, c2_data: np.ndarray | jnp.ndarray, phi_angle: float, weights: np.ndarray | jnp.ndarray | None = None, angle_idx: int = 0) -> Any` |
| `KEEP` | extra_in_heterodyne | `core.heterodyne_model.HeterodyneModel.dt` | heterodyne-only: `dt(self) -> float` |
| `KEEP` | extra_in_heterodyne | `core.heterodyne_model.HeterodyneModel.from_config` | heterodyne-only: `from_config(cls, config: dict[str, Any]) -> HeterodyneModel` |
| `KEEP` | extra_in_heterodyne | `core.heterodyne_model.HeterodyneModel.get_params` | heterodyne-only: `get_params(self) -> np.ndarray` |
| `KEEP` | extra_in_heterodyne | `core.heterodyne_model.HeterodyneModel.get_params_dict` | heterodyne-only: `get_params_dict(self) -> dict[str, float]` |
| `KEEP` | extra_in_heterodyne | `core.heterodyne_model.HeterodyneModel.n_params` | heterodyne-only: `n_params(self) -> int` |
| `KEEP` | extra_in_heterodyne | `core.heterodyne_model.HeterodyneModel.n_times` | heterodyne-only: `n_times(self) -> int` |
| `KEEP` | extra_in_heterodyne | `core.heterodyne_model.HeterodyneModel.n_varying` | heterodyne-only: `n_varying(self) -> int` |
| `KEEP` | extra_in_heterodyne | `core.heterodyne_model.HeterodyneModel.param_names` | heterodyne-only: `param_names(self) -> tuple[str, ...]` |
| `KEEP` | extra_in_heterodyne | `core.heterodyne_model.HeterodyneModel.q` | heterodyne-only: `q(self) -> float` |
| `KEEP` | extra_in_heterodyne | `core.heterodyne_model.HeterodyneModel.set_params` | heterodyne-only: `set_params(self, params: np.ndarray | dict[str, float]) -> None` |
| `KEEP` | extra_in_heterodyne | `core.heterodyne_model.HeterodyneModel.summary` | heterodyne-only: `summary(self) -> str` |
| `KEEP` | extra_in_heterodyne | `core.heterodyne_model.HeterodyneModel.sync_time_axis` | heterodyne-only: `sync_time_axis(self, t: np.ndarray) -> None` |
| `KEEP` | extra_in_heterodyne | `core.heterodyne_model.HeterodyneModel.t` | heterodyne-only: `t(self) -> jnp.ndarray` |
| `KEEP` | extra_in_heterodyne | `core.heterodyne_model.HeterodyneModel.varying_names` | heterodyne-only: `varying_names(self) -> list[str]` |
| `KEEP` | extra_in_heterodyne | `core.model_mixins.FractionMixin.compute_fraction_evolution` | heterodyne-only: `compute_fraction_evolution(self, f0: float, f1: float, f2: float, f3: float) -> jnp.ndarray` |
| `KEEP` | extra_in_heterodyne | `core.model_mixins.FractionMixin.compute_fraction_matrices` | heterodyne-only: `compute_fraction_matrices(self, f_sample: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]` |
| `KEEP` | extra_in_heterodyne | `core.model_mixins.TransportMixin.compute_half_transport` | heterodyne-only: `compute_half_transport(self, D0: float, alpha: float, offset: float) -> jnp.ndarray` |
| `KEEP` | extra_in_heterodyne | `core.model_mixins.TransportMixin.compute_transport_integral` | heterodyne-only: `compute_transport_integral(self, D0: float, alpha: float, offset: float) -> jnp.ndarray` |
| `KEEP` | extra_in_heterodyne | `core.model_mixins.TransportMixin.compute_transport_rate` | heterodyne-only: `compute_transport_rate(self, D0: float, alpha: float, offset: float) -> jnp.ndarray` |
| `KEEP` | extra_in_heterodyne | `core.model_mixins.VelocityMixin.compute_phase_factor` | heterodyne-only: `compute_phase_factor(self, v_integral: jnp.ndarray, phi: float) -> jnp.ndarray` |
| `KEEP` | extra_in_heterodyne | `core.model_mixins.VelocityMixin.compute_velocity_field` | heterodyne-only: `compute_velocity_field(self, v0: float, beta: float, v_offset: float) -> jnp.ndarray` |
| `KEEP` | extra_in_heterodyne | `core.model_mixins.VelocityMixin.compute_velocity_integral` | heterodyne-only: `compute_velocity_integral(self, v0: float, beta: float, v_offset: float) -> jnp.ndarray` |
| `KEEP` | extra_in_heterodyne | `core.models.HeterodyneModelBase.compute_correlation` | heterodyne-only: `compute_correlation(self, params: jnp.ndarray, t: jnp.ndarray, q: float, dt: float, phi_angle: float, contrast: float = 1.0, offset: float = 1.0) -> jnp.ndarray` |
| `KEEP` | extra_in_heterodyne | `core.models.HeterodyneModelBase.get_default_params` | heterodyne-only: `get_default_params(self) -> np.ndarray` |
| `KEEP` | extra_in_heterodyne | `core.models.HeterodyneModelBase.n_params` | heterodyne-only: `n_params(self) -> int` |
| `KEEP` | extra_in_heterodyne | `core.models.HeterodyneModelBase.param_names` | heterodyne-only: `param_names(self) -> tuple[str, ...]` |
| `KEEP` | extra_in_heterodyne | `core.models.ReducedModel.compute_correlation` | heterodyne-only: `compute_correlation(self, params: jnp.ndarray, t: jnp.ndarray, q: float, dt: float, phi_angle: float, contrast: float = 1.0, offset: float = 1.0) -> jnp.ndarray` |
| `KEEP` | extra_in_heterodyne | `core.models.ReducedModel.get_default_params` | heterodyne-only: `get_default_params(self) -> np.ndarray` |
| `KEEP` | extra_in_heterodyne | `core.models.ReducedModel.n_params` | heterodyne-only: `n_params(self) -> int` |
| `KEEP` | extra_in_heterodyne | `core.models.ReducedModel.param_names` | heterodyne-only: `param_names(self) -> tuple[str, ...]` |
| `KEEP` | extra_in_heterodyne | `core.models.TwoComponentModel.compute_correlation` | heterodyne-only: `compute_correlation(self, params: jnp.ndarray, t: jnp.ndarray, q: float, dt: float, phi_angle: float, contrast: float = 1.0, offset: float = 1.0) -> jnp.ndarray` |
| `KEEP` | extra_in_heterodyne | `core.models.TwoComponentModel.compute_fraction` | heterodyne-only: `compute_fraction(self, params: np.ndarray | jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray` |
| `KEEP` | extra_in_heterodyne | `core.models.TwoComponentModel.compute_g1_reference` | heterodyne-only: `compute_g1_reference(self, params: np.ndarray | jnp.ndarray, t: jnp.ndarray, q: float) -> jnp.ndarray` |
| `KEEP` | extra_in_heterodyne | `core.models.TwoComponentModel.compute_g1_sample` | heterodyne-only: `compute_g1_sample(self, params: np.ndarray | jnp.ndarray, t: jnp.ndarray, q: float) -> jnp.ndarray` |
| `KEEP` | extra_in_heterodyne | `core.models.TwoComponentModel.dict_to_params` | heterodyne-only: `dict_to_params(self, param_dict: dict[str, float]) -> np.ndarray` |
| `KEEP` | extra_in_heterodyne | `core.models.TwoComponentModel.get_default_params` | heterodyne-only: `get_default_params(self) -> np.ndarray` |
| `KEEP` | extra_in_heterodyne | `core.models.TwoComponentModel.n_params` | heterodyne-only: `n_params(self) -> int` |
| `KEEP` | extra_in_heterodyne | `core.models.TwoComponentModel.param_names` | heterodyne-only: `param_names(self) -> tuple[str, ...]` |
| `KEEP` | extra_in_heterodyne | `core.models.TwoComponentModel.params_to_dict` | heterodyne-only: `params_to_dict(self, params: np.ndarray | jnp.ndarray) -> dict[str, float]` |
| `KEEP` | extra_in_heterodyne | `core.numpy_gradients.compute_adaptive_gradient` | heterodyne-only: `compute_adaptive_gradient(fn: Callable[[np.ndarray], float], params: np.ndarray, initial_step: float | None = None, rtol: float = 1e-08) -> GradientResult` |
| `KEEP` | extra_in_heterodyne | `core.numpy_gradients.compute_gradient` | heterodyne-only: `compute_gradient(fn: Callable[[np.ndarray], float], params: np.ndarray, config: DifferentiationConfig | None = None) -> GradientResult` |
| `KEEP` | extra_in_heterodyne | `core.numpy_gradients.compute_gradient_finite_diff` | heterodyne-only: `compute_gradient_finite_diff(fn: Callable[[np.ndarray], float], params: np.ndarray, step_sizes: np.ndarray | None = None) -> np.ndarray` |
| `KEEP` | extra_in_heterodyne | `core.numpy_gradients.compute_gradient_parallel` | heterodyne-only: `compute_gradient_parallel(fn: Callable[[np.ndarray], float], params: np.ndarray, step_sizes: np.ndarray | None = None, n_workers: int | None = None) -> GradientResult` |
| `KEEP` | extra_in_heterodyne | `core.numpy_gradients.compute_hessian_finite_diff` | heterodyne-only: `compute_hessian_finite_diff(cost_fn: Callable[[np.ndarray], float], params: np.ndarray, step_sizes: np.ndarray | None = None) -> np.ndarray` |
| `KEEP` | extra_in_heterodyne | `core.numpy_gradients.compute_jacobian_chunked` | heterodyne-only: `compute_jacobian_chunked(residual_fn: Callable[[np.ndarray], np.ndarray], params: np.ndarray, step_sizes: np.ndarray | None = None, chunk_size: int = 10) -> np.ndarray` |
| `KEEP` | extra_in_heterodyne | `core.numpy_gradients.compute_jacobian_finite_diff` | heterodyne-only: `compute_jacobian_finite_diff(residual_fn: Callable[[np.ndarray], np.ndarray], params: np.ndarray, step_sizes: np.ndarray | None = None) -> np.ndarray` |
| `KEEP` | extra_in_heterodyne | `core.numpy_gradients.validate_gradient` | heterodyne-only: `validate_gradient(analytic_grad: np.ndarray, numerical_grad: np.ndarray, rtol: float = 0.0001, atol: float = 1e-06) -> dict[str, object]` |
| `KEEP` | extra_in_heterodyne | `core.physics_factors.PhysicsFactors.get_q_cosine` | heterodyne-only: `get_q_cosine(self, phi0: float = 0.0) -> jnp.ndarray` |
| `KEEP` | extra_in_heterodyne | `core.physics_factors.PhysicsFactors.time_extent` | heterodyne-only: `time_extent(self) -> float` |
| `KEEP` | extra_in_heterodyne | `core.physics_factors.create_cached_matrices` | heterodyne-only: `create_cached_matrices(factors: PhysicsFactors) -> CachedMatrices` |
| `KEEP` | extra_in_heterodyne | `core.physics_factors.create_physics_factors` | heterodyne-only: `create_physics_factors(n_times: int, dt: float, q: float, phi_angle: float = 0.0, t_start: float = 0.0) -> PhysicsFactors` |
| `KEEP` | extra_in_heterodyne | `core.physics_factors.create_physics_factors_from_config` | heterodyne-only: `create_physics_factors_from_config(config: dict) -> PhysicsFactors` |
| `KEEP` | extra_in_heterodyne | `core.physics_kernel.compute_c2_unified` | heterodyne-only: `compute_c2_unified(params: jnp.ndarray, q: float, dt: float, phi_angle: float, contrast: float = 1.0, offset: float = 1.0, *, eval_strategy: EvalStrategy, t: jnp.ndarray | None = None, shard_grid: Any = None) -> jnp.ndarray` |
| `KEEP` | extra_in_heterodyne | `core.scaling_utils.PerAngleScaling.from_config` | heterodyne-only: `from_config(cls, config: ScalingConfig) -> PerAngleScaling` |
| `KEEP` | extra_in_heterodyne | `core.scaling_utils.PerAngleScaling.get_bounds` | heterodyne-only: `get_bounds(self) -> tuple[np.ndarray, np.ndarray]` |
| `KEEP` | extra_in_heterodyne | `core.scaling_utils.PerAngleScaling.get_for_angle` | heterodyne-only: `get_for_angle(self, angle_idx: int) -> tuple[float, float]` |
| `KEEP` | extra_in_heterodyne | `core.scaling_utils.PerAngleScaling.get_scaling_array` | heterodyne-only: `get_scaling_array(self) -> np.ndarray` |
| `KEEP` | extra_in_heterodyne | `core.scaling_utils.PerAngleScaling.get_varying_values` | heterodyne-only: `get_varying_values(self) -> np.ndarray` |
| `KEEP` | extra_in_heterodyne | `core.scaling_utils.PerAngleScaling.initialize_from_data` | heterodyne-only: `initialize_from_data(self, c2_data: np.ndarray, t1: np.ndarray, t2: np.ndarray, phi_indices: np.ndarray) -> None` |
| `KEEP` | extra_in_heterodyne | `core.scaling_utils.PerAngleScaling.n_scaling_params` | heterodyne-only: `n_scaling_params(self) -> int` |
| `KEEP` | extra_in_heterodyne | `core.scaling_utils.PerAngleScaling.n_varying_scaling` | heterodyne-only: `n_varying_scaling(self) -> int` |
| `KEEP` | extra_in_heterodyne | `core.scaling_utils.PerAngleScaling.update_from_varying` | heterodyne-only: `update_from_varying(self, varying_values: np.ndarray) -> None` |
| `KEEP` | extra_in_heterodyne | `core.scaling_utils.PerAngleScaling.varying_indices` | heterodyne-only: `varying_indices(self) -> np.ndarray` |
| `KEEP` | extra_in_heterodyne | `data.angle_filtering.compute_angle_quality` | heterodyne-only: `compute_angle_quality(c2_3d: np.ndarray, phi_angles: np.ndarray) -> dict[str, Any]` |
| `KEEP` | extra_in_heterodyne | `data.angle_filtering.filter_by_angle_range` | heterodyne-only: `filter_by_angle_range(c2_3d: np.ndarray, phi_angles: np.ndarray, angle_range: AngleRange) -> tuple[np.ndarray, np.ndarray]` |
| `KEEP` | extra_in_heterodyne | `data.angle_filtering.find_nearest_angle` | heterodyne-only: `find_nearest_angle(phi_angles: np.ndarray, target: float) -> int` |
| `KEEP` | extra_in_heterodyne | `data.angle_filtering.select_angles` | heterodyne-only: `select_angles(phi_angles: np.ndarray, indices: np.ndarray | list[int]) -> np.ndarray` |
| `KEEP` | extra_in_heterodyne | `data.config.DataConfig.from_dict` | heterodyne-only: `from_dict(cls, d: dict[str, Any]) -> DataConfig` |
| `KEEP` | extra_in_heterodyne | `data.config.DataConfig.to_dict` | heterodyne-only: `to_dict(self) -> dict[str, Any]` |
| `KEEP` | extra_in_heterodyne | `data.filtering_utils.apply_q_range_filter` | heterodyne-only: `apply_q_range_filter(data: np.ndarray, q_values: np.ndarray, q_range: QRange) -> FilterResult` |
| `KEEP` | extra_in_heterodyne | `data.filtering_utils.apply_sigma_clip` | heterodyne-only: `apply_sigma_clip(c2: np.ndarray, sigma: float = 3.0) -> FilterResult` |
| `KEEP` | extra_in_heterodyne | `data.filtering_utils.apply_time_window` | heterodyne-only: `apply_time_window(c2: np.ndarray, t: np.ndarray, t_min: float, t_max: float) -> FilterResult` |
| `KEEP` | extra_in_heterodyne | `data.filtering_utils.compute_data_mask` | heterodyne-only: `compute_data_mask(c2: np.ndarray, conditions: Sequence[np.ndarray]) -> np.ndarray` |
| `KEEP` | extra_in_heterodyne | `data.memory_manager.AdaptiveChunker.compute_chunks` | heterodyne-only: `compute_chunks(self, total_elements: int, element_bytes: int, prioritize_near_diagonal: bool = False) -> list[ChunkInfo]` |
| `KEEP` | extra_in_heterodyne | `data.memory_manager.MemoryManager.estimate_array_size` | heterodyne-only: `estimate_array_size(shape: tuple[int, ...], dtype: np.dtype | type = np.float64) -> int` |
| `KEEP` | extra_in_heterodyne | `data.memory_manager.MemoryManager.get_budget` | heterodyne-only: `get_budget(self) -> MemoryBudget` |
| `KEEP` | extra_in_heterodyne | `data.memory_manager.MemoryManager.release` | heterodyne-only: `release(self, label: str) -> None` |
| `KEEP` | extra_in_heterodyne | `data.memory_manager.MemoryManager.request` | heterodyne-only: `request(self, n_bytes: int, label: str) -> bool` |
| `KEEP` | extra_in_heterodyne | `data.memory_manager.MemoryManager.suggest_chunk_size` | heterodyne-only: `suggest_chunk_size(self, total_elements: int, element_bytes: int) -> int` |
| `KEEP` | extra_in_heterodyne | `data.memory_manager.MemoryMapManager.close_all` | heterodyne-only: `close_all(self) -> None` |
| `KEEP` | extra_in_heterodyne | `data.memory_manager.MemoryMapManager.estimate_dataset_size` | heterodyne-only: `estimate_dataset_size(self, file_path: Path | str, dataset_path: str) -> int` |
| `KEEP` | extra_in_heterodyne | `data.memory_manager.MemoryMapManager.open_dataset` | heterodyne-only: `open_dataset(self, file_path: Path | str, dataset_path: str) -> np.ndarray` |
| `KEEP` | extra_in_heterodyne | `data.memory_manager.MemoryMapManager.read_slice` | heterodyne-only: `read_slice(self, file_path: Path | str, dataset_path: str, slices: tuple[slice, ...]) -> np.ndarray` |
| `KEEP` | extra_in_heterodyne | `data.memory_manager.MemoryPressureMonitor.available_bytes` | heterodyne-only: `available_bytes(self) -> int` |
| `KEEP` | extra_in_heterodyne | `data.memory_manager.MemoryPressureMonitor.current_pressure` | heterodyne-only: `current_pressure(self) -> MemoryPressureLevel` |
| `KEEP` | extra_in_heterodyne | `data.memory_manager.MemoryPressureMonitor.recommended_budget_fraction` | heterodyne-only: `recommended_budget_fraction(self) -> float` |
| `KEEP` | extra_in_heterodyne | `data.memory_manager.MemoryPressureMonitor.should_reduce_allocation` | heterodyne-only: `should_reduce_allocation(self) -> bool` |
| `KEEP` | extra_in_heterodyne | `data.optimization.categorize_dataset` | heterodyne-only: `categorize_dataset(shape: tuple[int, ...], dtype: np.dtype | type = np.float64, available_memory: int | None = None) -> DatasetSizeCategory` |
| `KEEP` | extra_in_heterodyne | `data.optimization.compute_dataset_statistics` | heterodyne-only: `compute_dataset_statistics(c2: np.ndarray, t: np.ndarray) -> dict[str, float | int]` |
| `KEEP` | extra_in_heterodyne | `data.optimization.create_loading_plan` | heterodyne-only: `create_loading_plan(file_path: Path | str, dataset_shape: tuple[int, ...], dtype: np.dtype | type = np.float64) -> dict[str, Any]` |
| `KEEP` | extra_in_heterodyne | `data.optimization.estimate_optimal_time_range` | heterodyne-only: `estimate_optimal_time_range(c2: np.ndarray, t: np.ndarray, snr_threshold: float = 2.0) -> tuple[float, float]` |
| `KEEP` | extra_in_heterodyne | `data.optimization.process_chunks_parallel` | heterodyne-only: `process_chunks_parallel(c2: np.ndarray, process_fn: Callable[[np.ndarray], np.ndarray], chunk_size: int = 100, max_workers: int | None = None) -> np.ndarray` |
| `KEEP` | extra_in_heterodyne | `data.optimization.recommend_strategy` | heterodyne-only: `recommend_strategy(c2: np.ndarray, t: np.ndarray) -> dict[str, object]` |
| `KEEP` | extra_in_heterodyne | `data.optimization.subsample_correlation` | heterodyne-only: `subsample_correlation(c2: np.ndarray, config: SubsamplingConfig) -> tuple[np.ndarray, np.ndarray]` |
| `KEEP` | extra_in_heterodyne | `data.performance_engine.PerformanceEngine.cache_dataset` | heterodyne-only: `cache_dataset(self, key: str, data: np.ndarray) -> None` |
| `KEEP` | extra_in_heterodyne | `data.performance_engine.PerformanceEngine.clear_cache` | heterodyne-only: `clear_cache(self) -> None` |
| `KEEP` | extra_in_heterodyne | `data.performance_engine.PerformanceEngine.get_cached` | heterodyne-only: `get_cached(self, key: str) -> np.ndarray | None` |
| `KEEP` | extra_in_heterodyne | `data.performance_engine.PerformanceEngine.get_stats` | heterodyne-only: `get_stats(self) -> dict[str, int | float]` |
| `KEEP` | extra_in_heterodyne | `data.performance_engine.PerformanceEngine.invalidate` | heterodyne-only: `invalidate(self, key: str) -> None` |
| `KEEP` | extra_in_heterodyne | `data.performance_engine.TieredCache.clear` | heterodyne-only: `clear(self) -> None` |
| `KEEP` | extra_in_heterodyne | `data.performance_engine.TieredCache.get` | heterodyne-only: `get(self, key: str) -> np.ndarray | None` |
| `KEEP` | extra_in_heterodyne | `data.performance_engine.TieredCache.get_stats` | heterodyne-only: `get_stats(self) -> dict[str, Any]` |
| `KEEP` | extra_in_heterodyne | `data.performance_engine.TieredCache.invalidate` | heterodyne-only: `invalidate(self, key: str) -> None` |
| `KEEP` | extra_in_heterodyne | `data.performance_engine.TieredCache.put` | heterodyne-only: `put(self, key: str, data: np.ndarray) -> None` |
| `KEEP` | extra_in_heterodyne | `data.phi_filtering.PhiAngleFilter.average_symmetric_angles` | heterodyne-only: `average_symmetric_angles(self, c2: np.ndarray, symmetry_center: float = 0.0) -> PhiFilterResult` |
| `KEEP` | extra_in_heterodyne | `data.phi_filtering.PhiAngleFilter.select_angle_range` | heterodyne-only: `select_angle_range(self, c2: np.ndarray, phi_min: float, phi_max: float) -> PhiFilterResult` |
| `KEEP` | extra_in_heterodyne | `data.phi_filtering.PhiAngleFilter.select_angles` | heterodyne-only: `select_angles(self, c2: np.ndarray, target_angles: list[float] | np.ndarray | None = None, angle_tolerance: float = 5.0) -> PhiFilterResult` |
| `KEEP` | extra_in_heterodyne | `data.phi_filtering.PhiAngleFilter.set_available_angles` | heterodyne-only: `set_available_angles(self, phi_angles: np.ndarray) -> None` |
| `KEEP` | extra_in_heterodyne | `data.phi_filtering.filter_by_phi` | heterodyne-only: `filter_by_phi(data: XPCSData, target_angles: list[float] | None = None, angle_tolerance: float = 5.0) -> PhiFilterResult` |
| `KEEP` | extra_in_heterodyne | `data.preprocessing.PreprocessingPipeline.add_step` | heterodyne-only: `add_step(self, name: str, func: Callable[[np.ndarray], np.ndarray]) -> PreprocessingPipeline` |
| `KEEP` | extra_in_heterodyne | `data.preprocessing.PreprocessingPipeline.clip_values` | heterodyne-only: `clip_values(self, min_val: float | None = None, max_val: float | None = None) -> PreprocessingPipeline` |
| `KEEP` | extra_in_heterodyne | `data.preprocessing.PreprocessingPipeline.crop_time` | heterodyne-only: `crop_time(self, t_start: int = 0, t_end: int | None = None) -> PreprocessingPipeline` |
| `KEEP` | extra_in_heterodyne | `data.preprocessing.PreprocessingPipeline.normalize_diagonal` | heterodyne-only: `normalize_diagonal(self) -> PreprocessingPipeline` |
| `KEEP` | extra_in_heterodyne | `data.preprocessing.PreprocessingPipeline.process_with_provenance` | heterodyne-only: `process_with_provenance(self, c2: np.ndarray, source_file: str | None = None) -> tuple[PreprocessingResult, PreprocessingProvenance]` |
| `KEEP` | extra_in_heterodyne | `data.preprocessing.PreprocessingPipeline.remove_outliers` | heterodyne-only: `remove_outliers(self, n_sigma: float = 5.0, replace_with: str = 'median') -> PreprocessingPipeline` |
| `KEEP` | extra_in_heterodyne | `data.preprocessing.PreprocessingPipeline.subtract_baseline` | heterodyne-only: `subtract_baseline(self, baseline: float = 1.0) -> PreprocessingPipeline` |
| `KEEP` | extra_in_heterodyne | `data.preprocessing.PreprocessingPipeline.symmetrize` | heterodyne-only: `symmetrize(self) -> PreprocessingPipeline` |
| `KEEP` | extra_in_heterodyne | `data.preprocessing.PreprocessingProvenance.add_record` | heterodyne-only: `add_record(self, record: TransformationRecord) -> None` |
| `KEEP` | extra_in_heterodyne | `data.preprocessing.PreprocessingProvenance.from_dict` | heterodyne-only: `from_dict(cls, d: dict[str, Any]) -> PreprocessingProvenance` |
| `KEEP` | extra_in_heterodyne | `data.preprocessing.apply_baseline_correction` | heterodyne-only: `apply_baseline_correction(c2: np.ndarray, baseline: np.ndarray | float | None = None, method: str = 'subtract') -> np.ndarray` |
| `KEEP` | extra_in_heterodyne | `data.preprocessing.apply_noise_reduction` | heterodyne-only: `apply_noise_reduction(c2: np.ndarray, method: NoiseReductionMethod = NoiseReductionMethod.GAUSSIAN_SMOOTH, **kwargs: Any) -> np.ndarray` |
| `KEEP` | extra_in_heterodyne | `data.preprocessing.normalize_minmax` | heterodyne-only: `normalize_minmax(c2: np.ndarray) -> np.ndarray` |
| `KEEP` | extra_in_heterodyne | `data.preprocessing.normalize_robust` | heterodyne-only: `normalize_robust(c2: np.ndarray) -> np.ndarray` |
| `KEEP` | extra_in_heterodyne | `data.preprocessing.normalize_zscore` | heterodyne-only: `normalize_zscore(c2: np.ndarray) -> np.ndarray` |
| `KEEP` | extra_in_heterodyne | `data.preprocessing.preprocess_correlation` | heterodyne-only: `preprocess_correlation(c2: np.ndarray, normalize: bool = True, remove_outliers: bool = True, symmetrize: bool = True) -> PreprocessingResult` |
| `KEEP` | extra_in_heterodyne | `data.preprocessing.process_chunked` | heterodyne-only: `process_chunked(c2: np.ndarray, pipeline: PreprocessingPipeline, chunk_size: int = 100) -> PreprocessingResult` |
| `KEEP` | extra_in_heterodyne | `data.quality_controller.QualityController.assess` | heterodyne-only: `assess(self, c2: np.ndarray, t: np.ndarray, q: np.ndarray | None = None, phi_angles: np.ndarray | None = None) -> QualityReport` |
| `KEEP` | extra_in_heterodyne | `data.quality_controller.QualityReport.summary` | heterodyne-only: `summary(self) -> str` |
| `KEEP` | extra_in_heterodyne | `data.quality_controller.apply_auto_corrections` | heterodyne-only: `apply_auto_corrections(c2: np.ndarray, t: np.ndarray, report: QualityReport, config: QualityControlConfig) -> tuple[np.ndarray, list[str]]` |
| `KEEP` | extra_in_heterodyne | `data.quality_controller.assess_stage` | heterodyne-only: `assess_stage(controller: QualityController, c2: np.ndarray, t: np.ndarray, stage: QualityControlStage, config: QualityControlConfig | None = None) -> QualityControlResult` |
| `KEEP` | extra_in_heterodyne | `data.quality_controller.export_report` | heterodyne-only: `export_report(result: QualityControlResult, format: str = 'text') -> str | dict[str, Any]` |
| `KEEP` | extra_in_heterodyne | `data.quality_controller.run_4_stage_pipeline` | heterodyne-only: `run_4_stage_pipeline(controller: QualityController, c2: np.ndarray, t: np.ndarray, config: QualityControlConfig | None = None) -> list[QualityControlResult]` |
| `KEEP` | extra_in_heterodyne | `data.quality_controller.suggest_fixes` | heterodyne-only: `suggest_fixes(report: QualityReport) -> list[dict[str, Any]]` |
| `KEEP` | extra_in_heterodyne | `data.quality_controller.track_quality_history` | heterodyne-only: `track_quality_history(history: list[QualityControlResult], new_result: QualityControlResult) -> list[QualityControlResult]` |
| `KEEP` | extra_in_heterodyne | `data.validation.DataQualityReport.summary` | heterodyne-only: `summary(self) -> str` |
| `KEEP` | extra_in_heterodyne | `data.validation.IncrementalValidationCache.clear_validation_cache` | heterodyne-only: `clear_validation_cache(self) -> None` |
| `KEEP` | extra_in_heterodyne | `data.validation.IncrementalValidationCache.get_cache_stats` | heterodyne-only: `get_cache_stats(self) -> dict[str, int]` |
| `KEEP` | extra_in_heterodyne | `data.validation.validate_time_consistency` | heterodyne-only: `validate_time_consistency(t: np.ndarray, c2_shape: tuple[int, ...], dt_expected: float | None = None) -> DataQualityReport` |
| `KEEP` | extra_in_heterodyne | `data.validators.validate_correlation_shape` | heterodyne-only: `validate_correlation_shape(c2: np.ndarray, expected_shape: tuple[int, ...] | None = None) -> list[str]` |
| `KEEP` | extra_in_heterodyne | `data.validators.validate_no_nan` | heterodyne-only: `validate_no_nan(arr: np.ndarray, name: str) -> list[str]` |
| `KEEP` | extra_in_heterodyne | `data.validators.validate_q_range` | heterodyne-only: `validate_q_range(q: np.ndarray, q_min: float, q_max: float) -> list[str]` |
| `KEEP` | extra_in_heterodyne | `data.validators.validate_time_arrays` | heterodyne-only: `validate_time_arrays(t1: np.ndarray, t2: np.ndarray) -> list[str]` |
| `KEEP` | extra_in_heterodyne | `data.validators.validate_weights` | heterodyne-only: `validate_weights(weights: np.ndarray, data_shape: tuple[int, ...]) -> list[str]` |
| `KEEP` | extra_in_heterodyne | `data.xpcs_loader.XPCSData.has_multi_phi` | heterodyne-only: `has_multi_phi(self) -> bool` |
| `KEEP` | extra_in_heterodyne | `data.xpcs_loader.XPCSData.has_multi_q` | heterodyne-only: `has_multi_q(self) -> bool` |
| `KEEP` | extra_in_heterodyne | `data.xpcs_loader.XPCSData.n_times` | heterodyne-only: `n_times(self) -> int` |
| `KEEP` | extra_in_heterodyne | `data.xpcs_loader.XPCSData.shape` | heterodyne-only: `shape(self) -> tuple[int, ...]` |
| `KEEP` | extra_in_heterodyne | `data.xpcs_loader.XPCSDataLoader.load` | heterodyne-only: `load(self, c2_key: str = 'c2', time_key: str = 't', q_key: str | None = 'q', phi_key: str | None = 'phi', use_cache: bool = False, frame_range: tuple[int, int] | None = None, select_q: float | None = None, q_tolerance: float | None = None, cache_dir: Path | None = None, cache_template: str | None = None, template_vars: dict[str, str] | None = None, cache_compression: bool = True) -> XPCSData` |
| `KEEP` | extra_in_heterodyne | `data.xpcs_loader.load_xpcs_batch` | heterodyne-only: `load_xpcs_batch(file_paths: list[Path | str], c2_key: str = 'c2', time_key: str = 't', format: str | None = None, use_cache: bool = False, validate: bool = False, apply_diag_correction: bool = False, diag_correction_width: int = 1, diag_correction_method: str = 'interpolate', frame_range: tuple[int, int] | None = None, select_q: float | None = None, q_tolerance: float | None = None) -> list[XPCSData]` |
| `KEEP` | extra_in_heterodyne | `data.xpcs_loader.probe_hdf5_structure` | heterodyne-only: `probe_hdf5_structure(file_path: Path | str) -> dict[str, Any]` |
| `KEEP` | extra_in_heterodyne | `data.xpcs_loader.select_optimal_wavevector` | heterodyne-only: `select_optimal_wavevector(q_values: np.ndarray, target_q: float, tolerance: float | None = None) -> tuple[np.ndarray, np.ndarray]` |
| `KEEP` | extra_in_heterodyne | `data.xpcs_loader.validate_loaded_data` | heterodyne-only: `validate_loaded_data(data: XPCSData) -> list[str]` |
| `KEEP` | extra_in_heterodyne | `device.config.configure_optimal_device` | heterodyne-only: `configure_optimal_device(mode: str = 'auto', num_chains: int | None = None) -> HardwareConfig` |
| `KEEP` | extra_in_heterodyne | `device.config.detect_cluster_type` | heterodyne-only: `detect_cluster_type() -> ClusterType` |
| `KEEP` | extra_in_heterodyne | `device.config.get_available_memory` | heterodyne-only: `get_available_memory() -> float` |
| `KEEP` | extra_in_heterodyne | `device.config.get_backend_name` | heterodyne-only: `get_backend_name(backend: CMCBackend) -> Literal['pjit', 'multiprocessing', 'pbs', 'slurm']` |
| `KEEP` | extra_in_heterodyne | `device.config.get_device_status` | heterodyne-only: `get_device_status() -> dict[str, object]` |
| `KEEP` | extra_in_heterodyne | `device.cpu.configure_jax_cpu` | heterodyne-only: `configure_jax_cpu(cpu_info: CPUInfo | None = None, num_devices: int | None = None) -> Mapping[str, str]` |
| `KEEP` | extra_in_heterodyne | `device.cpu.get_jax_cpu_flags` | heterodyne-only: `get_jax_cpu_flags(cpu_info: CPUInfo | None = None, num_devices: int | None = None) -> str` |
| `KEEP` | extra_in_heterodyne | `io.json_utils.load_json` | heterodyne-only: `load_json(path: Path | str) -> dict[str, Any]` |
| `KEEP` | extra_in_heterodyne | `io.json_utils.save_json` | heterodyne-only: `save_json(data: Any, path: Path | str) -> None` |
| `KEEP` | extra_in_heterodyne | `io.mcmc_writers.format_mcmc_summary` | heterodyne-only: `format_mcmc_summary(result: CMCResult) -> str` |
| `KEEP` | extra_in_heterodyne | `io.mcmc_writers.save_mcmc_diagnostics` | heterodyne-only: `save_mcmc_diagnostics(result: CMCResult, output_path: Path | str, r_hat_threshold: float = 1.1, min_bfmi: float = 0.3) -> Path` |
| `KEEP` | extra_in_heterodyne | `io.mcmc_writers.save_mcmc_results` | heterodyne-only: `save_mcmc_results(result: CMCResult, output_dir: Path | str, prefix: str = 'mcmc') -> dict[str, Path]` |
| `KEEP` | extra_in_heterodyne | `io.nlsq_writers.format_nlsq_summary` | heterodyne-only: `format_nlsq_summary(result: NLSQResult) -> str` |
| `KEEP` | extra_in_heterodyne | `io.nlsq_writers.load_nlsq_npz_file` | heterodyne-only: `load_nlsq_npz_file(path: Path | str) -> NLSQResult` |
| `KEEP` | extra_in_heterodyne | `optimization.batch_statistics.compute_batch_statistics` | heterodyne-only: `compute_batch_statistics(results: list[NLSQResult]) -> BatchResult` |
| `KEEP` | extra_in_heterodyne | `optimization.batch_statistics.format_batch_report` | heterodyne-only: `format_batch_report(batch_result: BatchResult) -> str` |
| `KEEP` | extra_in_heterodyne | `optimization.batch_statistics.identify_outlier_fits` | heterodyne-only: `identify_outlier_fits(results: list[NLSQResult], sigma: float = 3.0) -> list[int]` |
| `KEEP` | extra_in_heterodyne | `optimization.checkpoint_manager.CheckpointData.compute_checksum` | heterodyne-only: `compute_checksum(parameters: np.ndarray, cost: float) -> str` |
| `KEEP` | extra_in_heterodyne | `optimization.checkpoint_manager.CheckpointData.from_dict` | heterodyne-only: `from_dict(cls, d: dict[str, Any]) -> CheckpointData` |
| `KEEP` | extra_in_heterodyne | `optimization.checkpoint_manager.CheckpointData.to_dict` | heterodyne-only: `to_dict(self) -> dict[str, Any]` |
| `KEEP` | extra_in_heterodyne | `optimization.checkpoint_manager.CheckpointData.verify_integrity` | heterodyne-only: `verify_integrity(self) -> bool` |
| `KEEP` | extra_in_heterodyne | `optimization.checkpoint_manager.CheckpointManager.cleanup` | heterodyne-only: `cleanup(self, keep: int) -> None` |
| `KEEP` | extra_in_heterodyne | `optimization.checkpoint_manager.CheckpointManager.find_latest_valid` | heterodyne-only: `find_latest_valid(self) -> CheckpointData | None` |
| `KEEP` | extra_in_heterodyne | `optimization.checkpoint_manager.CheckpointManager.list_checkpoints` | heterodyne-only: `list_checkpoints(self) -> list[Path]` |
| `KEEP` | extra_in_heterodyne | `optimization.checkpoint_manager.CheckpointManager.load` | heterodyne-only: `load(self, path: Path) -> CheckpointData` |
| `KEEP` | extra_in_heterodyne | `optimization.checkpoint_manager.CheckpointManager.load_latest` | heterodyne-only: `load_latest(self) -> CheckpointData | None` |
| `KEEP` | extra_in_heterodyne | `optimization.checkpoint_manager.CheckpointManager.save` | heterodyne-only: `save(self, data: CheckpointData) -> Path` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.backends.base.CMCBackend.cleanup` | heterodyne-only: `cleanup(self) -> None` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.backends.base.CMCBackend.estimate_memory` | heterodyne-only: `estimate_memory(self, n_data: int, n_params: int, n_chains: int) -> float` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.backends.base.CMCBackend.get_capabilities` | heterodyne-only: `get_capabilities(self) -> BackendCapabilities` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.backends.base.CMCBackend.validate_resources` | heterodyne-only: `validate_resources(self) -> None` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.backends.base.MCMCBackend.run` | heterodyne-only: `run(self, model: Callable[..., Any], config: CMCConfig, rng_key: jnp.ndarray, init_params: dict[str, jnp.ndarray] | None = None) -> dict[str, Any]` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.backends.base.consensus_mc` | heterodyne-only: `consensus_mc(shard_posteriors: list[ShardPosterior]) -> tuple[np.ndarray, np.ndarray]` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.backends.base.robust_consensus_mc` | heterodyne-only: `robust_consensus_mc(shard_posteriors: list[ShardPosterior], *, outlier_sigma: float = 3.0) -> tuple[np.ndarray, np.ndarray]` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.backends.cpu_backend.CPUBackend.cleanup` | heterodyne-only: `cleanup(self) -> None` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.backends.cpu_backend.CPUBackend.estimate_memory` | heterodyne-only: `estimate_memory(self, n_data: int, n_params: int, n_chains: int) -> float` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.backends.cpu_backend.CPUBackend.get_capabilities` | heterodyne-only: `get_capabilities(self) -> BackendCapabilities` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.backends.cpu_backend.CPUBackend.run` | heterodyne-only: `run(self, model: Callable[..., Any], config: CMCConfig, rng_key: jnp.ndarray, init_params: dict[str, jnp.ndarray] | None = None) -> dict[str, Any]` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.backends.cpu_backend.CPUBackend.validate_resources` | heterodyne-only: `validate_resources(self) -> None` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.backends.multiprocessing_backend.LPTScheduler.as_deque` | heterodyne-only: `as_deque(self) -> deque[int]` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.backends.multiprocessing_backend.LPTScheduler.from_shard_data` | heterodyne-only: `from_shard_data(cls, shard_data_list: list[dict[str, Any]], n_workers: int, n_params: int = _N_PARAMS_HETERODYNE, n_samples: int = 1000) -> LPTScheduler` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.backends.multiprocessing_backend.LPTScheduler.next_shard` | heterodyne-only: `next_shard(self) -> int | None` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.backends.multiprocessing_backend.LPTScheduler.remaining` | heterodyne-only: `remaining(self) -> int` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.backends.multiprocessing_backend.MultiprocessingBackend.cleanup` | heterodyne-only: `cleanup(self) -> None` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.backends.multiprocessing_backend.MultiprocessingBackend.estimate_memory` | heterodyne-only: `estimate_memory(self, n_data: int, n_params: int, n_chains: int) -> float` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.backends.multiprocessing_backend.MultiprocessingBackend.get_capabilities` | heterodyne-only: `get_capabilities(self) -> BackendCapabilities` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.backends.multiprocessing_backend.MultiprocessingBackend.is_available` | heterodyne-only: `is_available(self) -> bool` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.backends.multiprocessing_backend.MultiprocessingBackend.run` | heterodyne-only: `run(self, model: Callable[..., Any], config: CMCConfig, rng_key: jnp.ndarray, init_params: dict[str, jnp.ndarray] | None = None) -> dict[str, Any]` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.backends.multiprocessing_backend.MultiprocessingBackend.run_shards` | heterodyne-only: `run_shards(self, shards: list[dict[str, Any]], config: CMCConfig, initial_values: dict[str, Any] | None = None, parameter_space: Any | None = None, prior_width_multiplier: float = 1.0, nlsq_uncertainties: dict[str, float] | None = None, nlsq_prior_width_factor: float = 2.0, progress_bar: bool = True) -> list[dict[str, Any]]` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.backends.multiprocessing_backend.MultiprocessingBackend.validate_resources` | heterodyne-only: `validate_resources(self) -> None` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.backends.multiprocessing_backend.SharedDataManager.cleanup` | heterodyne-only: `cleanup(self) -> None` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.backends.multiprocessing_backend.SharedDataManager.create_shared_array` | heterodyne-only: `create_shared_array(self, name: str, array: np.ndarray) -> dict[str, Any]` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.backends.multiprocessing_backend.SharedDataManager.create_shared_bytes` | heterodyne-only: `create_shared_bytes(self, name: str, data: bytes) -> dict[str, Any]` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.backends.multiprocessing_backend.SharedDataManager.create_shared_dict` | heterodyne-only: `create_shared_dict(self, name: str, d: dict[str, Any]) -> dict[str, Any]` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.backends.multiprocessing_backend.SharedDataManager.create_shared_shard_arrays` | heterodyne-only: `create_shared_shard_arrays(self, shard_data_list: list[dict[str, Any]]) -> list[dict[str, Any]]` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.backends.pbs.PBSBackend.cleanup` | heterodyne-only: `cleanup(self, job_ids: list[str] | None = None) -> None` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.backends.pbs.PBSBackend.estimate_memory` | heterodyne-only: `estimate_memory(self, n_data: int, n_params: int, n_chains: int) -> float` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.backends.pbs.PBSBackend.get_capabilities` | heterodyne-only: `get_capabilities(self) -> BackendCapabilities` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.backends.pbs.PBSBackend.run_shards` | heterodyne-only: `run_shards(self, shards: list[dict[str, Any]], model_fn: Callable[..., Any], config: CMCConfig, seeds: list[int]) -> list[ShardResult]` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.backends.pbs.PBSBackend.submit_shard` | heterodyne-only: `submit_shard(self, shard_data: dict[str, Any], model_fn: Callable[..., Any], config_dict: dict[str, Any], shard_id: int, seed: int) -> str` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.backends.pbs.PBSBackend.validate_resources` | heterodyne-only: `validate_resources(self) -> None` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.backends.pbs.PBSBackend.wait_for_jobs` | heterodyne-only: `wait_for_jobs(self, job_ids: list[str], timeout: float | None = None) -> list[ShardResult]` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.backends.pjit_backend.PjitBackend.cleanup` | heterodyne-only: `cleanup(self) -> None` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.backends.pjit_backend.PjitBackend.estimate_memory` | heterodyne-only: `estimate_memory(self, n_data: int, n_params: int, n_chains: int) -> float` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.backends.pjit_backend.PjitBackend.get_capabilities` | heterodyne-only: `get_capabilities(self) -> BackendCapabilities` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.backends.pjit_backend.PjitBackend.run` | heterodyne-only: `run(self, model: Callable[..., Any], config: CMCConfig, rng_key: jnp.ndarray, init_params: dict[str, jnp.ndarray] | None = None) -> dict[str, Any]` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.backends.pjit_backend.PjitBackend.validate_resources` | heterodyne-only: `validate_resources(self) -> None` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.backends.pjit_backend.combine_shard_samples` | heterodyne-only: `combine_shard_samples(shard_results: list[dict[str, Any]]) -> dict[str, Any]` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.backends.worker_pool.PersistentWorkerPool.get_result` | heterodyne-only: `get_result(self, timeout: float = 300.0) -> dict[str, Any]` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.backends.worker_pool.PersistentWorkerPool.is_alive` | heterodyne-only: `is_alive(self) -> bool` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.backends.worker_pool.PersistentWorkerPool.n_workers` | heterodyne-only: `n_workers(self) -> int` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.backends.worker_pool.PersistentWorkerPool.result_queue` | heterodyne-only: `result_queue(self) -> multiprocessing.Queue` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.backends.worker_pool.PersistentWorkerPool.results_pending` | heterodyne-only: `results_pending(self) -> bool` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.backends.worker_pool.PersistentWorkerPool.shutdown` | heterodyne-only: `shutdown(self, timeout: float = 10.0) -> None` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.backends.worker_pool.PersistentWorkerPool.submit` | heterodyne-only: `submit(self, task: dict[str, Any]) -> None` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.backends.worker_pool.WorkerPoolBackend.get_name` | heterodyne-only: `get_name(self) -> str` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.backends.worker_pool.WorkerPoolBackend.n_workers` | heterodyne-only: `n_workers(self) -> int` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.backends.worker_pool.WorkerPoolBackend.run` | heterodyne-only: `run(self, model: Callable[..., Any], config: CMCConfig, rng_key: jnp.ndarray, init_params: dict[str, jnp.ndarray] | None = None) -> dict[str, Any]` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.backends.worker_pool.WorkerPoolBackend.should_use_pool` | heterodyne-only: `should_use_pool(n_shards: int, n_workers: int) -> bool` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.backends.worker_pool.should_use_persistent_pool` | heterodyne-only: `should_use_persistent_pool(n_shards: int, n_workers: int) -> bool` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.config.effective_warmup_floor` | heterodyne-only: `effective_warmup_floor(requested: int, *, dense_mass: bool, fast_warmup: bool = False) -> int` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.core.fit_cmc_jax` | heterodyne-only: `fit_cmc_jax(model: HeterodyneModel, c2_data: np.ndarray | jnp.ndarray, phi_angle: float = 0.0, config: CMCConfig | None = None, sigma: np.ndarray | float | None = None, nlsq_result: NLSQResult | None = None, t_override: np.ndarray | None = None, priors_override: dict | None = None, prior_width_multiplier: float = 1.0) -> CMCResult` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.core.fit_cmc_sharded` | heterodyne-only: `fit_cmc_sharded(model: HeterodyneModel, c2_data: np.ndarray | jnp.ndarray, phi_angle: float = 0.0, config: CMCConfig | None = None, sigma: np.ndarray | float | None = None, nlsq_result: NLSQResult | None = None, num_shards: int = 4, sharding_strategy: str = 'random', shard_seed: int | None = None) -> CMCResult` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.data_prep.create_shard_grid` | heterodyne-only: `create_shard_grid(n_times: int, n_shards: int) -> list[tuple[int, int]]` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.data_prep.create_shards` | heterodyne-only: `create_shards(prepared_data: PreparedData, n_shards: int, strategy: ShardingStrategy = ShardingStrategy.ANGLE_BALANCED, *, seed: int = 42) -> list[PreparedData]` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.data_prep.estimate_shard_memory` | heterodyne-only: `estimate_shard_memory(shard: PreparedData) -> int` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.data_prep.merge_shard_results` | heterodyne-only: `merge_shard_results(shard_results: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.data_prep.prepare_cmc_data` | heterodyne-only: `prepare_cmc_data(c2_data: np.ndarray | jnp.ndarray, sigma: np.ndarray | float | None = None, weights: np.ndarray | jnp.ndarray | None = None) -> tuple[jnp.ndarray, jnp.ndarray | float, jnp.ndarray | None]` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.data_prep.prepare_data` | heterodyne-only: `prepare_data(raw_data: dict[str, Any], config: dict[str, Any] | None = None) -> PreparedData` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.data_prep.shard_correlation_data` | heterodyne-only: `shard_correlation_data(c2_data: np.ndarray | jnp.ndarray, shard_grid: list[tuple[int, int]]) -> list[jnp.ndarray]` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.data_prep.validate_shard_data` | heterodyne-only: `validate_shard_data(shard: PreparedData) -> None` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.diagnostics.analyze_divergences` | heterodyne-only: `analyze_divergences(samples: dict[str, np.ndarray] | CMCResult) -> DivergenceReport` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.diagnostics.compute_bfmi` | heterodyne-only: `compute_bfmi(energy: np.ndarray) -> float` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.diagnostics.compute_pair_correlations` | heterodyne-only: `compute_pair_correlations(samples: dict[str, np.ndarray]) -> dict[str, dict[str, float]]` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.diagnostics.compute_trace_diagnostics` | heterodyne-only: `compute_trace_diagnostics(samples: np.ndarray, lags: tuple[int, ...] = (1, 5, 10)) -> dict[str, object]` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.diagnostics.validate_convergence` | heterodyne-only: `validate_convergence(result: CMCResult, r_hat_threshold: float = 1.1, min_ess: int = 100, min_bfmi: float = 0.3) -> ConvergenceReport` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.diagnostics.validate_convergence_sharded` | heterodyne-only: `validate_convergence_sharded(results: list[CMCResult], r_hat_threshold: float = 1.1, min_ess: int = 100, min_bfmi: float = 0.3) -> ConvergenceReport` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.io.list_shards` | heterodyne-only: `list_shards(output_dir: str | Path) -> list[int]` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.io.load_inference_data` | heterodyne-only: `load_inference_data(path: str | Path) -> object` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.io.load_shard_results` | heterodyne-only: `load_shard_results(output_dir: str | Path, shard_id: int) -> dict[str, np.ndarray]` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.io.save_inference_data` | heterodyne-only: `save_inference_data(idata: object, path: str | Path) -> Path` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.io.save_shard_results` | heterodyne-only: `save_shard_results(results: dict[str, np.ndarray], output_dir: str | Path, shard_id: int) -> Path` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.model.estimate_sigma` | heterodyne-only: `estimate_sigma(c2_data: jnp.ndarray, method: str = 'diagonal', nlsq_result: NLSQResult | None = None, n_bootstrap: int = 200, bootstrap_seed: int = 0) -> jnp.ndarray` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.model.get_heterodyne_model` | heterodyne-only: `get_heterodyne_model(t: jnp.ndarray, q: float, dt: float, phi_angle: float, c2_data: jnp.ndarray, noise_scale: float, space: ParameterSpace, contrast: float = 1.0, offset: float = 1.0, shard_grid: ShardGrid | None = None, priors_override: dict | None = None, num_shards: int = 1)` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.model.get_heterodyne_model_constant` | heterodyne-only: `get_heterodyne_model_constant(t: jnp.ndarray, q: float, dt: float, phi_angle: float, c2_data: jnp.ndarray, noise_scale: float, space: ParameterSpace, fixed_contrast: jnp.ndarray, fixed_offset: jnp.ndarray, shard_grid: ShardGrid | None = None, num_shards: int = 1)` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.model.get_heterodyne_model_constant_averaged` | heterodyne-only: `get_heterodyne_model_constant_averaged(t: jnp.ndarray, q: float, dt: float, phi_angle: float, c2_data: jnp.ndarray, noise_scale: float, space: ParameterSpace, mean_contrast: float, mean_offset: float, shard_grid: ShardGrid | None = None, num_shards: int = 1)` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.model.get_heterodyne_model_individual` | heterodyne-only: `get_heterodyne_model_individual(t: jnp.ndarray, q: float, dt: float, phi_angles: jnp.ndarray, c2_data: jnp.ndarray, noise_scale: float, space: ParameterSpace, contrast_prior_loc: jnp.ndarray | float = 0.5, contrast_prior_scale: float = 0.25, offset_prior_loc: jnp.ndarray | float = 1.0, offset_prior_scale: float = 0.25, shard_grids: list[ShardGrid] | None = None, num_shards: int = 1)` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.model.get_heterodyne_model_reparam` | heterodyne-only: `get_heterodyne_model_reparam(t: jnp.ndarray, q: float, dt: float, phi_angle: float, c2_data: jnp.ndarray, noise_scale: float, space: ParameterSpace, nlsq_params: jnp.ndarray | None = None, reparam_config: ReparamConfig | None = None, scalings: dict[str, ParameterScaling] | None = None, contrast: float = 1.0, offset: float = 1.0, shard_grid: ShardGrid | None = None, num_shards: int = 1)` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.model.get_model_for_mode` | heterodyne-only: `get_model_for_mode(per_angle_mode: str, t: jnp.ndarray, q: float, dt: float, phi_angle: float, c2_data: jnp.ndarray, noise_scale: float, space: ParameterSpace, nlsq_result: NLSQResult | None = None, reparam_config: ReparamConfig | None = None, num_shards: int = 1, **kwargs: object) -> Callable[[], None]` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.plotting.plot_diagnostics_summary` | heterodyne-only: `plot_diagnostics_summary(idata: object) -> Figure` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.plotting.plot_pair_plot` | heterodyne-only: `plot_pair_plot(idata: object, var_names: list[str] | None = None, divergences: bool = True) -> Figure` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.plotting.plot_posterior_predictive` | heterodyne-only: `plot_posterior_predictive(idata: object, c2_data: np.ndarray, times: np.ndarray, ax: Axes | None = None) -> Axes` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.plotting.plot_trace_summary` | heterodyne-only: `plot_trace_summary(idata: object, var_names: list[str] | None = None, figsize: tuple[float, float] | None = None) -> Figure` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.prior_builder.PriorBuilder.build` | heterodyne-only: `build(self, param_space: ParameterSpace) -> dict[str, dist.Distribution]` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.prior_builder.build_default_priors_via_builder` | heterodyne-only: `build_default_priors_via_builder(param_space: ParameterSpace, registry: ParameterRegistry | None = None, use_log_space_priors: bool = True) -> dict[str, dist.Distribution]` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.prior_builder.build_log_space_priors_via_builder` | heterodyne-only: `build_log_space_priors_via_builder(param_names: list[str], registry: ParameterRegistry | None = None) -> dict[str, dist.Distribution]` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.priors.build_default_priors` | heterodyne-only: `build_default_priors(param_space: ParameterSpace, registry: ParameterRegistry | None = None, use_log_space_priors: bool = True) -> dict[str, dist.Distribution]` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.priors.build_log_space_priors` | heterodyne-only: `build_log_space_priors(param_names: list[str], registry: ParameterRegistry | None = None) -> dict[str, dist.Distribution]` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.priors.summarize_priors` | heterodyne-only: `summarize_priors(priors: dict[str, dist.Distribution]) -> str` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.priors.temper_priors` | heterodyne-only: `temper_priors(priors: dict[str, dist.Distribution], num_shards: int) -> dict[str, dist.Distribution]` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.priors.validate_priors` | heterodyne-only: `validate_priors(priors: dict[str, dist.Distribution], param_space: ParameterSpace) -> list[str]` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.reparameterization.ReparamConfig.enabled_pairs` | heterodyne-only: `enabled_pairs(self) -> list[tuple[str, str]]` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.reparameterization.ReparamConfig.get_reparam_name` | heterodyne-only: `get_reparam_name(self, prefactor: str) -> str` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.reparameterization.ReparamConfig.is_reparameterized` | heterodyne-only: `is_reparameterized(self, name: str) -> bool` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.reparameterization.d_offset_to_ratio` | heterodyne-only: `d_offset_to_ratio(d_offset: float, d_ref: float) -> float` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.reparameterization.heterodyne_offset_ratios_from_physics` | heterodyne-only: `heterodyne_offset_ratios_from_physics(params: dict[str, float], t_ref: float) -> dict[str, float]` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.reparameterization.heterodyne_physics_offsets_from_ratios` | heterodyne-only: `heterodyne_physics_offsets_from_ratios(ratios: dict[str, float], physics: dict[str, float], t_ref: float) -> dict[str, float]` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.reparameterization.ratio_to_d_offset` | heterodyne-only: `ratio_to_d_offset(ratio: float, d_ref: float) -> float` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.reparameterization.reparam_to_physics_jax` | heterodyne-only: `reparam_to_physics_jax(log_at_tref: jnp.ndarray, alpha: jnp.ndarray, t_ref: float) -> jnp.ndarray` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.results.CMCResult.get_param_summary` | heterodyne-only: `get_param_summary(self, name: str) -> dict[str, float]` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.results.CMCResult.get_samples` | heterodyne-only: `get_samples(self, name: str) -> np.ndarray | None` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.results.CMCResult.n_params` | heterodyne-only: `n_params(self) -> int` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.results.CMCResult.params_dict` | heterodyne-only: `params_dict(self) -> dict[str, float]` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.results.CMCResult.summary` | heterodyne-only: `summary(self) -> str` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.results.CMCResult.validate_convergence` | heterodyne-only: `validate_convergence(self, r_hat_threshold: float = 1.1, min_ess: int = 100, min_bfmi: float = 0.3) -> list[str]` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.results.cmc_result_summary_table` | heterodyne-only: `cmc_result_summary_table(result: CMCResult, ci_level: str = '95', width: int = 80) -> str` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.results.cmc_result_to_arviz` | heterodyne-only: `cmc_result_to_arviz(result: CMCResult) -> Any` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.results.compare_cmc_nlsq` | heterodyne-only: `compare_cmc_nlsq(cmc_result: CMCResult, nlsq_result: Any, consistency_sigma: float = 2.0) -> dict[str, Any]` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.results.merge_shard_cmc_results` | heterodyne-only: `merge_shard_cmc_results(shard_results: list[CMCResult], parameter_names: list[str] | None = None) -> CMCResult` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.sampler.AdaptiveSamplingPlan.get_plan` | heterodyne-only: `get_plan(self) -> SamplingPlan` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.sampler.NUTSSampler.from_plan` | heterodyne-only: `from_plan(cls, plan: SamplingPlan, model: Callable[..., Any], init_strategy: str = 'init_to_median', chain_method: str | None = None) -> NUTSSampler` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.sampler.NUTSSampler.get_diagnostics` | heterodyne-only: `get_diagnostics(self) -> az.InferenceData` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.sampler.NUTSSampler.get_divergence_stats` | heterodyne-only: `get_divergence_stats(self) -> dict[str, float]` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.sampler.NUTSSampler.log_adapter_diagnostics` | heterodyne-only: `log_adapter_diagnostics(self, run_logger: Any | None = None) -> None` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.sampler.NUTSSampler.mcmc` | heterodyne-only: `mcmc(self) -> MCMC` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.sampler.NUTSSampler.plan` | heterodyne-only: `plan(self) -> SamplingPlan` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.sampler.NUTSSampler.run` | heterodyne-only: `run(self, rng_key: jnp.ndarray | None = None, init_params: dict[str, jnp.ndarray] | None = None) -> dict[str, Any]` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.sampler.NUTSSampler.run_with_init_values` | heterodyne-only: `run_with_init_values(self, init_values: dict[str, float], rng_key: jnp.ndarray | None = None) -> dict[str, Any]` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.sampler.SamplingPlan.effective_seed` | heterodyne-only: `effective_seed(self) -> int` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.sampler.SamplingPlan.for_shard` | heterodyne-only: `for_shard(self, shard_size: int, full_size: int) -> SamplingPlan` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.sampler.SamplingStats.is_healthy` | heterodyne-only: `is_healthy(self) -> bool` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.sampler.compute_mcmc_safe_initial_values` | heterodyne-only: `compute_mcmc_safe_initial_values(initial_values: dict[str, float] | None, *, q: float, dt: float, time_grid: Any | None, target_g1: float = 0.5, g1_threshold: float = 0.1) -> dict[str, float] | None` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.scaling.smooth_bound` | heterodyne-only: `smooth_bound(raw: jnp.ndarray, low: float, high: float) -> jnp.ndarray` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.scaling.smooth_bound_inverse` | heterodyne-only: `smooth_bound_inverse(value: float, low: float, high: float) -> float` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.warmstart.clamp_params_to_interior` | heterodyne-only: `clamp_params_to_interior(params: np.ndarray, parameter_names: list[str], *, margin: float = BOUNDARY_INTERIOR_MARGIN) -> tuple[np.ndarray, list[str]]` |
| `KEEP` | extra_in_heterodyne | `optimization.cmc.warmstart.clamp_to_interior` | heterodyne-only: `clamp_to_interior(result: NLSQResult, fixed_param_overrides: dict[str, float] | None = None, *, margin: float = BOUNDARY_INTERIOR_MARGIN) -> NLSQResult` |
| `KEEP` | extra_in_heterodyne | `optimization.gradient_diagnostics.compute_gradient_norm` | heterodyne-only: `compute_gradient_norm(jacobian: np.ndarray) -> float` |
| `KEEP` | extra_in_heterodyne | `optimization.gradient_diagnostics.compute_per_parameter_sensitivity` | heterodyne-only: `compute_per_parameter_sensitivity(jacobian: np.ndarray, param_names: list[str]) -> dict[str, float]` |
| `KEEP` | extra_in_heterodyne | `optimization.gradient_diagnostics.diagnose_gradients` | heterodyne-only: `diagnose_gradients(jacobian: np.ndarray, residuals: np.ndarray, param_names: list[str]) -> GradientHealth` |
| `KEEP` | extra_in_heterodyne | `optimization.gradient_diagnostics.suggest_step_sizes` | heterodyne-only: `suggest_step_sizes(jacobian: np.ndarray, param_names: list[str]) -> dict[str, float]` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.adapter.NLSQAdapter.fit_jax` | heterodyne-only: `fit_jax(self, jax_residual_fn: Callable[..., Any], initial_params: np.ndarray, bounds: tuple[np.ndarray, np.ndarray], config: NLSQConfig, n_data: int) -> NLSQResult` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.adapter.NLSQAdapter.name` | heterodyne-only: `name(self) -> str` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.adapter.NLSQAdapter.supports_bounds` | heterodyne-only: `supports_bounds(self) -> bool` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.adapter.NLSQAdapter.supports_jacobian` | heterodyne-only: `supports_jacobian(self) -> bool` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.adapter.NLSQWrapper.fit` | heterodyne-only: `fit(self, residual_fn: Callable[[np.ndarray], np.ndarray], initial_params: np.ndarray, bounds: tuple[np.ndarray, np.ndarray], config: NLSQConfig, jacobian_fn: Callable[[np.ndarray], np.ndarray] | None = None) -> NLSQResult` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.adapter.NLSQWrapper.name` | heterodyne-only: `name(self) -> str` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.adapter.NLSQWrapper.supports_bounds` | heterodyne-only: `supports_bounds(self) -> bool` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.adapter.NLSQWrapper.supports_jacobian` | heterodyne-only: `supports_jacobian(self) -> bool` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.adapter.get_or_create_fitter` | heterodyne-only: `get_or_create_fitter(n_data: int, n_params: int, phi_angles: tuple[float, ...] | None = None, scaling_mode: str = 'auto', callable_scope: object | None = None) -> tuple[object, bool]` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.adapter_base.NLSQAdapterBase.fit_typed` | heterodyne-only: `fit_typed(self, residual_fn: Callable[[np.ndarray], np.ndarray], initial_params: np.ndarray, bounds: tuple[np.ndarray, np.ndarray], config: NLSQConfig, jacobian_fn: Callable[[np.ndarray], np.ndarray] | None = None) -> NLSQResult` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.adapter_base.NLSQAdapterBase.name` | heterodyne-only: `name(self) -> str` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.adapter_base.NLSQAdapterBase.supports_bounds` | heterodyne-only: `supports_bounds(self) -> bool` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.adapter_base.NLSQAdapterBase.supports_jacobian` | heterodyne-only: `supports_jacobian(self) -> bool` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.adaptive_regularization.AdaptiveRegularizer.adapt_lambda` | heterodyne-only: `adapt_lambda(self, cost_new: float, cost_old: float, lambda_: float | None = None) -> float` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.adaptive_regularization.AdaptiveRegularizer.compute_regularized_step` | heterodyne-only: `compute_regularized_step(self, jacobian: np.ndarray, residuals: np.ndarray, lambda_: float | None = None) -> np.ndarray` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.adaptive_regularization.AdaptiveRegularizer.current_lambda` | heterodyne-only: `current_lambda(self) -> float` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.adaptive_regularization.AdaptiveRegularizer.regularize_covariance` | heterodyne-only: `regularize_covariance(self, covariance: np.ndarray, lambda_: float | None = None) -> np.ndarray` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.adaptive_regularization.AdaptiveRegularizer.reset` | heterodyne-only: `reset(self) -> None` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.anti_degeneracy_controller.GradientCollapseDetector.reset` | heterodyne-only: `reset(self) -> None` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.anti_degeneracy_controller.GradientCollapseDetector.update` | heterodyne-only: `update(self, jacobian: np.ndarray) -> bool` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.anti_degeneracy_controller.compute_effective_lambda` | heterodyne-only: `compute_effective_lambda(base_lambda: float, iteration: int, decay_rate: float = 0.95) -> float` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.anti_degeneracy_controller.detect_hierarchical_trigger` | heterodyne-only: `detect_hierarchical_trigger(degeneracy_checks: list[DegeneracyCheck], cost_history: list[float]) -> bool` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.anti_degeneracy_controller.suggest_regularization` | heterodyne-only: `suggest_regularization(degeneracy_checks: list[DegeneracyCheck], base_lambda: float = 0.0001) -> float` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.cmaes_wrapper.adjust_covariance_for_bounds` | heterodyne-only: `adjust_covariance_for_bounds(cov: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.cmaes_wrapper.build_anti_degeneracy_objective` | heterodyne-only: `build_anti_degeneracy_objective(base_objective: Callable[[np.ndarray], float], bounds: tuple[np.ndarray, np.ndarray], parameter_names: list[str], penalty_weight: float = 0.01) -> Callable[[np.ndarray], float]` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.cmaes_wrapper.compute_adaptive_cmaes_params` | heterodyne-only: `compute_adaptive_cmaes_params(bounds: tuple[np.ndarray, np.ndarray]) -> tuple[int, int]` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.cmaes_wrapper.denormalize_from_unit_cube` | heterodyne-only: `denormalize_from_unit_cube(x_norm: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.cmaes_wrapper.normalize_to_unit_cube` | heterodyne-only: `normalize_to_unit_cube(x: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.core.fit_nlsq_multi_phi` | heterodyne-only: `fit_nlsq_multi_phi(model: HeterodyneModel, c2_data: np.ndarray, phi_angles: list[float] | np.ndarray, config: NLSQConfig | None = None, weights: np.ndarray | None = None) -> list[NLSQResult]` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.data_prep.compute_degrees_of_freedom` | heterodyne-only: `compute_degrees_of_freedom(n_data: int, n_params: int) -> int` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.data_prep.compute_weights` | heterodyne-only: `compute_weights(c2_data: np.ndarray, method: str = 'uniform', sigma: np.ndarray | None = None, exclude_diagonal: bool = False) -> np.ndarray` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.data_prep.flatten_upper_triangle` | heterodyne-only: `flatten_upper_triangle(matrix: np.ndarray, include_diagonal: bool = True) -> np.ndarray` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.data_prep.prepare_fit_data` | heterodyne-only: `prepare_fit_data(c2_data: np.ndarray, weights: np.ndarray | None = None, use_upper_triangle: bool = True, exclude_diagonal: bool = False) -> tuple[np.ndarray, np.ndarray, int]` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.data_prep.unflatten_upper_triangle` | heterodyne-only: `unflatten_upper_triangle(flat: np.ndarray, n: int, include_diagonal: bool = True) -> np.ndarray` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.fit_computation.compute_c2_batch` | heterodyne-only: `compute_c2_batch(params: jnp.ndarray, t: jnp.ndarray, phi_angles: jnp.ndarray, q: float, dt: float, contrast: float = 1.0, offset: float = 1.0) -> jnp.ndarray` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.fit_computation.compute_c2_batch_with_per_angle_scaling` | heterodyne-only: `compute_c2_batch_with_per_angle_scaling(params: jnp.ndarray, t: jnp.ndarray, phi_angles: jnp.ndarray, q: float, dt: float, contrasts: jnp.ndarray, offsets: jnp.ndarray) -> jnp.ndarray` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.gradient_monitor.GradientMonitor.check_exploding` | heterodyne-only: `check_exploding(self, threshold: float = 10000000000.0) -> bool` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.gradient_monitor.GradientMonitor.check_vanishing` | heterodyne-only: `check_vanishing(self, threshold: float = 1e-12) -> bool` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.gradient_monitor.GradientMonitor.get_summary` | heterodyne-only: `get_summary(self) -> dict[str, object]` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.gradient_monitor.GradientMonitor.history` | heterodyne-only: `history(self) -> list[GradientSnapshot]` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.gradient_monitor.GradientMonitor.n_records` | heterodyne-only: `n_records(self) -> int` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.gradient_monitor.GradientMonitor.record` | heterodyne-only: `record(self, iteration: int, gradients: np.ndarray) -> None` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.hierarchical.HierarchicalFitter.fit` | heterodyne-only: `fit(self, model: HeterodyneModel, c2_data: np.ndarray, phi_angle: float, config: NLSQConfig, residual_fn: Callable[[np.ndarray], np.ndarray] | None = None, jacobian_fn: Callable[[np.ndarray], np.ndarray] | None = None) -> NLSQResult` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.hierarchical.HierarchicalResult.convergence_trajectory` | heterodyne-only: `convergence_trajectory(self) -> list[float]` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.jacobian.compare_jacobians` | heterodyne-only: `compare_jacobians(analytic: np.ndarray, numerical: np.ndarray, rtol: float = 0.0001, atol: float = 1e-08) -> dict[str, object]` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.jacobian.compute_numerical_jacobian` | heterodyne-only: `compute_numerical_jacobian(residual_fn: Callable[[np.ndarray], np.ndarray], params: np.ndarray, step_sizes: np.ndarray | None = None) -> np.ndarray` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.jacobian.validate_jacobian` | heterodyne-only: `validate_jacobian(jac: np.ndarray, param_names: list[str] | None = None) -> list[str]` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.multistart.MultiStartConfig.from_dict` | heterodyne-only: `from_dict(cls, d: dict[str, Any]) -> MultiStartConfig` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.multistart.MultiStartOptimizer.fit` | heterodyne-only: `fit(self, residual_fn: Callable[[np.ndarray], np.ndarray], initial_params: np.ndarray, bounds: tuple[np.ndarray, np.ndarray], config: NLSQConfig, jacobian_fn: Callable[[np.ndarray], np.ndarray] | None = None) -> MultiStartResult` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.multistart.MultiStartOptimizer.generate_starting_points` | heterodyne-only: `generate_starting_points(self, initial_params: np.ndarray, bounds: tuple[np.ndarray, np.ndarray]) -> np.ndarray` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.multistart.MultiStartResult.all_results` | heterodyne-only: `all_results(self) -> list[NLSQResult]` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.multistart.MultiStartResult.to_nlsq_result` | heterodyne-only: `to_nlsq_result(self) -> NLSQResult` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.parameter_index_mapper.ParameterIndexMapper.build_from_manager` | heterodyne-only: `build_from_manager(cls, pm: ParameterManager, use_log: bool = False) -> ParameterIndexMapper` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.parameter_index_mapper.ParameterIndexMapper.fixed_full_indices` | heterodyne-only: `fixed_full_indices(self) -> list[int]` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.parameter_index_mapper.ParameterIndexMapper.full_to_varying` | heterodyne-only: `full_to_varying(self, full_idx: int) -> int` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.parameter_index_mapper.ParameterIndexMapper.get_name` | heterodyne-only: `get_name(self, varying_idx: int) -> str` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.parameter_index_mapper.ParameterIndexMapper.is_log_transformed` | heterodyne-only: `is_log_transformed(self, varying_idx: int) -> bool` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.parameter_index_mapper.ParameterIndexMapper.log_mask` | heterodyne-only: `log_mask(self) -> list[bool]` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.parameter_index_mapper.ParameterIndexMapper.n_full` | heterodyne-only: `n_full(self) -> int` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.parameter_index_mapper.ParameterIndexMapper.n_varying` | heterodyne-only: `n_varying(self) -> int` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.parameter_index_mapper.ParameterIndexMapper.name_to_varying` | heterodyne-only: `name_to_varying(self, name: str) -> int` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.parameter_index_mapper.ParameterIndexMapper.varying_full_indices` | heterodyne-only: `varying_full_indices(self) -> list[int]` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.parameter_index_mapper.ParameterIndexMapper.varying_names` | heterodyne-only: `varying_names(self) -> list[str]` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.parameter_index_mapper.ParameterIndexMapper.varying_to_full` | heterodyne-only: `varying_to_full(self, varying_idx: int) -> int` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.parameter_utils.clip_to_bounds` | heterodyne-only: `clip_to_bounds(params: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.parameter_utils.compute_parameter_sensitivity` | heterodyne-only: `compute_parameter_sensitivity(residual_fn: Callable[[np.ndarray], np.ndarray], params: np.ndarray, step_sizes: np.ndarray | None = None) -> np.ndarray` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.parameter_utils.format_parameter_table` | heterodyne-only: `format_parameter_table(names: list[str], values: np.ndarray, uncertainties: np.ndarray | None = None, bounds: tuple[np.ndarray, np.ndarray] | None = None) -> str` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.parameter_utils.perturb_parameters` | heterodyne-only: `perturb_parameters(params: np.ndarray, scale: float, bounds: tuple[np.ndarray, np.ndarray], rng: np.random.Generator | None = None) -> np.ndarray` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.progress.ProgressTracker.get_history` | heterodyne-only: `get_history(self) -> list[ProgressRecord]` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.progress.ProgressTracker.is_stalled` | heterodyne-only: `is_stalled(self, patience: int = 10, min_improvement: float = 1e-10) -> bool` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.progress.ProgressTracker.n_records` | heterodyne-only: `n_records(self) -> int` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.progress.ProgressTracker.record` | heterodyne-only: `record(self, iteration: int, cost: float, params: np.ndarray | None = None, gradient: np.ndarray | None = None) -> None` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.progress.ProgressTracker.summary` | heterodyne-only: `summary(self) -> str` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.result_builder.build_failed_result` | heterodyne-only: `build_failed_result(parameter_names: list[str], message: str, initial_params: np.ndarray | None = None, wall_time: float | None = None, metadata: dict[str, Any] | None = None) -> NLSQResult` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.result_builder.build_result_from_arrays` | heterodyne-only: `build_result_from_arrays(parameters: np.ndarray, parameter_names: list[str], residuals: np.ndarray, n_data: int, success: bool = True, message: str = '', jacobian: np.ndarray | None = None, n_iterations: int = 0, n_function_evals: int = 0, wall_time: float | None = None, metadata: dict[str, Any] | None = None) -> NLSQResult` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.result_builder.build_result_from_nlsq` | heterodyne-only: `build_result_from_nlsq(nlsq_result: Any, parameter_names: list[str], n_data: int, wall_time: float = 0.0, metadata: dict[str, Any] | None = None) -> NLSQResult` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.result_builder.build_result_from_scipy` | heterodyne-only: `build_result_from_scipy(opt_result: OptimizeResult, parameter_names: list[str], n_data: int, wall_time: float | None = None, metadata: dict[str, Any] | None = None) -> NLSQResult` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.results.NLSQResult.get_correlation_matrix` | heterodyne-only: `get_correlation_matrix(self) -> np.ndarray | None` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.results.NLSQResult.get_param` | heterodyne-only: `get_param(self, name: str) -> float` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.results.NLSQResult.get_uncertainty` | heterodyne-only: `get_uncertainty(self, name: str) -> float | None` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.results.NLSQResult.n_params` | heterodyne-only: `n_params(self) -> int` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.results.NLSQResult.params_dict` | heterodyne-only: `params_dict(self) -> dict[str, float]` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.results.NLSQResult.summary` | heterodyne-only: `summary(self) -> str` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.results.NLSQResult.validate` | heterodyne-only: `validate(self) -> list[str]` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.strategies.base.FittingStrategy.fit` | heterodyne-only: `fit(self, model: HeterodyneModel, c2_data: np.ndarray, phi_angle: float, config: NLSQConfig, weights: np.ndarray | None = None) -> StrategyResult` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.strategies.base.FittingStrategy.name` | heterodyne-only: `name(self) -> str` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.strategies.base.select_strategy` | heterodyne-only: `select_strategy(n_data: int, n_params: int, config: NLSQConfig, *, available_memory_gb: float | None = None) -> FittingStrategy` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.strategies.chunked.ChunkedStrategy.fit` | heterodyne-only: `fit(self, model: HeterodyneModel, c2_data: np.ndarray, phi_angle: float, config: NLSQConfig, weights: np.ndarray | None = None) -> StrategyResult` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.strategies.chunked.ChunkedStrategy.from_config` | heterodyne-only: `from_config(config: NLSQConfig) -> ChunkedStrategy` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.strategies.chunked.ChunkedStrategy.name` | heterodyne-only: `name(self) -> str` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.strategies.executors.ExecutionResult.result` | heterodyne-only: `result(self) -> NLSQResult` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.strategies.executors.ExecutionResult.success` | heterodyne-only: `success(self) -> bool` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.strategies.hybrid_streaming.HybridStreamingStrategy.fit` | heterodyne-only: `fit(self, model: HeterodyneModel, c2_data: np.ndarray, phi_angle: float, config: NLSQConfig, weights: np.ndarray | None = None) -> StrategyResult` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.strategies.hybrid_streaming.HybridStreamingStrategy.name` | heterodyne-only: `name(self) -> str` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.strategies.jit_strategy.JITStrategy.clear_cache` | heterodyne-only: `clear_cache(self) -> None` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.strategies.jit_strategy.JITStrategy.fit` | heterodyne-only: `fit(self, model: HeterodyneModel, c2_data: np.ndarray, phi_angle: float, config: NLSQConfig, weights: np.ndarray | None = None) -> StrategyResult` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.strategies.jit_strategy.JITStrategy.n_cached` | heterodyne-only: `n_cached(self) -> int` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.strategies.jit_strategy.JITStrategy.name` | heterodyne-only: `name(self) -> str` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.strategies.out_of_core.OutOfCoreStrategy.fit` | heterodyne-only: `fit(self, model: HeterodyneModel, c2_data: np.ndarray, phi_angle: float, config: NLSQConfig, weights: np.ndarray | None = None) -> StrategyResult` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.strategies.out_of_core.OutOfCoreStrategy.name` | heterodyne-only: `name(self) -> str` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.strategies.residual.ResidualStrategy.fit` | heterodyne-only: `fit(self, model: HeterodyneModel, c2_data: np.ndarray, phi_angle: float, config: NLSQConfig, weights: np.ndarray | None = None) -> StrategyResult` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.strategies.residual.ResidualStrategy.name` | heterodyne-only: `name(self) -> str` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.strategies.residual_jit.ResidualJITStrategy.fit` | heterodyne-only: `fit(self, model: HeterodyneModel, c2_data: np.ndarray, phi_angle: float, config: NLSQConfig, weights: np.ndarray | None = None) -> StrategyResult` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.strategies.residual_jit.ResidualJITStrategy.name` | heterodyne-only: `name(self) -> str` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.strategies.sequential.SequentialStrategy.fit` | heterodyne-only: `fit(self, model: HeterodyneModel, c2_data: np.ndarray, phi_angle: float, config: NLSQConfig, weights: np.ndarray | None = None) -> StrategyResult` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.strategies.sequential.SequentialStrategy.fit_multi_angle` | heterodyne-only: `fit_multi_angle(self, model: HeterodyneModel, c2_data: np.ndarray, phi_angles: list[float] | np.ndarray, config: NLSQConfig, weights: np.ndarray | None = None, weighting: str = 'inverse_variance') -> MultiAngleResult` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.strategies.sequential.SequentialStrategy.fit_multi_angle_list` | heterodyne-only: `fit_multi_angle_list(self, model: HeterodyneModel, c2_data: np.ndarray, phi_angles: list[float] | np.ndarray, config: NLSQConfig, weights: np.ndarray | None = None) -> list[StrategyResult]` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.strategies.sequential.SequentialStrategy.name` | heterodyne-only: `name(self) -> str` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.strategies.stratified_ls.StratifiedLSStrategy.fit` | heterodyne-only: `fit(self, model: HeterodyneModel, c2_data: np.ndarray, phi_angle: float, config: NLSQConfig, weights: np.ndarray | None = None) -> StrategyResult` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.strategies.stratified_ls.StratifiedLSStrategy.name` | heterodyne-only: `name(self) -> str` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.transforms.ParameterTransform.get_optimizer_bounds` | heterodyne-only: `get_optimizer_bounds(self) -> tuple[np.ndarray, np.ndarray]` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.transforms.ParameterTransform.get_optimizer_x0` | heterodyne-only: `get_optimizer_x0(self) -> np.ndarray` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.transforms.ParameterTransform.jacobian_correction` | heterodyne-only: `jacobian_correction(self, optimizer_params: np.ndarray) -> np.ndarray` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.transforms.ParameterTransform.log_mask` | heterodyne-only: `log_mask(self) -> np.ndarray` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.transforms.ParameterTransform.n_varying` | heterodyne-only: `n_varying(self) -> int` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.transforms.ParameterTransform.to_model` | heterodyne-only: `to_model(self, optimizer_params: np.ndarray) -> np.ndarray` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.transforms.ParameterTransform.to_optimizer` | heterodyne-only: `to_optimizer(self, full_params: np.ndarray) -> np.ndarray` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.transforms.ParameterTransform.varying_names` | heterodyne-only: `varying_names(self) -> list[str]` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.transforms.compress_to_varying` | heterodyne-only: `compress_to_varying(full_params: np.ndarray, vary_flags: dict[str, bool]) -> np.ndarray` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.transforms.expand_to_full` | heterodyne-only: `expand_to_full(varying_params: np.ndarray, fixed_values: np.ndarray, vary_flags: dict[str, bool]) -> np.ndarray` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.validation.bounds.BoundsValidator.validate` | heterodyne-only: `validate(self, result: NLSQResult) -> ValidationReport` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.validation.convergence.ConvergenceValidator.validate` | heterodyne-only: `validate(self, result: NLSQResult, config: NLSQConfig | None = None) -> ValidationReport` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.validation.fit_quality.FitQualityValidator.validate` | heterodyne-only: `validate(self, result: NLSQResult) -> ValidationReport` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.validation.fit_quality.FitQualityValidator.validate_extended` | heterodyne-only: `validate_extended(self, result: Any, bounds: tuple[np.ndarray, np.ndarray] | None = None, config: FitQualityConfig | None = None, param_labels: list[str] | None = None) -> FitQualityReport` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.validation.fit_quality.classify_fit_quality` | heterodyne-only: `classify_fit_quality(reduced_chi_squared: float | None, n_at_bounds: int = 0) -> str` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.validation.input_validator.InputValidator.validate` | heterodyne-only: `validate(self, data: np.ndarray, initial_params: np.ndarray, bounds: tuple[np.ndarray, np.ndarray]) -> ValidationReport` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.validation.result.ResultValidator.validate` | heterodyne-only: `validate(self, result: NLSQResult) -> ValidationReport` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.validation.result.ValidationReport.errors` | heterodyne-only: `errors(self) -> list[ValidationIssue]` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.validation.result.ValidationReport.summary` | heterodyne-only: `summary(self) -> str` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.validation.result.ValidationReport.warnings` | heterodyne-only: `warnings(self) -> list[ValidationIssue]` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.validation.result_validator.ValidationReport.summary` | heterodyne-only: `summary(self) -> str` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.validation.result_validator.check_bound_saturation` | heterodyne-only: `check_bound_saturation(result: NLSQResult, registry: ParameterRegistry, tolerance: float = 0.01) -> list[str]` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.validation.result_validator.check_chi_squared` | heterodyne-only: `check_chi_squared(result: NLSQResult, max_reduced_chi2: float = 10.0, min_reduced_chi2: float = 0.01) -> list[str]` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.validation.result_validator.check_covariance_health` | heterodyne-only: `check_covariance_health(result: NLSQResult) -> list[str]` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.validation.result_validator.check_uncertainty_ratios` | heterodyne-only: `check_uncertainty_ratios(result: NLSQResult, max_relative_uncertainty: float = 1.0) -> list[str]` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.validation.result_validator.validate_result` | heterodyne-only: `validate_result(result: NLSQResult, registry: ParameterRegistry | None = None) -> ValidationReport` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.wrapper.NLSQWrapper.name` | heterodyne-only: `name(self) -> str` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.wrapper.NLSQWrapper.supports_bounds` | heterodyne-only: `supports_bounds(self) -> bool` |
| `KEEP` | extra_in_heterodyne | `optimization.nlsq.wrapper.NLSQWrapper.supports_jacobian` | heterodyne-only: `supports_jacobian(self) -> bool` |
| `KEEP` | extra_in_heterodyne | `optimization.numerical_validation.safe_compute` | heterodyne-only: `safe_compute(fn: Callable[..., np.ndarray], *args: Any, fallback: np.ndarray | None = None, **kwargs: Any) -> np.ndarray` |
| `KEEP` | extra_in_heterodyne | `optimization.numerical_validation.validate_array` | heterodyne-only: `validate_array(data: Any, *, name: str = 'array') -> np.ndarray` |
| `KEEP` | extra_in_heterodyne | `optimization.numerical_validation.validate_parameters` | heterodyne-only: `validate_parameters(values: Any, *, names: list[str], bounds: list[tuple[float, float]] | None = None) -> np.ndarray` |
| `KEEP` | extra_in_heterodyne | `optimization.recovery_strategies.apply_recovery` | heterodyne-only: `apply_recovery(plan: RecoveryPlan, config: NLSQConfig) -> NLSQConfig` |
| `KEEP` | extra_in_heterodyne | `optimization.recovery_strategies.diagnose_failure` | heterodyne-only: `diagnose_failure(result: NLSQResult, config: NLSQConfig) -> RecoveryPlan` |
| `KEEP` | extra_in_heterodyne | `optimization.recovery_strategies.suggest_fixed_parameters` | heterodyne-only: `suggest_fixed_parameters(result: NLSQResult) -> list[str]` |
| `KEEP` | extra_in_heterodyne | `runtime.utils.system_validator.SystemValidator.run_all` | heterodyne-only: `run_all(self) -> list[ValidationResult]` |
| `KEEP` | extra_in_heterodyne | `runtime.utils.system_validator.SystemValidator.test_heterodyne_installation` | heterodyne-only: `test_heterodyne_installation(self) -> ValidationResult` |
| `KEEP` | extra_in_heterodyne | `runtime.utils.system_validator.SystemValidator.test_jax_cpu_backend` | heterodyne-only: `test_jax_cpu_backend(self) -> ValidationResult` |
| `KEEP` | extra_in_heterodyne | `runtime.utils.system_validator.SystemValidator.test_jax_x64_precision` | heterodyne-only: `test_jax_x64_precision(self) -> ValidationResult` |
| `KEEP` | extra_in_heterodyne | `runtime.utils.system_validator.SystemValidator.test_numpy_installation` | heterodyne-only: `test_numpy_installation(self) -> ValidationResult` |
| `KEEP` | extra_in_heterodyne | `runtime.utils.system_validator.SystemValidator.test_numpyro_installation` | heterodyne-only: `test_numpyro_installation(self) -> ValidationResult` |
| `KEEP` | extra_in_heterodyne | `runtime.utils.system_validator.SystemValidator.test_python_version` | heterodyne-only: `test_python_version(self) -> ValidationResult` |
| `KEEP` | extra_in_heterodyne | `runtime.utils.system_validator.run_validation` | heterodyne-only: `run_validation(verbose: bool = True) -> list[ValidationResult]` |
| `KEEP` | extra_in_heterodyne | `utils.async_io.AsyncWriter.write_json` | heterodyne-only: `write_json(self, path: str | Path, data: dict[str, Any]) -> Future[None]` |
| `KEEP` | extra_in_heterodyne | `utils.async_io.AsyncWriter.write_npz` | heterodyne-only: `write_npz(self, path: str | Path, **arrays: np.ndarray) -> Future[None]` |
| `KEEP` | extra_in_heterodyne | `utils.logging.AnalysisSummaryLogger.convergence_status` | heterodyne-only: `convergence_status(self) -> str | None` |
| `KEEP` | extra_in_heterodyne | `utils.logging.AnalysisSummaryLogger.is_failure` | heterodyne-only: `is_failure(self) -> bool` |
| `KEEP` | extra_in_heterodyne | `utils.logging.ConvergenceLogger.log_convergence` | heterodyne-only: `log_convergence(self, reason: str, final_loss: float) -> None` |
| `KEEP` | extra_in_heterodyne | `utils.logging.ConvergenceLogger.log_diagnostic` | heterodyne-only: `log_diagnostic(self, metric_name: str, value: float, threshold: float, higher_is_better: bool = True) -> None` |
| `KEEP` | extra_in_heterodyne | `utils.logging.ConvergenceLogger.log_iteration` | heterodyne-only: `log_iteration(self, iteration: int, loss: float, gradient_norm: float | None = None, step_size: float | None = None) -> None` |
| `KEEP` | extra_in_heterodyne | `utils.path_validation.ensure_directory` | heterodyne-only: `ensure_directory(path: str | Path) -> Path` |
| `KEEP` | extra_in_heterodyne | `utils.path_validation.resolve_path` | heterodyne-only: `resolve_path(path: str | Path) -> Path` |
| `KEEP` | extra_in_heterodyne | `utils.path_validation.validate_file_exists` | heterodyne-only: `validate_file_exists(path: str | Path, description: str = 'File') -> Path` |
| `KEEP` | extra_in_heterodyne | `utils.path_validation.validate_output_path` | heterodyne-only: `validate_output_path(path: str | Path, create_parents: bool = True) -> Path` |
| `KEEP` | extra_in_heterodyne | `viz.datashader_backend.render_correlation_heatmap` | heterodyne-only: `render_correlation_heatmap(c2: np.ndarray, times: np.ndarray, width: int = 800, height: int = 800) -> PILImage.Image | np.ndarray` |
| `KEEP` | extra_in_heterodyne | `viz.datashader_backend.render_multi_angle_grid` | heterodyne-only: `render_multi_angle_grid(c2_data: np.ndarray, phi_angles: np.ndarray, times: np.ndarray, ncols: int = 3) -> PILImage.Image | np.ndarray` |
| `KEEP` | extra_in_heterodyne | `viz.datashader_backend.render_residual_heatmap` | heterodyne-only: `render_residual_heatmap(residuals: np.ndarray, times: np.ndarray, width: int = 800, height: int = 800) -> PILImage.Image | np.ndarray` |
| `KEEP` | extra_in_heterodyne | `viz.diagnostics.plot_convergence_trace` | heterodyne-only: `plot_convergence_trace(losses: np.ndarray, ax: Axes | None = None, log_scale: bool = True) -> Axes` |
| `KEEP` | extra_in_heterodyne | `viz.diagnostics.plot_diagonal_overlay` | heterodyne-only: `plot_diagonal_overlay(c2: np.ndarray, corrected_c2: np.ndarray, times: np.ndarray, ax: Axes | None = None) -> Axes` |
| `KEEP` | extra_in_heterodyne | `viz.diagnostics.plot_pair_correlation` | heterodyne-only: `plot_pair_correlation(samples: dict[str, np.ndarray], param_names: list[str] | None = None, ax: Axes | None = None) -> Axes` |
| `KEEP` | extra_in_heterodyne | `viz.diagnostics.plot_parameter_sensitivity` | heterodyne-only: `plot_parameter_sensitivity(sensitivity_dict: dict[str, float], ax: Axes | None = None) -> Axes` |
| `KEEP` | extra_in_heterodyne | `viz.diagnostics.plot_residual_histogram` | heterodyne-only: `plot_residual_histogram(residuals: np.ndarray, ax: Axes | None = None) -> Axes` |
| `KEEP` | extra_in_heterodyne | `viz.diagnostics.plot_residual_map` | heterodyne-only: `plot_residual_map(residuals: np.ndarray, times: np.ndarray, ax: Axes | None = None) -> Axes` |
| `KEEP` | extra_in_heterodyne | `viz.diagnostics.plot_trace_posterior` | heterodyne-only: `plot_trace_posterior(samples: dict[str, np.ndarray], param_names: list[str] | None = None, figsize: tuple[float, float] | None = None) -> plt.Figure` |
| `KEEP` | extra_in_heterodyne | `viz.diagnostics.plot_weight_map` | heterodyne-only: `plot_weight_map(weights: np.ndarray, times: np.ndarray, ax: Axes | None = None) -> Axes` |
| `KEEP` | extra_in_heterodyne | `viz.experimental_plots.plot_correlation` | heterodyne-only: `plot_correlation(c2: np.ndarray, t: np.ndarray | None = None, title: str = 'Two-Time Correlation', save_path: Path | str | None = None, cmap: str = 'viridis', vmin: float | None = None, vmax: float | None = None) -> plt.Figure` |
| `KEEP` | extra_in_heterodyne | `viz.experimental_plots.plot_diagonal_decay` | heterodyne-only: `plot_diagonal_decay(c2: np.ndarray, t: np.ndarray | None = None, fitted_c2: np.ndarray | None = None, save_path: Path | str | None = None) -> plt.Figure` |
| `KEEP` | extra_in_heterodyne | `viz.experimental_plots.plot_g1_components` | heterodyne-only: `plot_g1_components(model: HeterodyneModel, params: np.ndarray | None = None, save_path: Path | str | None = None) -> plt.Figure` |
| `KEEP` | extra_in_heterodyne | `viz.experimental_plots.plot_phi_dependence` | heterodyne-only: `plot_phi_dependence(c2_multi_phi: np.ndarray, phi_angles: np.ndarray, t_slice: int | None = None, save_path: Path | str | None = None) -> plt.Figure` |
| `KEEP` | extra_in_heterodyne | `viz.mcmc_arviz.to_inference_data` | heterodyne-only: `to_inference_data(cmc_result: CMCResult) -> Any` |
| `KEEP` | extra_in_heterodyne | `viz.mcmc_comparison.plot_multi_angle_comparison` | heterodyne-only: `plot_multi_angle_comparison(results_by_phi: dict[float, CMCResult], save_path: Path | str | None = None) -> Figure` |
| `KEEP` | extra_in_heterodyne | `viz.mcmc_comparison.plot_nlsq_vs_cmc` | heterodyne-only: `plot_nlsq_vs_cmc(nlsq_result: NLSQResult, cmc_result: CMCResult, save_path: Path | str | None = None) -> Figure` |
| `KEEP` | extra_in_heterodyne | `viz.mcmc_diagnostics.plot_adaptation_summary` | heterodyne-only: `plot_adaptation_summary(result: CMCResult, save_path: Path | str | None = None) -> Figure` |
| `KEEP` | extra_in_heterodyne | `viz.mcmc_diagnostics.plot_divergence_scatter` | heterodyne-only: `plot_divergence_scatter(result: CMCResult, save_path: Path | str | None = None) -> Figure` |
| `KEEP` | extra_in_heterodyne | `viz.mcmc_diagnostics.plot_ess_evolution` | heterodyne-only: `plot_ess_evolution(result: CMCResult, save_path: Path | str | None = None) -> Figure` |
| `KEEP` | extra_in_heterodyne | `viz.mcmc_plots.plot_autocorrelation` | heterodyne-only: `plot_autocorrelation(samples: dict[str, np.ndarray], param_names: list[str] | None = None, max_lag: int = 50, save_path: Path | str | None = None, figsize: tuple[float, float] | None = None, dpi: int = 150) -> Figure` |
| `KEEP` | extra_in_heterodyne | `viz.mcmc_plots.plot_corner` | heterodyne-only: `plot_corner(result: CMCResult, params: list[str] | None = None, save_path: Path | str | None = None) -> Figure` |
| `KEEP` | extra_in_heterodyne | `viz.mcmc_plots.plot_divergence_scatter` | heterodyne-only: `plot_divergence_scatter(samples: dict[str, np.ndarray], divergent_mask: np.ndarray, param_pairs: list[tuple[str, str]] | None = None, save_path: Path | str | None = None, figsize: tuple[float, float] | None = None, dpi: int = 150) -> Figure` |
| `KEEP` | extra_in_heterodyne | `viz.mcmc_plots.plot_energy` | heterodyne-only: `plot_energy(samples: dict[str, np.ndarray], save_path: Path | str | None = None, figsize: tuple[float, float] = (8, 5), dpi: int = 150) -> Figure` |
| `KEEP` | extra_in_heterodyne | `viz.mcmc_plots.plot_forest` | heterodyne-only: `plot_forest(samples: dict[str, np.ndarray], param_names: list[str] | None = None, credible_interval: float = 0.94, save_path: Path | str | None = None, figsize: tuple[float, float] | None = None, dpi: int = 150) -> Figure` |
| `KEEP` | extra_in_heterodyne | `viz.mcmc_plots.plot_posterior` | heterodyne-only: `plot_posterior(result: CMCResult, params: list[str] | None = None, save_path: Path | str | None = None, figsize: tuple[float, float] | None = None) -> Figure` |
| `KEEP` | extra_in_heterodyne | `viz.mcmc_plots.plot_posterior_predictive` | heterodyne-only: `plot_posterior_predictive(c2_observed: np.ndarray, c2_predicted: np.ndarray, times: np.ndarray, phi_angle: float | None = None, n_samples_overlay: int = 50, save_path: Path | str | None = None, figsize: tuple[float, float] = (14, 5), dpi: int = 150) -> Figure` |
| `KEEP` | extra_in_heterodyne | `viz.mcmc_plots.plot_rank_histogram` | heterodyne-only: `plot_rank_histogram(samples: dict[str, np.ndarray], param_names: list[str] | None = None, save_path: Path | str | None = None, figsize: tuple[float, float] | None = None, dpi: int = 150) -> Figure` |
| `KEEP` | extra_in_heterodyne | `viz.mcmc_plots.plot_rhat_summary` | heterodyne-only: `plot_rhat_summary(rhat_dict: dict[str, float], threshold: float = 1.01, save_path: Path | str | None = None, figsize: tuple[float, float] | None = None, dpi: int = 150) -> Figure` |
| `KEEP` | extra_in_heterodyne | `viz.mcmc_plots.plot_shard_comparison` | heterodyne-only: `plot_shard_comparison(shard_results: list[dict[str, np.ndarray]], param_names: list[str] | None = None, save_path: Path | str | None = None, figsize: tuple[float, float] | None = None, dpi: int = 150) -> Figure` |
| `KEEP` | extra_in_heterodyne | `viz.mcmc_plots.plot_trace` | heterodyne-only: `plot_trace(result: CMCResult, params: list[str] | None = None, save_path: Path | str | None = None) -> Figure` |
| `KEEP` | extra_in_heterodyne | `viz.mcmc_report.generate_report` | heterodyne-only: `generate_report(nlsq_results: list[NLSQResult] | None = None, cmc_results: list[CMCResult] | None = None, output_dir: Path | str | None = None, config: ReportConfig | None = None) -> Path | str` |
| `KEEP` | extra_in_heterodyne | `viz.nlsq_plots.plot_chi_squared_landscape` | heterodyne-only: `plot_chi_squared_landscape(chi2_values: np.ndarray, param_values: np.ndarray, param_name: str, best_value: float | None = None, save_path: Path | str | None = None, figsize: tuple[float, float] = (8, 5), dpi: int = 150) -> Figure` |
| `KEEP` | extra_in_heterodyne | `viz.nlsq_plots.plot_fit_surface` | heterodyne-only: `plot_fit_surface(c2_exp: np.ndarray, c2_fit: np.ndarray, times: np.ndarray, phi_angle: float | None = None, save_path: Path | str | None = None, figsize: tuple[float, float] = (14, 5), dpi: int = 150) -> Figure` |
| `KEEP` | extra_in_heterodyne | `viz.nlsq_plots.plot_multistart_summary` | heterodyne-only: `plot_multistart_summary(results: list[dict[str, Any]], save_path: Path | str | None = None, figsize: tuple[float, float] = (12, 5), dpi: int = 150) -> Figure` |
| `KEEP` | extra_in_heterodyne | `viz.nlsq_plots.plot_nlsq_fit` | heterodyne-only: `plot_nlsq_fit(c2_data: np.ndarray, result: NLSQResult, t: np.ndarray | None = None, save_path: Path | str | None = None, figsize: tuple[float, float] = (15, 5)) -> Figure` |
| `KEEP` | extra_in_heterodyne | `viz.nlsq_plots.plot_parameter_evolution` | heterodyne-only: `plot_parameter_evolution(history: list[dict[str, Any]], param_names: list[str], save_path: Path | str | None = None, figsize: tuple[float, float] | None = None, dpi: int = 150) -> Figure` |
| `KEEP` | extra_in_heterodyne | `viz.nlsq_plots.plot_parameter_uncertainties` | heterodyne-only: `plot_parameter_uncertainties(result: NLSQResult, save_path: Path | str | None = None) -> Figure` |
| `KEEP` | extra_in_heterodyne | `viz.nlsq_plots.plot_per_angle_residuals` | heterodyne-only: `plot_per_angle_residuals(residuals: np.ndarray, phi_angles: np.ndarray, times: np.ndarray, save_path: Path | str | None = None, figsize: tuple[float, float] | None = None, dpi: int = 150) -> Figure` |
| `KEEP` | extra_in_heterodyne | `viz.nlsq_plots.plot_residual_map` | heterodyne-only: `plot_residual_map(result: NLSQResult, c2_data: np.ndarray, t: np.ndarray | None = None, save_path: Path | str | None = None) -> Figure` |
| `KEEP` | extra_in_heterodyne | `viz.nlsq_plots.plot_scaling_comparison` | heterodyne-only: `plot_scaling_comparison(solver_scaling: np.ndarray, lstsq_scaling: np.ndarray, phi_angles: np.ndarray, param_labels: tuple[str, str] = ('contrast', 'offset'), save_path: Path | str | None = None, figsize: tuple[float, float] = (10, 5), dpi: int = 150) -> Figure` |
| `KEEP` | extra_in_heterodyne | `viz.validation.plot_bounds_check` | heterodyne-only: `plot_bounds_check(param_values: np.ndarray, param_names: list[str], bounds: list[tuple[float, float]], ax: Axes | None = None, edge_fraction: float = 0.05) -> Axes` |
| `KEEP` | extra_in_heterodyne | `viz.validation.plot_quality_report` | heterodyne-only: `plot_quality_report(quality_report: QualityReport, ax: Axes | None = None) -> Axes` |
| `KEEP` | extra_in_heterodyne | `viz.validation.plot_validation_report` | heterodyne-only: `plot_validation_report(report: ValidationReport, ax: Axes | None = None) -> Axes` |

## P2 — Observable drift — log formats, error stems, docs heading drift (1475 gaps)

### docs (837)

| Disposition | Kind | Qualname / Path | Detail |
|---|---|---|---|
| `KEEP` | heading_drift | `api/cli.rst` | heading `Arguments` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/cli.rst` | heading `Internal CLI Modules` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/cli.rst` | heading `Shell Alias` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/cli.rst` | heading `Shell Aliases` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/cli.rst` | heading `Shell Completion System` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/cli.rst` | heading `Usage Example` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/cli.rst` | heading `Usage Examples` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/cli.rst` | heading `homodyne` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/cli.rst` | heading `homodyne CLI` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/cli.rst` | heading `homodyne-cleanup` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/cli.rst` | heading `homodyne-config` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/cli.rst` | heading `homodyne-config-xla` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/cli.rst` | heading `homodyne-post-install` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/cmc.rst` | heading `Accessing diagnostics` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/cmc.rst` | heading `CMC without warm-start (exploratory)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/cmc.rst` | heading `CMCConfig` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/cmc.rst` | heading `CMCResult` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/cmc.rst` | heading `Per-Angle Mode` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/cmc.rst` | heading `Quality Filtering` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/cmc.rst` | heading `Shard Size Selection` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/cmc.rst` | heading `Standard NLSQ → CMC workflow` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/cmc.rst` | heading `Usage Examples` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/cmc.rst` | heading `YAML Configuration Reference` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/cmc.rst` | heading `fit\_mcmc\_jax` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/cmc.rst` | heading `homodyne.optimization.cmc` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/config.rst` | heading `ConfigManager` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/config.rst` | heading `Default Parameter Values` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/config.rst` | heading `ParameterRegistry` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/config.rst` | heading `ParameterSpace` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/config.rst` | heading `Usage Example` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/config.rst` | heading `YAML Configuration Schema` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/config.rst` | heading `homodyne.config` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/core.rst` | heading `Usage Example` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/core.rst` | heading `homodyne.core` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/core.rst` | heading `homodyne.core.physics` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/core.rst` | heading `homodyne.core.physics\_utils` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/core.rst` | heading `homodyne.core.scaling\_utils` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/data.rst` | heading `Data Validation` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/data.rst` | heading `From a ConfigManager` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/data.rst` | heading `From a YAML config file` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/data.rst` | heading `HDF5 Format Requirements` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/data.rst` | heading `NPZ Caching` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/data.rst` | heading `Output Data Structure` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/data.rst` | heading `Supplementary Modules` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/data.rst` | heading `Usage Examples` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/data.rst` | heading `Using the convenience function` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/data.rst` | heading `XPCSDataLoader` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/data.rst` | heading `homodyne.data` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/device.rst` | heading `Benchmark Metrics` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/device.rst` | heading `CPU Information` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/device.rst` | heading `CPU Module` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/device.rst` | heading `CPU-Specific Functions` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/device.rst` | heading `Configuration Module` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/device.rst` | heading `Configuration Result` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/device.rst` | heading `Device Configuration` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/device.rst` | heading `Device Module` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/device.rst` | heading `Device Status` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/device.rst` | heading `Environment Variables` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/device.rst` | heading `HPC Best Practices` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/device.rst` | heading `HPC Configuration` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/device.rst` | heading `HPC Optimization` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/device.rst` | heading `JAX Configuration` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/device.rst` | heading `Module Contents` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/device.rst` | heading `Optimal Batch Size` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/device.rst` | heading `Overview` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/device.rst` | heading `Performance Benchmarking` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/device.rst` | heading `Performance Estimates` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/device.rst` | heading `Primary Functions` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/device.rst` | heading `See Also` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/device.rst` | heading `Status Information` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/device.rst` | heading `Troubleshooting` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/device.rst` | heading `Usage Example` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/index.rst` | heading `Module Overview` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/io.rst` | heading `I/O Module` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/io.rst` | heading `Module Contents` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/io.rst` | heading `Output Files` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/io.rst` | heading `Output Structure` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/io.rst` | heading `Overview` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/io.rst` | heading `Primary Functions` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/io.rst` | heading `See Also` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/io.rst` | heading `Supported Types` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/io.rst` | heading `Usage Examples` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/jax_backend.rst` | heading `Automatic Differentiation` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/jax_backend.rst` | heading `Backend Availability` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/jax_backend.rst` | heading `Batched Computations` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/jax_backend.rst` | heading `Cache management` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/jax_backend.rst` | heading `Chi-Squared` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/jax_backend.rst` | heading `Computing gradients for optimization` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/jax_backend.rst` | heading `Computing g₂ for a single angle` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/jax_backend.rst` | heading `Diagnostics` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/jax_backend.rst` | heading `Meshgrid Cache` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/jax_backend.rst` | heading `Usage Examples` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/jax_backend.rst` | heading `batch\_chi\_squared` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/jax_backend.rst` | heading `compute\_chi\_squared` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/jax_backend.rst` | heading `compute\_g1\_diffusion` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/jax_backend.rst` | heading `compute\_g1\_shear` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/jax_backend.rst` | heading `compute\_g1\_total` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/jax_backend.rst` | heading `compute\_g2\_scaled` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/jax_backend.rst` | heading `compute\_g2\_scaled\_with\_factors` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/jax_backend.rst` | heading `get\_device\_info` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/jax_backend.rst` | heading `get\_performance\_summary` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/jax_backend.rst` | heading `g₁ Correlation Functions` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/jax_backend.rst` | heading `g₂ Correlation Functions` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/jax_backend.rst` | heading `homodyne.core.jax\_backend` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/jax_backend.rst` | heading `validate\_backend` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/jax_backend.rst` | heading `vectorized\_g2\_computation` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/models.rst` | heading `CombinedModel` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/models.rst` | heading `DiffusionModel` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/models.rst` | heading `Module-level factory` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/models.rst` | heading `Physical Model Summary` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/models.rst` | heading `PhysicsModelBase` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/models.rst` | heading `ShearModel` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/models.rst` | heading `Usage Example` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/models.rst` | heading `homodyne.core.models` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/models.rst` | heading `|                  | :math:`\dot\gamma_\text{offset}, \varphi_0`                       |       |` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/models.rst` | heading `| Mode             | Parameters                                                        | Count |` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/models.rst` | heading `| ``static``       | :math:`D_0, \alpha, D_\text{offset}`                              | 3     |` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/optimization.rst` | heading `Choosing Between NLSQ and CMC` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/optimization.rst` | heading `Common Workflow Patterns` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/optimization.rst` | heading `Package Structure` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/optimization.rst` | heading `Pattern 1: NLSQ only (fast, no uncertainties)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/optimization.rst` | heading `Pattern 2: NLSQ warm-start → CMC (recommended for publications)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/optimization.rst` | heading `Pattern 3: CLI workflow (two-step)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/optimization.rst` | heading `homodyne.optimization` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/utils.rst` | heading `Basic logger` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/utils.rst` | heading `CLI log configuration` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/utils.rst` | heading `Contextual logging (CMC shard)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/utils.rst` | heading `Exception logging` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/utils.rst` | heading `Performance decorator` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/utils.rst` | heading `Phase timing` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/utils.rst` | heading `Usage Example` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/utils.rst` | heading `Usage Examples` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/utils.rst` | heading `detect\_cpu\_info` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/utils.rst` | heading `homodyne.device` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/utils.rst` | heading `homodyne.utils.async\_io` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/utils.rst` | heading `homodyne.utils.logging` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/utils.rst` | heading `homodyne.utils.path\_validation` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/viz.rst` | heading `Best Practices` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/viz.rst` | heading `C2 Heatmap Plotting` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/viz.rst` | heading `CMC Summary Dashboard` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/viz.rst` | heading `Color Maps` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/viz.rst` | heading `Convergence Diagnostics` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/viz.rst` | heading `Datashader Backend (Optional)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/viz.rst` | heading `Diagnostic Functions` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/viz.rst` | heading `Diagonal Overlay Stats` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/viz.rst` | heading `Experimental Data Plots` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/viz.rst` | heading `Fast Plotting` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/viz.rst` | heading `KL Divergence Matrix` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/viz.rst` | heading `Key Functions` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/viz.rst` | heading `MCMC Diagnostic Plots` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/viz.rst` | heading `Module Contents` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/viz.rst` | heading `NLSQ Optimization Plots` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/viz.rst` | heading `Overview` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/viz.rst` | heading `Plot Generation` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/viz.rst` | heading `Posterior Distributions` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/viz.rst` | heading `Publication-Quality Figures` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/viz.rst` | heading `See Also` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/viz.rst` | heading `Trace Plots` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/viz.rst` | heading `Usage Example` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/viz.rst` | heading `Visualization Diagnostics` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `api/viz.rst` | heading `Visualization Module` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `1. Entry Point & Orchestration` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `10. Result Creation & Diagnostics` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `2. Data Preparation & Sharding` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `3. Auto Shard Size Selection` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `4. Time Grid Construction` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `5. Physics Model` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `6. Gradient Balancing (Z-Space)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `7. NUTS Sampling` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `8. Backend Execution` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `9. Sample Combination` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `Additional Diagnostics Functions` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `Angle-Aware Scaling` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `Apply per-angle scaling` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `Auto Shard Size Selection (v2.20.0)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `CMC Configuration Defaults (v2.22.2)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `CMCConfig Fields Added in v2.20.0` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `CMCResult Structure` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `CMCResult.from_mcmc_samples() Workflow` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `Chain Parallelism (chain_method)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `Combination Methods` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `Complete Data Flow` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `Compute g1 using exact same physics as NLSQ` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `Config field in per_shard_mcmc:` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `Convergence Status Determination` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `Critical Design Principles (v2.20.0)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `Data Validation (prepare_mcmc_data)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `December 2025: Proper Time Grid Construction` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `December 2025: Smooth Bounding` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `Decision Logic by Mode` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `Dense Mass Matrix` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `Dynamic max_shards by Dataset Size` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `Emits: shard count, divergence rate, R-hat table, NLSQ comparison (if provided)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `End-of-run structured log summary` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `Estimate per-angle contrast/offset from raw C2 data` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `February 2026: Adaptive Sampling & SamplingPlan (v2.22.2)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `February 2026: Constant-Averaged Mode & NLSQ Parity (v2.22.2)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `February 2026: JAX Profiling Support (v2.22.2)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `February 2026: Mode-Aware Consensus MC (v2.22.0)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `February 2026: NLSQ-Informed Priors & Prior Tempering (v2.22.2)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `Five Model Variants (v2.22.2)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `Flags parameters exceeding tolerance (default 3σ)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `For AUTO mode: average the estimates` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `Hierarchical Combination` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `High-Level Architecture` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `Homodyne CMC (Consensus Monte Carlo) Fitting Architecture` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `January 2026: CMC Divergence & Precision Loss Fix (v2.20.0)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `January 2026: Heterogeneity Prevention (v2.21.0)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `January 2026: Quality Filtering & Warm-Start (v2.19.0)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `Key Dataclasses` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `Laminar Flow Mode (with n_phi ≤ 3, angle_factor = 0.6)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `Map per-point g1 using phi indices` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `Memory Capping` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `Mode-Specific Parameters` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `MultiprocessingBackend (Primary)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `Orchestration Flow` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `Per-Angle Mode Selection (v2.22.2)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `Physics Computation` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `Posterior quality vs NLSQ baseline` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `Posterior uncertainty contraction (CMC vs prior)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `Precision analysis for parameter absorption detection` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `Quick Reference Tables` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `Returns: per-parameter {diff_pct, z_score, status} table` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `Returns: {"contrast_0": 0.4, "offset_0": 0.95, ...}` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `Sharding Strategies` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `Static Mode` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `The Problem` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `The Solution: Non-Centered Reparameterization` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `Values near 1.0 = strong data constraint; near 0.0 = prior-dominated` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `WorkerPool Manager` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `fit_mcmc_jax() Signature` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `g1_all_phi shape: (n_phi, n_points)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `run_nuts_sampling() Workflow` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `v2.12.0: Correct Consensus MC` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `v2.14.2: Diagonal Point Filtering` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `xpcs_model_averaged() Structure (v2.22.2)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `xpcs_model_constant() Structure (v2.18.0)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `xpcs_model_constant_averaged() Structure (v2.22.2)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `xpcs_model_reparameterized() Structure (v2.23.0)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/cmc-fitting-architecture.md` | heading `xpcs_model_scaled() Structure` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/data-handler-architecture.md` | heading `1. Configuration System` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/data-handler-architecture.md` | heading `10. Result Writing (CMC)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/data-handler-architecture.md` | heading `11. CLI Orchestration` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/data-handler-architecture.md` | heading `2. Data Loading` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/data-handler-architecture.md` | heading `3. HDF5 Format Detection & Loading` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/data-handler-architecture.md` | heading `4. Data Filtering` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/data-handler-architecture.md` | heading `5. Preprocessing Pipeline` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/data-handler-architecture.md` | heading `6. Quality Control` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/data-handler-architecture.md` | heading `7. Caching & Performance` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/data-handler-architecture.md` | heading `8. Memory Management` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/data-handler-architecture.md` | heading `9. Result Writing (NLSQ)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/data-handler-architecture.md` | heading `AdaptiveChunker` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/data-handler-architecture.md` | heading `AdvancedDatasetOptimizer` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/data-handler-architecture.md` | heading `AdvancedMemoryManager` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/data-handler-architecture.md` | heading `Apply default values to partially-specified config dict` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/data-handler-architecture.md` | heading `Async I/O Utilities` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/data-handler-architecture.md` | heading `CMC JSON Output Files` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/data-handler-architecture.md` | heading `CMC-Specific I/O (`homodyne/optimization/cmc/`)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/data-handler-architecture.md` | heading `Cache NPZ Format` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/data-handler-architecture.md` | heading `Complete Data Flow` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/data-handler-architecture.md` | heading `ConfigManager Key Methods` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/data-handler-architecture.md` | heading `Configuration (`homodyne/config/`)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/data-handler-architecture.md` | heading `Configuration Defaults` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/data-handler-architecture.md` | heading `Convenience Function: load_xpcs_data()` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/data-handler-architecture.md` | heading `Data Loading (`homodyne/data/`)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/data-handler-architecture.md` | heading `DataQualityController` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/data-handler-architecture.md` | heading `Diagonal Correction` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/data-handler-architecture.md` | heading `Equivalent to: XPCSDataLoader(...).load_experimental_data()` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/data-handler-architecture.md` | heading `Filtering Pipeline` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/data-handler-architecture.md` | heading `Format Detection` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/data-handler-architecture.md` | heading `Generate a complete example YAML config (for homodyne-config)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/data-handler-architecture.md` | heading `Half-Triangle Reconstruction` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/data-handler-architecture.md` | heading `Homodyne Data Handler Architecture` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/data-handler-architecture.md` | heading `JSON Configuration Support` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/data-handler-architecture.md` | heading `JSON Output Summary` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/data-handler-architecture.md` | heading `JSON Serialization Safety` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/data-handler-architecture.md` | heading `Load JSON config (same structure as YAML)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/data-handler-architecture.md` | heading `Metadata Extraction` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/data-handler-architecture.md` | heading `Migrate legacy JSON config to modern nested YAML structure` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/data-handler-architecture.md` | heading `Module-level convenience wrapper — avoids constructing XPCSDataLoader directly` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/data-handler-architecture.md` | heading `Multi-Level Cache` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/data-handler-architecture.md` | heading `Note: q_vector_hash and dt are NOT stored in the cache NPZ.` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/data-handler-architecture.md` | heading `Output Directory Structure` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/data-handler-architecture.md` | heading `Parameter Name Constants` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/data-handler-architecture.md` | heading `ParameterManager` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/data-handler-architecture.md` | heading `ParameterRegistry (Singleton)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/data-handler-architecture.md` | heading `ParameterSpace (for CMC)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/data-handler-architecture.md` | heading `Preferred for one-shot scripts; XPCSDataLoader for repeated loads (caches state)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/data-handler-architecture.md` | heading `PreprocessingPipeline` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/data-handler-architecture.md` | heading `Provenance Tracking` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/data-handler-architecture.md` | heading `Q-Vector Selection` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/data-handler-architecture.md` | heading `Quality Control Result` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/data-handler-architecture.md` | heading `QualityLevel Enum` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/data-handler-architecture.md` | heading `Result Saving Flow` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/data-handler-architecture.md` | heading `Result Writing (`homodyne/io/`)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/data-handler-architecture.md` | heading `Return Data Structure` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/data-handler-architecture.md` | heading `Returns:` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/data-handler-architecture.md` | heading `Saved via _save_to_cache() [xpcs_loader.py]` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/data-handler-architecture.md` | heading `Supports wrapped phi ranges (min > max)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/data-handler-architecture.md` | heading `T=0 Exclusion` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/data-handler-architecture.md` | heading `Top-level sections` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/data-handler-architecture.md` | heading `Validate config structure, return result with issues list` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/data-handler-architecture.md` | heading `Validators` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/data-handler-architecture.md` | heading `XPCSDataLoader` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/data-handler-architecture.md` | heading `YAML Configuration Schema` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/data-handler-architecture.md` | heading `cmc/io.py (Lower-Level CMC I/O)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/data-handler-architecture.md` | heading `config/parameter_names.py` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/data-handler-architecture.md` | heading `data/validators.py - Input validation at I/O boundaries` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/data-handler-architecture.md` | heading `load_experimental_data() Flow` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/data-handler-architecture.md` | heading `mcmc_writers.py (High-Level Dictionaries)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/data-handler-architecture.md` | heading `save_nlsq_json_files()` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/data-handler-architecture.md` | heading `save_nlsq_npz_file()` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/data-handler-architecture.md` | heading `→ ["D0", "alpha", "D_offset", "gamma_dot_t0", "beta", "gamma_dot_t_offset", "phi0"]` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/data-handler-architecture.md` | heading `→ ["contrast_0", ..., "contrast_22", "offset_0", ..., "offset_22", "D0", ...]` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/nlsq-fitting-architecture.md` | heading `10. Error Recovery` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/nlsq-fitting-architecture.md` | heading `11. Result Building` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/nlsq-fitting-architecture.md` | heading `2. Global Optimization Selection` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/nlsq-fitting-architecture.md` | heading `3-Attempt Recovery System` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/nlsq-fitting-architecture.md` | heading `3. CMA-ES Global Optimization` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/nlsq-fitting-architecture.md` | heading `4. Adapter Selection` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/nlsq-fitting-architecture.md` | heading `5. Memory & Strategy Selection` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/nlsq-fitting-architecture.md` | heading `6. Stratification Decision` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/nlsq-fitting-architecture.md` | heading `7. Residual Function Setup` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/nlsq-fitting-architecture.md` | heading `8. Anti-Degeneracy Defense System` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/nlsq-fitting-architecture.md` | heading `9. Strategy Execution` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/nlsq-fitting-architecture.md` | heading `API Usage` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/nlsq-fitting-architecture.md` | heading `Adapter Comparison` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/nlsq-fitting-architecture.md` | heading `Adaptive Threshold` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/nlsq-fitting-architecture.md` | heading `Alias used in some contexts (same type)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/nlsq-fitting-architecture.md` | heading `Angle-Stratified Chunking` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/nlsq-fitting-architecture.md` | heading `Architecture` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/nlsq-fitting-architecture.md` | heading `Automatic Warm-Start via CLI (v2.20.0)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/nlsq-fitting-architecture.md` | heading `Benefits of NLSQ Warm-Start` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/nlsq-fitting-architecture.md` | heading `CMA-ES vs NLSQ Internal Refinement` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/nlsq-fitting-architecture.md` | heading `CMAESWrapper.compute_scale_ratio()` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/nlsq-fitting-architecture.md` | heading `Check if CMA-ES is appropriate for this problem` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/nlsq-fitting-architecture.md` | heading `Comparison` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/nlsq-fitting-architecture.md` | heading `Create wrapper with custom config` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/nlsq-fitting-architecture.md` | heading `Decision Flow` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/nlsq-fitting-architecture.md` | heading `Decision Logic` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/nlsq-fitting-architecture.md` | heading `Example: laminar_flow mode` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/nlsq-fitting-architecture.md` | heading `Global Optimization Comparison` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/nlsq-fitting-architecture.md` | heading `HYBRID_STREAMING Strategy (StreamingExecutor)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/nlsq-fitting-architecture.md` | heading `Homodyne NLSQ Fitting Architecture` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/nlsq-fitting-architecture.md` | heading `Input Validation & Conversion` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/nlsq-fitting-architecture.md` | heading `JIT Kernel Factory (create_ooc_kernels)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/nlsq-fitting-architecture.md` | heading `Jacobian Overhead Factor Evolution` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/nlsq-fitting-architecture.md` | heading `Layer 1: Fourier Reparameterization` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/nlsq-fitting-architecture.md` | heading `Layer 2: Hierarchical Optimization` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/nlsq-fitting-architecture.md` | heading `Layer 3: Adaptive CV Regularization` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/nlsq-fitting-architecture.md` | heading `Layer 4: Gradient Collapse Monitor` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/nlsq-fitting-architecture.md` | heading `Layer 5: Shear-Sensitivity Weighting` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/nlsq-fitting-architecture.md` | heading `Memory Estimation` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/nlsq-fitting-architecture.md` | heading `Mode-Specific Parameters` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/nlsq-fitting-architecture.md` | heading `NLSQ and Bimodal CMC Posteriors (v2.22.0)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/nlsq-fitting-architecture.md` | heading `NLSQ as CMC Warm-Start Provider (v2.20.0)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/nlsq-fitting-architecture.md` | heading `OOCComputePool — Persistent Worker Pool` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/nlsq-fitting-architecture.md` | heading `OUT_OF_CORE Strategy (LargeDatasetExecutor)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/nlsq-fitting-architecture.md` | heading `OptimizationResult Structure` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/nlsq-fitting-architecture.md` | heading `Optimizations` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/nlsq-fitting-architecture.md` | heading `Parallel Reduction (accumulate_chunks_parallel)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/nlsq-fitting-architecture.md` | heading `Parameter Initialization` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/nlsq-fitting-architecture.md` | heading `Quality Flag Determination` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/nlsq-fitting-architecture.md` | heading `Related Dataclasses` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/nlsq-fitting-architecture.md` | heading `Residual Computation Flow` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/nlsq-fitting-architecture.md` | heading `ResultBuilder Fluent Interface` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/nlsq-fitting-architecture.md` | heading `Root Problem: Structural Degeneracy` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/nlsq-fitting-architecture.md` | heading `STANDARD Strategy (StandardExecutor)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/nlsq-fitting-architecture.md` | heading `Scale Ratio Computation` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/nlsq-fitting-architecture.md` | heading `Shared Memory Layout (OOCSharedArrays)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/nlsq-fitting-architecture.md` | heading `Step 1: Run NLSQ first` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/nlsq-fitting-architecture.md` | heading `Step 2: Pass NLSQ result to CMC for warm-start` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/nlsq-fitting-architecture.md` | heading `Strategy Decision Tree` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/nlsq-fitting-architecture.md` | heading `Strategy Selection` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/nlsq-fitting-architecture.md` | heading `Strategy to Executor Mapping` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/nlsq-fitting-architecture.md` | heading `The Problem: Cold-Start CMC Divergence` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/nlsq-fitting-architecture.md` | heading `Usage Example` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/nlsq-fitting-architecture.md` | heading `When to Use CMA-ES` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `1. Mathematical Foundation` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `10. HomodyneModel Unified Interface` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `11. TheoryEngine High-Level API` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `12. Fitting Infrastructure` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `2. Physical Constants & Validation` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `3. Model Hierarchy` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `4. Physics Factors Pre-Computation` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `5. JIT-Compiled Computation Kernels` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `6. Shadow-Copy Architecture` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `7. Per-Angle Scaling System` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `8. Numerical Stability Techniques` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `9. Automatic Differentiation & Fallback` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `Analysis Mode Detection` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `Automatic Differentiation (Module-Level)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `Broadcast and multiply` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `CMC Shard Computation` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `CMC-Specific Path` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `CORRECT: preserves gradients (d/dx = 1 for x < eps)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `Class Hierarchy` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `CombinedModel (Primary Model)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `Complete Computation Flow` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `Consistency Invariants (Verified Across All 5 Paths)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `Construction` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `Convenience Functions` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `Core Kernel: g1 Diffusion` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `Core Kernel: g1 Shear` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `Core Kernel: g1 Total` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `Core Kernel: g2 Scaled` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `Core Physics Model (~9,100 lines)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `Correlation Functions` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `D(t) = D0 * t^alpha → undefined at t=0 when alpha < 0` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `Design Pattern` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `Diagonal Correction` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `DiffusionModel` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `Factory Functions` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `FitResult` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `Float64 Requirement` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `For integral matrices: need |cumsum[i] - cumsum[j]|` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `Functions` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `Gradient-Safe Floors (CLAUDE.md Rule 7)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `Gradient-safe lower floor` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `High-Level Architecture` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `Homodyne Physical Model Architecture` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `INCORRECT: zeros gradients (d/dx = 0 for x < eps)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `Instead of: g1 = exp(-wq_dt * D_integral)   → underflow for large D_integral` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `Integration Method` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `Integration Points` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `JAX Automatic Differentiation (Primary)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `JIT Compilation Points` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `JIT-Compiled Least Squares Solvers` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `Key Files Reference` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `Key Methods` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `Laminar Flow Mode (7 Parameters)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `Log-Space Computation` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `Memory Layout` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `Meshgrid Caching` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `Methods` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `Model Mixins (Gradient Dispatch)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `NLSQ-Specific Path` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `NumPy Numerical Differentiation (Fallback)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `Numerical Constants` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `Parameter Bounds` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `Parameter Summary` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `ParameterSpace` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `PhysicsConstants` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `PhysicsFactors Dataclass` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `PhysicsModelBase (ABC)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `Public API` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `Public Wrapper Functions` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `Purpose` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `Quantile-Based Estimation Algorithm` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `Quick Reference Tables` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `Shadow-Copy Registry` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `ShardGrid (NamedTuple)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `Shared Utilities` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `ShearModel` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `Singularity Floor for Power Laws` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `Smooth Absolute Value` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `Static Mode (3 Parameters)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `Table of Contents` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `Taylor Expansion for sinc` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `The 4 Per-Angle Modes` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `The Homodyne XPCS Model` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `Time-Dependent Coefficients` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `UnifiedHomodyneEngine` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `Use:` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `Validation Functions` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `ValidationResult Dataclass` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `Vectorization (Module-Level)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `architecture/physical-model-architecture.md` | heading `Why Shadow Copies Exist` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `configuration/index.rst` | heading `Analysis Modes` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `configuration/index.rst` | heading `Configuration Examples` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `configuration/index.rst` | heading `Configuration Features` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `configuration/index.rst` | heading `Configuration Sections` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `configuration/index.rst` | heading `Data Filtering and Caching` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `configuration/index.rst` | heading `Environment Validation` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `configuration/index.rst` | heading `For Questions` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `configuration/index.rst` | heading `Interactive Configuration Builder` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `configuration/index.rst` | heading `Laminar Flow Mode (7 physical parameters)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `configuration/index.rst` | heading `Optimization Configuration` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `configuration/index.rst` | heading `Output Configuration` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `configuration/index.rst` | heading `Overview` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `configuration/index.rst` | heading `Per-Angle Scaling` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `configuration/index.rst` | heading `Performance Tuning` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `configuration/index.rst` | heading `Quick Links` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `configuration/index.rst` | heading `Running Analysis with Configuration` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `configuration/index.rst` | heading `Static Mode (3 physical parameters)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `configuration/index.rst` | heading `Validating Configurations` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `configuration/index.rst` | heading `Validation and Quality Control` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `configuration/index.rst` | heading `Version Information` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `configuration/index.rst` | heading `Working with Configurations` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `configuration/options.rst` | heading `Adaptive Sampling` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `configuration/options.rst` | heading `Analysis Mode` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `configuration/options.rst` | heading `Analyzer Parameters Section` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `configuration/options.rst` | heading `Anti-Degeneracy Configuration` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `configuration/options.rst` | heading `Backend Configuration` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `configuration/options.rst` | heading `CMA-ES Configuration` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `configuration/options.rst` | heading `Common Configuration Patterns` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `configuration/options.rst` | heading `Configuration Structure` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `configuration/options.rst` | heading `Convergence Validation` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `configuration/options.rst` | heading `Experimental Data Section` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `configuration/options.rst` | heading `For Examples` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `configuration/options.rst` | heading `For Validation` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `configuration/options.rst` | heading `Initial Parameters Section` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `configuration/options.rst` | heading `Initial Values Configuration` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `configuration/options.rst` | heading `JAX Profiling` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `configuration/options.rst` | heading `Laminar Flow Mode Bounds` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `configuration/options.rst` | heading `Logging Section` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `configuration/options.rst` | heading `MCMC Configuration` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `configuration/options.rst` | heading `Metadata Section` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `configuration/options.rst` | heading `NLSQ Configuration` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `configuration/options.rst` | heading `Optimization Section` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `configuration/options.rst` | heading `Output Section` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `configuration/options.rst` | heading `Parameter Space Section` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `configuration/options.rst` | heading `Per-Shard MCMC` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `configuration/options.rst` | heading `Performance Section` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `configuration/options.rst` | heading `Phi Angle Filtering Section` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `configuration/options.rst` | heading `Plotting Section` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `configuration/options.rst` | heading `Quality Filtering` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `configuration/options.rst` | heading `Quick Reference Table` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `configuration/options.rst` | heading `Sequential Optimization` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `configuration/options.rst` | heading `Sharding Configuration (CMC)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `configuration/options.rst` | heading `Static Mode Bounds` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `configuration/options.rst` | heading `Stratification Configuration` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `configuration/options.rst` | heading `Subposterior Combination` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `configuration/options.rst` | heading `Validation Section` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `configuration/templates.rst` | heading `Getting Started` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `configuration/templates.rst` | heading `Laminar Flow Mode Template` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `configuration/templates.rst` | heading `NLSQ → MCMC Workflow` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `configuration/templates.rst` | heading `Per-Angle Scaling Details` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `configuration/templates.rst` | heading `See Also` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `configuration/templates.rst` | heading `Static Mode Template` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `configuration/templates.rst` | heading `Template Overview` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `configuration/templates.rst` | heading `Workflow Guidance` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `contributing.rst` | heading `Contributing to Homodyne` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `developer/architecture.rst` | heading `CLI Module Map` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `developer/architecture.rst` | heading `CMC Module Map` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `developer/architecture.rst` | heading `Component Overview` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `developer/architecture.rst` | heading `Core Module Map` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `developer/architecture.rst` | heading `Key Design Decisions` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `developer/architecture.rst` | heading `Module Dependency Map` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `developer/architecture.rst` | heading `NLSQ Module Map` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `developer/architecture.rst` | heading `System Architecture` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `developer/architecture.rst` | heading `Visualization Module Map` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `developer/contributing_guide.rst` | heading `Before Committing` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `developer/contributing_guide.rst` | heading `Before Submitting` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `developer/contributing_guide.rst` | heading `Black - Code Formatting` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `developer/contributing_guide.rst` | heading `CPU Performance Tips` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `developer/contributing_guide.rst` | heading `Code Coverage Requirements` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `developer/contributing_guide.rst` | heading `Code Quality Standards` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `developer/contributing_guide.rst` | heading `Commit Message Guidelines` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `developer/contributing_guide.rst` | heading `Common JAX Debugging Commands` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `developer/contributing_guide.rst` | heading `Complete Quality Check` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `developer/contributing_guide.rst` | heading `Contributing to Homodyne` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `developer/contributing_guide.rst` | heading `Development Setup with uv` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `developer/contributing_guide.rst` | heading `Getting Help` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `developer/contributing_guide.rst` | heading `Installing Development Environment` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `developer/contributing_guide.rst` | heading `JAX and CPU Debugging` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `developer/contributing_guide.rst` | heading `MyPy - Type Checking` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `developer/contributing_guide.rst` | heading `PR Description Template` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `developer/contributing_guide.rst` | heading `Pull Request Guidelines` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `developer/contributing_guide.rst` | heading `Reference Material` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `developer/contributing_guide.rst` | heading `Ruff - Linting` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `developer/contributing_guide.rst` | heading `Running Tests` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `developer/contributing_guide.rst` | heading `Standard Commit Prefixes` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `developer/contributing_guide.rst` | heading `Test Organization` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `developer/contributing_guide.rst` | heading `Testing with Pytest` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `developer/contributing_guide.rst` | heading `Verifying Installation` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `developer/contributing_guide.rst` | heading `Writing Tests` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `developer/index.rst` | heading `Architecture Overview` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `developer/parameter_bounds_verification.rst` | heading `CMC Prior Construction` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `developer/parameter_bounds_verification.rst` | heading `Documentation Corrections Applied (2026-03-03)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `developer/parameter_bounds_verification.rst` | heading `Mode Selection Logic` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `developer/parameter_bounds_verification.rst` | heading `NLSQ Bounds Flow` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `developer/parameter_bounds_verification.rst` | heading `Parameter Bounds Codebase Verification` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `developer/parameter_bounds_verification.rst` | heading `Physics Validators (Soft Constraints)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `developer/parameter_bounds_verification.rst` | heading `Registry Defaults and Priors` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `developer/parameter_bounds_verification.rst` | heading `Verified Bounds (All Sources Consistent)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `developer/parameter_bounds_verification.rst` | heading `Verified Source Files` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `developer/parameter_bounds_verification.rst` | heading `YAML Template Prior Overrides` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `developer/testing_guide.rst` | heading `Advanced Pytest Options` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `developer/testing_guide.rst` | heading `Code Coverage` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `developer/testing_guide.rst` | heading `Common JAX Debugging Issues` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `developer/testing_guide.rst` | heading `Continuous Integration` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `developer/testing_guide.rst` | heading `Coverage Requirements` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `developer/testing_guide.rst` | heading `Environment Variables` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `developer/testing_guide.rst` | heading `Flaky Tests` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `developer/testing_guide.rst` | heading `For Development` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `developer/testing_guide.rst` | heading `Improving Coverage` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `developer/testing_guide.rst` | heading `Integration Tests` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `developer/testing_guide.rst` | heading `JAX Debugging` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `developer/testing_guide.rst` | heading `JAX Device Configuration` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `developer/testing_guide.rst` | heading `MCMC Tests` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `developer/testing_guide.rst` | heading `Measuring Coverage` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `developer/testing_guide.rst` | heading `Performance Tests` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `developer/testing_guide.rst` | heading `Profiling JAX Code` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `developer/testing_guide.rst` | heading `Quick Reference Table` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `developer/testing_guide.rst` | heading `Quick Test Commands` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `developer/testing_guide.rst` | heading `Reference Commands` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `developer/testing_guide.rst` | heading `Test Failures` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `developer/testing_guide.rst` | heading `Test Organization` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `developer/testing_guide.rst` | heading `Test-Specific Commands` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `developer/testing_guide.rst` | heading `Troubleshooting Tests` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `developer/testing_guide.rst` | heading `Understanding JAX Compilation` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `developer/testing_guide.rst` | heading `Unit Tests` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `examples/index.rst` | heading `See Also` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `index.rst` | heading `Community and Support` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `index.rst` | heading `Homodyne` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `installation.rst` | heading `CPU Optimization Notes` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `installation.rst` | heading `Development Install` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `installation.rst` | heading `Install with uv (Recommended)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `installation.rst` | heading `Shell Completion Setup` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `installation.rst` | heading `Uninstall` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `quickstart.rst` | heading `Interpreting Results` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `quickstart.rst` | heading `Quick Start — 5 Minutes to First Analysis` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `quickstart.rst` | heading `Step 1: Install` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `quickstart.rst` | heading `Step 2: Generate a Configuration File` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `quickstart.rst` | heading `Step 3: Load Data` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `quickstart.rst` | heading `Step 4: Run Static NLSQ Analysis` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `quickstart.rst` | heading `Step 5: Run from the Command Line` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `theory/analysis_modes.rst` | heading `Configuration Example` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `theory/analysis_modes.rst` | heading `Diffusion Integral` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `theory/analysis_modes.rst` | heading `Example YAML snippet` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `theory/analysis_modes.rst` | heading `Laminar Flow Mode` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `theory/analysis_modes.rst` | heading `Mode Overview` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `theory/analysis_modes.rst` | heading `Mode Selection Guidelines` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `theory/analysis_modes.rst` | heading `Parameter Bounds (Laminar Flow Mode)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `theory/analysis_modes.rst` | heading `Parameter Bounds (Static Mode)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `theory/analysis_modes.rst` | heading `Parameter Bounds and Priors (NLSQ & CMC)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `theory/analysis_modes.rst` | heading `Parameter Ordering` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `theory/analysis_modes.rst` | heading `Per-Angle Scaling` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `theory/analysis_modes.rst` | heading `Physical Constraints` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `theory/analysis_modes.rst` | heading `Physical Parameters (3)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `theory/analysis_modes.rst` | heading `Physical Parameters (7)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `theory/analysis_modes.rst` | heading `Scaling Parameters (Per Angle)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `theory/analysis_modes.rst` | heading `Shear Integral` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `theory/analysis_modes.rst` | heading `Static Mode` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `theory/analysis_modes.rst` | heading `Total Parameter Count` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `theory/analysis_modes.rst` | heading `When to Use Laminar Flow Mode` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `theory/analysis_modes.rst` | heading `When to Use Static Mode` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `theory/citations.rst` | heading `Acknowledgments` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `theory/citations.rst` | heading `Bayesian Methods` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `theory/citations.rst` | heading `Citing Homodyne` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `theory/citations.rst` | heading `Contact` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `theory/citations.rst` | heading `JAX and Scientific Stack` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `theory/citations.rst` | heading `Optimization` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `theory/citations.rst` | heading `Primary References` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `theory/citations.rst` | heading `References and Citations` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `theory/citations.rst` | heading `Rheology` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `theory/citations.rst` | heading `Stochastic Processes` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `theory/citations.rst` | heading `XPCS Methodology` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `theory/classical_processes.rst` | heading `Brownian Oscillator (Overdamped)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `theory/classical_processes.rst` | heading `Brownian Oscillator (Underdamped)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `theory/classical_processes.rst` | heading `Classical Langevin Processes` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `theory/classical_processes.rst` | heading `Comparison Table` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `theory/classical_processes.rst` | heading `Non-Gaussian Corrections` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `theory/classical_processes.rst` | heading `Ornstein-Uhlenbeck Process (Inertial Brownian Motion)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `theory/classical_processes.rst` | heading `The Langevin Equation` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `theory/computational_methods.rst` | heading `Adaptive Sampling` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `theory/computational_methods.rst` | heading `CMA-ES for Multi-Scale Problems` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `theory/computational_methods.rst` | heading `Consensus Monte Carlo (CMC)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `theory/computational_methods.rst` | heading `Cumulative Trapezoid Integration` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `theory/computational_methods.rst` | heading `Diagnostics` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `theory/computational_methods.rst` | heading `Jacobian Computation` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `theory/computational_methods.rst` | heading `Levenberg-Marquardt Algorithm` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `theory/computational_methods.rst` | heading `Memory Management` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `theory/computational_methods.rst` | heading `NUTS (No-U-Turn Sampler)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `theory/computational_methods.rst` | heading `Non-Linear Least Squares (NLSQ)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `theory/computational_methods.rst` | heading `Reparameterization (Log-Space Priors)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `theory/correlation_functions.rst` | heading `Correlation Functions in XPCS` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `theory/correlation_functions.rst` | heading `Equilibrium Approximation: g₂(q, τ)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `theory/correlation_functions.rst` | heading `First-Order Correlation Function` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `theory/correlation_functions.rst` | heading `Model Fitting` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `theory/correlation_functions.rst` | heading `Position Density and Scattered Field` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `theory/correlation_functions.rst` | heading `Second-Order Correlation Function` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `theory/correlation_functions.rst` | heading `Siegert Relation` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `theory/correlation_functions.rst` | heading `Two-Time Correlation Matrix` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `theory/heterodyne_scattering.rst` | heading `Comparison to Homodyne` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `theory/heterodyne_scattering.rst` | heading `Heterodyne Scattering: Multi-Component Models` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `theory/heterodyne_scattering.rst` | heading `N-Component General Formula` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `theory/heterodyne_scattering.rst` | heading `Normalization Factor f²` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `theory/heterodyne_scattering.rst` | heading `Oscillatory Patterns as Diagnostic` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `theory/heterodyne_scattering.rst` | heading `Physical Motivation` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `theory/heterodyne_scattering.rst` | heading `Three-Component Case: Two Flowing + One Static` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `theory/heterodyne_scattering.rst` | heading `Two-Component Case: Static + Flowing` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `theory/index.rst` | heading `Overview of Sections` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `theory/index.rst` | heading `Quick Physics Reference` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `theory/transport_coefficient.rst` | heading `Connection to Physical Diffusion Coefficient` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `theory/transport_coefficient.rst` | heading `Homodyne Implementation` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `theory/transport_coefficient.rst` | heading `Physical Interpretation` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `theory/transport_coefficient.rst` | heading `Relationship to Rheology` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `theory/transport_coefficient.rst` | heading `Table of J(t) for Classical Processes` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/01_fundamentals/parameter_guide.rst` | heading `Contrast (beta_coherence)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/01_fundamentals/parameter_guide.rst` | heading `Cross-System Bounds Consistency` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/01_fundamentals/parameter_guide.rst` | heading `D_offset — Baseline Diffusion` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/01_fundamentals/parameter_guide.rst` | heading `Diffusion Parameters` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/01_fundamentals/parameter_guide.rst` | heading `D₀ — Reference Diffusion Coefficient` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/01_fundamentals/parameter_guide.rst` | heading `Offset` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/01_fundamentals/parameter_guide.rst` | heading `Optical Parameters` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/01_fundamentals/parameter_guide.rst` | heading `Overriding Parameter Bounds via YAML` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/01_fundamentals/parameter_guide.rst` | heading `Overview` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/01_fundamentals/parameter_guide.rst` | heading `Parameter Interpretation Guide` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/01_fundamentals/parameter_guide.rst` | heading `Parameter Reference Table` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/01_fundamentals/parameter_guide.rst` | heading `See Also` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/01_fundamentals/parameter_guide.rst` | heading `Setting Initial Values` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/01_fundamentals/parameter_guide.rst` | heading `Shear Parameters (laminar_flow only)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/01_fundamentals/parameter_guide.rst` | heading `alpha — Diffusion Time-Dependence Exponent` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/01_fundamentals/parameter_guide.rst` | heading `beta — Shear Rate Time-Dependence Exponent` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/01_fundamentals/parameter_guide.rst` | heading `gamma_dot_0 — Reference Shear Rate` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/01_fundamentals/parameter_guide.rst` | heading `gamma_dot_t_offset — Baseline Shear Rate` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/01_fundamentals/parameter_guide.rst` | heading `phi_0 — Angular Offset` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/01_fundamentals/what_is_xpcs.rst` | heading `Connection to Particle Dynamics` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/01_fundamentals/what_is_xpcs.rst` | heading `Experimental Setup` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/01_fundamentals/what_is_xpcs.rst` | heading `From Speckles to Correlation Functions` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/01_fundamentals/what_is_xpcs.rst` | heading `Homodyne Detection` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/01_fundamentals/what_is_xpcs.rst` | heading `Introduction` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/01_fundamentals/what_is_xpcs.rst` | heading `See Also` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/01_fundamentals/what_is_xpcs.rst` | heading `The Coherent X-ray Requirement` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/01_fundamentals/what_is_xpcs.rst` | heading `Two-Time Correlation Functions` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/01_fundamentals/what_is_xpcs.rst` | heading `What is X-ray Photon Correlation Spectroscopy?` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/01_fundamentals/what_is_xpcs.rst` | heading `Why XPCS for Soft Matter?` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/02_data_and_fitting/data_loading.rst` | heading `Common Data Issues and Fixes` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/02_data_and_fitting/data_loading.rst` | heading `Data Validation` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/02_data_and_fitting/data_loading.rst` | heading `HDF5 File Requirements` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/02_data_and_fitting/data_loading.rst` | heading `Loaded Data Dictionary` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/02_data_and_fitting/data_loading.rst` | heading `Loading XPCS Data` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/02_data_and_fitting/data_loading.rst` | heading `Multiple q-Values` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/02_data_and_fitting/data_loading.rst` | heading `Overview` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/02_data_and_fitting/data_loading.rst` | heading `Phi Angle Filtering` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/02_data_and_fitting/data_loading.rst` | heading `See Also` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/02_data_and_fitting/data_loading.rst` | heading `Supported File Formats` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/02_data_and_fitting/data_loading.rst` | heading `YAML Configuration for Data Loading` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/02_data_and_fitting/nlsq_fitting.rst` | heading `Common Fitting Issues` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/02_data_and_fitting/nlsq_fitting.rst` | heading `Complete Workflow` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/02_data_and_fitting/nlsq_fitting.rst` | heading `Convergence Diagnostics` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/02_data_and_fitting/nlsq_fitting.rst` | heading `Memory-Adaptive Strategy Selection` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/02_data_and_fitting/nlsq_fitting.rst` | heading `Multi-Start Optimization` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/02_data_and_fitting/nlsq_fitting.rst` | heading `NLSQ Fitting Guide` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/02_data_and_fitting/nlsq_fitting.rst` | heading `NLSQAdapter vs NLSQWrapper` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/02_data_and_fitting/nlsq_fitting.rst` | heading `Overview` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/02_data_and_fitting/nlsq_fitting.rst` | heading `See Also` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/02_data_and_fitting/nlsq_fitting.rst` | heading `Setting Initial Parameters` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/02_data_and_fitting/nlsq_fitting.rst` | heading `The fit_nlsq_jax Function` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/02_data_and_fitting/nlsq_fitting.rst` | heading `Understanding the Result Object` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/02_data_and_fitting/result_interpretation.rst` | heading `CMC Result Structure` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/02_data_and_fitting/result_interpretation.rst` | heading `Convergence Diagnostics` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/02_data_and_fitting/result_interpretation.rst` | heading `Interpreting Results` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/02_data_and_fitting/result_interpretation.rst` | heading `NLSQ Result Structure` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/02_data_and_fitting/result_interpretation.rst` | heading `Reduced Chi-Squared Interpretation` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/02_data_and_fitting/result_interpretation.rst` | heading `Residual Analysis` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/02_data_and_fitting/result_interpretation.rst` | heading `Saving and Loading Results` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/02_data_and_fitting/result_interpretation.rst` | heading `See Also` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/03_advanced_topics/bayesian_inference.rst` | heading `Bayesian Inference with CMC` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/03_advanced_topics/bayesian_inference.rst` | heading `CMC Configuration` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/03_advanced_topics/bayesian_inference.rst` | heading `Chain Execution Methods` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/03_advanced_topics/bayesian_inference.rst` | heading `Checkpointing for Long Runs` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/03_advanced_topics/bayesian_inference.rst` | heading `Heterogeneity Detection` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/03_advanced_topics/bayesian_inference.rst` | heading `Parameter Reparameterization` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/03_advanced_topics/bayesian_inference.rst` | heading `Posterior Comparison: NLSQ vs CMC` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/03_advanced_topics/bayesian_inference.rst` | heading `Quality Filtering` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/03_advanced_topics/bayesian_inference.rst` | heading `SamplingPlan and Adaptive Sampling` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/03_advanced_topics/bayesian_inference.rst` | heading `See Also` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/03_advanced_topics/bayesian_inference.rst` | heading `Shard Scheduling` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/03_advanced_topics/bayesian_inference.rst` | heading `The NLSQ Warm-Start Pipeline` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/03_advanced_topics/bayesian_inference.rst` | heading `When to Use Bayesian Inference` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/03_advanced_topics/bayesian_inference.rst` | heading `Worker Pool and Shared Memory` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/03_advanced_topics/cmaes_optimization.rst` | heading `CMA-ES for Multi-Scale Problems` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/03_advanced_topics/cmaes_optimization.rst` | heading `CMA-ES → NLSQ Refinement Pipeline` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/03_advanced_topics/cmaes_optimization.rst` | heading `Configuration` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/03_advanced_topics/cmaes_optimization.rst` | heading `Performance Considerations` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/03_advanced_topics/cmaes_optimization.rst` | heading `Python API` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/03_advanced_topics/cmaes_optimization.rst` | heading `See Also` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/03_advanced_topics/cmaes_optimization.rst` | heading `Warm-Start Optimization` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/03_advanced_topics/diagnostics.rst` | heading `ArviZ Comprehensive Check` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/03_advanced_topics/diagnostics.rst` | heading `Convergence Diagnostics` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/03_advanced_topics/diagnostics.rst` | heading `Convergence Status` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/03_advanced_topics/diagnostics.rst` | heading `Diagnosing Common Failures` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/03_advanced_topics/diagnostics.rst` | heading `Error Recovery Actions` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/03_advanced_topics/diagnostics.rst` | heading `Quality Filtering in CMC` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/03_advanced_topics/diagnostics.rst` | heading `R-hat (Gelman-Rubin Statistic)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/03_advanced_topics/diagnostics.rst` | heading `Reduced Chi-Squared` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/03_advanced_topics/diagnostics.rst` | heading `Residual Analysis` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/03_advanced_topics/diagnostics.rst` | heading `See Also` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/03_advanced_topics/per_angle_modes.rst` | heading `Configuration Examples` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/03_advanced_topics/per_angle_modes.rst` | heading `Diagnosing Degeneracy` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/03_advanced_topics/per_angle_modes.rst` | heading `Fixing Degeneracy Issues` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/03_advanced_topics/per_angle_modes.rst` | heading `Parameter Count Summary` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/03_advanced_topics/per_angle_modes.rst` | heading `Per-Angle Scaling Modes` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/03_advanced_topics/per_angle_modes.rst` | heading `See Also` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/03_advanced_topics/per_angle_modes.rst` | heading `The Four Modes` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/03_advanced_topics/per_angle_modes.rst` | heading `Why Per-Angle Scaling Matters` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/03_advanced_topics/per_angle_modes.rst` | heading `auto (Recommended)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/03_advanced_topics/per_angle_modes.rst` | heading `constant` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/03_advanced_topics/per_angle_modes.rst` | heading `fourier` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/03_advanced_topics/per_angle_modes.rst` | heading `individual` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/04_practical_guides/configuration.rst` | heading `Additional Configuration Details` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/04_practical_guides/configuration.rst` | heading `Complete Configuration Schema` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/04_practical_guides/configuration.rst` | heading `Configuration Overview` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/04_practical_guides/configuration.rst` | heading `Configuration Precedence` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/04_practical_guides/configuration.rst` | heading `Configuration Validation` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/04_practical_guides/configuration.rst` | heading `See Also` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/04_practical_guides/configuration.rst` | heading `Template Generation` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/04_practical_guides/configuration.rst` | heading `YAML Configuration Reference` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/04_practical_guides/performance_tuning.rst` | heading `CMC Worker Count` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/04_practical_guides/performance_tuning.rst` | heading `JAX Compilation Caching` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/04_practical_guides/performance_tuning.rst` | heading `JAX Profiling (Advanced)` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/04_practical_guides/performance_tuning.rst` | heading `Memory Profiling` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/04_practical_guides/performance_tuning.rst` | heading `OMP Thread Configuration` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/04_practical_guides/performance_tuning.rst` | heading `Overview` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/04_practical_guides/performance_tuning.rst` | heading `Performance Checklist` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/04_practical_guides/performance_tuning.rst` | heading `Performance Tuning: CPU/NUMA Optimization` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/04_practical_guides/performance_tuning.rst` | heading `See Also` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/04_practical_guides/performance_tuning.rst` | heading `XLA Flags` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/04_practical_guides/performance_tuning.rst` | heading `homodyne-config-xla` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/04_practical_guides/visualization.rst` | heading `CMC Posterior Plots` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/04_practical_guides/visualization.rst` | heading `Fitted vs Experimental Cuts` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/04_practical_guides/visualization.rst` | heading `NLSQ Fit Quality Plot` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/04_practical_guides/visualization.rst` | heading `Parameter Evolution Across Samples` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/04_practical_guides/visualization.rst` | heading `Plotting Results` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/04_practical_guides/visualization.rst` | heading `Publication-Quality Figures` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/04_practical_guides/visualization.rst` | heading `See Also` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/04_practical_guides/visualization.rst` | heading `Two-Time Correlation Heatmaps` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/05_appendices/faq.rst` | heading `Analysis Mode Selection` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/05_appendices/faq.rst` | heading `CMC / Bayesian Questions` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/05_appendices/faq.rst` | heading `Configuration Questions` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/05_appendices/faq.rst` | heading `General Questions` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/05_appendices/faq.rst` | heading `NLSQ Optimization` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/05_appendices/faq.rst` | heading `Results Questions` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/05_appendices/faq.rst` | heading `See Also` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/05_appendices/troubleshooting.rst` | heading `CMC Divergence Issues` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/05_appendices/troubleshooting.rst` | heading `Data Loading Errors` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/05_appendices/troubleshooting.rst` | heading `Fitting Convergence Failures` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/05_appendices/troubleshooting.rst` | heading `Getting Help` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/05_appendices/troubleshooting.rst` | heading `Installation Issues` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/05_appendices/troubleshooting.rst` | heading `JAX Compilation Issues` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/05_appendices/troubleshooting.rst` | heading `See Also` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/05_appendices/troubleshooting.rst` | heading `Shell Completion Issues` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/05_appendices/troubleshooting.rst` | heading `Troubleshooting Guide` present in homodyne, absent in heterodyne |
| `KEEP` | heading_drift | `user_guide/index.rst` | heading `Sections Overview` present in homodyne, absent in heterodyne |

### logs_errors (638)

| Disposition | Kind | Qualname / Path | Detail |
|---|---|---|---|
| `KEEP` | log_format_drift | `cli.commands [debug]` | homodyne logs 'MCMC will use mid-point defaults (no initial_parameters.values in config)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `cli.commands [debug]` | homodyne logs 'Using NLSQ native large dataset handling'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `cli.commands [info]` | homodyne logs 'Generating ArviZ diagnostic plots...'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `cli.commands [info]` | homodyne logs 'Loading data from configuration'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `cli.commands [info]` | homodyne logs 'Loading experimental data...'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `cli.commands [info]` | homodyne logs 'NLSQ results saved, proceeding to CMC...'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `cli.commands [info]` | homodyne logs 'Phase 1/2: NLSQ optimization...'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `cli.commands [info]` | homodyne logs 'Phase 2/2: CMC optimization with NLSQ warm-start...'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `cli.commands [info]` | homodyne logs 'Running sequential NLSQ -> CMC pipeline...'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `cli.commands [info]` | homodyne logs '[CLI] Plotting experimental data only (skipping optimization)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `cli.commands [info]` | homodyne logs '[CLI] Plotting simulated data only (skipping data loading and optimization)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `cli.commands [warning]` | homodyne logs 'Cannot generate ArviZ diagnostic plots: inference_data not available'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `cli.commands [warning]` | homodyne logs 'No diagnostic plots were generated'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `cli.commands [warning]` | homodyne logs 'No inference_data available in result - skipping ArviZ diagnostic plots'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `cli.config_handling [info]` | homodyne logs 'Configuration file not found, using defaults'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `cli.config_handling [info]` | homodyne logs 'Configuring computational device...'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `cli.config_handling [warning]` | homodyne logs 'Device configuration failed, using defaults'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `cli.data_pipeline [debug]` | homodyne logs 'Phi filtering not enabled, using all angles for optimization'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `cli.data_pipeline [debug]` | homodyne logs 'Skipping t=0 exclusion: arrays too small to slice'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `cli.data_pipeline [debug]` | homodyne logs 'Skipping t=0 exclusion: missing t1, t2, or c2_exp arrays'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `cli.data_pipeline [info]` | homodyne logs 'Normalized phi angles to [-180, 180] deg range (flow direction at 0 deg)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `cli.data_pipeline [warning]` | homodyne logs "Multiple shards requested but backend is 'auto/jax'; defaulting to multiprocessing. Use --cmc-backend to override explicitly."; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `cli.data_pipeline [warning]` | homodyne logs 'No angles matched phi_filtering criteria, using all angles'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `cli.data_pipeline [warning]` | homodyne logs 'No phi angles or C2 data available, cannot apply filtering'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `cli.data_pipeline [warning]` | homodyne logs 'Phi filtering enabled but no target_ranges specified, using all angles'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `cli.main [info]` | homodyne logs 'Analysis completed successfully'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `cli.main [info]` | homodyne logs 'Starting homodyne analysis...'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `cli.optimization_runner [info]` | homodyne logs 'Running NLSQ optimization for CMC warm-start...'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `cli.optimization_runner [warning]` | homodyne logs 'NLSQ warm-start disabled (--no-nlsq-warmstart). CMC may have higher divergence rates without warm-start.'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `cli.plot_dispatch [error]` | homodyne logs 'Configuration required for simulated data plotting'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `cli.plot_dispatch [info]` | homodyne logs 'Generating plots...'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `cli.plot_dispatch [warning]` | homodyne logs 'Plotting requested but matplotlib not installed. Install with: pip install matplotlib'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `cli.result_saving [debug]` | homodyne logs 'Analysis results saving error:'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `cli.result_saving [debug]` | homodyne logs 'Could not stat fitted data file'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `cli.result_saving [debug]` | homodyne logs 'Could not stat samples file'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `cli.result_saving [debug]` | homodyne logs 'Diagnostics saving error:'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `cli.result_saving [debug]` | homodyne logs 'Extracting metadata (L, dt, q)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `cli.result_saving [debug]` | homodyne logs 'Inferred n_angles=%s for _prepare_parameter_data (mode=%s, params=%s)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `cli.result_saving [debug]` | homodyne logs 'Parameter saving error:'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `cli.result_saving [debug]` | homodyne logs 'Plot error details:'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `cli.result_saving [debug]` | homodyne logs 'Samples saving error:'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `cli.result_saving [debug]` | homodyne logs 'Shard diagnostics saving error:'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `cli.result_saving [debug]` | homodyne logs 'Using L from experimental_data.geometry'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `cli.result_saving [debug]` | homodyne logs 'Using time arrays in seconds (already converted by data loader)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `cli.result_saving [debug]` | homodyne logs 'Waiting for background result writes to complete...'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `cli.result_saving [info]` | homodyne logs '  - 3 JSON files (parameters, analysis results, convergence metrics)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `cli.result_saving [info]` | homodyne logs '  - NPZ and plots skipped (theoretical fits unavailable)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `cli.result_saving [info]` | homodyne logs 'Computing theoretical C2 with posterior means:'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `cli.result_saving [info]` | homodyne logs 'Computing theoretical fits with per-angle scaling'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `cli.result_saving [info]` | homodyne logs 'Generating comparison heatmap plots'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `cli.result_saving [info]` | homodyne logs 'Generating heatmap plots'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `cli.result_saving [info]` | homodyne logs 'Saving JSON files (parameters, analysis, convergence)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `cli.result_saving [info]` | homodyne logs 'Saving NPZ file with all arrays'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `cli.result_saving [info]` | homodyne logs 'Saving legacy results summary for backward compatibility'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `cli.result_saving [info]` | homodyne logs 'Using comprehensive CMC result saving (async)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `cli.result_saving [info]` | homodyne logs 'Using comprehensive NLSQ result saving (async)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `cli.result_saving [info]` | homodyne logs 'Using per-angle least squares estimation for contrast/offset (fixes CMC sharding aggregation issue)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `cli.result_saving [warning]` | homodyne logs 'Background save errors (%d): %s'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `cli.result_saving [warning]` | homodyne logs 'Cannot cleanly infer n_angles: parameter count %d - n_physical %d = %d (odd). Falling back to n_angles=1.'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `cli.result_saving [warning]` | homodyne logs 'Empty wavevector_q_list'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `cli.result_saving [warning]` | homodyne logs 'Failed to save diagnostics.json (%s). Payload keys: %s'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `cli.result_saving [warning]` | homodyne logs 'Recorded scalar_per_angle_expansion=true in diagnostics (scalar contrast/offset replicated per angle).'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `cli.result_saving [warning]` | homodyne logs 'dt not found in config - may need manual specification'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `config.manager [debug]` | homodyne logs 'Applying default CMC configuration values'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `config.manager [debug]` | homodyne logs 'Applying default configuration values (fallback)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `config.manager [debug]` | homodyne logs 'CMC configuration validation passed'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `config.manager [debug]` | homodyne logs 'Configuration validation completed'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `config.manager [debug]` | homodyne logs 'No CMC configuration found, using defaults'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `config.manager [debug]` | homodyne logs 'Skipping normalization: data_folder_path or data_file_name is None'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `config.manager [info]` | homodyne logs 'Configuration loaded from override data'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `config.manager [info]` | homodyne logs 'No initial_parameters section in config, using mid-point defaults'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `config.manager [info]` | homodyne logs 'No parameter_names in initial_parameters, using active parameters from mode'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `config.manager [info]` | homodyne logs 'Using default configuration...'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `config.manager [warning]` | homodyne logs "Configuration file '%s' is empty or null; using defaults"; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `config.manager [warning]` | homodyne logs 'Configuration is empty'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `config.manager [warning]` | homodyne logs "Deprecated sharding key 'optimal_shard_size' detected. Use 'max_points_per_shard' instead."; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `config.manager [warning]` | homodyne logs 'No configuration loaded, using empty initial parameters'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `config.manager [warning]` | homodyne logs 'Unknown top-level config keys (possible typo): %s'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `config.manager [warning]` | homodyne logs 'force_stratified_ls=True enabled. This uses full Jacobian (high memory) - ensure sufficient RAM.'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `config.manager [warning]` | homodyne logs 'hierarchical.enable=False for laminar_flow may cause gradient cancellation issues with many phi angles.'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `config.manager [warning]` | homodyne logs 'memory_fraction=%s outside valid range (0, 1); should be between 0 and 1'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `config.manager [warning]` | homodyne logs 'optimization.angle_filtering must be a dict, ignoring (got %s)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `config.parameter_manager [warning]` | homodyne logs 'fixed_parameters must be a dict, ignoring'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `config.parameter_manager [warning]` | homodyne logs 'parameter_space.bounds must be a list, ignoring'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `config.parameter_space [debug]` | homodyne logs 'Converted %s prior to BetaScaled on [%s, %s] (alpha=%.3f, beta=%.3f)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `config.parameter_space [warning]` | homodyne logs 'parameter_space.bounds must be a list, using package defaults'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `core.diagonal_correction [warning]` | homodyne logs 'JAX not available, falling back to NumPy'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `core.fitting [error]` | homodyne logs 'Could not import core modules - fitting engine disabled'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `core.fitting [info]` | homodyne logs '  - Balanced iteration counts and memory usage'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `core.fitting [info]` | homodyne logs '  - Conservative iteration counts to manage memory'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `core.fitting [info]` | homodyne logs '  - Distributed processing with intelligent chunking'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `core.fitting [info]` | homodyne logs '  - Efficient batching with VI+JAX/MCMC+JAX'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `core.fitting [info]` | homodyne logs '  - Full JAX acceleration without chunking'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `core.fitting [info]` | homodyne logs '  - Higher iteration counts for better convergence'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `core.fitting [info]` | homodyne logs '  - In-memory VI+JAX processing for instant fits'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `core.fitting [info]` | homodyne logs '  - Moderate chunking for memory efficiency'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `core.fitting [info]` | homodyne logs '  - Progressive loading and compression'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `core.fitting [info]` | homodyne logs 'Large dataset optimization:'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `core.fitting [info]` | homodyne logs 'Medium dataset optimization:'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `core.fitting [info]` | homodyne logs 'Small dataset optimization:'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `core.homodyne_model [info]` | homodyne logs 'HomodyneModel initialized successfully'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `core.homodyne_model [info]` | homodyne logs 'Initializing HomodyneModel with hybrid architecture'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `core.homodyne_model [warning]` | homodyne logs 'Data was saved successfully, continuing...'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `core.homodyne_model [warning]` | homodyne logs 'matplotlib not available, skipping plots'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `core.model_mixins [info]` | homodyne logs 'Benchmarking gradient computation performance...'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `core.model_mixins [info]` | homodyne logs 'Gradient accuracy validation completed'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `core.model_mixins [info]` | homodyne logs 'Gradient performance benchmark completed'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `core.model_mixins [info]` | homodyne logs 'Validating gradient accuracy...'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `core.models [debug]` | homodyne logs 'CombinedModel.compute_g1: calling compute_g1_diffusion with params.shape=%s'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `core.models [debug]` | homodyne logs 'CombinedModel.compute_g1: calling compute_g1_total with params.shape=%s, t1.shape=%s, t2.shape=%s, phi.shape=%s, q=%s, L=%s, dt=%s'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `core.models [error]` | homodyne logs 'CombinedModel.compute_g1: traceback:'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `core.numpy_gradients [debug]` | homodyne logs 'Numeric step-size computation failed for param %d, using initial step'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `core.numpy_gradients [debug]` | homodyne logs 'Step-size estimation failed in Hessian computation, using default step'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.angle_filtering [debug]` | homodyne logs 'No config available for angle filtering, plotting all angles'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.angle_filtering [debug]` | homodyne logs 'Phi filtering not enabled, plotting all angles'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.angle_filtering [debug]` | homodyne logs 'Phi filtering not enabled, using all angles for optimization'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.angle_filtering [info]` | homodyne logs 'Normalized phi angles to [-180, 180] deg range (flow direction at 0 deg)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.angle_filtering [warning]` | homodyne logs 'No angles matched phi_filtering criteria, using all angles'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.angle_filtering [warning]` | homodyne logs 'No angles matched target ranges, plotting all angles'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.angle_filtering [warning]` | homodyne logs 'No phi angles or C2 data available, cannot apply filtering'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.angle_filtering [warning]` | homodyne logs 'Phi filtering enabled but no target_ranges specified, plotting all angles'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.angle_filtering [warning]` | homodyne logs 'Phi filtering enabled but no target_ranges specified, using all angles'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.config [info]` | homodyne logs 'Consider migrating to YAML format for improved readability'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.config [info]` | homodyne logs 'Migrated JSON configuration to YAML format'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.filtering_utils [info]` | homodyne logs 'Data filtering disabled - returning all indices'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.filtering_utils [info]` | homodyne logs 'No filtering criteria specified - returning all indices'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.filtering_utils [warning]` | homodyne logs 'Falling back to all indices due to empty filter result'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.filtering_utils [warning]` | homodyne logs 'Falling back to all indices due to filtering error'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.filtering_utils [warning]` | homodyne logs 'Filtering resulted in no selected data points'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.memory_manager [critical]` | homodyne logs 'Critical memory pressure - performing emergency cleanup'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.memory_manager [debug]` | homodyne logs 'Cleared JAX compilation cache'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.memory_manager [debug]` | homodyne logs 'JAX clear_caches() not available (older JAX version)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.memory_manager [debug]` | homodyne logs 'Skipping GC - previous calls freed 0 objects (memory likely in JAX/NumPy arrays)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.memory_manager [info]` | homodyne logs 'Advanced memory manager initialized'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.memory_manager [info]` | homodyne logs 'Advanced memory manager shutdown complete'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.memory_manager [info]` | homodyne logs 'Memory pressure monitoring started'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.memory_manager [info]` | homodyne logs 'Memory pressure monitoring stopped'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.memory_manager [info]` | homodyne logs 'Memory pressure recovered - restoring normal operation'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.memory_manager [info]` | homodyne logs 'Shutting down advanced memory manager'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.memory_manager [warning]` | homodyne logs 'Memory allocation succeeded after emergency cleanup'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.memory_manager [warning]` | homodyne logs 'Memory monitoring already active'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.memory_manager [warning]` | homodyne logs 'Memory pressure warning - triggering optimization'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.memory_manager [warning]` | homodyne logs 'Performing emergency memory cleanup'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.memory_manager [warning]` | homodyne logs 'Zero-size virtual memory allocation requested'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.optimization [debug]` | homodyne logs 'Scheduling background optimization for potential CMC follow-up'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.optimization [debug]` | homodyne logs 'Set JAX env var %s=%s (may be ignored if JAX already imported)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.optimization [info]` | homodyne logs 'Advanced dataset optimizer cleanup complete'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.optimization [info]` | homodyne logs 'Advanced dataset optimizer initialized with performance engine integration'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.optimization [info]` | homodyne logs 'Advanced memory manager initialized'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.optimization [info]` | homodyne logs 'Cleaning up advanced dataset optimizer'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.optimization [info]` | homodyne logs 'Dataset analysis complete:'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.optimization [info]` | homodyne logs 'Dataset optimizer initialized:'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.optimization [info]` | homodyne logs 'Performance engine initialized'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.performance_engine [debug]` | homodyne logs 'Background processing started'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.performance_engine [debug]` | homodyne logs 'Performance monitoring started'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.performance_engine [info]` | homodyne logs 'All memory mappings closed'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.performance_engine [info]` | homodyne logs 'Performance engine initialized with advanced optimizations'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.performance_engine [info]` | homodyne logs 'Performance engine shutdown complete'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.performance_engine [info]` | homodyne logs 'Shutting down performance engine'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.phi_filtering [debug]` | homodyne logs 'Angle filtering disabled, returning all angles'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.phi_filtering [warning]` | homodyne logs 'Falling back to using all angles for optimization'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.preprocessing [debug]` | homodyne logs 'Output validation completed'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.preprocessing [debug]` | homodyne logs 'Standardizing data format'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.preprocessing [debug]` | homodyne logs 'Validating output data integrity'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.preprocessing [warning]` | homodyne logs 'Scipy not available for Savitzky-Golay filtering, falling back to median'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.preprocessing [warning]` | homodyne logs 'Scipy not available for Wiener filtering, falling back to gaussian'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.preprocessing [warning]` | homodyne logs 'Scipy not available for gaussian filtering, skipping'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.preprocessing [warning]` | homodyne logs 'Scipy not available for median filtering, skipping'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.preprocessing [warning]` | homodyne logs 't1 array is not monotonically increasing'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.quality_controller [debug]` | homodyne logs 'Quality control disabled - creating minimal result'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.quality_controller [debug]` | homodyne logs 'Re-validating data after auto-repair'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.quality_controller [debug]` | homodyne logs 'Validating filtered data stage'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.quality_controller [debug]` | homodyne logs 'Validating final data stage'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.quality_controller [debug]` | homodyne logs 'Validating preprocessed data stage'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.quality_controller [debug]` | homodyne logs 'Validating raw data stage'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.quality_controller [debug]` | homodyne logs 'Validation cache cleared'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.quality_controller [info]` | homodyne logs 'Generating comprehensive quality assessment report'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.quality_controller [warning]` | homodyne logs 'Could not compare data sizes before/after filtering'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.quality_controller [warning]` | homodyne logs 'Could not perform advanced data quality checks'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.quality_controller [warning]` | homodyne logs 'Could not perform basic data quality checks'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.quality_controller [warning]` | homodyne logs 'Could not perform data consistency checks'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.validation [debug]` | homodyne logs 'No data changes detected - using cached results'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.validation [debug]` | homodyne logs 'Performing incremental validation'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.validation [debug]` | homodyne logs 'Using cached validation result'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.validation [debug]` | homodyne logs 'Validation cache cleared'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.validation [info]` | homodyne logs 'Validation disabled - skipping all checks'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.xpcs_loader [debug]` | homodyne logs 'Applying comprehensive data filtering'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.xpcs_loader [debug]` | homodyne logs 'Applying mandatory diagonal correction to correlation matrices'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.xpcs_loader [debug]` | homodyne logs 'Applying phi-only filtering (no quality filtering)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.xpcs_loader [debug]` | homodyne logs 'Auto-selecting JAX format (available)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.xpcs_loader [debug]` | homodyne logs 'Auto-selecting numpy format (JAX not available)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.xpcs_loader [debug]` | homodyne logs 'Converting arrays to JAX format'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.xpcs_loader [debug]` | homodyne logs 'Data filtering disabled in configuration'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.xpcs_loader [debug]` | homodyne logs 'Legacy phi filtering not enabled'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.xpcs_loader [debug]` | homodyne logs 'No frame slicing needed - using full range'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.xpcs_loader [debug]` | homodyne logs 'Phi filtering already applied in main filtering system'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.xpcs_loader [debug]` | homodyne logs 'Phi filtering system not available - using original selection'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.xpcs_loader [debug]` | homodyne logs 'Preprocessing pipeline disabled in configuration'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.xpcs_loader [debug]` | homodyne logs 'Quality control disabled in configuration'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.xpcs_loader [debug]` | homodyne logs 'Quality filtering enabled - running metadata-only pre-filter'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.xpcs_loader [debug]` | homodyne logs 'Selecting optimal q-vector for caching'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.xpcs_loader [debug]` | homodyne logs 'Starting HDF5 format detection'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.xpcs_loader [debug]` | homodyne logs 'Transformed flat config structure to nested structure'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.xpcs_loader [error]` | homodyne logs 'Correlation data contains non-finite values (NaN or Inf)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.xpcs_loader [error]` | homodyne logs 'Preprocessing pipeline failed'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.xpcs_loader [info]` | homodyne logs 'Advanced dataset optimizer initialized'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.xpcs_loader [info]` | homodyne logs 'Advanced memory manager initialized'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.xpcs_loader [info]` | homodyne logs 'Applying preprocessing pipeline to loaded data'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.xpcs_loader [info]` | homodyne logs 'Consider migrating to YAML format for better readability'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.xpcs_loader [info]` | homodyne logs 'Data quality validation completed'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.xpcs_loader [info]` | homodyne logs 'Falling back to basic optimization'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.xpcs_loader [info]` | homodyne logs 'File will be checked again during data loading'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.xpcs_loader [info]` | homodyne logs 'Filtering statistics:'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.xpcs_loader [info]` | homodyne logs 'Initializing data quality control system'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.xpcs_loader [info]` | homodyne logs 'Performance engine disabled in configuration'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.xpcs_loader [info]` | homodyne logs 'Performance engine initialized'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.xpcs_loader [info]` | homodyne logs 'Performing comprehensive data quality analysis...'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.xpcs_loader [info]` | homodyne logs 'Physics validation completed'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.xpcs_loader [info]` | homodyne logs 'Preprocessing pipeline completed successfully'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.xpcs_loader [info]` | homodyne logs 'Raw data was modified by quality control auto-repair'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.xpcs_loader [warning]` | homodyne logs 'Correlation data contains negative values'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.xpcs_loader [warning]` | homodyne logs 'Falling back to no filtering due to error'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.xpcs_loader [warning]` | homodyne logs 'Falling back to original data after preprocessing error'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.xpcs_loader [warning]` | homodyne logs 'Falling back to original data after preprocessing failure'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.xpcs_loader [warning]` | homodyne logs 'Filtering used fallback - all data points included'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.xpcs_loader [warning]` | homodyne logs 'Loading legacy cache without selective q-vector optimization'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.xpcs_loader [warning]` | homodyne logs 'No data filtering criteria matched - returning all angles. Check filter configuration if this is unexpected.'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.xpcs_loader [warning]` | homodyne logs 'No valid indices found, using first available entry as fallback'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.xpcs_loader [warning]` | homodyne logs 'Performance engine not available - falling back to basic optimization'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `data.xpcs_loader [warning]` | homodyne logs 'Physics validation requested but v2 physics module not available'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `device.__init__ [info]` | homodyne logs 'Configuring CPU optimization...'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `device.__init__ [info]` | homodyne logs 'Configuring optimal CPU device for homodyne analysis'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `device.__init__ [info]` | homodyne logs 'Running CPU benchmark...'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `device.__init__ [info]` | homodyne logs '[OK] Basic CPU configuration completed'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `device.config [debug]` | homodyne logs 'PBS_JOBID present but PBS_NODEFILE not found'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `device.config [info]` | homodyne logs 'Detecting hardware configuration for CMC...'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `device.config [info]` | homodyne logs 'Standalone system detected (no cluster scheduler)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `device.config [warning]` | homodyne logs 'Failed to parse SLURM_JOB_NUM_NODES'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `device.config [warning]` | homodyne logs 'psutil not available. Assuming 32 GB system memory'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `device.config [warning]` | homodyne logs 'psutil not available. Using multiprocessing for CPU count'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `device.cpu [debug]` | homodyne logs 'NUMA detection via lscpu failed: %s'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `device.cpu [info]` | homodyne logs 'Configuring CPU optimization for HPC environment'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `device.cpu [info]` | homodyne logs 'Intel oneDNN enabled (experimental for XPCS workloads). Benchmark to verify performance improvements.'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `device.cpu [info]` | homodyne logs 'JAX CPU configuration completed successfully'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `device.cpu [warning]` | homodyne logs 'oneDNN requested but CPU is not Intel. Skipping oneDNN.'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.__init__ [warning]` | homodyne logs 'Could not import CMC optimization: %s'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.__init__ [warning]` | homodyne logs 'Could not import NLSQ optimization: %s'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.checkpoint_manager [debug]` | homodyne logs 'No valid checkpoint found for recovery'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.cmc.backends.base [warning]` | homodyne logs "CMC backend 'jax' is deprecated; mapping to 'multiprocessing' for parallel execution. Set backend_config.name to 'multiprocessing' or 'auto' instead."; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.cmc.backends.base [warning]` | homodyne logs 'PBS backend not available, falling back to multiprocessing'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.cmc.backends.base [warning]` | homodyne logs 'pjit backend not available, falling back to multiprocessing'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.cmc.backends.multiprocessing [debug]` | homodyne logs 'Per-shard spawn: %d shards < 3, pool not beneficial'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.cmc.backends.multiprocessing [error]` | homodyne logs "ParameterSpace._config_dict is absent and no 'config_dict' in model_kwargs. Workers will reconstruct ParameterSpace from an empty dict (default bounds). This may produce unconstrained or incorrect NUTS proposals. Ensure ParameterSpace exposes _config_dict or pass config_dict in model_kwargs."; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.cmc.backends.multiprocessing [info]` | homodyne logs 'No shards reported adapted n_warmup; CMCResult will use config default'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.cmc.backends.multiprocessing [info]` | homodyne logs 'Running single-shard MCMC (no parallelization)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.cmc.backends.multiprocessing [info]` | homodyne logs 'WorkerPool dispatched %d shards to %d persistent workers'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.cmc.backends.multiprocessing [warning]` | homodyne logs 'Interrupted - terminating all active processes'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.cmc.backends.multiprocessing [warning]` | homodyne logs 'WorkerPool creation failed (%s), falling back to per-shard spawn'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.cmc.backends.multiprocessing [warning]` | homodyne logs 'WorkerPool stall detected: no result for %.0fs (per_shard_timeout=%.0fs). Shutting down pool.'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.cmc.backends.multiprocessing [warning]` | homodyne logs 'parameter_space is None - bounds-aware CV disabled; heterogeneity detection may produce false positives for near-zero parameters'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.cmc.backends.pbs [info]` | homodyne logs 'PBSBackend: All jobs completed successfully'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.cmc.backends.pbs [info]` | homodyne logs 'PBSBackend: PBS scheduler detected'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.cmc.backends.pbs [warning]` | homodyne logs 'PBSBackend: qstat not found. PBS commands may not be available.'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.cmc.backends.pbs [warning]` | homodyne logs 'PBSBackend: qstat timed out'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.cmc.backends.pjit [info]` | homodyne logs 'PjitBackend: Running single MCMC (no sharding)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.cmc.backends.pjit [warning]` | homodyne logs 'PjitBackend: Only 1 device available. Consider using multiprocessing backend for better parallelism.'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.cmc.backends.worker_pool [debug]` | homodyne logs 'detect_cpu_info unavailable, falling back to os.cpu_count'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.cmc.backends.worker_pool [info]` | homodyne logs 'WorkerPool shut down'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.cmc.backends.worker_pool [info]` | homodyne logs 'WorkerPool started: %d/%d workers ready'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.cmc.backends.worker_pool [warning]` | homodyne logs 'Worker %d did not exit gracefully, terminating'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.cmc.backends.worker_pool [warning]` | homodyne logs 'Worker startup timed out after %.0fs (%d/%d ready)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.cmc.config [debug]` | homodyne logs 'Note: CLI applies base mcmc settings to per_shard_mcmc. If using CLI, ensure base mcmc and per_shard_mcmc are aligned.'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.cmc.config [info]` | homodyne logs 'CMC per-angle mode: auto -> constant_averaged (NLSQ warm-start present, fixing scaling for stability)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.cmc.config [warning]` | homodyne logs "CMC backend 'jax' is deprecated; mapping to 'multiprocessing' for parallel execution. Set backend_config.name to 'multiprocessing' or 'auto' instead."; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.cmc.core [info]` | homodyne logs '  -> Actual runtime much faster than expected - estimate may be conservative'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.cmc.core [info]` | homodyne logs '  -> Consider reducing num_samples or num_chains for faster runs'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.cmc.core [info]` | homodyne logs 'No initial values provided, using midpoint defaults'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.cmc.core [info]` | homodyne logs '[CMC] Reference time: t_ref=1.0 (fallback)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.cmc.core [warning]` | homodyne logs 'CMC WARM-START ADVISORY: Running laminar_flow without NLSQ warm-start. This is strongly discouraged for production use because:\n  1. 7 parameters span 6+ orders of magnitude (D0~1e4, gamma_dot_t0~1e-3)\n  2. NUTS adaptation may waste warmup exploring implausible regions\n  3. Higher divergence rates and inflated posterior uncertainty expected\nRecommendation: Run NLSQ first and pass nlsq_result to fit_mcmc_jax()\nTo enforce this, set validation.require_nlsq_warmstart=true in CMC config.'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.cmc.core [warning]` | homodyne logs 'No positive time differences found; falling back to dt=1.0'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.cmc.core [warning]` | homodyne logs "Overriding sharding_strategy='stratified' -> 'random' for multi-angle data. Stratified sharding violates Consensus MC assumptions for global parameters."; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.cmc.core [warning]` | homodyne logs 'SamplingPlan invariant violated: stats without plan. Using config defaults.'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.cmc.data_prep [info]` | homodyne logs 'Single angle detected - falling back to random sharding'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.cmc.diagnostics [debug]` | homodyne logs 'sklearn not available, skipping bimodality detection'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.cmc.model [info]` | homodyne logs 'CMC: Using auto mode model (sampled averaged scaling, 10 params)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.cmc.model [info]` | homodyne logs 'CMC: Using constant mode model (fixed per-angle scaling, 8 params)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.cmc.model [info]` | homodyne logs 'CMC: Using constant_averaged mode model (fixed averaged scaling, 8 params, NLSQ parity)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.cmc.model [info]` | homodyne logs 'CMC: Using individual mode model (sampled per-angle scaling)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.cmc.model [info]` | homodyne logs 'CMC: Using reparameterized auto mode model (log_D_ref + log_gamma_ref sampling)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.cmc.priors [warning]` | homodyne logs 'Ambiguous n_params=9: could be static-individual (3 angles) or laminar_flow auto_averaged. Defaulting to laminar_flow. Pass analysis_mode_hint to disambiguate.'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.cmc.sampler [debug]` | homodyne logs 'Using init_to_median (no initial values)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.cmc.sampler [info]` | homodyne logs 'Extracting samples + extra_fields...'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.cmc.sampler [info]` | homodyne logs 'NUTS phase: JIT compile + sampling started (may take minutes)...'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.cmc.sampler [warning]` | homodyne logs 'CRITICAL: Acceptance rate is essentially 0% - all proposals rejected! This indicates severe sampling problems. Possible causes:\n  1. Initial values are outside prior support or at boundaries\n  2. Likelihood returns -inf due to numerical issues (NaN/overflow)\n  3. Prior is too narrow for the data\n  4. Step size adaptation failed during warmup\nConsider: checking initial values, widening priors, or running NLSQ first.'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.cmc.scaling [info]` | homodyne logs 'Parameter scaling factors for gradient balancing:'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.gradient_diagnostics [info]` | homodyne logs '\nGradient Norms (SSE):'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.gradient_diagnostics [info]` | homodyne logs '\nThis can cause:'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.gradient_diagnostics [info]` | homodyne logs '\nTo apply these fixes:'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.gradient_diagnostics [info]` | homodyne logs '  - Missing fine-scale features (oscillations)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.gradient_diagnostics [info]` | homodyne logs '  - Poor fit quality despite low chi-squared'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.gradient_diagnostics [info]` | homodyne logs '  - Premature convergence'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.gradient_diagnostics [info]` | homodyne logs '  1. Add x_scale_map to your configuration file'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.gradient_diagnostics [info]` | homodyne logs '  2. Re-run optimization with updated config'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.gradient_diagnostics [info]` | homodyne logs '  3. Verify improved convergence and fit quality'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.gradient_diagnostics [info]` | homodyne logs 'GRADIENT DIAGNOSTIC REPORT'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.gradient_diagnostics [info]` | homodyne logs 'No significant gradient imbalance'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.gradient_diagnostics [info]` | homodyne logs 'RECOMMENDATIONS'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.gradient_diagnostics [warning]` | homodyne logs 'GRADIENT IMBALANCE DETECTED'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.gradient_diagnostics [warning]` | homodyne logs 'data.dt is missing or None; using dt=1.0 for gradient diagnostics. Gradient norms will be correct only if the true frame interval is 1.0 s.'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.adapter [debug]` | homodyne logs 'Model cache stats: hits=%d, misses=%d, size=%d'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.adapter [debug]` | homodyne logs 'Model created in %.3fs (JIT=%s)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.adapter [debug]` | homodyne logs 'Selected workflow: %s'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.adapter [error]` | homodyne logs 'NLSQ optimization failed: %s'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.adapter [info]` | homodyne logs 'NLSQAdapter.fit completed: chi2=%.6g, reduced_chi2=%.6g, status=%s, time=%.2fs'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.adapter [info]` | homodyne logs 'NLSQAdapter.fit: n_data=%d, n_params=%d, n_phi=%d, mode=%s'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.adaptive_regularization [info]` | homodyne logs 'Adaptive Regularization Summary:'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.adaptive_regularization [info]` | homodyne logs 'Adaptive regularization: DISABLED'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.anti_degeneracy_controller [info]` | homodyne logs '  Behavior: Quantile estimates -> per-angle values FIXED (NOT optimized)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.anti_degeneracy_controller [info]` | homodyne logs '  Parameters: 7 physical + 2 averaged scaling = 9 total'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.anti_degeneracy_controller [info]` | homodyne logs '  Parameters: 7 physical only (scaling FIXED from quantiles)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.anti_degeneracy_controller [info]` | homodyne logs 'ANTI-DEGENERACY: Layer 5 - Shear-Sensitivity Weighting'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.cmaes_wrapper [debug]` | homodyne logs '[CMA-ES] Parameter bounds (canonical order): lower=%s, upper=%s'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.cmaes_wrapper [debug]` | homodyne logs '[CMA-ES] Parameters denormalized from [0,1] to physical space'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.cmaes_wrapper [debug]` | homodyne logs '[CMA-ES] Post-refinement disabled, using global search result'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.cmaes_wrapper [info]` | homodyne logs '[CMA-ES] Global search phase starting...'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.cmaes_wrapper [info]` | homodyne logs '[CMA-ES] Method unavailable: evosax not installed. Install with: pip install nlsq[evosax]'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.cmaes_wrapper [info]` | homodyne logs "[CMA-ES] Warm-start: overriding restart_strategy='bipop' -> 'none' (BIPOP large-population restarts are incoherent with small sigma_warmstart)"; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.core [debug]` | homodyne logs 'Attempting optimization with NLSQAdapter'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.core [debug]` | homodyne logs 'No global optimization enabled, using local optimization'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.core [debug]` | homodyne logs 'Using NLSQAdapter (CurveFit class) for optimization'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.core [error]` | homodyne logs 'Both NLSQAdapter and NLSQWrapper failed: adapter=%s, wrapper=%s'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.core [info]` | homodyne logs '  Averaged to: 1 contrast + 1 offset (OPTIMIZED)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.core [info]` | homodyne logs '  Per-angle values: FIXED (not optimized)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.core [info]` | homodyne logs '  Physical parameters:'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.core [info]` | homodyne logs '  Total parameters: 7 physical + 2 averaged scaling = 9'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.core [info]` | homodyne logs '  Total parameters: 7 physical only'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.core [info]` | homodyne logs 'ANTI-DEGENERACY: Enabled for CMA-ES (Auto Averaged Mode)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.core [info]` | homodyne logs 'ANTI-DEGENERACY: Enabled for CMA-ES (Fixed Constant Mode)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.core [info]` | homodyne logs 'CMA-ES OPTIMIZATION COMPLETE'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.core [info]` | homodyne logs 'CMA-ES enabled, delegating to fit_nlsq_cmaes'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.core [info]` | homodyne logs 'CONSTANT MODE: Computing per-angle scaling from quantiles'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.core [info]` | homodyne logs 'Fitted parameters:'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.core [info]` | homodyne logs 'Fixed constant mode: per-angle scaling will be FIXED'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.core [info]` | homodyne logs 'Including user-specified initial parameters as custom start point'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.core [info]` | homodyne logs 'Loaded initial parameters from configuration for multi-start optimization'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.core [info]` | homodyne logs 'Multi-start enabled, delegating to fit_nlsq_multistart'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.core [info]` | homodyne logs 'NLSQWrapper fallback optimization succeeded'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.core [info]` | homodyne logs 'Using default initial parameters'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.core [info]` | homodyne logs 'Using initial parameters from configuration'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.core [info]` | homodyne logs '[CMA-ES] NLSQ warm-start did not improve fit, using original starting point'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.core [info]` | homodyne logs '[CMA-ES] Phase 1: Running NLSQ warm-start...'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.core [warning]` | homodyne logs "Could not estimate contrast/offset: no 'g2' or 'c2_exp' in data. Using generic defaults (0.5, 1.0)"; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.core [warning]` | homodyne logs "Initial parameters in config missing 'parameter_names' or 'values'"; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.core [warning]` | homodyne logs 'NLSQAdapter failed, falling back to NLSQWrapper: %s'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.core [warning]` | homodyne logs 'NLSQAdapter requested but not available, falling back to NLSQWrapper'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.core [warning]` | homodyne logs '[CMA-ES] Enabled in config but not available (evosax not installed). Install with: pip install nlsq[evosax]. Falling back to multi-start or local optimization.'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.core [warning]` | homodyne logs '[Multi-Start] Enabled in config but not available. Falling back to local optimization.'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.core [warning]` | homodyne logs 'per_angle_scaling in initial_parameters must provide equal-length contrast/offset arrays; ignoring overrides'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.data_prep [info]` | homodyne logs 'Expanding scaling parameters for per-angle scaling:'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.fallback_chain [info]` | homodyne logs 'NLSQ Result Analysis:'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.fallback_chain [info]` | homodyne logs 'Using NLSQ AdaptiveHybridStreamingOptimizer for large datasets...'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.fallback_chain [warning]` | homodyne logs 'No pcov attribute in result object. Using identity matrix.'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.fallback_chain [warning]` | homodyne logs 'Optimization failure: Parameters unchanged from initial guess!\n   This suggests curve_fit returned immediately without optimizing.\n   Possible causes: (1) Already at optimum, (2) Singular Jacobian, (3) Bounds too tight'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.fit_computation [debug]` | homodyne logs 'Unable to infer analysis_mode from params=%s angles=%s; defaulting to static'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.fit_computation [info]` | homodyne logs 'Note: lstsq contrast/offset values may differ from NLSQ-optimized values. lstsq re-fits scaling to raw theory (contrast=1, offset=1) post-hoc; NLSQ values are authoritative as they are jointly optimized with physical parameters.'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.fit_computation [warning]` | homodyne logs 'Solver returned scalar contrast/offset (parameter count %d). Expanding scalars across %d filtered angles for result saving.'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.gradient_monitor [info]` | homodyne logs '  Status: No collapse detected'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.gradient_monitor [info]` | homodyne logs 'Gradient Collapse Monitor Summary:'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.gradient_monitor [info]` | homodyne logs 'Gradient monitoring: DISABLED'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.gradient_monitor [info]` | homodyne logs 'Gradient monitoring: No checks performed'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.hierarchical [info]` | homodyne logs 'HIERARCHICAL OPTIMIZATION'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.hierarchical [info]` | homodyne logs 'HIERARCHICAL OPTIMIZATION COMPLETE'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.multistart [debug]` | homodyne logs 'Screening disabled, proceeding with all starting points'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.multistart [error]` | homodyne logs 'All multi-start optimizations failed!'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.multistart [info]` | homodyne logs 'Falling back to sequential execution'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.multistart [info]` | homodyne logs 'MULTI-START NLSQ OPTIMIZATION'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.multistart [info]` | homodyne logs 'MULTI-START OPTIMIZATION COMPLETE'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.multistart [info]` | homodyne logs 'PHASE 1: Generating starting points'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.multistart [info]` | homodyne logs 'PHASE 2: Screening starting points'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.multistart [info]` | homodyne logs 'PHASE 3: Running optimizations'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.multistart [info]` | homodyne logs 'PHASE 4: Analyzing results'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.multistart [info]` | homodyne logs 'Strategy: FULL (all starting points run complete optimization)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.multistart [warning]` | homodyne logs 'Parameter bounds have zero volume (all lower == upper). Falling back to single-start at bounds center.'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.parallel_accumulator [info]` | homodyne logs 'OOCComputePool shut down'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.parallel_accumulator [info]` | homodyne logs 'OOCComputePool started: %d workers, %d chunks'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.parallel_accumulator [warning]` | homodyne logs 'Parallel chunk accumulation failed (%s), falling back to sequential'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.progress [debug]` | homodyne logs 'NLSQ ProgressBar not available'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.progress [warning]` | homodyne logs 'tqdm not available for progress bar display. Install with: pip install tqdm'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.recovery [debug]` | homodyne logs 'Using curve_fit_large with NLSQ automatic memory management'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.recovery [error]` | homodyne logs 'Optimization returned unchanged parameters after all retries. This may indicate a bug in NLSQ or an intractable problem.'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.recovery [info]` | homodyne logs '  bounds=None (unbounded)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.recovery [info]` | homodyne logs 'NLSQ curve_fit RESULT DIAGNOSTICS'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.recovery [info]` | homodyne logs 'Retrying with perturbed parameters...'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.recovery [warning]` | homodyne logs '   Affected parameters were likely NOT optimized by NLSQ.'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.recovery [warning]` | homodyne logs '   This indicates singular/ill-conditioned Jacobian matrix!'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.result_builder [warning]` | homodyne logs 'No pcov attribute in result object. Using identity matrix.'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.shear_weighting [debug]` | homodyne logs 'Shear-sensitivity weighting disabled by config'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.shear_weighting [debug]` | homodyne logs 'phi0 not in physical params -- shear weighting disabled'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.chunking [info]` | homodyne logs 'Single phi angle detected, no stratification needed'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.chunking [warning]` | homodyne logs 'psutil not available and os.sysconf failed, using conservative default of 16 GB'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.chunking [warning]` | homodyne logs 'psutil not available, cannot check memory safety'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.executors [debug]` | homodyne logs 'Using curve_fit_large for memory-efficient optimization'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.executors [debug]` | homodyne logs 'Using standard curve_fit'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.executors [info]` | homodyne logs 'Using NLSQ AdaptiveHybridStreamingOptimizer for large dataset...'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.executors [warning]` | homodyne logs 'OptimizeResult has no pcov attribute - using identity matrix as covariance placeholder. Uncertainties will be unreliable.'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.executors [warning]` | homodyne logs "Streaming optimizer result has no 'pcov' key - using identity matrix as covariance placeholder. Uncertainties will be unreliable."; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [debug]` | homodyne logs 'Fixed-constant mode: No per-angle regularization (scaling is fixed)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [info]` | homodyne logs '  4-Layer Defense Strategy (NLSQ 0.3.6):'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [info]` | homodyne logs '  Behavior: Quantile estimates -> AVERAGED -> OPTIMIZED'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [info]` | homodyne logs '  Behavior: Quantile estimates -> per-angle values FIXED (NOT optimized)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [info]` | homodyne logs '  Covariance expanded: per-angle=0 (fixed), physical=preserved'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [info]` | homodyne logs '  Covariance transformed from Fourier to per-angle space'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [info]` | homodyne logs '  Covariance transformed from constant to per-angle space'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [info]` | homodyne logs '  Fixed scaling mode: skipping group variance regularization (no per-angle params)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [info]` | homodyne logs '  Group Variance Regularization (NLSQ 0.3.8):'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [info]` | homodyne logs '  Initial values: averaged from per-angle quantile estimates'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [info]` | homodyne logs '  Method: Quantile estimates -> averaged -> OPTIMIZED'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [info]` | homodyne logs '  Method: Quantile-based per-angle scaling (FIXED, not optimized)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [info]` | homodyne logs '  Parameters: 7 physical + 2 averaged (contrast, offset) = 9 total'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [info]` | homodyne logs '  Parameters: 7 physical only (scaling FIXED from quantiles)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [info]` | homodyne logs '  Per-angle contrast/offset will be estimated from c2 data quantiles'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [info]` | homodyne logs '  Reason: Only 2 per-angle DoF (vs 46), no need for hierarchical alternation'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [info]` | homodyne logs '  Residual Weighting (Shear-Sensitivity):'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [info]` | homodyne logs '  Shear weighting: WILL BE APPLIED via hierarchical loss function'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [info]` | homodyne logs '  Skipped: constant scaling mode already prevents per-angle absorption'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [info]` | homodyne logs '  These values are FIXED (not optimized) during fitting'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [info]` | homodyne logs '  These will be OPTIMIZED along with 7 physical params (9 total)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [info]` | homodyne logs '  Using Fourier-wrapped model function'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [info]` | homodyne logs '  Using parameter-based averaged initial values (OPTIMIZED)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [info]` | homodyne logs '  Using quantile-based averaged initial values (OPTIMIZED)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [info]` | homodyne logs "ANTI-DEGENERACY DEFENSE: Auto-selected 'auto_averaged' mode"; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [info]` | homodyne logs "ANTI-DEGENERACY DEFENSE: Auto-selected 'individual' mode"; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [info]` | homodyne logs "ANTI-DEGENERACY DEFENSE: Explicit 'constant' mode -> fixed_constant"; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [info]` | homodyne logs 'ANTI-DEGENERACY DEFENSE: Layer 1 - Auto Averaged Mode (v2.18.0)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [info]` | homodyne logs 'ANTI-DEGENERACY DEFENSE: Layer 1 - Fixed Constant Mode (v2.18.0)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [info]` | homodyne logs 'ANTI-DEGENERACY DEFENSE: Layer 1 - Fourier Reparameterization'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [info]` | homodyne logs 'ANTI-DEGENERACY DEFENSE: Layer 2 - Hierarchical Optimization'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [info]` | homodyne logs 'ANTI-DEGENERACY DEFENSE: Layer 3 - Adaptive Regularization'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [info]` | homodyne logs 'ANTI-DEGENERACY DEFENSE: Layer 4 - Gradient Collapse Monitor'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [info]` | homodyne logs 'ANTI-DEGENERACY DEFENSE: Layer 5 - Shear-Sensitivity Weighting'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [info]` | homodyne logs 'ANTI-DEGENERACY EXECUTION: Auto Averaged Scaling Mode'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [info]` | homodyne logs 'ANTI-DEGENERACY EXECUTION: Fixed Per-Angle Scaling (v2.17.0)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [info]` | homodyne logs 'ANTI-DEGENERACY EXECUTION: Fourier Reparameterization'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [info]` | homodyne logs 'ANTI-DEGENERACY EXECUTION: Hierarchical Two-Stage Optimization'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [info]` | homodyne logs 'ANTI-DEGENERACY EXECUTION: Inverse Auto Averaged Transform'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [info]` | homodyne logs 'ANTI-DEGENERACY EXECUTION: Inverse Fixed Scaling Transform (v2.17.0)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [info]` | homodyne logs 'ANTI-DEGENERACY EXECUTION: Inverse Fourier Transform'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [info]` | homodyne logs 'Averaged scaling computed (initial values for optimization):'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [info]` | homodyne logs 'Computing quantile-based per-angle scaling estimates...'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [info]` | homodyne logs 'Fixed per-angle scaling computed (FIXED, not optimized):'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [info]` | homodyne logs 'Fixes: 1) Shear-term gradients, 2) Convergence, 3) Covariance'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [info]` | homodyne logs 'HYBRID STREAMING OPTIMIZATION COMPLETE'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [info]` | homodyne logs 'Hybrid streaming config:'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [info]` | homodyne logs 'Hybrid streaming optimization completed successfully'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [info]` | homodyne logs 'Initializing NLSQ AdaptiveHybridStreamingOptimizer...'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [info]` | homodyne logs 'Note: StreamingOptimizer was removed in NLSQ 0.4.0. Using AdaptiveHybridStreamingOptimizer instead.'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [info]` | homodyne logs 'Preparing hybrid streaming data...'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [info]` | homodyne logs 'Starting hybrid optimization (L-BFGS + Gauss-Newton)...'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [warning]` | homodyne logs ''; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [warning]` | homodyne logs '  - Check if gamma_dot_t0 ~ 0 means shear contribution is missing'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [warning]` | homodyne logs '  - Check reduced chi-squared: if worse than expected, re-run optimization'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [warning]` | homodyne logs '  - Consider static_isotropic mode if shear is truly absent'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [warning]` | homodyne logs '  - Enable multi-start optimization to explore parameter basins'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [warning]` | homodyne logs '  - Enable phi_filtering to use only angles near 0 and 90 deg for laminar flow'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [warning]` | homodyne logs '  - Use multi-start optimization to explore multiple parameter basins'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [warning]` | homodyne logs '  - Verify per-angle contrast/offset are not varying excessively'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [warning]` | homodyne logs '  1. Per-angle contrast/offset absorbed the shear signal'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [warning]` | homodyne logs '  1. The optimizer cannot find gradient information for these parameters'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [warning]` | homodyne logs '  2. Inconsistent initialization of per-angle vs physical params'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [warning]` | homodyne logs '  2. The initial guess was already at or near the bounds'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [warning]` | homodyne logs '  3. Physical parameters at bounds with weak gradient signal'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [warning]` | homodyne logs '  3. The model is insensitive to these parameters with this data coverage'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [warning]` | homodyne logs '  4. The data may genuinely have no measurable shear'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [warning]` | homodyne logs '  Auto-enabling hierarchical optimization to apply shear weights.'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [warning]` | homodyne logs '  Without this, gradient cancellation will collapse gamma_dot_t0.'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [warning]` | homodyne logs 'ANTI-DEGENERACY: Shear weighting enabled but hierarchical disabled!'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [warning]` | homodyne logs 'Failed to compute quantile-based scaling, falling back to standard constant mode (optimizing 2 params)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [warning]` | homodyne logs 'GRADIENT COLLAPSE WAS DETECTED DURING OPTIMIZATION'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [warning]` | homodyne logs 'PARAMETER BOUNDS WARNING'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [warning]` | homodyne logs 'POSSIBLE CAUSES:'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [warning]` | homodyne logs 'RECOMMENDED ACTIONS:'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [warning]` | homodyne logs 'SHEAR COLLAPSE WARNING'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [warning]` | homodyne logs 'Singular Hessian in hierarchical path, using pseudo-inverse'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [warning]` | homodyne logs 'The following parameters are stuck at bounds with zero uncertainty:'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [warning]` | homodyne logs 'The model has effectively collapsed to static_isotropic mode.'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [warning]` | homodyne logs 'This may indicate:'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.hybrid_streaming [warning]` | homodyne logs 'This means the shear contribution to g1 is negligible.'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.out_of_core [debug]` | homodyne logs 'Sequential chunk reduction: %d chunks'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.out_of_core [info]` | homodyne logs 'Initializing Out-of-Core Global Stratified Optimization (Full Physics)...'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.out_of_core [info]` | homodyne logs 'No per-point sigma available - using unit weighting for OOC'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.out_of_core [info]` | homodyne logs 'Parallel OOC compute: %d chunks across %d workers'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.out_of_core [info]` | homodyne logs 'Parallel chunk reduction: %d chunks'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.out_of_core [warning]` | homodyne logs 'Could not find better step. Stopping.'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.out_of_core [warning]` | homodyne logs 'Gradient/Hessian contains NaNs/Infs. Checking params.'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.out_of_core [warning]` | homodyne logs 'Parallel OOC pool creation failed (%s), using sequential'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.out_of_core [warning]` | homodyne logs 'Singular J^T J in OOC - using pseudo-inverse for covariance'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.out_of_core [warning]` | homodyne logs '_fit_with_stratified_least_squares (OOC): dt not found in data or config; using dt=0.001 s as fallback.'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.residual [debug]` | homodyne logs 'Inline chunk structure validation passed'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.residual [info]` | homodyne logs 'Chunk structure validation passed'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.residual [info]` | homodyne logs 'Chunk structure validation passed (cached -- validated during build)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.residual [warning]` | homodyne logs 'StratifiedResidualFunction: dt not set (chunk_dt is None); using dt=0.001 s as fallback. Physics factors may be incorrect.'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.residual_jit [info]` | homodyne logs 'CONSTANT MODE: Using fixed per-angle scaling from quantiles. Parameter vector contains ONLY physical parameters.'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.residual_jit [info]` | homodyne logs 'Chunk structure validation passed: all chunks angle-complete'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.residual_jit [info]` | homodyne logs 'JIT-compiling residual function...'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.residual_jit [info]` | homodyne logs 'Stratified Residual Function Diagnostics:'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.residual_jit [warning]` | homodyne logs 'StratifiedResidualFunctionJIT: dt is None; using dt=0.001 s as fallback. Physics factors may be incorrect.'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.sequential [debug]` | homodyne logs 'Angle %.2f deg dtype check: init=%s%s lower=%s%s upper=%s%s'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.sequential [debug]` | homodyne logs 'Sequential least_squares kwargs sanitized: %s'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.sequential [warning]` | homodyne logs "%s mapping key '%s' not found in parameter_names; ignoring"; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.sequential [warning]` | homodyne logs 'Dropping non-numeric %s due to %s; reverting to default'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.sequential [warning]` | homodyne logs 'Dropping non-numeric max_nfev due to %s; reverting to default'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.sequential [warning]` | homodyne logs 'Dropping non-numeric x_scale due to %s; reverting to default'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.sequential [warning]` | homodyne logs 'Singular J^T J - used pinv fallback for covariance'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.stratified_ls [error]` | homodyne logs ''; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.stratified_ls [error]` | homodyne logs '  - Optimization will fail with 0 iterations'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.stratified_ls [error]` | homodyne logs '  - Parameter initialization issue (likely wrong parameter count)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.stratified_ls [error]` | homodyne logs '  - Residual function not sensitive to parameter changes'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.stratified_ls [error]` | homodyne logs 'Diagnostic information:'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.stratified_ls [error]` | homodyne logs 'GRADIENT SANITY CHECK FAILED'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.stratified_ls [error]` | homodyne logs 'This indicates:'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.stratified_ls [info]` | homodyne logs 'ANTI-DEGENERACY DEFENSE: Enabled for Stratified LS'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.stratified_ls [info]` | homodyne logs 'AUTO_AVERAGED MODE: Computing averaged scaling initial values'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.stratified_ls [info]` | homodyne logs 'Auto averaged mode: parameter transformation deferred to quantile-based averaged scaling computation'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.stratified_ls [info]` | homodyne logs 'Computing covariance matrix from Jacobian...'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.stratified_ls [info]` | homodyne logs 'Creating JIT-compatible stratified residual function...'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.stratified_ls [info]` | homodyne logs 'FIXED_CONSTANT MODE: Computing fixed per-angle scaling from quantiles'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.stratified_ls [info]` | homodyne logs 'Fixed constant mode: parameter transformation deferred to quantile-based fixed scaling computation'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.stratified_ls [info]` | homodyne logs 'GRADIENT SANITY CHECK'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.stratified_ls [info]` | homodyne logs 'OPTIMIZATION RESULTS'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.stratified_ls [info]` | homodyne logs 'STRATIFIED LEAST-SQUARES OPTIMIZATION'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.stratified_ls [info]` | homodyne logs 'Starting NLSQ least_squares() optimization...'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.stratified_ls [info]` | homodyne logs "Using NLSQ's least_squares() with angle-stratified chunks"; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.stratified_ls [info]` | homodyne logs 'Using covariance matrix from NLSQ result'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.stratified_ls [info]` | homodyne logs 'Validating chunk structure...'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.stratified_ls [warning]` | homodyne logs '  - Insufficient constraints (consider constrained optimizer)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.stratified_ls [warning]` | homodyne logs '  - Optimizer exploring unphysical parameter space'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.stratified_ls [warning]` | homodyne logs '  - Poor initial conditions (check config initial_parameters.values)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.stratified_ls [warning]` | homodyne logs 'BOUNDS VIOLATION DETECTED'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.stratified_ls [warning]` | homodyne logs 'Failed to compute fixed per-angle scaling, falling back to standard mode'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.stratified_ls [warning]` | homodyne logs 'Failed to compute per-angle scaling estimates, falling back to mean of initial per-angle values'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.stratified_ls [warning]` | homodyne logs 'Fixed constant mode but no fixed scaling available. Unexpected state - results may be unreliable.'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.stratified_ls [warning]` | homodyne logs 'One or more parameters violated physical bounds.'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.stratified_ls [warning]` | homodyne logs 'Parameters have been clipped to valid ranges.'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.stratified_ls [warning]` | homodyne logs 'Proceeding with optimization, but this may fail'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.stratified_ls [warning]` | homodyne logs 'RECOMMENDED: Use phi_filtering for angles near 0 and 90 deg'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.stratified_ls [warning]` | homodyne logs 'SHEAR COLLAPSE WARNING'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.stratified_ls [warning]` | homodyne logs 'Singular Jacobian, using pseudo-inverse for covariance'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.stratified_ls [warning]` | homodyne logs 'The model has effectively collapsed to static_isotropic mode.'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.strategies.stratified_ls [warning]` | homodyne logs 'This may indicate:'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.validation.fit_quality [info]` | homodyne logs '[FitQuality] All quality checks passed'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.validation.fit_quality [info]` | homodyne logs '[FitQuality] Chi-squared quality: acceptable (%.4g <= %.4g)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.validation.fit_quality [info]` | homodyne logs '[FitQuality] Chi-squared quality: good (%.4g <= %.4g)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.validation.fit_quality [warning]` | homodyne logs '[FitQuality] Chi-squared quality: poor (%.4g > %.4g)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.validation.input_validator [warning]` | homodyne logs 'xdata is empty'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.validation.input_validator [warning]` | homodyne logs 'ydata is empty'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.wrapper [debug]` | homodyne logs 'Empty 2D t1 array converted to empty 1D array'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.wrapper [debug]` | homodyne logs 'Empty 2D t2 array converted to empty 1D array'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.wrapper [debug]` | homodyne logs 'Iteration count not available from NLSQ (curve_fit_large does not return this info)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.wrapper [debug]` | homodyne logs 'Sequential bounds dtype: lower=%s upper=%s'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.wrapper [debug]` | homodyne logs 'Sequential bounds values: lower=%s upper=%s'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.wrapper [debug]` | homodyne logs 'Sequential residual call: params_shape=%s, phi_unique=%d'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.wrapper [error]` | homodyne logs 'Legacy scalar contrast/offset mode (per_angle_scaling=False) is no longer supported. Single contrast/offset parameters are not physically meaningful as each scattering angle has different optical properties and detector responses. Per-angle scaling is required for physically correct NLSQ optimization.'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.wrapper [error]` | homodyne logs 'Streaming mode requested but AdaptiveHybridStreamingOptimizer not available. Upgrade NLSQ to >= 0.3.2. Falling back to stratified least-squares.'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.wrapper [info]` | homodyne logs 'ADAPTIVE HYBRID STREAMING MODE (Preferred)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.wrapper [info]` | homodyne logs 'Computing consistent per-angle initialization for laminar_flow mode...'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.wrapper [info]` | homodyne logs 'Diagnostics enabled: loss=%s, x_scale=%s, sample_size=%d'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.wrapper [info]` | homodyne logs 'Diagnostics: nfev reported=%s actual=%s'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.wrapper [info]` | homodyne logs 'Expanded scalar contrast/offset to per-angle layout for sequential solver (%d angles)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.wrapper [info]` | homodyne logs 'Expanding scaling parameters for per-angle scaling:'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.wrapper [info]` | homodyne logs 'Final Jacobian column norms: %s'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.wrapper [info]` | homodyne logs 'HYBRID STREAMING MODE (Strategy Re-check)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.wrapper [info]` | homodyne logs 'HYBRID STREAMING OPTIMIZATION COMPLETE'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.wrapper [info]` | homodyne logs 'Initial Jacobian column norms: %s'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.wrapper [info]` | homodyne logs 'Loaded parameter bounds from config'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.wrapper [info]` | homodyne logs 'OUT-OF-CORE ACCUMULATION MODE (Re-check)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.wrapper [info]` | homodyne logs 'SEQUENTIAL OPTIMIZATION COMPLETE'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.wrapper [info]` | homodyne logs 'SEQUENTIAL PER-ANGLE OPTIMIZATION'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.wrapper [info]` | homodyne logs 'STRATIFIED LEAST-SQUARES COMPLETE'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.wrapper [info]` | homodyne logs 'STRATIFIED LEAST-SQUARES PATH ACTIVATED (v2.2.1)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.wrapper [info]` | homodyne logs 'Sequential per-angle fallback forced via configuration'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.wrapper [info]` | homodyne logs "Solving double-chunking problem with NLSQ's least_squares()"; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.wrapper [info]` | homodyne logs 'Stratification disabled via configuration'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.wrapper [info]` | homodyne logs 'Using NLSQ AdaptiveHybridStreamingOptimizer for better convergence and parameter estimation'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.wrapper [info]` | homodyne logs 'Using chunk-wise J^T J accumulation for memory efficiency'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.wrapper [info]` | homodyne logs 'Using full-copy stratification'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.wrapper [info]` | homodyne logs 'Using index-based stratification (zero-copy)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.wrapper [warning]` | homodyne logs 'Diagnostics: parameters at bounds -> %s'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.wrapper [warning]` | homodyne logs 'Invalid per-angle contrast override; ignoring'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.wrapper [warning]` | homodyne logs 'Invalid per-angle offset override; ignoring'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.wrapper [warning]` | homodyne logs 'Invalid sequential per-angle contrast override; ignoring'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.wrapper [warning]` | homodyne logs 'Invalid sequential per-angle offset override; ignoring'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.wrapper [warning]` | homodyne logs 'Per-angle scaling requested but parameter vector has %d entries (expected %d); sequential solver will operate with scalar scaling'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.wrapper [warning]` | homodyne logs 'Sequential per-angle contrast override has %d entries (expected %d); ignoring override'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.wrapper [warning]` | homodyne logs 'Sequential per-angle offset override has %d entries (expected %d); ignoring override'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.wrapper [warning]` | homodyne logs 'per_angle_scaling contrast override has %d entries (expected %d); ignoring override'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `optimization.nlsq.wrapper [warning]` | homodyne logs 'per_angle_scaling offset override has %d entries (expected %d); ignoring override'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `runtime.utils.system_validator [debug]` | homodyne logs 'Shell alias validation failed; continuing without alias checks'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `runtime.utils.system_validator [debug]` | homodyne logs 'Version parsing for hybrid streaming optimizer failed: %s'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `utils.async_io [debug]` | homodyne logs 'Background write traceback:'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `utils.async_io [error]` | homodyne logs 'Failed to write JSON: %s'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `utils.async_io [error]` | homodyne logs 'Failed to write NPZ: %s'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `utils.async_io [info]` | homodyne logs 'Background write still in progress after %.0fs (will complete during shutdown)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `utils.async_io [warning]` | homodyne logs 'Background write failed (%s): %s'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `viz.mcmc_arviz [warning]` | homodyne logs 'ArviZ not available. Cannot create pair plots.'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `viz.mcmc_arviz [warning]` | homodyne logs 'ArviZ not available. Cannot create posterior plots.'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `viz.mcmc_arviz [warning]` | homodyne logs 'ArviZ not available. Falling back to custom trace plots.'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `viz.mcmc_arviz [warning]` | homodyne logs 'No parameter samples available for pair plots'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `viz.mcmc_arviz [warning]` | homodyne logs 'No parameter samples available for posterior plots'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `viz.mcmc_arviz [warning]` | homodyne logs 'No parameter samples available for trace plots'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `viz.mcmc_diagnostics [warning]` | homodyne logs 'No parameter samples available for trace plots'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `viz.mcmc_report [debug]` | homodyne logs 'BFMI: potential_energy not available in result'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `viz.mcmc_report [info]` | homodyne logs '    %20s: %.1f %s'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `viz.mcmc_report [info]` | homodyne logs '    %20s: %.4f %s'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `viz.mcmc_report [info]` | homodyne logs '  %20s: %12.4f +/- %8.4f'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `viz.mcmc_report [info]` | homodyne logs '  %20s: %12.4f +/- %8.4f  [%.4f, %.4f]'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `viz.mcmc_report [info]` | homodyne logs '  - Combination Method: %s'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `viz.mcmc_report [info]` | homodyne logs '  - Number of Shards: %s'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `viz.mcmc_report [info]` | homodyne logs '  BFMI (target >= 0.3):'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `viz.mcmc_report [info]` | homodyne logs '  ESS (target > 400):'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `viz.mcmc_report [info]` | homodyne logs '  R-hat (target < 1.1):'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `viz.mcmc_report [info]` | homodyne logs 'Analysis Mode: %s'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `viz.mcmc_report [info]` | homodyne logs 'CMC Information:'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `viz.mcmc_report [info]` | homodyne logs 'Computation Time: %.2fs'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `viz.mcmc_report [info]` | homodyne logs 'Converged: %s'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `viz.mcmc_report [info]` | homodyne logs 'Convergence Diagnostics'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `viz.mcmc_report [info]` | homodyne logs 'MCMC Results Summary'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `viz.mcmc_report [info]` | homodyne logs 'Parameter Estimates (95%% CI)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `viz.mcmc_report [info]` | homodyne logs 'Sampler: %s'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `viz.nlsq_plots [info]` | homodyne logs 'Falling back to matplotlib backend (publication quality)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `viz.nlsq_plots [info]` | homodyne logs 'Falling back to sequential plotting...'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `viz.nlsq_plots [info]` | homodyne logs 'Generating fitted C2 simulations...'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `viz.nlsq_plots [info]` | homodyne logs 'Simulated data plots generated successfully'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `viz.nlsq_plots [info]` | homodyne logs 'Using Datashader backend (preview mode, fast rendering)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `viz.nlsq_plots [info]` | homodyne logs 'Using matplotlib backend (publication quality)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `viz.nlsq_plots [warning]` | homodyne logs 'Cannot extract fitted parameters from result'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `viz.nlsq_plots [warning]` | homodyne logs 'Missing experimental data structure (phi_angles_list, t1, t2)'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `viz.nlsq_plots [warning]` | homodyne logs 'Preview mode (Datashader) requested but Datashader not available. Install with: pip install datashader xarray colorcet'; heterodyne does not (at this level) |
| `KEEP` | log_format_drift | `viz.nlsq_plots [warning]` | homodyne logs 'Theoretical plots using potentially filtered phi angles from experimental data. To use all angles, disable phi_filtering in config or provide --phi-angles explicitly.'; heterodyne does not (at this level) |

## P3 — Cosmetic — docstring/comment differences (0 gaps)

_(none)_

