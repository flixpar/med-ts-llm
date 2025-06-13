class TaskInterpretabilityMixin:
    """Mixin for tasks to log interpretability information."""

    def log_interpretability(self, inputs, prefix="val"):
        if not hasattr(self, "model"):
            return
        if not getattr(self.model, "interpretability_enabled", False):
            return
        report = self.model.generate_interpretability_report(inputs)
        fig = report.get("figure")
        if fig is not None and hasattr(self.logger, "log_figure"):
            self.logger.log_figure(fig, f"{prefix}/interpretability")
