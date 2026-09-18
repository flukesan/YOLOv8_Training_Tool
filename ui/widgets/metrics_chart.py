"""
Metrics Chart - reusable matplotlib canvas for plotting training metrics.

Renders two line charts (detection metrics + losses) from a history dict of
metric_name -> list of per-epoch values. Used both for the live view during
training (dark theme) and the Training Results dialog (light theme, fed from
results.csv). Drawing from data rather than a saved .png means the graphs
always render even when Ultralytics did not produce the image files.
"""
import matplotlib
matplotlib.use("QtAgg")  # Qt-compatible Agg backend (picks PyQt6 automatically)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PyQt6.QtWidgets import QWidget, QVBoxLayout


# Consistent line colors across both themes
_COLORS = {
    'train_loss': '#f44336',
    'val_loss': '#FF9800',
    'precision': '#2196F3',
    'recall': '#9C27B0',
    'mAP50': '#4CAF50',
    'mAP50-95': '#00BCD4',
}

_METRIC_KEYS = ['precision', 'recall', 'mAP50', 'mAP50-95']
_LOSS_KEYS = ['train_loss', 'val_loss']


class MetricsChart(QWidget):
    """Embeds two matplotlib line charts: detection metrics and losses."""

    def __init__(self, parent=None, dark: bool = True):
        super().__init__(parent)
        self.dark = dark

        self.figure = Figure(figsize=(5, 4))
        self.canvas = FigureCanvasQTAgg(self.figure)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.ax_metrics = self.figure.add_subplot(2, 1, 1)
        self.ax_loss = self.figure.add_subplot(2, 1, 2)

        self.clear()

    # ------------------------------------------------------------------ theme
    def _theme(self):
        if self.dark:
            return {'fg': '#dddddd', 'bg': '#1e2128', 'grid': '#3d4250'}
        return {'fg': '#212529', 'bg': '#ffffff', 'grid': '#dee2e6'}

    def _style_axis(self, ax, title, ylabel):
        c = self._theme()
        ax.set_facecolor(c['bg'])
        ax.set_title(title, fontsize=10, color=c['fg'])
        ax.set_xlabel('epoch', fontsize=8, color=c['fg'])
        ax.set_ylabel(ylabel, fontsize=8, color=c['fg'])
        ax.tick_params(colors=c['fg'], labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(c['grid'])
        ax.grid(True, color=c['grid'], linewidth=0.5, alpha=0.5)

    # ------------------------------------------------------------------ public
    def clear(self):
        """Reset both charts to an empty placeholder state."""
        c = self._theme()
        self.figure.set_facecolor(c['bg'])

        self.ax_metrics.clear()
        self.ax_loss.clear()
        self._style_axis(self.ax_metrics, 'Detection Metrics', 'value')
        self._style_axis(self.ax_loss, 'Loss', 'loss')

        self.ax_metrics.text(
            0.5, 0.5, 'Waiting for data...',
            ha='center', va='center',
            transform=self.ax_metrics.transAxes,
            color=c['fg'], fontsize=9
        )
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def update_data(self, history: dict):
        """Redraw both charts from a history dict.

        Args:
            history: mapping of metric name -> list of per-epoch values.
                     Recognised keys: precision, recall, mAP50, mAP50-95,
                     train_loss, val_loss. Missing/empty series are skipped.
        """
        if not history:
            return

        has_data = any(history.get(k) for k in _METRIC_KEYS + _LOSS_KEYS)
        if not has_data:
            return

        c = self._theme()
        self.figure.set_facecolor(c['bg'])

        # --- detection metrics ---
        self.ax_metrics.clear()
        plotted_metric = False
        for key in _METRIC_KEYS:
            values = history.get(key) or []
            if values:
                self.ax_metrics.plot(
                    range(1, len(values) + 1), values,
                    label=key, color=_COLORS[key], linewidth=1.5
                )
                plotted_metric = True
        self._style_axis(self.ax_metrics, 'Detection Metrics', 'value')
        self.ax_metrics.set_ylim(0, 1.0)
        if plotted_metric:
            self.ax_metrics.legend(
                fontsize=7, loc='lower right',
                framealpha=0.3, labelcolor=c['fg']
            )

        # --- losses ---
        self.ax_loss.clear()
        plotted_loss = False
        for key in _LOSS_KEYS:
            values = history.get(key) or []
            if values:
                self.ax_loss.plot(
                    range(1, len(values) + 1), values,
                    label=key, color=_COLORS[key], linewidth=1.5
                )
                plotted_loss = True
        self._style_axis(self.ax_loss, 'Loss', 'loss')
        if plotted_loss:
            self.ax_loss.legend(
                fontsize=7, loc='upper right',
                framealpha=0.3, labelcolor=c['fg']
            )

        self.figure.tight_layout()
        self.canvas.draw_idle()
