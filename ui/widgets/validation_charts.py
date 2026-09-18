"""
Validation Charts - interactive confusion matrix and P/R/F1/PR curves.

Drawn from the raw arrays returned by Ultralytics `model.val()`
(`metrics.confusion_matrix.matrix` and `metrics.curves_results`) instead of
the saved .png files, so the plots are dynamic and do not depend on
Ultralytics having written the image files.
"""
import matplotlib
matplotlib.use("QtAgg")  # Qt-compatible Agg backend

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np

from PyQt6.QtWidgets import QWidget, QVBoxLayout


class ValidationCharts(QWidget):
    """One figure holding the confusion matrix (top) and the four curves."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(8, 11))
        self.canvas = FigureCanvasQTAgg(self.figure)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self._show_message("Click 'Generate Interactive Charts' to run validation")

    # ------------------------------------------------------------------ helpers
    def _show_message(self, text):
        self.figure.clear()
        self.figure.set_facecolor("#ffffff")
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, text, ha='center', va='center',
                fontsize=11, color='#495057', wrap=True)
        ax.axis('off')
        self.canvas.draw_idle()

    def show_message(self, text):
        """Public: show a status/placeholder message."""
        self._show_message(text)

    def _plot_confusion(self, ax, matrix, names):
        matrix = np.asarray(matrix, dtype=float)
        n = matrix.shape[0]

        # Labels: class names plus a trailing 'background' bucket if present
        labels = list(names)
        if n == len(labels) + 1:
            labels = labels + ['background']
        elif n != len(labels):
            labels = [str(i) for i in range(n)]

        im = ax.imshow(matrix, cmap='Blues', aspect='auto')
        ax.set_title('Confusion Matrix', fontsize=11, color='#212529')
        ax.set_xlabel('Predicted', fontsize=9, color='#212529')
        ax.set_ylabel('True', fontsize=9, color='#212529')
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
        self.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Annotate cell counts only when the grid is small enough to be legible
        if n <= 15:
            thresh = matrix.max() / 2.0 if matrix.max() > 0 else 0.5
            for i in range(n):
                for j in range(n):
                    val = matrix[i, j]
                    ax.text(j, i, f"{int(val)}", ha='center', va='center',
                            fontsize=7,
                            color='white' if val > thresh else '#212529')

    def _plot_curve(self, ax, curve_result, title):
        x = np.asarray(curve_result[0], dtype=float)
        y = np.asarray(curve_result[1], dtype=float)
        xlabel = curve_result[2] if len(curve_result) > 2 else 'x'
        ylabel = curve_result[3] if len(curve_result) > 3 else 'y'

        if y.ndim == 2:
            # One line per class (faint) plus the mean (bold)
            for row in y:
                if len(row) == len(x):
                    ax.plot(x, row, linewidth=0.5, alpha=0.25, color='#9aa4b2')
            mean_y = y.mean(axis=0)
        else:
            mean_y = y

        if len(x) != len(mean_y):
            x = np.arange(len(mean_y))

        ax.plot(x, mean_y, linewidth=2.0, color='#2196F3', label='mean')
        ax.set_title(title, fontsize=10, color='#212529')
        ax.set_xlabel(xlabel, fontsize=8, color='#212529')
        ax.set_ylabel(ylabel, fontsize=8, color='#212529')
        ax.tick_params(labelsize=7)
        ax.grid(True, color='#dee2e6', linewidth=0.5, alpha=0.6)
        ax.legend(fontsize=7, loc='best', framealpha=0.4)

    # ------------------------------------------------------------------ public
    def plot(self, matrix, names, curves, curves_results):
        """Render the confusion matrix and up to four curves.

        Args:
            matrix: confusion matrix 2D array (may be None).
            names: iterable of class names.
            curves: list of curve titles (parallel to curves_results).
            curves_results: list of [x, y, xlabel, ylabel] from Ultralytics.
        """
        self.figure.clear()
        self.figure.set_facecolor("#ffffff")

        names = list(names) if names is not None else []
        curves = list(curves) if curves else []
        curves_results = list(curves_results) if curves_results else []

        has_cm = matrix is not None and np.asarray(matrix).size > 0
        n_curves = min(4, len(curves_results))

        gs = self.figure.add_gridspec(3, 2, height_ratios=[1.5, 1, 1])

        if has_cm:
            ax_cm = self.figure.add_subplot(gs[0, :])
            try:
                self._plot_confusion(ax_cm, matrix, names)
            except Exception:
                ax_cm.axis('off')
                ax_cm.text(0.5, 0.5, 'Confusion matrix unavailable',
                           ha='center', va='center', color='#495057')

        curve_positions = [gs[1, 0], gs[1, 1], gs[2, 0], gs[2, 1]]
        for i in range(n_curves):
            ax = self.figure.add_subplot(curve_positions[i])
            title = curves[i] if i < len(curves) else f'Curve {i + 1}'
            try:
                self._plot_curve(ax, curves_results[i], title)
            except Exception:
                ax.axis('off')
                ax.text(0.5, 0.5, f'{title}\nunavailable',
                        ha='center', va='center', color='#495057', fontsize=8)

        if not has_cm and n_curves == 0:
            self._show_message("Validation returned no plottable data.")
            return

        self.figure.tight_layout()
        self.canvas.draw_idle()
