import sys
import scipy
try:
    import cupy as cp
    import cupyx.scipy.special as css
    HAS_GPU = True
except ImportError:
    import numpy as cp  #Use numpy as the 'cp' alias
    import scipy.special as css
    HAS_GPU = False
import cupyx
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                              QHBoxLayout, QLabel, QLineEdit, QPushButton,
                              QStackedWidget, QSizePolicy)
from PyQt6.QtCore import (Qt, QPropertyAnimation, QEasingCurve, QPoint,
                           QParallelAnimationGroup, QThread, pyqtSignal)
from PyQt6.QtGui import QFont, QColor, QIcon
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtCore import QUrl
import plotly.graph_objects as go
import tempfile, os, sys, ctypes

# This tells Windows to treat this as a unique application
myappid = 'OrbitalViewer'
try:
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
except Exception as e:
    print(f"AppUserModelID Error: {e}")

# ── math (unchanged) ──────────────────────────────────────────────────────────
def sphHarm(l, m, theta, aziConst):
    return aziConst * cupyx.scipy.special.lpmv(abs(m), l, cp.cos(theta))

def normSphHarm(l, m):
    a = (2*l+1)*scipy.special.factorial(l-abs(m)) / (4*cp.pi*scipy.special.factorial(l-abs(m)))
    return cp.sqrt(cp.asarray(a))

def normRadial(n, l):
    a = ((2 / n) ** 3) * scipy.special.factorial(n - l + 1) / (2 * n * scipy.special.factorial(n + l))
    return cp.sqrt(cp.asarray(a))

def azi(m, phi_array):
    if m > 0:   return cp.sqrt(2) * cp.cos(m * phi_array)
    elif m < 0: return cp.sqrt(2) * cp.sin(abs(m) * phi_array)
    else:       return cp.ones_like(phi_array)

def probCalc(r, theta, phi, n, l, m, check_flag=0):
    phi = cp.asarray(phi)
    normConst = normRadial(n, l)
    normConstY = normSphHarm(l, m)
    aziConst = cp.sqrt(2) if (check_flag == 1 and m != 0) else azi(m, phi)
    rho = 2 * r / n
    poly_coeffs = scipy.special.genlaguerre(n - l - 1, 2 * l + 1).coeffs
    gpu_coeffs = cp.asarray(poly_coeffs)
    L_poly = cp.polyval(gpu_coeffs, rho)
    R = (normConst * cp.exp(-r / n) * (rho) ** l * L_poly)  # laguerre poly
    Y = normConstY * sphHarm(l, m, theta, aziConst)
    return (R * Y) ** 2

def maxProb(n, l, m, delta=1.05):
    rlist, thetalist = cp.linspace(0, n*n+10, 200), cp.linspace(0, cp.pi, 50)
    rGrid, tGrid = cp.meshgrid(rlist, thetalist)
    return cp.max(probCalc(rGrid, tGrid, cp.zeros_like(rGrid), n, l, m, 1)) * delta

def generateCloud(n, l, m, sampleSize=100_000, delta=1.05):
    L = n*n + n*l +10
    x, y, z = (cp.random.uniform(-L, L, sampleSize) for _ in range(3))
    testMaxProbs = cp.random.uniform(0, maxProb(n, l, m, delta), sampleSize)
    r     = cp.sqrt(x*x + y*y + z*z)
    theta = cp.arccos(z / r)
    phi   = cp.arctan2(y, x)
    probs = probCalc(r, theta, phi, n, l, m, 0)
    mask  = probs >= testMaxProbs
    return (x[mask].get() if HAS_GPU else x[mask],
            y[mask].get() if HAS_GPU else y[mask],
            z[mask].get() if HAS_GPU else z[mask],
            probs[mask].get() if HAS_GPU else probs[mask])

#NEW TEST FOR REPRESENTATION
def buildFigureHTML(n, l, m, sampleSize=100_000, delta=1.05):
    x, y, z, density = generateCloud(n, l, m, sampleSize=sampleSize, delta=delta)

    fig = go.Figure(go.Scatter3d(
        x=x, y=y, z=z, mode='markers',
        marker=dict(
            size=3.5,
            color=density,
            # 'Inferno', 'Plasma', 'Turbo'
            colorscale='Turbo',
            opacity=0.1,  # Bumped slightly so the glowing edges are more visible
            colorbar=dict(
                title='Probability Density',
                tickfont=dict(color='white'),
            ),
            line=dict(width=0),
        ),
    ))

    # A reusable style dictionary so we don't repeat ourselves for x, y, z
    axis_style = dict(
        showbackground=False,  # Keeps the 3D pane background transparent
        showgrid=True,  # Turns the grid lines BACK ON
        gridcolor='rgba(255, 255, 255, 0.15)',  # Faint white grid
        zeroline=True,  # Turns the main origin lines ON
        zerolinecolor='rgba(255, 255, 255, 0.4)',  # Brighter zero lines
        tickfont=dict(color='white')
    )

    fig.update_layout(
        title=dict(text=f"Hydrogen Orbital Density (n={n}, l={l}, m={m})<br>On all axes: 1 unit = 1 Bohr radius (a<sub>0</sub>) = 5.29×10<sup>−11</sup> m",
                   font=dict(color='white')),
        scene=dict(
            xaxis=axis_style,
            yaxis=axis_style,
            zaxis=axis_style,
            bgcolor='black',
            aspectmode='cube'  # Forces the 3D box to stay a perfect cube
        ),
        paper_bgcolor='black',
        plot_bgcolor='black',
        font_color='white',
        margin=dict(l=0, r=0, t=40, b=0),
    )

    # Keeping the WebEngine fix from earlier!
    fd, file_path = tempfile.mkstemp(suffix='.html')
    os.close(fd)
    fig.write_html(file_path, include_plotlyjs=True)
    return file_path


# ── worker thread ─────────────────────────────────────────────────────────────
class RenderWorker(QThread):
    finished = pyqtSignal(str)
    error    = pyqtSignal(str)

    def __init__(self, n, l, m, sampleSize, delta):
        super().__init__()
        self.n, self.l, self.m = n, l, m
        self.sampleSize = sampleSize
        self.delta = delta

    def run(self):
        try:
            path = buildFigureHTML(self.n, self.l, self.m, self.sampleSize, self.delta)
            self.finished.emit(path)
        except Exception as e:
            self.error.emit(str(e))

# ── stylesheet ────────────────────────────────────────────────────────────────
DARK = """
QWidget {
    background-color: #000000;
    color: #ffffff;
    font-family: -apple-system, 'SF Pro Display', 'Segoe UI', sans-serif;
}
QLabel { background: transparent; color: #ffffff; }

QLineEdit {
    background-color: rgba(255,255,255,0.07);
    border: 0.5px solid rgba(255,255,255,0.18);
    border-radius: 10px;
    padding: 8px 14px;
    color: #ffffff;
    font-size: 15px;
}
QLineEdit:focus {
    border: 0.5px solid rgba(255,255,255,0.55);
    background-color: rgba(255,255,255,0.11);
}

QPushButton {
    background-color: rgba(255,255,255,0.10);
    border: 0.5px solid rgba(255,255,255,0.22);
    border-radius: 12px;
    padding: 10px 28px;
    color: #ffffff;
    font-size: 14px;
    letter-spacing: 0.3px;
}
QPushButton:hover {
    background-color: rgba(255,255,255,0.18);
    border: 0.5px solid rgba(255,255,255,0.40);
}
QPushButton:pressed { background-color: rgba(255,255,255,0.06); }
QPushButton:disabled { color: rgba(255,255,255,0.25); }
QStackedWidget { background-color: #000000; }
"""

# ── slide transition ──────────────────────────────────────────────────────────
def slide_transition(stack, new_widget):
    current = stack.currentWidget()
    w = stack.width()
    new_widget.move(w, 0)
    stack.setCurrentWidget(new_widget)
    new_widget.show()
    new_widget.raise_()

    anim_out = QPropertyAnimation(current, b"pos")
    anim_out.setDuration(500)
    anim_out.setStartValue(QPoint(0, 0))
    anim_out.setEndValue(QPoint(-w, 0))
    anim_out.setEasingCurve(QEasingCurve.Type.InOutCubic)

    anim_in = QPropertyAnimation(new_widget, b"pos")
    anim_in.setDuration(500)
    anim_in.setStartValue(QPoint(w, 0))
    anim_in.setEndValue(QPoint(0, 0))
    anim_in.setEasingCurve(QEasingCurve.Type.InOutCubic)

    group = QParallelAnimationGroup()
    group.addAnimation(anim_out)
    group.addAnimation(anim_in)
    group.start()
    stack._anim_group = group

# ── title screen ──────────────────────────────────────────────────────────────
class TitleScreen(QWidget):
    def __init__(self, on_go):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(0)
        layout.setContentsMargins(60, 0, 60, 0)

        # 1. Main Title
        title = QLabel("Stochastic Electron Density Mapping")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setFont(QFont("Segoe UI", 36, QFont.Weight.Bold))
        title.setStyleSheet("color: #ffffff; letter-spacing: 1px;")

        # 2. Subtitle (Italicized)
        subtitle = QLabel("<i>(via Monte Carlo Rejection Sampling)</i>")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: rgba(255,255,255,0.7); font-size: 20px; margin-bottom: 20px;")

        # 3. Members Table - Centered around the colons
        # We split the names into two rows for better readability
        members = QLabel(
            "<div style='color: rgba(255,255,255,0.45); font-size: 16px;'>"
            "<p style='text-align: center; font-weight: bold; margin-bottom: 15px; color: rgba(255,255,255,0.6);'>EP Project Members</p>"
            "<table align='center' style='border-spacing: 10px 5px;'>"
            "<tr><td style='text-align: right;'>By</td><td>:</td><td style='text-align: left;'>Neel Tendulkar</td></tr>"
            "<tr><td style='text-align: right;'>Group member</td><td>:</td><td style='text-align: left;'>2</td></tr>"
            "<tr><td style='text-align: right;'>Group member</td><td>:</td><td style='text-align: left;'>3</td></tr>"
            "<tr><td style='text-align: right;'>Group member</td><td>:</td><td style='text-align: left;'>4</td></tr>"
            "</table>"
            "</div>"
        )
        members.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # 4. Description line
        desc = QLabel("\nInteractive 3D electron density clouds")
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc.setStyleSheet("color: rgba(255,255,255,0.25); font-size: 14px;")

        go_btn = QPushButton("Get started")
        go_btn.setFixedWidth(180)
        go_btn.setFixedHeight(46)
        go_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        go_btn.clicked.connect(on_go)

        # Add to layout
        layout.addStretch(2)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addSpacing(10)
        layout.addWidget(members)
        layout.addWidget(desc)
        layout.addSpacing(40)
        layout.addWidget(go_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addStretch(3)

# ── plot screen ───────────────────────────────────────────────────────────────
class PlotScreen(QWidget):
    def __init__(self):
        super().__init__()
        self._worker = None
        self._tmp    = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)

        row = QHBoxLayout()
        row.setSpacing(16)

        def labeled_input(label_text, width=72):
            col = QVBoxLayout()
            col.setSpacing(6)
            lbl = QLabel(label_text)
            lbl.setStyleSheet(
                "color: rgba(255,255,255,0.38); font-size: 11px; letter-spacing: 1.2px;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            inp = QLineEdit()
            inp.setPlaceholderText("—")
            inp.setAlignment(Qt.AlignmentFlag.AlignCenter)
            inp.setFixedWidth(width)
            inp.setFixedHeight(40)
            col.addWidget(lbl, alignment=Qt.AlignmentFlag.AlignCenter)
            col.addWidget(inp)
            return col, inp

        n_col, self.n_in   = labeled_input("N")
        l_col, self.l_in   = labeled_input("L")
        m_col, self.m_in   = labeled_input("M")
        # sample size input
        s_col, self.s_in   = labeled_input("POINTS  (10^n)", width=110)
        self.s_in.setPlaceholderText("5")
        del_col, self.del_in = labeled_input("Delta", width=110)
        self.del_in.setPlaceholderText("1.05")

        self.plot_btn = QPushButton("Render")
        self.plot_btn.setFixedWidth(130)
        self.plot_btn.setFixedHeight(40)
        self.plot_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.plot_btn.clicked.connect(self.render)

        self.status = QLabel("")
        self.status.setStyleSheet("color: rgba(255,255,255,0.32); font-size: 12px;")

        row.addStretch()
        row.addLayout(n_col)
        row.addLayout(l_col)
        row.addLayout(m_col)
        row.addLayout(s_col)
        row.addLayout(del_col)
        row.addSpacing(8)
        row.addWidget(self.plot_btn, alignment=Qt.AlignmentFlag.AlignBottom)
        row.addWidget(self.status,   alignment=Qt.AlignmentFlag.AlignBottom)
        row.addStretch()

        self.web = QWebEngineView()
        self.web.page().setBackgroundColor(QColor(0, 0, 0))
        self.web.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.web.setStyleSheet("border-radius: 10px;")
        self.web.page().profile().downloadRequested.connect(self._handle_download, type=Qt.ConnectionType.QueuedConnection)

        layout.addLayout(row)
        layout.addWidget(self.web)

    def _handle_download(self, item):
        from PyQt6.QtWidgets import QFileDialog
        import os

        # Force the app to focus on the window so the dialog doesn't hang
        self.setFocus()

        suggested_name = item.suggestedFileName()

        # Open the dialog
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Orbital Image",
            suggested_name,
            "PNG Image (*.png)"
        )

        if path:
            # Crucial: set these before accepting
            item.setDownloadDirectory(os.path.dirname(path))
            item.setDownloadFileName(os.path.basename(path))
            item.accept()
        else:
            item.cancel()

    def render(self):
        try:
            n = int(self.n_in.text())
            l = int(self.l_in.text())
            m = int(self.m_in.text())
            assert 0 < n and 0 <= l < n and abs(m) <= l

            s_text = self.s_in.text().strip()
            exp = float(s_text) if s_text else 5
            assert 0 <= exp, "n must be positive" #floats work, they are just truncated
            sampleSize = int(10 ** exp) #said truncating logic

            del_text = self.del_in.text().strip()
            delta = float(del_text) if del_text else 1.05
            assert 1 <= delta <=2, "delta must be between 1 and 2" #less than one and it will accept too many points in the orbital. More than 2 and it will reject way too many

        except AssertionError as e:
            self.status.setText(str(e) if str(e) else "invalid n, l, m values")
            return
        except Exception:
            self.status.setText("invalid input — check values")
            return

        self.status.setText(f"computing  {sampleSize:,} points…")
        self.plot_btn.setEnabled(False)

        self._worker = RenderWorker(n, l, m, sampleSize, delta)
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_done(self, path):
        if self._tmp and os.path.exists(self._tmp):
            os.unlink(self._tmp)
        self._tmp = path

        def on_load(ok):
            self.status.setText("" if ok else "failed to render")
            self.plot_btn.setEnabled(True)
            try: self.web.loadFinished.disconnect(on_load)
            except: pass

        self.web.loadFinished.connect(on_load)
        self.web.load(QUrl.fromLocalFile(path))
        self.status.setText("loading…")

    def _on_error(self, msg):
        self.status.setText(f"error: {msg}")
        self.plot_btn.setEnabled(True)

# ── main window ───────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hydrogen Atom Visualizer")
        self.resize(1060, 780)
        self.setStyleSheet("QMainWindow { background: #000000; }")

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        if hasattr(sys, '_MEIPASS'): icon_path = os.path.join(sys._MEIPASS, "3d0.ico")
        else: icon_path = "3d0.ico"
        self.setWindowIcon(QIcon(icon_path))

        self.title_screen = TitleScreen(on_go=self.go_to_plot)
        self.plot_screen  = PlotScreen()

        self.stack.addWidget(self.title_screen)
        self.stack.addWidget(self.plot_screen)

    def go_to_plot(self):
        slide_transition(self.stack, self.plot_screen)
    try: generateCloud(1,0,0,sampleSize=1,delta=1) #so that it "warms up" and doesnt freeze on first run
    except: pass

if __name__ == "__main__":
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except Exception as e:
        print(f"AppUserModelID Error: {e}")
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
