import scipy
import scipy.special
try:
    import cupy as cp
    import cupyx.scipy.special as css
    # Test if a CUDA device is actually responsive
    _ = cp.cuda.runtime.getDeviceCount()
    HAS_GPU = True
except:
    import numpy as cp
    import scipy.special as css
    HAS_GPU = False
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                              QHBoxLayout, QLabel, QLineEdit, QComboBox, QPushButton,
                              QStackedWidget, QSizePolicy)
from PyQt6.QtCore import (Qt, QPropertyAnimation, QEasingCurve, QPoint,
                           QParallelAnimationGroup, QThread, pyqtSignal)
from PyQt6.QtGui import QFont, QColor, QIcon
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtCore import QUrl
import plotly.graph_objects as go
import tempfile, os, sys, ctypes, gc

# This tells Windows to treat this as a unique application
myappid = 'OrbitalViewer'
try:
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
except Exception as e:
    print(f"AppUserModelID Error: {e}")

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    # Qt stylesheets prefer forward slashes / even on Windows
    return os.path.join(base_path, relative_path).replace("\\", "/")

icon_path = resource_path("3d0.ico")
arrow_path = resource_path("arrow.png")

# ── math ──────────────────────────────────────────────────────────
def sphHarm(l, m, theta, aziConst):
    return aziConst * css.lpmv(abs(m), l, cp.cos(theta))

# def normSphHarm(l, m):
#     a = (2*l+1)*scipy.special.factorial(l-abs(m)) / (4*cp.pi*scipy.special.factorial(l+abs(m)))
#     return cp.sqrt(cp.asarray(a))
#
# def normRadial(n, l):
#     a = ((2 / n) ** 3) * scipy.special.factorial(n - l - 1) / (2 * n * scipy.special.factorial(n + l))
#     return cp.sqrt(cp.asarray(a))

def normRadial(n, l):
    # n-l-1 factorial using log-gamma to prevent 'inf'
    log_num = css.gammaln(n - l)
    # 2n * (n+l)! in log space
    log_den = cp.log(2 * n) + css.gammaln(n + l + 1)
    # Full normalization constant in log space
    log_a = 3 * cp.log(2 / n) + log_num - log_den
    return cp.sqrt(cp.exp(log_a))

def normSphHarm(l, m):
    # sqrt((2l+1)/4pi * (l-m)!/(l+m)!)
    log_num = css.gammaln(l - abs(m) + 1)
    log_den = css.gammaln(l + abs(m) + 1)
    log_const = cp.log((2 * l + 1) / (4 * cp.pi)) + log_num - log_den
    return cp.sqrt(cp.exp(log_const))

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
    return (R * Y) ** 2, cp.sign(R*Y)

def maxProb(n, l, m, delta=1.05):
    rlist, thetalist = cp.linspace(0, n*n+15, 500), cp.linspace(0, cp.pi, 200)
    rGrid, tGrid = cp.meshgrid(rlist, thetalist)
    # return cp.max(probCalc(rGrid, tGrid, cp.zeros_like(rGrid), n, l, m, 1)) * delta
    p, _ = probCalc(rGrid, tGrid, cp.zeros_like(rGrid), n, l, m, 1)
    m_val = cp.max(p)

    # Safety: If maxProb is 0, the math is broken
    if m_val == 0:
        return 1.0
    return m_val * delta


def generateCloud(n, l, m, sampleSize=100_000, delta=1.05,option=0):
    L = n * n + n * l + 10
    p_max = maxProb(n, l, m, delta)

    # Pre-allocate CPU arrays using NumPy to avoid GPU memory pressure
    # We use sampleSize as the maximum possible buffer size
    final_x = np.empty(sampleSize, dtype=np.float32)
    final_y = np.empty(sampleSize, dtype=np.float32)
    final_z = np.empty(sampleSize, dtype=np.float32)
    final_probs = np.empty(sampleSize, dtype=np.float32)
    final_phases = np.empty(sampleSize, dtype=np.float32)

    total_accepted = 0
    chunk_size = 5_000_000  # Size of each chunk of VRAM

    for start in range(0, sampleSize, chunk_size):
        # Determine size for the current chunk
        current_chunk = min(chunk_size, sampleSize - start)

        # 1. Generate candidate points
        x = cp.random.uniform(-L, L, current_chunk)
        y = cp.random.uniform(-L, L, current_chunk)
        z = cp.random.uniform(-L, L, current_chunk)
        testMaxProbs = cp.random.uniform(0, p_max, current_chunk)

        # 2. Coordinate conversion
        r = cp.sqrt(x * x + y * y + z * z)
        r = cp.where(r == 0, 1e-10, r)  # Avoid division by zero
        theta = cp.arccos(cp.clip(z / r, -1, 1))
        phi = cp.arctan2(y, x)

        # 3. Probability check
        probs, phases = probCalc(r, theta, phi, n, l, m, 0)
        mask = probs >= testMaxProbs

        # 4. Extract accepted values directly to CPU
        accepted_count = int(cp.sum(mask))
        if accepted_count > 0:
            # must not exceed the pre-allocated buffer
            space_left = sampleSize - total_accepted
            to_copy = min(accepted_count, space_left)

            # Slice and get() only the accepted points
            final_x[total_accepted: total_accepted + to_copy] = x[mask][:to_copy].get() if HAS_GPU else x[mask][:to_copy]
            final_y[total_accepted: total_accepted + to_copy] = y[mask][:to_copy].get() if HAS_GPU else y[mask][:to_copy]
            final_z[total_accepted: total_accepted + to_copy] = z[mask][:to_copy].get() if HAS_GPU else z[mask][:to_copy]

            if option == 0:
                final_probs[total_accepted: total_accepted + to_copy] = probs[mask][:to_copy].get() if HAS_GPU else probs[mask][:to_copy]
            else:
                p_val = probs[mask][:to_copy]
                ph_val = phases[mask][:to_copy]
                if option == 1:
                    final_probs[total_accepted: total_accepted + to_copy] = (cp.sqrt(p_val) * ph_val).get() if HAS_GPU else (cp.sqrt(p_val) * ph_val) # this function is normalized to show the phases more distinctly for higher values of l, m, n. The actual function is p_val * ph_val
                    # final_probs[total_accepted: total_accepted + to_copy] = (p_val * ph_val).get() if HAS_GPU else (p_val * ph_val) # shows the density along with the sign (the actual wave function), but it just shows everything as white in higher orbital since the centre is extremely dense
                else: final_probs[total_accepted: total_accepted + to_copy] = ph_val.get() if HAS_GPU else ph_val  # this shows ONLY the sign


            total_accepted += to_copy

        # 5. Force clear GPU cache for this chunk
        if HAS_GPU:
            del x, y, z, testMaxProbs, r, theta, phi, probs, mask
            cp.get_default_memory_pool().free_all_blocks()

    # print(f"DEBUG: Accepted {total_accepted} points out of {sampleSize}")

    # Return only the portions of the arrays that were actually filled
    return (final_x[:total_accepted],
            final_y[:total_accepted],
            final_z[:total_accepted],
            final_probs[:total_accepted])

#NEW TEST FOR REPRESENTATION
def buildFigureHTML(n, l, m, sampleSize=100_000, delta=1.05,option = 0):
    x, y, z, density = generateCloud(n, l, m, sampleSize=sampleSize, delta=delta,option=option)

    marker_setting=dict(
        size=3.5,
        color=density,
        opacity=0.1,  # Bumped slightly so the glowing edges are more visible
        colorbar=dict(
            title='Probability Density (Ψ<sup>2</sup>)',
            tickfont=dict(color='white'),
        ),
        line=dict(width=0),)

    if option == 0:
        marker_setting['colorscale'] = 'Turbo' # Turbo and Inferno look good for this
        title = 'Probability Density'
    else:
        marker_setting['colorscale'] = 'RdBu' # only this really fits here sadly
        marker_setting['cmid'] = 0
        if option == 1:
            marker_setting['colorbar'] = dict(
                title='Wavefunction (Ψ)',)
            title = 'Wavefunction'
        else:
            marker_setting['colorbar'] = dict(
                title='Phase (±)',)
            title = 'Phase Topology'

    fig = go.Figure(go.Scatter3d(
        x=x, y=y, z=z, mode='markers',
        marker=marker_setting,
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
        title=dict(text=f"Hydrogen Orbital {title} (n={n}, l={l}, m={m})<br>On all axes: 1 unit = 1 Bohr radius (a<sub>0</sub>) = 5.29×10<sup>−11</sup> m",
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

    def __init__(self, n, l, m, sampleSize, delta, option):
        super().__init__()
        self.n, self.l, self.m = n, l, m
        self.sampleSize = sampleSize
        self.delta = delta
        self.option = option

    def run(self):
        try:
            path = buildFigureHTML(self.n, self.l, self.m, self.sampleSize, self.delta, self.option)

            #frees VRAM
            gc.collect()
            if HAS_GPU:
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()

            self.finished.emit(path)
        except Exception as e:
            self.error.emit(str(e))

# ── stylesheet ────────────────────────────────────────────────────────────────
def stylesheet(arrow = arrow_path):
    return f'''
QWidget {{
    background-color: #000000;
    color: #ffffff;
    font-family: -apple-system, 'SF Pro Display', 'Segoe UI', sans-serif;
}}
QLabel {{ background: transparent; color: #ffffff; }}

QLineEdit {{
    background-color: rgba(255,255,255,0.07);
    border: 0.5px solid rgba(255,255,255,0.18);
    border-radius: 10px;
    padding: 8px 14px;
    color: #ffffff;
    font-size: 15px;
}}
QLineEdit:focus {{
    border: 0.5px solid rgba(255,255,255,0.55);
    background-color: rgba(255,255,255,0.11);
}}

QPushButton {{
    background-color: rgba(255,255,255,0.10);
    border: 0.5px solid rgba(255,255,255,0.22);
    border-radius: 12px;
    padding: 10px 28px;
    color: #ffffff;
    font-size: 14px;
    letter-spacing: 0.3px;
}}
QPushButton:hover {{
    background-color: rgba(255,255,255,0.18);
    border: 0.5px solid rgba(255,255,255,0.40);
}}
QPushButton:pressed {{ background-color: rgba(255,255,255,0.06); }}
QPushButton:disabled {{ color: rgba(255,255,255,0.25); }}

QComboBox {{
    background-color: rgba(255,255,255,0.07);
    border: 0.5px solid rgba(255,255,255,0.18);
    border-radius: 10px;
    padding: 0px 14px;
    color: #ffffff;
}}
QComboBox::drop-down {{
    border: 0px;
}}
QComboBox::down-arrow {{
    /* Use the relative path to your file */
    image: url("{arrow_path}");
    width: 9px;
    height: 9px;
    margin-right: 15px;
}}
QComboBox:focus {{
    border: 0.5px solid rgba(255,255,255,0.55);
    background-color: rgba(255,255,255,0.11);
}}
/* Style for the list that pops out */
QComboBox QAbstractItemView {{
    background-color: #121212;
    color: white;
    selection-background-color: #333333;
    border: 0.5px solid rgba(255,255,255,0.2);
    outline: none;
}}

QStackedWidget {{ background-color: #000000; }}
'''

# DARK =

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
            "<tr><td style='text-align: right;'>Neel Tendulkar</td><td>:</td><td style='text-align: left;'>1262251509</td></tr>"
            "<tr><td style='text-align: right;'>Medhansh Singh</td><td>:</td><td style='text-align: left;'>1262250758</td></tr>"
            "<tr><td style='text-align: right;'>Hemal Trivedi</td><td>:</td><td style='text-align: left;'>1262251292</td></tr>"
            "<tr><td style='text-align: right;'>Mohak Chaurasia</td><td>:</td><td style='text-align: left;'>1262253680</td></tr>"
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

        def labeled_dropdown(label_text, items, width=150):
            col = QVBoxLayout()
            col.setSpacing(6)

            lbl = QLabel(label_text)
            lbl.setStyleSheet("color: rgba(255,255,255,0.38); font-size: 11px; letter-spacing: 1.2px;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

            combo = QComboBox()
            combo.addItems(items)
            combo.setFixedWidth(width)
            combo.setFixedHeight(40)
            # Apply your custom styling here or in the main DARK stylesheet
            col.addWidget(lbl, alignment=Qt.AlignmentFlag.AlignCenter)
            col.addWidget(combo)
            return col, combo

        n_col, self.n_in   = labeled_input("N")
        l_col, self.l_in   = labeled_input("L")
        m_col, self.m_in   = labeled_input("M")
        # sample size input
        s_col, self.s_in   = labeled_input("POINTS  (10^n)", width=110)
        self.s_in.setPlaceholderText("5")
        del_col, self.del_in = labeled_input("Delta", width=110)
        self.del_in.setPlaceholderText("1.05")
        option_col, self.option_in = labeled_dropdown("Render Mode", ["Probability Density", "Wavefunction", "Phase ONLY"])
        self.option_in.setPlaceholderText("0")

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
        row.addLayout(option_col)
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
            assert 0 <= exp <= 8, "n must be between 1 and 8" #floats work, they are just truncated
            sampleSize = int(10 ** exp) #said truncating logic

            del_text = self.del_in.text().strip()
            delta = float(del_text) if del_text else 1.05
            assert 1 <= delta <=2, "delta must be between 1 and 2" #less than one and it will accept too many points in the orbital. More than 2 and it will reject way too many

            option = self.option_in.currentIndex()

        except AssertionError as e:
            self.status.setText(str(e) if str(e) else "invalid n, l, m values")
            return
        except Exception:
            self.status.setText("invalid input — check values")
            return

        self.status.setText(f"computing  {sampleSize:,} points…")
        self.plot_btn.setEnabled(False)

        self._worker = RenderWorker(n, l, m, sampleSize, delta,option)
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
    app.setStyleSheet(stylesheet(arrow_path))
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
