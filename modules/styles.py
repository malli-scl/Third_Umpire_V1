# modules/styles.py
# -----------------
# Centralized style definitions for Third Umpire UI

BIG_BUTTON_STYLE = """
QPushButton {
    background-color: #1565C0;
    color: white;
    font-size: 14px;
    font-weight: bold;
    padding: 8px 16px;
    border-radius: 10px;
}
QPushButton:hover { background-color: #1E88E5; }
QPushButton:pressed { background-color: #0D47A1; }
"""

TITLE_LABEL_STYLE = """
QLabel {
    color: #64B5F6;
    font-size: 24px;
    font-weight: bold;
}
"""

SUBTITLE_LABEL_STYLE = """
QLabel {
    color: white;
    font-size: 16px;
}
"""

CARD_STYLE = """
QFrame {
    background-color: #203a43;
    border-radius: 12px;
    padding: 10px;
}
"""

BACKGROUND_GRADIENT = """
QWidget {
    background-color: qlineargradient(
        x1:0, y1:0, x2:1, y2:1,
        stop:0 #0f2027, stop:0.5 #203a43, stop:1 #2c5364
    );
}
"""
