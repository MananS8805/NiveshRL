"""GLOS: the searchable glossary of every market, technical, model and chart term in the app."""
from __future__ import annotations

from PySide6.QtCore import Qt, QUrl
from PySide6.QtWidgets import QHBoxLayout, QLineEdit, QTextBrowser, QTreeWidget, QTreeWidgetItem, QVBoxLayout

from ... import glossary as gl
from ..explain import render_html
from ..widgets import muted
from . import Panel, vbox


class GlossaryPanel(Panel):
    title = "Glossary"
    code = "GLOS"

    def __init__(self, ctx, parent=None):
        super().__init__(ctx, parent)
        lay = vbox(self)
        row = QHBoxLayout()
        left = QVBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search terms: RSI, Sharpe, FinBERT, decile, VIX…")
        self.search.textChanged.connect(self._fill)
        left.addWidget(self.search)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setMinimumWidth(300)
        self.tree.setMaximumWidth(380)
        self.tree.currentItemChanged.connect(self._picked)
        left.addWidget(self.tree, 1)
        self.count = muted("")
        left.addWidget(self.count)
        row.addLayout(left)
        self.view = QTextBrowser()
        self.view.setOpenLinks(False)
        self.view.anchorClicked.connect(self._link)
        row.addWidget(self.view, 1)
        lay.addLayout(row, 1)
        lay.addWidget(muted("Everywhere in the app: click a label marked ⓘ (tiles, section titles, chart buttons), or "
                            "right-click a table header or a value, to see it explained with your value highlighted."))
        self._fill()

    def _fill(self) -> None:
        q = self.search.text().strip()
        hits = gl.search(q)
        self.tree.clear()
        groups, best = {}, None
        for c in gl.CATEGORIES:
            items = [e for e in hits if e.category == c]
            if not items:
                continue
            g = QTreeWidgetItem([f"{c}  ({len(items)})"])
            g.setFlags(g.flags() & ~Qt.ItemIsSelectable)
            self.tree.addTopLevelItem(g)
            for e in sorted(items, key=lambda x: x.title.lower()):
                it = QTreeWidgetItem([e.title])
                it.setData(0, Qt.UserRole, e.key)
                g.addChild(it)
                if hits and e is hits[0]:
                    best = it
            g.setExpanded(bool(q) or c in ("Models", "Model quality"))
            groups[c] = g
        self.count.setText(f"{len(hits)} of {len(gl.G)} terms")
        first = best or next((g.child(0) for g in groups.values() if g.childCount()), None)
        if first is not None and (q or self.view.toPlainText() == ""):
            self.tree.setCurrentItem(first)            # the best match (title hits rank first)

    def _picked(self, it, _prev=None) -> None:
        if it is not None and it.data(0, Qt.UserRole):
            self.show_key(it.data(0, Qt.UserRole))

    def show_key(self, key: str) -> None:
        if key in gl.G:
            self.view.setHtml(render_html(gl.G[key]))

    def _link(self, url: QUrl) -> None:
        s = url.toString()
        if s.startswith("term:"):
            self.show_key(s[5:])
