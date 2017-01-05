# -*- coding: utf-8 -*-
"""
Custom sliders, derived from QSliders.
Features:
 -- dynamic valid range,
 -- direct jumping to mouse position,
 -- integer and float version.

@author: Robin Roussel
"""
from PyQt5.QtCore import Qt, QRect, pyqtSignal
from PyQt5.QtWidgets import QSlider, QStyleOptionSlider, QStyle
from PyQt5.QtGui import QPainter, QBrush, QColor


class IntSlider(QSlider):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._minValid = self.minimum()
        self._maxValid = self.maximum()

        self.actionTriggered.connect(self.checkValidRange)

    def checkValidRange(self, action):
        if action in range(1,8):
            if self.sliderPosition() < self._minValid:
                self.setSliderPosition(self._minValid)
            elif self.sliderPosition() > self._maxValid:
                self.setSliderPosition(self._maxValid)

    def setValidRange(self, min_, max_):
        self._minValid = min_
        self._maxValid = max_

    def paintEvent(self, event):
        """Paint the valid range."""
        super().paintEvent(event)

        opt = QStyleOptionSlider()
        self.initStyleOption(opt) # initialize opt with the values from self
#        opt.subControls = QStyle.SC_SliderGroove | QStyle.SC_SliderHandle
#        if (self.tickPosition() != QSlider.NoTicks):
#            opt.subControls |= QStyle.SC_SliderTickmarks
        groove_rect = self.style().subControlRect(
            QStyle.CC_Slider, opt, QStyle.SC_SliderGroove, self)
        minval = ((self._minValid - self.minimum())
                  / (self.maximum() - self.minimum()))
        maxval = ((self._maxValid - self.minimum())
                  / (self.maximum() - self.minimum()))
        # Rectangle coords = left, top, width, height
        rec1 = QRect(groove_rect.left()+2,
                     groove_rect.top()+(2 if self.tickPosition() else 1),
                     minval*groove_rect.width()-2,
                     groove_rect.height()-2)
        rec2 = QRect(groove_rect.left()+maxval*groove_rect.width()+3,
                     groove_rect.top()+(2 if self.tickPosition() else 1),
                     (1.-maxval)*groove_rect.width()-3,
                     groove_rect.height()-2)
        painter = QPainter(self)
        brush = QBrush(QColor(51, 51, 51, 255), Qt.Dense4Pattern)
        painter.fillRect(rec1, brush)
        painter.fillRect(rec2, brush)

    def mousePressEvent(self, ev):
        """Jump to click position."""
        self.setSliderPosition(QStyle.sliderValueFromPosition(
            self.minimum(), self.maximum(), ev.x(), self.width()))
        super().mousePressEvent(ev)


class FloatSlider(IntSlider):

    valueChanged = pyqtSignal(float)

    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self._resol = 20
        self.setMinimum(self.minimum())
        self.setMaximum(self.maximum())
        super().valueChanged.connect(self._onValueChanged)

    def value(self):
        return super().value() / self._resol

    def setMinimum(self, value):
        return super().setMinimum(value * self._resol)

    def setMaximum(self, value):
        return super().setMaximum(value * self._resol)

    def setRange(self, min, max):
        super().setRange(min * self._resol, max * self._resol)

    def setValue(self, value):
        super().setValue(int(value * self._resol))

    def _onValueChanged(self, value):
        value /= self._resol
        self.valueChanged.emit(value)

    def setValidRange(self, min_, max_):
        self._minValid = min_ * self._resol
        self._maxValid = max_ * self._resol
