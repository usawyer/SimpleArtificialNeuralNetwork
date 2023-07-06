QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = paint
TEMPLATE = app


SOURCES += main.cpp\
    controller/controller.cc \
    model/network.cc \
    model/s21_matrix_oop.cpp \
    view/mlp_app.cc \
    view/paintscene.cc

HEADERS  += \
    controller/controller.h \
    model/network.h \
    model/s21_matrix_oop.h \
    view/mlp_app.h \
    view/paintscene.h

FORMS    += \
    view/ui/mlp_app.ui \
    view/ui/metrics_dialog.ui \
    view/ui/train_dialog.ui
