#-------------------------------------------------
#
# Project created by QtCreator 2017-07-30T13:14:28
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = open_cv_examples
TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
#CONFIG -= qt

# OpenCV paths for headers and libs
INCLUDEPATH += /usr/local/include/opencv

LIBS += -L/usr/local/lib \
-lopencv_core \
-lopencv_imgcodecs \
-lopencv_imgproc \
-lopencv_highgui \
-lopencv_ml \
-lopencv_video \
-lopencv_videoio \
-lopencv_features2d \
-lopencv_calib3d \
-lopencv_objdetect \
-lopencv_flann
# --- ENDOF --- OpenCV paths for headers and libs

SOURCES += \
    main.cpp \
    examples.cpp \
    utils.cpp

HEADERS += \
    common.h \
    utils.h
