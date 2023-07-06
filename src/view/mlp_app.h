#ifndef MLP_APP_H
#define MLP_APP_H

#include <QWidget>
#include <QTimer>
#include <QResizeEvent>

#include "paintscene.h"

namespace Ui {
class Paint;
}

class Paint : public QWidget
{
    Q_OBJECT

public:
    explicit Paint(QWidget *parent = 0);
    ~Paint();

private:
    Ui::Paint *ui;
    QTimer *timer;      /* Определяем таймер для подготовки актуальных размеров
                         * графической сцены
                         * */
    paintScene *scene;  // Объявляем кастомную графическую сцену

private:
    /* Переопределяем событие изменения размера окна
     * для пересчёта размеров графической сцены
     * */
    void resizeEvent(QResizeEvent * event);
    //void Clear();
   // bool par;

private slots:
    void slotTimer();
};

#endif // MLP_APP_H
