#include "view/mlp_app.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);    
    Paint w;
    w.show();
    return a.exec();
}
