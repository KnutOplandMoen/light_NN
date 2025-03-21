#include "subprojects/animationwindow/include/AnimationWindow.h"

void plot_progress(std::vector<double> loss, int epochs) {
    TDT4102::AnimationWindow window(800, 600, 500, 500);
    window.wait_for_close();
}