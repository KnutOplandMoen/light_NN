#pragma once
#include <vector>
#include "network.h"
#include "AnimationWindow.h"

class training_visualise : public TDT4102::AnimationWindow {
    public:
    training_visualise(int x, int y, int width, int height, const std::string& title) : TDT4102::AnimationWindow(x, y, width, height, title) {}
    void update(double* epochs_n, double* loss_n, double* accuracy_n, double current_accuracy, double current_loss, int epochs, int i);
    void initialise();
    void finish();
    void callbackFunction();
    void quit() {close();}

};

class feed_forward_visualise : public TDT4102::AnimationWindow {
    public:
    int x;
    int y;
    feed_forward_visualise(int x, int y, int width, int height, const std::string& title) : TDT4102::AnimationWindow(x, y, width, height, title) {
        this -> x = x;
        this -> y = y;
    }
    void visualize_feed_forward(std::vector<Matrix>& activated_layers, Matrix& x_labels, std::vector<std::string> x_labels_names = {}, std::vector<std::string> y_labels_names = {}, bool show_text = true);
};
