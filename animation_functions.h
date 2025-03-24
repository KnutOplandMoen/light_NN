#pragma once
#include <vector>
#include "network.h"
#include "AnimationWindow.h"

class training_visualise : public TDT4102::AnimationWindow {
    public:
    training_visualise(int x, int y, int width, int height, const std::string& title) : TDT4102::AnimationWindow(x, y, width, height, title) {}
    void update(std::vector <double>& epochs_n, std::vector <double>& loss_n, std::vector <double>& accuracy_n, double current_accuracy, double current_loss, int epochs);
    void initialise();
};

class feed_forward_visualise : public TDT4102::AnimationWindow {
    public:
    feed_forward_visualise(int x, int y, int width, int height, const std::string& title) : TDT4102::AnimationWindow(x, y, width, height, title) {}
    void visualize_feed_forward(std::vector<Matrix>& activated_layers, Matrix& x_labels);
};
