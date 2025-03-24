#pragma once
#include <vector>
#include "network.h"
#include "AnimationWindow.h"

void update(std::vector <double>& epochs_n, std::vector <double>& loss_n, std::vector <double>& accuracy_n, double current_accuracy, double current_loss, int width, int height, int epochs, TDT4102::AnimationWindow& window);
void visualize_feed_forward_2(std::vector<Matrix> activated_layers, Matrix x_labels);
void visualize_feed_forward_multiple(std::vector<std::vector<Matrix>> multiple_activated_layers, std::vector<Matrix> x_labels);

class feed_forward_visualise : public TDT4102::AnimationWindow {
    public:
    feed_forward_visualise(int x, int y, int width, int height, const std::string& title) : TDT4102::AnimationWindow(x, y, width, height, title) {
        std::cout << "Feed forward visualisation" << std::endl;
    }
    void visualize_feed_forward(std::vector<Matrix> activated_layers, Matrix x_labels);
};
