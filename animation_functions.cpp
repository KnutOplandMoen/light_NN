#include "animation_functions.h"
#include "network.h"

void update(std::vector <double>& epochs_n, std::vector <double>& loss_n, std::vector <double>& accuracy_n, double current_accuracy, double current_loss, int width, int height, int epochs, TDT4102::AnimationWindow& window) {
    for (int j = 0; j < epochs_n.size(); ++j) {
        window.draw_circle(TDT4102::Point(50 + (width - 100) / epochs * epochs_n[j], height - 50 - (height - 100) * loss_n[j]*0.1), 5, TDT4102::Color::navy);
        window.draw_circle(TDT4102::Point(50 + (width - 100) / epochs * epochs_n[j], height - 50 - (height - 100) * accuracy_n[j]*0.01), 5, TDT4102::Color::dark_orange);
        window.draw_text(TDT4102::Point(50 + (width - 100) / epochs * epochs_n[j], height - 40), std::to_string(j + 1), TDT4102::Color::black, 10);
        if (j > 0) {
            window.draw_line(TDT4102::Point(50 + (width - 100) / epochs * epochs_n[j - 1], height - 50 - (height - 100) * loss_n[j - 1]*0.1), TDT4102::Point(50 + (width - 100) / epochs * epochs_n[j], height - 50 - (height - 100) * loss_n[j]*0.1), TDT4102::Color::navy);
            window.draw_line(TDT4102::Point(50 + (width - 100) / epochs * epochs_n[j - 1], height - 50 - (height - 100) * accuracy_n[j - 1]*0.01), TDT4102::Point(50 + (width - 100) / epochs * epochs_n[j], height - 50 - (height - 100) * accuracy_n[j]*0.01), TDT4102::Color::dark_orange);
        }
    }   

    window.draw_line(TDT4102::Point(50, height - 50), TDT4102::Point(50, 50));
    window.draw_line(TDT4102::Point(50, height - 50), TDT4102::Point(width - 50, height - 50));
    window.draw_text(TDT4102::Point(52, 50), "100%", TDT4102::Color::black, 10);

    window.draw_text(TDT4102::Point(5, 20), "Accuracy: " + std::to_string(current_accuracy), TDT4102::Color::dark_orange, 20);
    window.draw_text(TDT4102::Point(5, 2), "Loss: " + std::to_string(current_loss), TDT4102::Color::navy, 20);
    window.next_frame();
}

void visualize_feed_forward(std::vector<Matrix> activated_layers, const Matrix& x_labels) {
    int width = 1000;
    int height = 500;
    TDT4102::AnimationWindow window(100, 100, width, height, "Feed forward pass");

    activated_layers.insert(activated_layers.begin(), x_labels);
    
    for (int i = 0; i < activated_layers.size(); ++i) {
        window.draw_text(TDT4102::Point(50 + (i * width/activated_layers.size()), 20), "Layer " + std::to_string(i), TDT4102::Color::black, 20);

        for (int j = 0; j < activated_layers[i].getRows(); ++j) {
            window.draw_circle(TDT4102::Point(50 + (i * width/activated_layers.size()), (height - 50) - (j*(height-50))/activated_layers[i].getRows()), 20, TDT4102::Color::black);
            window.draw_text(TDT4102::Point(50 + (i * width/activated_layers.size()),(height - 50) - (j*height)/activated_layers[i].getRows()), std::to_string(activated_layers[i][j][0]), TDT4102::Color::white, 20);

        }
    }

    window.wait_for_close();   
}