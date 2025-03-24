#include "animation_functions.h"
#include "network.h"
#include <windows.h>
#include <unistd.h>
#include "widgets/Button.h"

void training_visualise::update(std::vector <double>& epochs_n, std::vector <double>& loss_n, std::vector <double>& accuracy_n, double current_accuracy, double current_loss, int epochs) {
    for (int j = 0; j < epochs_n.size(); ++j) {
        draw_circle(TDT4102::Point(50 + (width() - 100) / epochs * epochs_n[j], height() - 50 - (height() - 100) * loss_n[j]*0.1), 2, TDT4102::Color::navy);
        draw_circle(TDT4102::Point(50 + (width() - 100) / epochs * epochs_n[j], height() - 50 - (height() - 100) * accuracy_n[j]*0.01), 2, TDT4102::Color::dark_orange);
        draw_text(TDT4102::Point(50 + (width() - 100) / epochs * epochs_n[j], height() - 40), std::to_string(j + 1), TDT4102::Color::black, 10);
        if (j > 0) {
            draw_line(TDT4102::Point(50 + (width() - 100) / epochs * epochs_n[j - 1], height() - 50 - (height() - 100) * loss_n[j - 1]*0.1), TDT4102::Point(50 + (width() - 100) / epochs * epochs_n[j], height() - 50 - (height() - 100) * loss_n[j]*0.1), TDT4102::Color::navy);
            draw_line(TDT4102::Point(50 + (width() - 100) / epochs * epochs_n[j - 1], height() - 50 - (height() - 100) * accuracy_n[j - 1]*0.01), TDT4102::Point(50 + (width() - 100) / epochs * epochs_n[j], height() - 50 - (height() - 100) * accuracy_n[j]*0.01), TDT4102::Color::dark_orange);
        }
    }   

    draw_line(TDT4102::Point(50, height() - 50), TDT4102::Point(50, 50));
    draw_line(TDT4102::Point(50, height() - 50), TDT4102::Point(width() - 50, height() - 50));
    draw_text(TDT4102::Point(52, 50), "100%", TDT4102::Color::black, 10);

    draw_text(TDT4102::Point(5, 20), "Accuracy: " + std::to_string(current_accuracy), TDT4102::Color::dark_orange, 20);
    draw_text(TDT4102::Point(5, 2), "Loss: " + std::to_string(current_loss), TDT4102::Color::navy, 20);

    next_frame();
}

void training_visualise::initialise() {
    draw_line(TDT4102::Point(50, height() - 50), TDT4102::Point(50, 50));
    draw_line(TDT4102::Point(50, height() - 50), TDT4102::Point(width() - 50, height() - 50));
    draw_text(TDT4102::Point(52, 50), "100%", TDT4102::Color::black, 10); 
}

void feed_forward_visualise::visualize_feed_forward(std::vector<Matrix>& activated_layers, Matrix& x_labels) {
    activated_layers.insert(activated_layers.begin(), x_labels);
    const int circle_size = 15;

    double max = 0;
    int max_index = 0;
    for (int idx = 0; idx < activated_layers.back().getRows(); ++idx) {
        if (activated_layers.back()[idx][0] > max) {
            max = activated_layers.back()[idx][0];
            max_index = idx;
        }
    }

    for (int i = 0; i < activated_layers.size(); ++i) {
        for (int j = 1; j <= activated_layers[i].getRows(); ++j) {

            for (int k = 1; k <= activated_layers[i+1].getRows(); ++k) { //lines between neurons
                if (!static_cast <double> (std::round(activated_layers[i+1][k-1][0] * 10) / 10) > 0 || !static_cast <double> (std::round(activated_layers[i][j-1][0] * 10) / 10) > 0) { 
                    draw_line(TDT4102::Point(50 + (i * width()/activated_layers.size()), height() - 50 - (((height() - 100 - 2* circle_size * activated_layers[i].getRows()))/(activated_layers[i].getRows() + 1))*j - (circle_size * 2 * j)), TDT4102::Point(50 + ((i +1) * width()/activated_layers.size()), height() - 50 - (((height() - 100 - 2* circle_size * activated_layers[i + 1].getRows()))/(activated_layers[i + 1].getRows() + 1))*k - (circle_size * 2 * k)), TDT4102::Color::light_gray);
                }

                else {//drwaing green lines between neurons
                    draw_line(TDT4102::Point(50 + (i * width()/activated_layers.size()), height() - 50 - (((height() - 100 - 2* circle_size * activated_layers[i].getRows()))/(activated_layers[i].getRows() + 1))*j - (circle_size * 2 * j)), TDT4102::Point(50 + ((i +1) * width()/activated_layers.size()), height() - 50 - (((height() - 100 - 2* circle_size * activated_layers[i + 1].getRows()))/(activated_layers[i + 1].getRows() + 1))*k - (circle_size * 2 * k)), TDT4102::Color::dark_green);
                    
                }
            }

            draw_circle(TDT4102::Point(50 + (i * width()/activated_layers.size()), height() - 50 - (((height() - 100 - 2* circle_size * activated_layers[i].getRows()))/(activated_layers[i].getRows() + 1))*j - (circle_size * 2 * j)), circle_size, TDT4102::Color::black);
            if (i == activated_layers.size() - 1) { //drawing output layer
                if (j == max_index +1) {
                    draw_circle(TDT4102::Point(50 + (i * width()/activated_layers.size()), height() - 50 - (((height() - 100 - 2* circle_size * activated_layers[i].getRows()))/(activated_layers[i].getRows() + 1))*j - (circle_size * 2 * j)), circle_size, TDT4102::Color::dark_green);
                    draw_text(TDT4102::Point(37 + (i * width()/activated_layers.size()), height() - 62 - (((height() - 100 - 2* circle_size * activated_layers[i].getRows()))/(activated_layers[i].getRows() + 1))*j - (circle_size * 2 * j)), std::format("{:.1f}", static_cast <double> (std::round(activated_layers[i][j-1][0] * 10) / 10)), TDT4102::Color::white, 17);
                }
                else {
                    if (activated_layers[i][j-1][0]< 0) {
                        draw_text(TDT4102::Point(32 + (i * width()/activated_layers.size()), height() - 62 - (((height() - 100 - 2* circle_size * activated_layers[i].getRows()))/(activated_layers[i].getRows() + 1))*j - (circle_size * 2 * j)), std::format("{:.1f}", static_cast <double> (std::round(activated_layers[i][j-1][0] * 10) / 10)), TDT4102::Color::white, 17);
                        }
                        else if (static_cast <int> (activated_layers[i][j-1][0]) >= 10) {
                            draw_text(TDT4102::Point(37 + (i * width()/activated_layers.size()), height() - 62 - (((height() - 100 - 2* circle_size * activated_layers[i].getRows()))/(activated_layers[i].getRows() + 1))*j - (circle_size * 2 * j)), std::to_string(static_cast <int> (std::round(activated_layers[i][j-1][0] * 10) / 10)), TDT4102::Color::white, 17);
                        }
                        else {
                        draw_text(TDT4102::Point(37 + (i * width()/activated_layers.size()), height() - 62 - (((height() - 100 - 2* circle_size * activated_layers[i].getRows()))/(activated_layers[i].getRows() + 1))*j - (circle_size * 2 * j)), std::format("{:.1f}", static_cast <double> (std::round(activated_layers[i][j-1][0] * 10) / 10)), TDT4102::Color::white, 17);
                        }
                    }
            }
            else {
                if (activated_layers[i][j-1][0]< 0) {
                draw_text(TDT4102::Point(32 + (i * width()/activated_layers.size()), height() - 62 - (((height() - 100 - 2* circle_size * activated_layers[i].getRows()))/(activated_layers[i].getRows() + 1))*j - (circle_size * 2 * j)), std::format("{:.1f}", static_cast <double> (std::round(activated_layers[i][j-1][0] * 10) / 10)), TDT4102::Color::white, 17);
                }
                else if (static_cast <int> (activated_layers[i][j-1][0]) >= 10) {
                    draw_text(TDT4102::Point(37 + (i * width()/activated_layers.size()), height() - 62 - (((height() - 100 - 2* circle_size * activated_layers[i].getRows()))/(activated_layers[i].getRows() + 1))*j - (circle_size * 2 * j)), std::to_string(static_cast <int> (std::round(activated_layers[i][j-1][0] * 10) / 10)), TDT4102::Color::white, 17);
                }
                else {
                draw_text(TDT4102::Point(37 + (i * width()/activated_layers.size()), height() - 62 - (((height() - 100 - 2* circle_size * activated_layers[i].getRows()))/(activated_layers[i].getRows() + 1))*j - (circle_size * 2 * j)), std::format("{:.1f}", static_cast <double> (std::round(activated_layers[i][j-1][0] * 10) / 10)), TDT4102::Color::white, 17);
                }
            }

        }
    }   
}