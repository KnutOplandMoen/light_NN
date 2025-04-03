#include "animation_functions.h"
#include "network.h"
#include <windows.h>
#include <unistd.h>
#include "widgets/Button.h"


void training_visualise::update(double* epochs_n, double* loss_n, double* accuracy_n, double current_accuracy, double current_loss, int epochs, int i) {
    for (int j = 0; j < i; ++j) {
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

}

void training_visualise::callbackFunction() {
    close();
}

void training_visualise::finish() {
    draw_text(TDT4102::Point(width() - 200, 2), "Training complete!", TDT4102::Color::black, 20);

    const TDT4102::Point buttonPosition {width() - 210, 30};
    const unsigned int buttonWidth = 200;
    const unsigned int buttonHeight = 20;
    const std::string buttonLabel = "Finish and close";
    TDT4102::Button button {buttonPosition, buttonWidth, buttonHeight, buttonLabel};

    button.setCallback(std::bind(&training_visualise::callbackFunction, this));
    add(button);
    wait_for_close();


}

void training_visualise::initialise() {
    draw_line(TDT4102::Point(50, height() - 50), TDT4102::Point(50, 50));
    draw_line(TDT4102::Point(50, height() - 50), TDT4102::Point(width() - 50, height() - 50));
    draw_text(TDT4102::Point(52, 50), "100%", TDT4102::Color::black, 10); 
    next_frame();
}


/**
 * @brief Visualizes the feed-forward process of a neural network.
 * 
 * This function generates a graphical representation of the feed-forward 
 * process in a neural network, including the connections between neurons 
 * across layers and the activation values of each neuron. It also highlights 
 * the output neuron with the highest activation value.
 * 
 * @param activated_layers A vector of matrices representing the activation 
 *        values of neurons in each layer of the neural network.
 * @param x_labels A matrix representing the input layer's activation values.
 * @param x_labels_names A vector of strings representing the names of the 
 *        input features (optional).
 * @param y_labels_names A vector of strings representing the names of the 
 *        output labels (optional).
 * @param show_text A boolean flag indicating whether to display text 
 *        annotations (e.g., activation values and labels) in the visualization.
 */
void feed_forward_visualise::visualize_feed_forward(std::vector<Matrix>& activated_layers, Matrix& x_labels, std::vector<std::string> x_labels_names, std::vector<std::string> y_labels_names, bool show_text) {
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

    bool x_labels_names_show = false;
    if (x_labels_names.size() > 0) {
        x_labels_names_show = true;
    }
    bool y_labels_names_show = false;
    if (y_labels_names.size() > 0) {
        y_labels_names_show = true;
    }


    int y_padding = 70;
    int x_padding = 50;

    for (int i = 0; i < activated_layers.size(); ++i) {
        for (int j = 1; j <= activated_layers[i].getRows(); ++j) {

            for (int k = 1; k <= activated_layers[i+1].getRows(); ++k) { //lines between neurons
                if (!static_cast <double> (std::round(activated_layers[i+1][k-1][0] * 10) / 10) > 0 || !static_cast <double> (std::round(activated_layers[i][j-1][0] * 10) / 10) > 0) { 
                    this -> draw_line(TDT4102::Point(x_padding + (i * width()/activated_layers.size()), height() - y_padding - (((height() - y_padding*2 - 2* circle_size * activated_layers[i].getRows()))/(activated_layers[i].getRows() + 1))*j - (circle_size * 2 * j)), TDT4102::Point(x_padding + ((i +1) * width()/activated_layers.size()), height() - y_padding - (((height() - y_padding*2 - 2* circle_size * activated_layers[i + 1].getRows()))/(activated_layers[i + 1].getRows() + 1))*k - (circle_size * 2 * k)), TDT4102::Color::light_gray);
                }

                else {//drwaing green lines between neurons
                    this -> draw_line(TDT4102::Point(x_padding + (i * width()/activated_layers.size()), height() - y_padding - (((height() - y_padding*2 - 2* circle_size * activated_layers[i].getRows()))/(activated_layers[i].getRows() + 1))*j - (circle_size * 2 * j)), TDT4102::Point(x_padding + ((i +1) * width()/activated_layers.size()), height() - y_padding - (((height() - y_padding*2 - 2* circle_size * activated_layers[i + 1].getRows()))/(activated_layers[i + 1].getRows() + 1))*k - (circle_size * 2 * k)), TDT4102::Color::dark_green);
                    
                }
            }
            
            if (i == activated_layers.size() - 1) { //drawing output layer

                if (y_labels_names_show && show_text) {
                    this -> draw_text(TDT4102::Point(x_padding + 25 + (i * width()/activated_layers.size()), height() - y_padding - 12 - (((height() - y_padding*2 - 2* circle_size * activated_layers[i].getRows()))/(activated_layers[i].getRows() + 1))*j - (circle_size * 2 * j)), y_labels_names[j-1], TDT4102::Color::black, 17);
                }

                if (j == max_index +1) {
                    this -> draw_circle(TDT4102::Point(x_padding + (i * width()/activated_layers.size()), height() - y_padding - (((height() - y_padding*2 - 2* circle_size * activated_layers[i].getRows()))/(activated_layers[i].getRows() + 1))*j - (circle_size * 2 * j)), circle_size, TDT4102::Color::dark_green);
                    if (show_text) {this -> draw_text(TDT4102::Point(x_padding - 15 + (i * width()/activated_layers.size()), height() - y_padding - 12 - (((height() - y_padding*2 - 2* circle_size * activated_layers[i].getRows()))/(activated_layers[i].getRows() + 1))*j - (circle_size * 2 * j)), std::format("{:.1f}", static_cast <double> (std::round(activated_layers[i][j-1][0] * 10) / 10)), TDT4102::Color::white, 17);} 
                }
                else {
                    this -> draw_circle(TDT4102::Point(x_padding + (i * width()/activated_layers.size()), height() - y_padding - (((height() - y_padding*2 - 2* circle_size * activated_layers[i].getRows()))/(activated_layers[i].getRows() + 1))*j - (circle_size * 2 * j)), circle_size, TDT4102::Color::black);
                    if(show_text) {
                    if (activated_layers[i][j-1][0] < 0) {
                        this -> draw_text(TDT4102::Point(x_padding - 18 + (i * width()/activated_layers.size()), height() - y_padding - 12 - (((height() - y_padding*2 - 2* circle_size * activated_layers[i].getRows()))/(activated_layers[i].getRows() + 1))*j - (circle_size * 2 * j)), std::format("{:.1f}", static_cast <double> (std::round(activated_layers[i][j-1][0] * 10) / 10)), TDT4102::Color::white, 17);
                        }
                        else if (static_cast <int> (activated_layers[i][j-1][0]) >= 100) {
                            this -> draw_text(TDT4102::Point(x_padding - 15 + (i * width()/activated_layers.size()), height() - y_padding - 12 - (((height() - y_padding*2 - 2* circle_size * activated_layers[i].getRows()))/(activated_layers[i].getRows() + 1))*j - (circle_size * 2 * j)), std::to_string(static_cast <int> (std::round(activated_layers[i][j-1][0] * 10) / 10)), TDT4102::Color::white, 17);
                        }
                        else if (static_cast <int> (activated_layers[i][j-1][0]) >= 10) {
                            this -> draw_text(TDT4102::Point(x_padding - 15 + (i * width()/activated_layers.size()), height() - y_padding - 12 - (((height() - y_padding*2 - 2* circle_size * activated_layers[i].getRows()))/(activated_layers[i].getRows() + 1))*j - (circle_size * 2 * j)), std::to_string(static_cast <int> (std::round(activated_layers[i][j-1][0] * 10) / 10)), TDT4102::Color::white, 17);
                        }
                        else {
                            this -> draw_text(TDT4102::Point(x_padding - 15 + (i * width()/activated_layers.size()), height() - y_padding -12 - (((height() - y_padding*2 - 2* circle_size * activated_layers[i].getRows()))/(activated_layers[i].getRows() + 1))*j - (circle_size * 2 * j)), std::format("{:.1f}", static_cast <double> (std::round(activated_layers[i][j-1][0] * 10) / 10)), TDT4102::Color::white, 17);
                        }
                    }}
            }
            else {
                
                if (i == 0 && x_labels_names_show && show_text) {
                    this -> draw_text(TDT4102::Point(10 + (i * width()/activated_layers.size()), height() - y_padding - 12 - (((height() - y_padding*2 - 2* circle_size * activated_layers[i].getRows()))/(activated_layers[i].getRows() + 1))*j - (circle_size * 2 * j)), x_labels_names[j-1], TDT4102::Color::black, 17);
                }

                if (activated_layers[i][j-1][0] >= 1) {
                    this -> draw_circle(TDT4102::Point(x_padding + (i * width()/activated_layers.size()), height() - y_padding - (((height() - y_padding*2 - 2* circle_size * activated_layers[i].getRows()))/(activated_layers[i].getRows() + 1))*j - (circle_size * 2 * j)), circle_size, TDT4102::Color::dark_green);
                }

                else {
                    this -> draw_circle(TDT4102::Point(x_padding + (i * width()/activated_layers.size()), height() - y_padding - (((height() - y_padding*2 - 2* circle_size * activated_layers[i].getRows()))/(activated_layers[i].getRows() + 1))*j - (circle_size * 2 * j)), circle_size, TDT4102::Color::black);
                }

                if (activated_layers[i][j-1][0] < 0 && show_text) {
                    this -> draw_text(TDT4102::Point(x_padding - 18 + (i * width()/activated_layers.size()), height() - y_padding - 12 - (((height() - y_padding*2 - 2* circle_size * activated_layers[i].getRows()))/(activated_layers[i].getRows() + 1))*j - (circle_size * 2 * j)), std::format("{:.1f}", static_cast <double> (std::round(activated_layers[i][j-1][0] * 10) / 10)), TDT4102::Color::white, 17);
                }
                else if (static_cast <int> (activated_layers[i][j-1][0]) >= 10 && show_text) {
                    this -> draw_text(TDT4102::Point(x_padding - 12 + (i * width()/activated_layers.size()), height() - y_padding -12 - (((height() - y_padding*2 - 2* circle_size * activated_layers[i].getRows()))/(activated_layers[i].getRows() + 1))*j - (circle_size * 2 * j)), std::to_string(static_cast <int> (std::round(activated_layers[i][j-1][0] * 10) / 10)), TDT4102::Color::white, 17);
                }
                else if (show_text) {
                    this -> draw_text(TDT4102::Point(x_padding - 15 + (i * width()/activated_layers.size()), height() - y_padding - 12 - (((height() - y_padding*2 - 2* circle_size * activated_layers[i].getRows()))/(activated_layers[i].getRows() + 1))*j - (circle_size * 2 * j)), std::format("{:.1f}", static_cast <double> (std::round(activated_layers[i][j-1][0] * 10) / 10)), TDT4102::Color::white, 17);
                }
            }

        }
    }   
}