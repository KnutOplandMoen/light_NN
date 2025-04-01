#include "q_network.h"
#include "game.h"
#include "map"

information q_network::get_information(Matrix input, int done, game game_play) {

    Matrix q_values = feed_forward_pass(input)[0].back();
    double q_value = q_values[q_values.getMaxRow()][0];

    game_play.take_action(q_values); //TODO: Here next state needs to be made.. in environment
    Matrix prev_state = game_play.get_state();
    double reward = game_play.get_reward();
    int done = game_play.is_over();

    Matrix next_action = feed_forward_pass(prev_state)[0].back();
    double max_next_q_value = next_action[next_action.getMaxRow()][0];

    double q_target_value = reward + (1 - done) * gamma * max_next_q_value;

    Matrix q_target = q_values;
    q_target[q_values.getMaxRow()][0] = q_target_value;
    information info;
    
    return m;

}

struct information {
    Matrix q_values;
    double q_value;
    double reward;
    int done;
    double q_target_value;
    Matrix q_target;
    information()
}

/* 
The net should be updates for when enough minibatches is done
*/

void q_network::update_net(int epochs, double learning_rate, int batch_size, std::vector<std::map<std::string, Matrix>> experiences) {
    double epochs_n[epochs];
    double loss_n[epochs];//bacon
    double accuracy_n[epochs];
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<Matrix> batch_activated_layers;
    std::vector<Matrix> batch_weighted_inputs;
    std::vector<Matrix> batch_predictions;
    std::vector<Matrix> batch_errors;
    for (int k = 0; k < batch_size && (j + k) < experiences.size(); ++k) { 
        int index = j + k;
        //here lets get the information we need from current state:

        std::vector<Matrix> activated_layers = feed_forward[0];
        activated_layers.insert(activated_layers.begin(), experiences[index]);
        std::vector<Matrix> weighted_inputs = feed_forward[1]; // Extract weighted inputs
        std::vector<Matrix> error = get_errors(experiences[index]["state"], experiences[index]["state"]);
        batch_errors.push_back(error);
        batch_activated_layers.push_back(activated_layers);
        batch_weighted_inputs.push_back(weighted_inputs); // Store weighted inputs
        batch_predictions.push_back(feed_forward[0].back());
    }
        gradient_descent_weights(batch_errors, learning_rate, train_x_labels[j], batch_activated_layers);
        gradient_descent_biases(batch_errors, learning_rate, train_x_labels[j], batch_activated_layers);
        for (int k = 0; k < batch_predictions.size(); ++k) {
            update_loss(batch_predictions[k], train_y_labels[j + k]);
        }
    }
        
        // Test the network with the test data
        std::vector <Matrix> predictions;
        for (int j = 0; j < test_x_labels.size(); ++j) {
            std::vector<std::vector<Matrix>> feed_forward = feed_forward_pass(test_x_labels[j]);
            predictions.push_back(feed_forward[0].back());
        }
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

        double current_accuracy = get_accuracy(predictions, test_y_labels);
        double current_loss = loss / train_x_labels.size();
        std::cout << "Epoch " << i + 1 << ": " << "\033[1;32mDone\033[0m\n";
        std::cout << "---------" << "\n";
        std::cout << "\033[1;30mAccuracy: \033[0m\n" << current_accuracy << "%\n";
        std::cout << "\033[1;30mLoss: \033[0m\n" << current_loss << "\n";
        std::cout << "\033[1;30mTime taken for epoch: \033[0m\n" << static_cast<double> (duration.count()) / 1000 << " s" << "\n";
        std::cout << "\033[1;30mEstimated time left: \033[0m\n" << static_cast<double> (duration.count()) / 1000 * (epochs - i - 1) << " s" << "\n";
        std::cout << "-----------------" << "\n";

    

        loss = 0; // Reset loss
    }}}

void q_network::train() {

    /*
    Plan: 

    Initialize loop with number of games
        For each game initialize a game
        do actions
        save experiences
        when enough actions is take (experience_size = X): 
            train network on experiences using update_network
        
    */
}