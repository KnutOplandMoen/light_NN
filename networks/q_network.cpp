#include "q_network.h"
#include "game.h"
#include "map"
#include "math_functions.h"
#include <deque>
#include <unistd.h> 
#include "animation_functions.h"

/**
 * @brief Select an action from one of the following:
 * 
 * 1. Random -> if epsilon is close to 1 this is likely
 * 
 * 2. Network -> if epsilon is close to 0 this is more likely
 * 
 * 3. User -> from_user needs to be togled manually to true
 * 
 * @param state Current state of the game.
 * @param game_play Game class with game information.
 */
int q_network::select_action(Matrix& state, Game& game_play) {
    double number = randDouble(0, 10000);

    bool from_user = false;
    
    if (!from_user) {
    if (number / 10000 < epsilon) {  
        return static_cast<int>(number) % output_layer_size;  // Random action (explore)
    } else {
        Matrix q_values = feed_forward_pass(state)[0].back();
        return q_values.getMaxRow();  // Best action (exploit)
    }
    }
    else { //Take move from user 
        int move;
        std::string move_l;
        std::cout <<"move:\n:";
        
        cin >> move_l;
        if (move_l == "w") {
            move = 0;
        }
        else if (move_l == "d") {
            move = 1;
        }
        else if (move_l == "a") {
            move = 3;
        }
        else if (move_l == "s") {
            move = 2;
        }
        else {
            move = 0;
        }
        std::cout << "chose move: " << move << std::endl;
        return move;
        }
    }

/**
 * @brief Specific function: Gets information by taking in the state, doing a action in this state with select action, 
 * then adding information needed for later training to an information struct
 * 
 * @param state Current state of game.
 * @param game_play Game class with game information.
 * @param nextState boolean representing if nextstate after new state should be predicted or not
 * @note This is a game specific function, for instance this is one for snake, make new one or change this one for specific game
 * 
 * @return A information struct with:
 * `q_values`, `q_value`, `reward`, `done`, `q_target_value`, `q_target` and `state`
 */
information q_network::get_information(Matrix& state, Game& game_play, bool nextState) {
    bool grow = false;
    bool collision = false;
    information info;
    TDT4102::Point lastPos = game_play.snake.getSnakeHead();
    std::vector<std::vector<Matrix>> ff = feed_forward_pass(state);
    Matrix q_values = ff[0].back();

    int action = select_action(state, game_play);
    double q_value = q_values[action][0];
    Matrix prev_state = game_play.getState();

    game_play.take_action(action); 

    if (game_play.snake.collisionFood(game_play.foodVec) != -1){
        grow = true;
        game_play.foodVec.clear();
        game_play.newFood();
    }

    game_play.snake.move(grow);
    
    if (game_play.snake.collisionFood(game_play.foodVec) != -1){
        grow = true;
    }

    std::deque<TDT4102::Point> body = game_play.snake.getSnakeBody();

    for (TDT4102::Point& part : body) {
        if ((part.x == game_play.snake.getSnakeHead().x) && (part.y == game_play.snake.getSnakeHead().y)) {
            collision = true;
            info.death_reason = "crashed into wall";
        }
    }

    if(!(game_play.snake.getSnakeHead().x >= 0 && game_play.snake.getSnakeHead().x < game_play.getWidth())){
        info.death_reason = "crashed into wall";
    }
    else if(!(game_play.snake.getSnakeHead().y >= 0 && game_play.snake.getSnakeHead().y < game_play.getHeight())){
        info.death_reason = "crashed into wall";
    }

    
    collision = game_play.snake.collision() || collision;
    //std::cout << "collision: " << collision << std::endl;
    
    double reward = game_play.getReward(grow, collision, lastPos);
    total_reward += reward;

    int done = game_play.is_over();

    info.q_values = q_values;
    info.q_value = q_value;
    info.reward = reward;
    info.done = done;
    info.state = state;


    if (nextState) {
    Matrix new_state = game_play.getState();

    Matrix next_action = feed_forward_pass(new_state)[0].back();
    double max_next_q_value = next_action[next_action.getMaxRow()][0];

    double q_target_value = reward + (1 - done) * gamma * max_next_q_value;

    Matrix q_target = q_values;
    q_target[action][0] = q_target_value;

    info.q_target_value = q_target_value;
    info.q_target = q_target;
    }

    info.activated_layers = ff[0];

    return info;

}


/**
 * @brief Updates network using normal backpropagation
 * 
 * @param Learning_rate Learning rate (0-1).
 * @param mini_batch Deque with information structs to train on.
 * @param mini_batch_size The size of the mini-batch used for training the network.
 * 
 */
void q_network::update_net(double learning_rate, int mini_batch_size, std::deque<information>& mini_batch) {
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::vector<Matrix>> batch_errors;
    std::vector<std::vector<Matrix>> batch_activated_layers;

    for (int k = 0; k < mini_batch.size(); ++k) {
        information& info = mini_batch[k];
        std::vector<std::vector<Matrix>> ff = feed_forward_pass(info.state);
        std::vector<Matrix> activated_layers = ff[0];
        activated_layers.insert(activated_layers.begin(), info.state); 
        std::vector<Matrix> error = get_errors(info.state, info.q_target); 
        batch_errors.push_back(error);
        batch_activated_layers.push_back(activated_layers);
    }

    gradient_descent_weights(batch_errors, learning_rate, mini_batch[0].state, batch_activated_layers);
    gradient_descent_biases(batch_errors, learning_rate, mini_batch[0].state, batch_activated_layers);

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
}


/**
 * @brief Trains the Network using the reinforcement learning paradigm.
 * 
 * This function simulates a specified number of games, collects experiences 
 * during gameplay, and updates the Q-network using mini-batches of experiences. 
 * It also applies epsilon-greedy exploration, decays epsilon over time, and 
 * optionally saves the network state at specified intervals.
 * 
 * @param games The number of games to simulate for training.
 * @param batch_size The maximum size of the experience replay buffer.
 * @param mini_batch_size The size of the mini-batch used for training the network.
 * @param learning_rate The learning rate for updating the network.
 * @param autosave_file A map containing file paths as keys and save intervals 
 *                      (in terms of games) as values. If provided, the network 
 *                      state is saved to the specified files at the given intervals.
 */
void q_network::train(int games, int batch_size, int mini_batch_size, double learning_rate, const std::map<std::string, int>& autosave_file) {
    std::cout << "starting training" << std::endl;
    std::deque<information> experiences;
    for (int game = 0; game < games; ++game) {
        std::cout << "Game: " << game+1 << std::endl;
        Game game_play(300, 100);
        int move = 0;
        total_reward = 0;
        while (!game_play.is_over()) {
            game_play.next_frame();


            Matrix state = game_play.getState(); //Get state
            information info = get_information(state, game_play, true); //Use state -> get info

            game_play.drawBoard(); //Draw board

            if (experiences.size() >= batch_size) {
                experiences.pop_front();
            }

            if (experiences.size() >= mini_batch_size) {
                std::vector<int> indices(experiences.size());
                std::iota(indices.begin(), indices.end(), 0);
                std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});
                
                std::deque<information> mini_batch;
                for (int i = 0; i < mini_batch_size && i < experiences.size(); ++i) {
                    mini_batch.push_back(experiences[indices[i]]);
                }
                update_net(learning_rate, mini_batch_size, mini_batch);  // Train on a single mini-batch
            }
            

            experiences.push_back(info); //Add experience
        }

        epsilon = std::max(min_epsilon, epsilon * epsilon_decay); // Decay epsilon over time

        //Print stats
        std::cout << "Game: " << game + 1 << ": " << "\033[1;32mDone\033[0m\n";
        std::cout << "---------" << "\n";
        std::cout << "\033[1;30mSnake size:: \033[0m\n" << game_play.snake.getSnakeBody().size() << "\n";
        std::cout << "\033[1;30mTotal reward: \033[0m\n" << total_reward << "\n";
        std::cout << "\033[1;30mEpsilon: \033[0m\n" << epsilon << "\n";
        std::cout << "-----------------" << "\n";

        if (!autosave_file.empty()) { 
            for (const auto& [key, value] : autosave_file) {
            if (game % value == 0) {
                save_state(key, true);
            }
            }
        }

        game_play.close();

    }
}

/**
 * @brief Test a network and play the game snake with model
 * 
 * @param games The number of games to simulate for training.
 */
void q_network::play(int games) { 
    
    set_epsilon(0); //No random moves with epsilon = 0
     //Initialize visualization
    std::vector<std::string> x_labels_names = {"D_UP", "D_RIGHT", "D_DOWN", "D_LEFT", "FOOD_UP", "FOOD_RIGHT", "FOOD_DOWN", "FOOD_LEFT", "DIR_UP", "DIR_RIGHT", "DIR_DOWN", "DIR_LEFT", "FOOD_DIR_UP", "FOOD_DIR_RIGHT", "FOOD_DIR_DOWN", "FOOD_DIR_LEFT"};
    std::vector<std::string> y_labels_names = {"Up", "Right", "Down", "Left"};
    feed_forward_visualise nn_vis(0, 20, 650, 750, "Feed forward pass");

    for (int game = 0; game < games; ++ game) {
        int move = 0;
        total_reward = 0;

        Game game_play(650, 100);

        while (!game_play.is_over()) {
            nn_vis.next_frame();
            game_play.next_frame();

            Matrix state = game_play.getState(); //Get state
            
            information info = get_information(state, game_play, false); //Use state -> make move and get info
            total_reward += info.reward;

            game_play.drawBoard(); //Draw board    
            nn_vis.visualize_feed_forward(info.activated_layers, state, x_labels_names, y_labels_names, false); //Vis feed forward
        }

        std::cout << "Game: " << game + 1 << ": " << "\033[1;32mDone\033[0m\n";
        std::cout << "---------" << "\n";
        std::cout << "\033[1;30mSnake size:: \033[0m\n" << game_play.snake.getSnakeBody().size() << "\n";
        std::cout << "\033[1;30mTotal reward: \033[0m\n" << total_reward << "\n";
        std::cout << "-----------------" << "\n";
    }

}