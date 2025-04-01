#include "matrix.h"

class game {
    private:
    Matrix state;
    int game_over = 0;
    public:
    Matrix get_state() {return state;}
    void take_action(Matrix action);
    double get_reward();
    int is_over();
    void train();
};