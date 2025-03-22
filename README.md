# light_NN

## Project setup:

### Optinion 1 (Using TDT4102 template):
1. Download as zip -> VScode
2. Install TDT4102 extention
3. CTRL + Shift + p
4. TDT4102: Create Project from TDT4102 Template
5. Choose blank project
6. Skip overwrite of main.cpp, but overwrite all other files.

### Option 2 (Not using template):
Use [old_arcithecture](https://github.com/KnutOplandMoen/light_NN/tree/old_arcitecture) branch.

## Network methods:

### network::network -> Constructor
The constructor initializes weights, biases and hidden layers. 
- initialise_biases() -> initializes biases to 0
- initialise_hidden_layers() -> makes an vector with matrixes representing each hidden layer
- initialize_weights() -> Makes weigths matrixes, all using HE-initialization
    - [ ] TODO: Add functionality for different weight initialization methods :shipit:
### network::train -> training function
training the network based on folowing params: 
- std::vector Matrix train_x_labels
- std::vector Matrix train_y_labels
- std::vector Matrix test_x_labels
- std::vector Matrix test_y_labels
- int epochs
- double learning_rate
- int batch_size
- bool animation -> set to false by default. Lets user see training progress as animation.

### network::feed_forward_pass
Does a feed fowrad pass in the network.  
  
**Input** -> Input of class matrix.  
**returns** -> {all activated layer, all layers without activation function applied}   
  
**get output layer** -> To get prediction use feed_forward_pass(input)[0].back()

### network::load_state and network::save_state
Lets user load and save models (The weights and biases)    
  
- **load_state(std::string file)** -> Updates the network with weights and biases stored in file  
- **save_state(std::string file)** -> saves to models weights and biases to file
  
## Other functionality:
TODO