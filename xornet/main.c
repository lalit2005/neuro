#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define NUM_INPUTS 2
#define NUM_HIDDEN_NODES 2
#define NUM_OUTPUTS 1
#define NUM_TRAINING_SETS 4

double init_weights() { return (double)rand() / (double)RAND_MAX; }

double sigmoid(double x) { return 1 / (1.0 + exp(-x)); }

double d_sigmoid(double x) { return x * (1 - x); }

void shuffle(int *array, size_t n) {
  if (n > 1) {
    for (size_t i = 0; i < n - 1; i++) {
      size_t j = i + rand() % (n - i);
      int t = array[j];
      array[j] = array[i];
      array[i] = t;
    }
  }
}

int main() {
  printf("start of program\n");
  // 46.875 gives a much better approximation for the xor function than 0.1
  // found this value by performing binary search from 1..100
  const double learning_rate = 0.1f;

  double hidden_layer[NUM_HIDDEN_NODES];
  double output_layer[NUM_OUTPUTS];

  double hidden_layer_bias[NUM_HIDDEN_NODES];
  double output_layer_bias[NUM_OUTPUTS];

  double hidden_weights[NUM_INPUTS][NUM_HIDDEN_NODES];
  double output_weights[NUM_HIDDEN_NODES][NUM_OUTPUTS];

  double training_inputs[NUM_TRAINING_SETS][NUM_INPUTS] = {
      {0.0f, 0.0f},
      {1.0f, 0.0f},
      {0.0f, 1.0f},
      {1.0f, 1.0f},
  };

  double training_outputs[NUM_TRAINING_SETS][NUM_OUTPUTS] = {
      {0.0f},
      {1.0f},
      {1.0f},
      {0.0f},
  };

  for (int i = 0; i < NUM_INPUTS; i++) {
    for (int j = 0; j < NUM_HIDDEN_NODES; j++) {
      hidden_weights[i][j] = init_weights();
    }
  }

  for (int i = 0; i < NUM_HIDDEN_NODES; i++) {
    for (int j = 0; j < NUM_OUTPUTS; j++) {
      output_weights[i][j] = init_weights();
    }
  }

  for (int i = 0; i < NUM_OUTPUTS; i++) {
    output_layer_bias[i] = init_weights();
  }

  for (int i = 0; i < NUM_HIDDEN_NODES; i++) {
    hidden_layer_bias[i] = init_weights();
  }

  int training_set_order[] = {0, 1, 2, 3};
  int number_of_epochs = 10000;

  // train the neural net for `number_of_epochs` times
  for (int epoch = 0; epoch < number_of_epochs; epoch++) {
    shuffle(training_set_order, NUM_TRAINING_SETS);
    for (int x = 0; x < NUM_TRAINING_SETS; x++) {
      int training_set_index = training_set_order[x];

      // forward pass
      // compute activation_fn(neuron_value) for hidden layer
      for (int hidden_node = 0; hidden_node < NUM_HIDDEN_NODES; hidden_node++) {
        double activation = hidden_layer_bias[hidden_node];

        // for each path from input_node -> hidden_node, activation is
        // calculated and summed
        for (int input_node = 0; input_node < NUM_INPUTS; input_node++) {
          activation += training_inputs[training_set_index][input_node] *
                        hidden_weights[input_node][hidden_node];
        }
        // assigning new value for the neuron
        hidden_layer[hidden_node] = sigmoid(activation);
      }

      // compute activation_fn(neuron_value) for output layer
      for (int output_node = 0; output_node < NUM_OUTPUTS; output_node++) {
        double activation = output_layer_bias[output_node];
        // for each path from hidden_node -> output_node, activation is
        // calculated and summed
        for (int hidden_node = 0; hidden_node < NUM_HIDDEN_NODES;
             hidden_node++) {
          activation += hidden_layer[hidden_node] *
                        output_weights[hidden_node][output_node];
        }
        // assigning new value for the neuron
        output_layer[output_node] = sigmoid(activation);
      }

      printf("input: %g %g; output: %g; expected: %g\n",
             training_inputs[training_set_index][0],
             training_inputs[training_set_index][1], output_layer[0],
             training_outputs[training_set_index][0]);

      // -------------------------------------

      // backpropogation
      // compute change in output weights
      double delta_output[NUM_OUTPUTS];
      for (int output_node = 0; output_node < NUM_OUTPUTS; output_node++) {
        double error = training_outputs[training_set_index][output_node] -
                       output_layer[output_node];
        delta_output[output_node] =
            error * d_sigmoid(output_layer[output_node]);
      }

      // compute change in hidden weights
      double delta_hidden[NUM_HIDDEN_NODES];
      for (int hidden_node = 0; hidden_node < NUM_HIDDEN_NODES; hidden_node++) {
        double error = 0.0f;
        for (int output_node = 0; output_node < NUM_OUTPUTS; output_node++) {
          error += delta_output[output_node] *
                   output_weights[hidden_node][output_node];
        }
        delta_hidden[hidden_node] =
            error * d_sigmoid(hidden_layer[hidden_node]);
      }

      // apply change in output weights
      for (int output_node = 0; output_node < NUM_OUTPUTS; output_node++) {
        output_layer_bias[output_node] +=
            delta_output[output_node] * learning_rate;
        for (int hidden_node = 0; hidden_node < NUM_HIDDEN_NODES;
             hidden_node++) {
          output_weights[hidden_node][output_node] +=
              hidden_layer[hidden_node] * delta_output[output_node] *
              learning_rate;
        }
      }

      // apply change in hidden weights
      for (int hidden_node = 0; hidden_node < NUM_HIDDEN_NODES; hidden_node++) {
        hidden_layer_bias[hidden_node] +=
            delta_hidden[hidden_node] * learning_rate;
        for (int input_node = 0; input_node < NUM_INPUTS; input_node++) {
          hidden_weights[input_node][hidden_node] +=
              training_inputs[training_set_index][input_node] *
              delta_hidden[hidden_node] * learning_rate;
        }
      }
    }
  }

  // Print final weights after training
  printf("\n\n\nFinal Hidden Weights\n[ ");
  for (int j = 0; j < NUM_HIDDEN_NODES; j++) {
    printf("[ ");
    for (int k = 0; k < NUM_INPUTS; k++) {
      printf("%f ", hidden_weights[k][j]);
    }
    printf("] ");
  }

  printf("]\n\nFinal Hidden Biases\n[ ");
  for (int j = 0; j < NUM_HIDDEN_NODES; j++) {
    printf("%f ", hidden_layer_bias[j]);
  }

  printf("]\n\nFinal Output Weights\n");
  for (int j = 0; j < NUM_OUTPUTS; j++) {
    printf("[ ");
    for (int k = 0; k < NUM_HIDDEN_NODES; k++) {
      printf("%f ", output_weights[k][j]);
    }
    printf("]\n");
  }

  printf("\n\nFinal Output Biases\n[ ");
  for (int j = 0; j < NUM_OUTPUTS; j++) {
    printf("%f ", output_layer_bias[j]);
  }

  printf("]\n");

  return 0;
}
