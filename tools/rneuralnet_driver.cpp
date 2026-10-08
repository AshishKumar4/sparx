// Runs RNeuralNet-Research's own classes and functions in one specified order, single-threaded.
//
// Compiled by tools/make_rneuralnet_fixtures.py next to the repository's
// header.h, neuron.h, neuron.cpp, Processor.h and Processor.cpp, unchanged,
// and a gnuplot-iostream.h that plots nothing. Main.cpp runs the forward
// pass, the neurons' refresh and the learning in three threads that share
// every neuron without synchronization; this driver runs one tick as
//
//   1. each input neuron takes its value (Main.cpp's InputThread),
//   2. Global_Refresher computes every internal neuron's output from its sum,
//   3. Global_ForwardProcessor sends and delivers along every connection,
//      in the order the connections were made,
//   4. on a tick with a reward, Global_Adjuster spreads it from the reward
//      feeder (Main.cpp leaves this call commented out), Global_Teacher
//      changes the weights and Global_Renew clears the rewards.
//
// It reads a network and its inputs and rewards on stdin and writes each
// tick's outputs, and each reward's local rewards and weights, to the file
// named by its argument (the classes print their own traces on stdout).
#include "neuron.cpp"
#include "Processor.cpp"

#include <cstdio>

static void initialized(Neuron_t* neuron) {
  // The constructors leave these unset; the order above reads none before
  // writing it, and they are set here so no run depends on stale memory.
  neuron->ValSum = 0;
  neuron->softmax_tmp = 0;
  neuron->Lock = 0;
}

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  FILE* out = fopen(argv[1], "w");
  int neurons, inputs, outputs, edges, ticks;
  if (scanf("%d %d %d %d %d", &neurons, &inputs, &outputs, &edges, &ticks) != 5) return 1;
  for (int i = 0; i < neurons; i++) {
    float threshold;
    scanf("%f", &threshold);
    Neuron_t* neuron = new Neuron_t(threshold);
    initialized(neuron);
    NeuralNet.push_back(neuron);
  }
  for (int i = 0; i < inputs; i++) {
    Neuron_t* neuron = new Neuron_t(0, 2);
    initialized(neuron);
    input_layer.push_back(neuron);
  }
  RewardFeederNeuron = new Neuron_t(0, 3);
  initialized(RewardFeederNeuron);
  for (int i = 0; i < outputs; i++) {
    int index;
    scanf("%d", &index);
    output_layer.push_back(NeuralNet[index]);
  }
  // kind 0: internal pre -> internal post; 1: internal pre -> the reward feeder; 2: input pre -> internal post.
  for (int e = 0; e < edges; e++) {
    int kind, pre, post;
    float weight, myelin;
    scanf("%d %d %d %f %f", &kind, &pre, &post, &weight, &myelin);
    Neuron_t* start = kind == 2 ? input_layer[pre] : NeuralNet[pre];
    Neuron_t* end = kind == 1 ? RewardFeederNeuron : NeuralNet[post];
    Neurite_t<Neuron_t, Neuron_t>* connection = Connect_Neurons(start, end, weight, myelin);
    connection->softmax_tmp = 0;
    connection->tmp_Out = 0;
    GreyMatter.push_back(connection);
  }
  vector<vector<float>> values(ticks, vector<float>(inputs));
  vector<float> rewards(ticks);
  for (int t = 0; t < ticks; t++)
    for (int i = 0; i < inputs; i++) scanf("%f", &values[t][i]);
  for (int t = 0; t < ticks; t++) scanf("%f", &rewards[t]);

  vector<Neuron_t*> all(NeuralNet);
  all.insert(all.end(), input_layer.begin(), input_layer.end());
  all.push_back(RewardFeederNeuron);
  for (int t = 0; t < ticks; t++) {
    for (int i = 0; i < inputs; i++) {
      input_layer[i]->ValSum = values[t][i];
      input_layer[i]->Output = values[t][i];
      input_layer[i]->Fired = true;
    }
    Global_Refresher(NeuralNet);
    Global_ForwardProcessor(GreyMatter);
    // Each neuron's last transmitted output, the activity the reward is shared by.
    for (Neuron_t* neuron : all) fprintf(out, "%.9g ", neuron->OldOutput);
    fprintf(out, "\n");
    if (rewards[t] != 0) {
      RewardGenerated = rewards[t];
      Global_Adjuster(input_layer);
      Global_Teacher();
      for (Neuron_t* neuron : all) fprintf(out, "%.9g ", neuron->LocalReward);
      fprintf(out, "\n");
      for (auto* connection : GreyMatter) fprintf(out, "%.9g ", connection->Weight);
      fprintf(out, "\n");
      Global_Renew();
    }
  }
  fclose(out);
  return 0;
}
