/*
* This program implements an N-layer perceptron with a specified number of nodes in each layer. The running 
* mode runs the network for a given training set, network configuration, and set of initial weights, and 
* then calculates and prints the outputs. The training mode repeatedly runs the network and adjusts the weights 
* using backpropagation/optimized gradient descent to minimize the error. Configuration, inputs, and weights 
* can be saved or loaded in from a file. Weights may be randomized.
* 
* @author Juliana Li
* @version 4/15/2024
*
* Table of contents (all methods):
* - void setConfig()
* - DARRAY2D allocate2DArray(int x, int y)
* - void allocateArrays()
* - double randNum(double min, double max)
* - void randWeights()
* - bool loadWeights()
* - bool loadCases()
* - bool populateArrays()
* - void printTruthTable(DARRAY2D outputs)
* - void echoParams()
* - double sigmoid(double num)
* - double func(double num)
* - double derivFunc(double num)
* - double calcError(DARRAY1D result, DARRAY1D truth)
* - void run1Set(int trainSet)
* - void runForTrain(int trainSet)
* - void run()
* - void train1Set(int trainSet)
* - void printTime(double seconds)
* - void printEnd()
* - void reportResults()
* - void train()
* - void trainOrNo()
* - void saveWeights()
* - int main(int argc, char *argv[])
*/
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <sstream>
#include <chrono>

#define MS_PER_SEC   1000.0
#define SEC_PER_MIN  60.0
#define MIN_PER_HOUR 60.0
#define HOUR_PER_DAY 24.0
#define DAY_PER_WEEK 6.0
#define DOUBLE_PREC  17.0

using namespace std;

/*
* Define types for 1D, 2D, and 3D double arrays for easier readability.
*/
typedef double*   DARRAY1D;
typedef double**  DARRAY2D;
typedef double*** DARRAY3D;

string configFile;    // File name of the configuration file

bool trainFlag;       // Flag for training or running; 1 = train, 0 = run
bool randFlag;        // Flag for randomizing or loading weights; 1 = rand, 0 = load.
bool saveFlag;        // Flag for saving or not saving the weights to a file; 1 = save, 0 = don't save
string loadFileName;  // Name of the file to load weights from
string saveFileName;  // Name of the file to save weights to
string inputFileName; // Name of file to read inputs for test cases from
string outputFileName;// Name of file to read outputs for test cases from
int* netConfig;       // Network configuration containing number of nodes in each layer
int numLayers;        // Number of connectivity layers
DARRAY2D a;           // Array of all activations
DARRAY2D inCases;     // Inputs for the test cases
DARRAY2D outCases;    // Outputs for the test cases
DARRAY2D allOutputs;  // Stores the outputs for each test case

DARRAY3D w;           // 3D jagged array of all weights

double minWeight;     // Minimum value of the random weights generated
double maxWeight;     // Maximum value of the random weights generated

int testCases;        // Number of test cases used for training
int iter;             // Current number of iterations in training
int maxIters;         // Maximum number of iterations before training is stopped
double totalError;    // The sum of the error across all test cases
double avgError;      // The average error across test cases
double errorThresh;   // Minimum error value to be reached for training to stop
double lambda;        // Learning factor used in training
int keepAlive;        // Number of iterations between keep-alive message, 0 if no output
double totalTime;     // The total time elapsed in training, in seconds
DARRAY2D thetas;      // Array of thetas
DARRAY2D psis;        // Array of psi values

/*
* Sets the configuration parameters for the network by reading from a configuration file.
*/
void setConfig()
{
   ifstream in(configFile);
   
   string line, property, value;
   string delim = "=";
   string hyphen = "-";
   int delimPos, n;

   while(getline(in, line))
   {
      delimPos = line.find(delim);

      if (delimPos == string::npos) continue;

      property = line.substr(0, delimPos - 1);
      value = line.substr(delimPos + 2);

      if (property == "TRAIN_FLAG") 
         trainFlag = stoi(value);
      else if (property == "RAND_FLAG")
         randFlag = stoi(value);
      else if (property == "SAVE_FLAG")
         saveFlag = stoi(value);
      else if (property == "NUM_LAYERS")
      {
         numLayers = stoi(value);
         netConfig = new int[numLayers + 1];
      }
      else if (property == "LAYER_CONFIG")
      {
         for (int n = 0; n <= numLayers; n++)
         {
            netConfig[n] = stoi(value.substr(0, value.find(hyphen)));
            value = value.substr(value.find(hyphen) + 1);
         }
      }
      else if (property == "MIN_WEIGHT")
         minWeight = stod(value);
      else if (property == "MAX_WEIGHT")
         maxWeight = stod(value);
      else if (property == "TEST_CASES")
         testCases = stoi(value);
      else if (property == "MAX_ITERATIONS")
         maxIters = stoi(value);
      else if (property == "ERROR_THRESHOLD")
         errorThresh = stod(value);
      else if (property == "LAMBDA")
         lambda = stod(value);
      else if (property == "KEEP_ALIVE")
         keepAlive = stoi(value);
      else if (!randFlag && property == "LOAD_FILE_NAME")
         loadFileName = value;
      else if (saveFlag && property == "SAVE_FILE_NAME")
         saveFileName = value;
      else if (property == "INPUT_FILE_NAME")
         inputFileName = value;
      else if (property == "OUTPUT_FILE_NAME")
         outputFileName = value;
   } // while(getline(in, line))
} // void setConfig()

/*
* Allocates memory for a 2D array with given dimensions.
*/
DARRAY2D allocate2DArray(int x, int y)
{
   DARRAY2D array = new DARRAY1D[x];

   for (int xi = 0; xi < x; xi++) 
      array[xi] = new double[y];

   return array;
}

/*
* Allocates memory for arrays used, and allocates certain arrays only if in training mode.
*/
void allocateArrays()
{
   a = new DARRAY1D[numLayers + 1];
   for (int n = 0; n <= numLayers; n++)
      a[n] = new double[netConfig[n]];

   w = new DARRAY2D[numLayers];
   for (int n = 0; n < numLayers; n++)
      w[n] = allocate2DArray(netConfig[n], netConfig[n + 1]);
   
   inCases = allocate2DArray(testCases, netConfig[0]);
   outCases = allocate2DArray(testCases, netConfig[numLayers]);
   allOutputs = allocate2DArray(testCases, netConfig[numLayers]);

   if (trainFlag)
   {
      thetas = new DARRAY1D[numLayers];
      for (int n = 1; n < numLayers; n++)
         thetas[n] = new double[netConfig[n]];
      
      psis = new DARRAY1D[numLayers + 1];
      for (int n = 1; n <= numLayers; n++)
         psis[n] = new double[netConfig[n]];
   } // if (trainFlag)
} // void allocateArrays()

/*
* Generates a random number between a given min and max using a C++ pseudo-random generator.
*/
double randNum(double min, double max)
{
   random_device random;
   mt19937 rng(random());
   uniform_real_distribution<double> distrib(min, max);
   return distrib(rng);
}

/*
* Generates random weights between the max and min weight configuration.
*/
void randWeights()
{
   for (int n = 0; n < numLayers; n++)
      for (int k = 0; k < netConfig[n]; k++)
         for (int j = 0; j < netConfig[n + 1]; j++)
            w[n][k][j] = randNum(minWeight, maxWeight);
}

/*
* Loads weights into the weight arrays from a file. If loading and the loaded weights arrays do 
* not match the network configuration, or if the file to be loaded does not exist, a message will 
* be printed and a value of false will be returned, meaning population of arrays has failed. 
* Otherwise returns true, meaning population of arrays has worked and the program will continue.
*/
bool loadWeights()
{
   cout << loadFileName << endl;
   ifstream in(loadFileName, ios::out | ios::binary);
   bool success = true;

   if (!in.good())
   {
      cout << "Weights file to be loaded does not exist. Running/training will not be executed." << endl;
      success = false;
   }

   int numNodes;
   for (int n = 0; n <= numLayers; n++)
   {
      in.read((char*) &numNodes, sizeof(int));
      
      if (success && netConfig[n] != numNodes)
      {
         cout << "Loaded weights do not match current network configuration. Running/training will not be executed." << endl;
         success = false;
      }
   } // for (int n = 0; n <= numLayers; n++)

   if (success)
      for (int n = 0; n < numLayers; n++)
         for (int k = 0; k < netConfig[n]; k++)
            for (int j = 0; j < netConfig[n + 1]; j++)
               in.read((char*) &w[n][k][j], sizeof(double));

   in.close();
   return success;
} // bool loadWeights()

/*
* Loads in the inputs & outputs for a truth table from a file. Only loads inputs for running mode. If files do not
* exist, an error message is printed and running/training is not executed.
*/
bool loadCases()
{
   ifstream in(inputFileName);
   string line;
   bool success = true;
   int set;

   if (!in.good())
   {
      cout << "Input file to be loaded does not exist. Running/training will not be executed." << endl;
      success = false;
   }

   set = 0;
   while (getline(in, line) && set < testCases)
   {
      istringstream iss(line);

      for (int k = 0; k < netConfig[0]; k++)
         iss >> inCases[set][k];

      set++;
   }

   if (trainFlag)
   {
      ifstream in(outputFileName);

      if (!in.good())
      {
         cout << "Output file to be loaded does not exist. Running/training will not be executed." << endl;
         success = false;
      }

      set = 0;
      while (getline(in, line) && set < testCases)
      {
         istringstream iss(line);

         for (int i = 0; i < netConfig[numLayers]; i++)
            iss >> outCases[set][i];

         set++;
      }
   } // if (trainFlag)

   return success;
} // void loadCases()

/*
* Populates the arrays, including the weights (randomized or loaded), and the training input and output cases, 
* loaded from a file. Returns true if arrays are populated successfully, false otherwise.
*/
bool populateArrays()
{
   bool success = true;

   if (randFlag)
      randWeights(); 
   else 
      success = loadWeights();

   success = success && loadCases();

   return success;
} // bool populateArrays()

/*
* Outputs each test case in a truth table along with a specified array of outputs.
*/
void printTruthTable(DARRAY2D outputs)
{
   cout << "Truth Table:" << endl;
   streamsize defaultPrecision = cout.precision();

   for (int set = 0; set < testCases; set++)
   {
      for (int k = 0; k < netConfig[0]; k++)
         cout << setprecision(DOUBLE_PREC) << inCases[set][k] << " ";

      cout << ": ";
      
      for (int i = 0; i < netConfig[numLayers]; i++)
         cout << setprecision(DOUBLE_PREC) << outputs[set][i] << " ";

      cout << endl;
   } // for (int set = 0; set < testCases; set++)

   cout.precision(defaultPrecision);
   cout << endl;
} // void printTruthTable(DARRAY2D outputs)

/*
* Prints all the outputs after running the network.
*/
void printOutputs(DARRAY2D outputs)
{
   streamsize defaultPrecision = cout.precision();

   for (int set = 0; set < testCases; set++)
   {
      for (int i = 0; i < netConfig[numLayers]; i++)
         cout << fixed << setprecision(3) << outputs[set][i] << "\t";

      cout << endl;
   } // for (int set = 0; set < testCases; set++)

   cout.precision(defaultPrecision);
   cout << endl;
} // void printTruthTable(DARRAY2D outputs)

/*
* Prints the network configurations and training cases, and if in trading mode, prints the random number range, 
* maximum number of iterations, error threshold, and lambda value. Also prints if loading or randomizing weights, 
* and if saving weights or not.
*/
void echoParams()
{
   cout << endl << "Network Configuration: ";
   
   for (int n = 0; n < numLayers; n++)
      cout << netConfig[n] << "-";
   
   cout << netConfig[numLayers] << endl << endl;
   
   if (!randFlag)
      cout << "Loading weights from: " << loadFileName << endl;
   else
      cout << "Randomizing weights." << endl;

   if (saveFlag)
      cout << "Saving weights to: " << saveFileName << endl;
   else
      cout << "Not saving weights." << endl << endl;

   if (trainFlag)
   {
      streamsize defaultPrecision = cout.precision();

      printOutputs(outCases);
      cout << setprecision(1) << "Random Num Range: " << minWeight << " to " << maxWeight << endl;
      cout << "Max Iterations:   " << maxIters << endl;
      cout.precision(defaultPrecision);
      cout << "Error Threshold:  " << errorThresh << endl;
      cout << setprecision(1) << "Lambda:           " << lambda << endl << endl;

      cout.precision(defaultPrecision);
   }
} // void echoParams()

/*
* Calculates the sigmoid function of a number, given by 1/(1+e^-num).
*/
double sigmoid(double num)
{
   return 1.0 / (1.0 + exp(-num));
}

/*
* Calculates the hyperbolic tangent of a number.
*/
double tanh(double num)
{
   int v = exp(num);
   return (v - 1.0 / v) / (v + 1.0 / v);
}

/*
* Applies the current activation function to a given number.
*/
double func(double num)
{
   return sigmoid(num);
}

/*
* Takes the partial derivative of the sigmoid of a number.
*/
double derivSigmoid(double num)
{
   num = sigmoid(num);
   return num * (1.0 - num);
}

/*
* Takes the partial derivative of the hyperbolic tangent of a number.
*/
double derivTanh(double num)
{
   num = tanh(num);
   return 1.0 - num * num;
}

/*
* Takes the partial derivative of the activation function of a number.
*/
double derivFunc(double num)
{
   return derivSigmoid(num);
}

/*
* Calculates the error for a given result using the formula Error = 1/2*(T-F)^2.
*/
double calcError(DARRAY1D result, DARRAY1D truth)
{
   double error = 0.0;

   for (int i = 0; i < netConfig[numLayers]; i++)
      error += (truth[i] - result[i]) * (truth[i] - result[i]) / 2.0;

   return error;
}

/*
* Runs the network for 1 test case by calculating activation values for each layer.
*/
void run1Set(int trainSet)
{
   double thetaTemp;

   for (int n = 1; n <= numLayers; n++)
   {
      for (int j = 0; j < netConfig[n]; j++)
      {
         thetaTemp = 0.0;

         for (int k = 0; k < netConfig[n - 1]; k++)
            thetaTemp += a[n - 1][k] * w[n - 1][k][j];

         a[n][j] = func(thetaTemp);
      }
   } // for (int n = 1; n <= numLayers; n++)
} // void run1Set(int trainSet)

/*
* In training mode, this method runs the network for 1 test case in the same way as for 
* running mode, except it saves values necessary for training.
*/
void runForTrain(int trainSet)
{
   for (int n = 1; n < numLayers; n++)
   {
      for (int j = 0; j < netConfig[n]; j++)
      {
         thetas[n][j] = 0.0;

         for (int k = 0; k < netConfig[n - 1]; k++)
            thetas[n][j] += a[n - 1][k] * w[n - 1][k][j];

         a[n][j] = func(thetas[n][j]);
      }
   } // for (int n = 1; n <= numLayers; n++)

   double thetaOut;
   for (int i = 0; i < netConfig[numLayers]; i++)
   {
      thetaOut = 0.0;

      for (int j = 0; j < netConfig[numLayers - 1]; j++)
         thetaOut += a[numLayers - 1][j] * w[numLayers - 1][j][i];

      a[numLayers][i] = func(thetaOut);
      psis[numLayers][i] = (outCases[trainSet][i] - a[numLayers][i]) * derivFunc(thetaOut);
   } // for (int i = 0; i < netConfig[numLayers]; i++)
} // void runForTrain(int trainSet)

/*
* Loads the inputs for a given test case into the input activations.
*/
void loadInputs(int set)
{
   for (int k = 0; k < netConfig[0]; k++)
      a[0][k] = inCases[set][k];
}

/*
* Runs the network for all the test cases.
*/
void run()
{
   for (int set = 0; set < testCases; set++)
   {
      loadInputs(set);

      run1Set(set);

      for (int i = 0; i < netConfig[numLayers]; i++)
         allOutputs[set][i] = a[numLayers][i];
   } // for (int set = 0; set < testCases; set++)
} // void run()

/*
* Trains the network for 1 test case by adjusting the weights using gradient (steepest) descent.
*/
void train1Set(int trainSet)
{
   double omega;
   for (int n = numLayers - 1; n > 1; n--)
   {
      for (int k = 0; k < netConfig[n]; k++)
      {
         omega = 0.0;
         for (int j = 0; j < netConfig[n + 1]; j++)
         {
            omega += psis[n + 1][j] * w[n][k][j];
            w[n][k][j] += lambda * a[n][k] * psis[n + 1][j];
         }

         psis[n][k] = omega * derivFunc(thetas[n][k]);
      } // for (int k = 0; k < netConfig[n]; k++)
   } // for (int n = numLayers - 1; n > 1; n--)

   int n = 1;
   for (int k = 0; k < netConfig[n]; k++)
   {
      omega = 0.0;
      for (int j = 0; j < netConfig[n + 1]; j++)
      {
         omega += psis[n + 1][j] * w[n][k][j];
         w[n][k][j] += lambda * a[n][k] * psis[n + 1][j];
      }

      psis[n][k] = omega * derivFunc(thetas[n][k]);

      for (int m = 0; m < netConfig[n - 1]; m++)
         w[n - 1][m][k] += lambda * a[n - 1][m] * psis[n][k];
   } // for (int k = 0; k < netConfig[n]; k++)

   run1Set(trainSet);

   totalError += calcError(a[numLayers], outCases[trainSet]);
} // void train1Set(int trainSet)

/*
* Accept a value representing seconds elapsed and print out a decimal value in easier to digest units.
* Code provided by Dr. Nelson on Schoology.
*/
void printTime(double seconds)
{
   double minutes, hours, days, weeks;

   printf("Elapsed time: ");

   if (seconds < 1.)
      printf("%g milliseconds", seconds * MS_PER_SEC);
   else if (seconds < SEC_PER_MIN)
      printf("%g seconds", seconds);
   else
   {
      minutes = seconds / SEC_PER_MIN;

      if (minutes < MIN_PER_HOUR)
         printf("%g minutes", minutes);
      else
      {
         hours = minutes / MIN_PER_HOUR;

         if (hours < HOUR_PER_DAY)
            printf("%g hours", hours);
         else
         {
            days = hours / HOUR_PER_DAY;

            if (days < DAY_PER_WEEK)
               printf("%g days", days);
            else
            {
               weeks = days / DAY_PER_WEEK;

               printf("%g weeks", weeks);
            }
         } // if (hours < HOUR_PER_DAY)...else
      } // if (minutes < MIN_PER_HOUR)...else
   } // if (seconds < 1.)...else if (seconds < SEC_PER_MIN)...else

   printf("\n\n");
   return;
} // void printTime(double seconds)

/*
* Prints the reason for exiting training, the iterations reached, and the average error reached.
*/
void printEnd()
{
   cout << "Reason for Exiting: ";

   if (avgError <= errorThresh) cout << "average error is less than " << errorThresh << endl;
   if (iter >= maxIters) cout << "iterations exceeded " << maxIters << endl;

   cout << "Iterations Reached: " << iter << endl;
   cout << "Avg Error Reached:  " << setprecision(DOUBLE_PREC) << avgError << endl << endl;
   printTime(totalTime / 1000.0);
} // void printEnd()

/*
* Reports the results of either running or training, including the
* end training info if training, and then the truth table.
*/
void reportResults()
{
   if (trainFlag) 
   {
      cout << "TRAINING RESULTS-------------------------------------" << endl;
      printEnd();
   }
   else 
      cout << "RUNNING RESULTS--------------------" << endl;

   printOutputs(allOutputs);
} // void reportResults()

/*
* Echoes the training parameters, trains the network by repeatedly adjusting the weights until
* either the average error is below the threshold or the number of iterations exceeds the maximum.
* Runs one more time after training so that the correct results will be reported for updated weights.
*/
void train()
{
   iter = 0;
   avgError = errorThresh + 1;

   while (avgError > errorThresh && iter < maxIters)
   {
      totalError = 0.0;

      for (int set = 0; set < testCases; set++)
      {
         loadInputs(set);

         runForTrain(set);
         train1Set(set);
      } // for (int set = 0; set < testCases; set++)

      avgError = totalError / ((double) testCases);
      iter++;

      if (keepAlive && !(iter % keepAlive))
         cout << "Iteration " << iter << ", Error = " << avgError << endl;
   } // while (avgError > errorThresh && iter < maxIters)
} // void train()

/*
* Enters training mode if the train flag is true, only runs otherwise.
*/
void trainOrNo()
{
   if (trainFlag) 
   {
      chrono::steady_clock::time_point beginT = std::chrono::steady_clock::now();
      train(); 
      chrono::steady_clock::time_point endT = std::chrono::steady_clock::now();
      totalTime = chrono::duration_cast<std::chrono::milliseconds>(endT - beginT).count();
   }
} // void trainOrRun()

/*
* Saves the weights to a binary file.
*/
void saveWeights()
{
   ofstream out(saveFileName, ios::out | ios::binary);

   for (int n = 0; n <= numLayers; n++)
      out.write((char*) &netConfig[n], sizeof(int));

   for (int n = 0; n < numLayers; n++)
      for (int k = 0; k < netConfig[n]; k++)
         for (int j = 0; j < netConfig[n + 1]; j++)
            out.write((char*) &w[n][k][j], sizeof(double));

   out.close();
} // void saveWeights()

/*
* The main method runs commands that allocates memory to the arrays, sets the network & training
* configurations, trains the network, and then prints the truth table with outputs. If population
* of arrays fails (loading weights from a file does not work), the rest of the program will not be 
* executed. Takes in a configuration file as an argument, and parses the file for network config.
*/
int main(int argc, char *argv[])
{
   if (argc > 1) 
      configFile = argv[1];
   else
      configFile = "Train_Config.txt"; // Defaults to N-Layer_Config.txt if no file given

   setConfig();
   allocateArrays();

   if (populateArrays())
   {
      echoParams();
      trainOrNo();
      run();

      if (saveFlag) saveWeights();

      reportResults();
   } // if (populateArrays())
} // int main()