/**
 * This code performs the backpropagation algorithm in a XOR neural network. It is a 2-2-1 network, with 2 bias nodes, one on the input layer,
 * and another on the hidden layer. The network runs the feedforward algorithm to calculate the outputs, then uses the backprop algorithm to determine
 * how much the weights need to be changed. The code is extremely janky, but it serves its purpose, to a degree. Sometimes it only goes through one 
 * iteration, and other times it takes over 100. I think it depends on the desired output compared to the actual inputs. Those can be easily changed
 * however.
 * @author Patrick Hogan
 */

import java.util.ArrayList;
import java.util.Random;

public class Backpropagation {
	public static double errorGoal = 0.05;
	public static double learningRate = 0.7;
	public static double momentum = 0.3;
	public static double desiredOut = 0.223;
	public static double input1 = 1.0;
	public static double input2 = 0.0;
	public static double bias = 1.0;
	public static int counter = 0;
	public static ArrayList<Double> g = new ArrayList<Double>();
	public static void main(String [] args) {
		//output weights
		ArrayList<Double> output = new ArrayList<Double>();
		
		//hidden 1 weights
		ArrayList<Double> hidden1 = new ArrayList<Double>();
		
		//hidden 2 weights
		ArrayList<Double> hidden2 = new ArrayList<Double>();
		
		//tracks the change in weights
		//the first 3 indices (0-2) track the output weights
		//3-5 tracks the hidden1 weights
		//6-8 tracks the hidden2 weights
		ArrayList<Double> iteration = new ArrayList<Double>();
		
	
		/**
		 * assigns random weights to each hidden/output layer
		 */
		for(int i = 0; i < 3; i++) {
			
			
			output.add((Math.random() * 20) - 10);
			//System.out.println("test1 " + output.get(i));
			
			hidden1.add((Math.random() * 20) - 10);
			//System.out.println("test2 " + hidden1.get(i));
			
			hidden2.add((Math.random() * 20) - 10);
			//System.out.println("test3 " + hidden2.get(i));
			
		}
		/**
		 * assigns 0 to each value in the iteration/weightchange list, as to make the backprop algorithm work on the first iteration
		 */
		for(int i = 0; i < 9; i++) {
			iteration.add(0.0);
			//System.out.println(iteration.size());
		}
		
		//goes through iterations, tracking the number of times looped. Ends when the error goal is met.
		while(true) {
			//System.out.println("test");
			double hiddenNeuron1 = feedForward(input1, input2, bias, hidden1);
			//System.out.println("Hidden 1: " + hiddenNeuron1);
			double hidden1Out = sigmoid(hiddenNeuron1);
			//System.out.println("Hidden 1 out: " + hidden1Out);
			double hiddenNeuron2 = feedForward(input1, input2, bias, hidden2);
			//System.out.println("Hidden 2: " + hiddenNeuron2);
			double hidden2Out = sigmoid(hiddenNeuron2);
			//System.out.println("Hidden 2 out: " + hidden2Out);
			double outputNeuron = feedForward(hidden1Out, hidden2Out, bias, output);
			//System.out.println("Output prelim:" + outputNeuron);
			double finalOut = sigmoid(outputNeuron);
			//System.out.println("Final output: " + finalOut);
			double error = calcError(finalOut);
			
			double outputND = nodeDelta(hidden1Out+hidden2Out+bias, error);
			double outputGradient1 = gradientCalc(hidden1Out, outputND);
			double outputGradient2 = gradientCalc(hidden2Out, outputND);
			double outputGradient3 = gradientCalc(bias, outputND);
			double hidden1ND = nodeDelta(input1+input2+bias, error);
			double hidden1Gradient1 = gradientCalc(input1, hidden1ND);
			double hidden1Gradient2 = gradientCalc(input2, hidden1ND);
			double hidden1Gradient3 = gradientCalc(bias, hidden1ND);
			double hidden2ND = nodeDelta(hidden2Out, error);
			double hidden2Gradient1 = gradientCalc(input1, hidden2ND);
			double hidden2Gradient2 = gradientCalc(input2, hidden2ND);
			double hidden2Gradient3 = gradientCalc(bias, hidden2ND);
			
			//prints the weights
			System.out.println("WEIGHTS: \n"+
							   "Hidden 1: \n" +
							   hidden1 +"\n"+
							   "Hidden 2: \n"+
							   hidden2 + "\n" +
							   "Output: \n" +
							   output);			
			//System.out.println(error);
			System.out.println("GRADIENTS: \n" +
								"Hidden 1: \n" +
								hidden1Gradient1 +"\n"+
								hidden1Gradient2 +"\n"+
								hidden1Gradient3 +"\n"+
								"Hidden 2: \n"+
								hidden2Gradient1 + "\n" +
								hidden2Gradient2 + "\n" +
								hidden2Gradient3 + "\n" +
								"Output: \n" +
								outputGradient1 + "\n" +
								outputGradient2 + "\n" +
								outputGradient3);			
			counter++;
			
			System.out.println(" \n"+
								" \n"+
								"ITERATIONS: " + counter);
			
			//backprop loops
			//absolute jank
			for(int i = 0; i < 3; i++) {
				if(i == 0) { 
					double weightChange = backProp(outputGradient1, iteration.get(i));
					iteration.set(i, weightChange);
					output.set(i, output.get(i)+weightChange);
				}
				if(i == 1) { 
					double weightChange = backProp(outputGradient2, iteration.get(i));
					iteration.set(i, weightChange);
					output.set(i, output.get(i)+weightChange);
				}
				if(i == 2) { 
					double weightChange = backProp(outputGradient3, iteration.get(i));
					iteration.set(i, weightChange);
					output.set(i, output.get(i)+weightChange);
				}
				
			}
			for(int i = 3; i < 6; i++) {
				if(i == 3) { 
					double weightChange = backProp(hidden1Gradient1, iteration.get(i));
					iteration.set(i, weightChange);
					hidden1.set(i-3, hidden1.get(0)+weightChange);
				}
				if(i == 4) { 
					double weightChange = backProp(hidden1Gradient2, iteration.get(i));
					iteration.set(i, weightChange);
					hidden1.set(i-3, hidden1.get(1)+weightChange);
				}
				if(i == 5) { 
					double weightChange = backProp(hidden1Gradient3, iteration.get(i));
					iteration.set(i, weightChange);
					hidden1.set(i-3, hidden1.get(2)+weightChange);
				}
			}
			for(int i = 6; i < 9; i++) {
				if(i == 6) { 
					double weightChange = backProp(hidden2Gradient1, iteration.get(i));
					iteration.set(i, weightChange);
					hidden2.set(i-6, hidden2.get(0)+weightChange);
				}
				if(i == 7) { 
					double weightChange = backProp(hidden2Gradient2, iteration.get(i));
					iteration.set(i, weightChange);
					hidden2.set(i-6, hidden2.get(1)+weightChange);
				}
				if(i == 8) { 
					double weightChange = backProp(hidden2Gradient3, iteration.get(i));
					iteration.set(i, weightChange);
					hidden2.set(i-6, hidden2.get(2)+weightChange);
				}
			}
			
			if (error < errorGoal) {
				break;
			}
		}
		
		
	}
	static double sigmoid(double x) {
		double sigmoidDenom = 1 + Math.pow(Math.E, -x);
		double sigmoidFinal = 1/sigmoidDenom;
		
		return sigmoidFinal;
	}
	static double calcError(double x) {
		double error1 = x - desiredOut;
		double error2 = Math.pow(error1, 2);
		return error2;
	}
	//x is input 1, y is input 2, z is bias, arraylist is weights
	static double feedForward(double x, double y, double z, ArrayList<Double> k) {
		int j = k.size();
		double part1 = 0.0;
		double part2 = 0.0;
		double part3 = 0.0;
		for (int i = 0; i < j; i++) {
			if(i == 0) {
				part1 = x*k.get(i);
			} else if (i == 1) {
				part2 = y*k.get(i);
			} else {
				part3 = z*k.get(i);
			}
		}
		double d = part1+part2+part3;
		//System.out.println(part1);
		//System.out.println(part2);
		//System.out.println(part3);
		//System.out.println(d);
		return d;
		
	}
	static double sigmoidDeriv(double x) {
		double sigDer = sigmoid(x);
		double sigDer2 = (1 - sigmoid(x));
		double sigDerFinal = sigDer*sigDer2;
		return sigDerFinal;
	}
	//where x is the value you're calculating the delta for and y is the error
	static double nodeDelta(double x, double y) {
		double node1 = sigmoidDeriv(x);
		double node2 = -y*node1;
		return node2;
	}
	//where x is the relevant node delta and y is the relevant output
	static double gradientCalc(double x, double y) {
		double gradient = x*y;
		return gradient;
	}
	
	//x is the gradient, y is the change in weight (begins as 0)
	static double backProp(double x, double y) {
		double weightChange = learningRate*x + momentum*y;
		return weightChange;
	}
	
}